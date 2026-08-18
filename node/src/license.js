'use strict';
// License verification — the same token format and semantics as anecho/sdk/license.py,
// so one keytool serves both runtimes. A token is
//
//     base64url(payload_json) + "." + base64url(signature)
//
// signed either with ed25519 (vendor holds the private key; only the PUBLIC key is
// embedded here, so a full reverse-engineering of this file forges nothing) or with
// hmac-sha256 (shared secret — dev/test only, the secret would ship in this file).
// Node >= 12.11 verifies ed25519 natively; there are no dependencies to audit.
const crypto = require('crypto');

class AnechoError extends Error {}
class LicenseError extends AnechoError {}
class LicenseFormatInvalidError extends LicenseError {}
class LicenseExpiredError extends LicenseError {}
class ProcessingNotAllowedError extends LicenseError {}

const b64d = (s) => Buffer.from(s, 'base64url');

/** Raw 32-byte ed25519 public key -> KeyObject, via the fixed SPKI DER prefix. */
function ed25519PublicKey(raw) {
  if (raw.length !== 32) throw new LicenseFormatInvalidError('verify key must be 32 bytes');
  const spki = Buffer.concat([Buffer.from('302a300506032b6570032100', 'hex'), raw]);
  return crypto.createPublicKey({ key: spki, format: 'der', type: 'spki' });
}

/**
 * Verify a license token. Returns the claims object, or throws
 * LicenseFormatInvalidError / LicenseExpiredError / ProcessingNotAllowedError —
 * the same three failure classes, for the same reasons, as the Python runtime.
 */
function verifyLicense(token, verifyKey, product, feature = 'enhance', now = null) {
  let payload, sig, claims;
  try {
    const [pjB64, sigB64, extra] = String(token).trim().split('.');
    if (!pjB64 || !sigB64 || extra !== undefined) throw new Error('shape');
    payload = b64d(pjB64);
    sig = b64d(sigB64);
    claims = JSON.parse(payload.toString('utf8'));
  } catch {
    throw new LicenseFormatInvalidError('malformed license token');
  }

  const key = Buffer.isBuffer(verifyKey) ? verifyKey : Buffer.from(verifyKey);
  let ok = false;
  if (claims.alg === 'ed25519') {
    try { ok = crypto.verify(null, payload, ed25519PublicKey(key), sig); } catch { ok = false; }
  } else if (claims.alg === 'hmac') {
    const mac = crypto.createHmac('sha256', key).update(payload).digest();
    ok = mac.length === sig.length && crypto.timingSafeEqual(mac, sig);
  } else {
    throw new LicenseFormatInvalidError(`unsupported alg '${claims.alg}'`);
  }
  if (!ok) throw new LicenseFormatInvalidError('signature does not verify');

  const t = now !== null ? now : Date.now() / 1000;
  if (!((claims.issued ?? 0) <= t && t <= (claims.expires ?? 0))) {
    throw new LicenseExpiredError('license outside its validity window');
  }
  if (claims.product !== product) {
    throw new ProcessingNotAllowedError(`license not valid for product '${product}'`);
  }
  if (!Array.isArray(claims.features) || !claims.features.includes(feature)) {
    throw new ProcessingNotAllowedError(`feature '${feature}' not licensed`);
  }
  return claims;
}

module.exports = {
  verifyLicense,
  AnechoError, LicenseError,
  LicenseFormatInvalidError, LicenseExpiredError, ProcessingNotAllowedError,
};
