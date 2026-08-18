"""
anecho licensing — offline, verifiable at runtime.

A license is a signed token:  base64url(payload_json) + "." + base64url(signature)

Two schemes:
  * ed25519 (recommended, needs `cryptography`): the vendor holds a private
    signing key; the runtime embeds only the PUBLIC key. Licenses cannot be forged
    without the private key, even by someone who fully reverse-engineers the SDK.
  * hmac (fallback, stdlib only): a shared secret both signs and verifies. Simpler,
    but the secret lives in the runtime and is extractable — use ed25519 for a
    shipped product.

Payload claims: product, licensee, features (list), issued (epoch), expires (epoch).
Runtime verification checks the signature, the product, the requested feature, and
the validity window.

This is YOUR license system with YOUR keys — not a reproduction of anyone else's.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from .errors import (LicenseFormatInvalidError, LicenseExpiredError,
                     ProcessingNotAllowedError)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey)
    from cryptography.hazmat.primitives import serialization
    _HAVE_ED = True
except Exception:                                    # cryptography not installed
    _HAVE_ED = False


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ---------------------------------------------------------------- vendor side
class LicenseAuthority:
    """Vendor-side signer. Holds the signing key; issues license tokens."""

    def __init__(self, alg: str, signing_key: bytes):
        self.alg = alg
        self.key = signing_key

    # ---- key generation -------------------------------------------------
    @staticmethod
    def generate(alg: str = "ed25519"):
        """Return (authority, verify_key_bytes). Store the authority's key safely."""
        if alg == "ed25519":
            if not _HAVE_ED:
                raise RuntimeError("install `cryptography` for ed25519, or use alg='hmac'")
            sk = Ed25519PrivateKey.generate()
            priv = sk.private_bytes(serialization.Encoding.Raw,
                                    serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
            pub = sk.public_key().public_bytes(serialization.Encoding.Raw,
                                               serialization.PublicFormat.Raw)
            return LicenseAuthority("ed25519", priv), pub
        elif alg == "hmac":
            secret = hashlib.sha256(_os_urandom(32)).digest()
            return LicenseAuthority("hmac", secret), secret          # verify key == secret
        raise ValueError(alg)

    # ---- issue ----------------------------------------------------------
    def issue(self, product: str, licensee: str, features=("enhance",),
              days_valid: int = 365, issued_at: float | None = None) -> str:
        iat = int(issued_at if issued_at is not None else time.time())
        payload = {
            "alg": self.alg, "product": product, "licensee": licensee,
            "features": list(features), "issued": iat,
            "expires": iat + days_valid * 86400,
        }
        pj = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        sig = self._sign(pj)
        return _b64e(pj) + "." + _b64e(sig)

    def _sign(self, msg: bytes) -> bytes:
        if self.alg == "ed25519":
            return Ed25519PrivateKey.from_private_bytes(self.key).sign(msg)
        return hmac.new(self.key, msg, hashlib.sha256).digest()


# --------------------------------------------------------------- runtime side
def verify_license(token: str, verify_key: bytes, product: str,
                   feature: str = "enhance", now: float | None = None) -> dict:
    """Verify a token against the embedded verify key. Returns the claims dict.

    Raises LicenseFormatInvalidError / LicenseExpiredError / ProcessingNotAllowedError.
    """
    try:
        pj_b64, sig_b64 = token.strip().split(".")
        pj = _b64d(pj_b64)
        sig = _b64d(sig_b64)
        claims = json.loads(pj)
    except Exception:
        raise LicenseFormatInvalidError("malformed license token")

    alg = claims.get("alg")
    ok = False
    if alg == "ed25519" and _HAVE_ED:
        try:
            Ed25519PublicKey.from_public_bytes(verify_key).verify(sig, pj)
            ok = True
        except Exception:
            ok = False
    elif alg == "hmac":
        ok = hmac.compare_digest(hmac.new(verify_key, pj, hashlib.sha256).digest(), sig)
    else:
        raise LicenseFormatInvalidError(f"unsupported alg {alg!r}")
    if not ok:
        raise LicenseFormatInvalidError("signature does not verify")

    t = now if now is not None else time.time()
    if not (claims.get("issued", 0) <= t <= claims.get("expires", 0)):
        raise LicenseExpiredError("license outside its validity window")
    if claims.get("product") != product:
        raise ProcessingNotAllowedError(f"license not valid for product {product!r}")
    if feature not in claims.get("features", []):
        raise ProcessingNotAllowedError(f"feature {feature!r} not licensed")
    return claims


def _os_urandom(n: int) -> bytes:
    import os
    return os.urandom(n)
