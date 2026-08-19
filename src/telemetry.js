'use strict';
// Usage reporting and license refresh — one loop, because they are one mechanism.
// The mirror of anecho/sdk/telemetry.py; see that file's docstring for the model.
// Short version: drain every attached Processor's counter, POST the batch to the
// license endpoint, cache the fresh token it returns. Stop reporting and the token
// stops refreshing; the model stops loading when the cache expires. Nothing here
// interrupts audio already flowing, and nothing derived from the signal ever
// leaves the process — durations, model id, SDK version, random keys. That's all.
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const DEFAULT_ENDPOINT = 'https://app.anecho.ai/api/v1/license/refresh';
const SDK_NAME = 'anecho-node';
const SDK_VERSION = require('../package.json').version;

class UsageReporter {
  constructor(apiKey, {
    endpoint = DEFAULT_ENDPOINT,
    intervalMs = 300_000,
    stateDir = path.join(os.homedir(), '.anecho'),
    timeoutMs = 10_000,
  } = {}) {
    if (!apiKey) throw new Error('apiKey is required');
    this.apiKey = apiKey;
    this.endpoint = endpoint;
    this.intervalMs = intervalMs;
    this.timeoutMs = timeoutMs;
    this.statePath = stateDir ? path.join(stateDir, 'license.json') : null;
    this._procs = [];
    this._carry = [];               // events that failed to send, retried next flush
    this._timer = null;
    this._license = this._loadCached();
  }

  // ---------------------------------------------------------------- license
  /** A valid token — cached when fresh enough, refreshed over the network when not. */
  async ensureLicense(minRemainingS = 600) {
    const lic = this._license;
    if (lic && lic.expires - Date.now() / 1000 > minRemainingS) return lic.token;
    await this.flush();             // refresh implies a report; empty is fine
    const fresh = this._license;
    if (fresh && fresh.expires > Date.now() / 1000) return fresh.token;
    throw new Error('could not obtain a license token (network down and cache expired?)');
  }

  _loadCached() {
    if (!this.statePath) return null;
    try {
      const lic = JSON.parse(fs.readFileSync(this.statePath, 'utf8'));
      return lic.expires > Date.now() / 1000 ? lic : null;
    } catch { return null; }
  }

  _store(lic) {
    this._license = lic;
    if (!this.statePath) return;
    try {
      fs.mkdirSync(path.dirname(this.statePath), { recursive: true });
      const tmp = this.statePath + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(lic));
      fs.renameSync(tmp, this.statePath);
    } catch { /* a cache miss on next start, not a failure */ }
  }

  // ---------------------------------------------------------------- metering
  /** Start counting this processor's output. Safe before or after start(). */
  attach(processor) { this._procs.push(processor); }

  _drain() {
    const events = [];
    for (const p of this._procs) {
      const ms = p.takeProcessedMs();
      if (ms > 0) {
        events.push({
          idempotencyKey: crypto.randomUUID().replace(/-/g, ''),
          modelId: p.model.id,
          durationMs: ms,
          occurredAt: new Date().toISOString(),
        });
      }
    }
    const carry = this._carry;
    this._carry = [];
    return carry.concat(events);
  }

  /** One report + refresh round trip. Returns the server reply, or null offline. */
  async flush() {
    const events = this._drain();
    let reply;
    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({ events, sdk: SDK_NAME, sdkVersion: SDK_VERSION }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      reply = await res.json();
      if (!res.ok && res.status !== 402) throw new Error(`HTTP ${res.status}`);
    } catch {
      // Requeue so the minutes survive the outage; the idempotency keys make an
      // eventual double-send harmless. Cap keeps a dead endpoint from growing
      // memory forever (~7 days at 5-minute flushes).
      this._carry = events.concat(this._carry).slice(0, 2048);
      return null;
    }
    if (reply && reply.token) this._store({ token: reply.token, expires: reply.expires || 0 });
    return reply;
  }

  // ---------------------------------------------------------------- lifecycle
  start() {
    if (this._timer) return;
    this._timer = setInterval(() => { this.flush().catch(() => {}); }, this.intervalMs);
    this._timer.unref();            // never the reason the process stays alive
  }

  async stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    await this.flush();             // the tail of the last interval
  }
}

module.exports = { UsageReporter };
