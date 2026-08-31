"""
Usage reporting and license refresh — one loop, because they are one mechanism.

The reporter drains every attached Processor's sample counter on an interval and
POSTs the batch to the license endpoint, which books the usage and returns a fresh
short-lived token. A process that stops reporting stops receiving tokens and the
model stops loading when the last cached token expires; the token TTL *is* the
offline grace period. Nothing here ever interrupts audio that is already flowing.

What leaves the process: durations in milliseconds, the model id, an SDK version
string, and random idempotency keys. Never audio, never text, never anything
derived from the signal. That is a product promise, not an implementation detail —
keep it true.

Typical wiring:

    reporter = UsageReporter(api_key="sk_...", state_dir="~/.anecho")
    token = reporter.ensure_license()          # cached, or fetched over the network
    proc = Processor(model, token)
    reporter.attach(proc)
    reporter.start()                           # daemon thread, interval flushes
    ...
    reporter.stop()                            # final flush on the way out

Offline behaviour: `ensure_license` serves the cached token while it is valid, so a
network outage shorter than the TTL is invisible. Failed flushes carry their events
over to the next attempt — the counter is drained once and requeued on failure, so
minutes are neither lost nor double-counted (the idempotency key survives the retry).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_ENDPOINT = "https://app.anecho.ai/api/v1/license/refresh"
SDK_NAME = "anecho-python"
SDK_VERSION = "1.0.0"


class UsageReporter:
    def __init__(self, api_key: str,
                 endpoint: str = DEFAULT_ENDPOINT,
                 interval_s: float = 300.0,
                 state_dir: str | Path | None = "~/.anecho",
                 timeout_s: float = 10.0):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.endpoint = endpoint
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self.state_path = (Path(state_dir).expanduser() / "license.json") if state_dir else None
        self._procs: list = []
        self._carry: list[dict] = []      # events that failed to send, retried next flush
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._license: dict | None = self._load_cached()

    # ------------------------------------------------------------------ license
    def ensure_license(self, min_remaining_s: float = 600.0) -> str:
        """A valid token — cached when fresh enough, refreshed over the network when not."""
        lic = self._license
        if lic and lic.get("expires", 0) - time.time() > min_remaining_s:
            return lic["token"]
        self.flush()                       # refresh implies a report; empty is fine
        lic = self._license
        if lic and lic.get("expires", 0) > time.time():
            return lic["token"]
        raise RuntimeError("could not obtain a license token (network down and cache expired?)")

    def _load_cached(self) -> dict | None:
        if not self.state_path or not self.state_path.exists():
            return None
        try:
            lic = json.loads(self.state_path.read_text())
            return lic if lic.get("expires", 0) > time.time() else None
        except Exception:
            return None

    def _store(self, lic: dict) -> None:
        self._license = lic
        if self.state_path:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(lic))
                tmp.replace(self.state_path)
            except Exception:
                pass                       # a cache miss on next start, not a failure

    # ------------------------------------------------------------------ metering
    def attach(self, processor) -> None:
        """Start counting this processor's output. Safe before or after start()."""
        with self._lock:
            self._procs.append(processor)

    def _drain(self) -> list[dict]:
        now = time.time()
        events = []
        with self._lock:
            procs, carry = list(self._procs), self._carry
            self._carry = []
        for p in procs:
            ms = p.take_processed_ms()
            if ms > 0:
                events.append({
                    "idempotencyKey": uuid.uuid4().hex,
                    "modelId": getattr(p.model, "model_id", "anecho"),
                    "durationMs": int(ms),
                    "occurredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                })
        return carry + events

    def flush(self) -> dict | None:
        """One report + refresh round trip. Returns the server reply, or None offline."""
        events = self._drain()
        body = json.dumps({"events": events, "sdk": SDK_NAME, "sdkVersion": SDK_VERSION}).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     # The edge blocks urllib's default Python-urllib/* agent as a
                     # bot (403 before the request reaches the app). Identify as
                     # ourselves — which is also just true.
                     "User-Agent": f"{SDK_NAME}/{SDK_VERSION}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                reply = json.loads(r.read().decode())
        except Exception:
            # Requeue so the minutes survive the outage; the idempotency keys make
            # an eventual double-send harmless.
            with self._lock:
                self._carry = events + self._carry
                # A dead endpoint must not grow memory forever. ~7 days at 5-minute
                # flushes; beyond that the token has long expired anyway.
                del self._carry[2048:]
            return None
        if "token" in reply:
            self._store({"token": reply["token"], "expires": reply.get("expires", 0)})
        return reply

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="anecho-usage", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.flush()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.timeout_s + 1)
            self._thread = None
        self.flush()                       # the tail of the last interval
