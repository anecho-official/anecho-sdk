# Anecho SDK

Primary-speaker isolation that runs in **your** process. 16 kHz, 15 ms of
algorithmic delay, real-time on one CPU core, no network in the audio path.

- **Node** (this repo root): pure JavaScript, zero dependencies, Node ≥ 18, CLI included
- **Python** (`python/`): numpy + torch, Python ≥ 3.10
- The two runtimes execute the same model file and are cross-tested to −58 dB
  against each other

Audio never leaves the process — you can read that invariant straight from
`src/telemetry.js` / `python/anecho/telemetry.py`: the usage reporter sends
processed **durations**, a model id and an SDK version. Nothing else, ever.

## Installation

Until the registry releases land, install straight from GitHub:

```bash
# Node — the repo root is the npm package
npm install github:anecho-official/anecho-sdk

# Python
pip install "git+https://github.com/anecho-official/anecho-sdk.git#subdirectory=python"
```

You also need an **API key** and your **model file** (a per-customer `.anecho`)
from [app.anecho.ai](https://app.anecho.ai).

## Quick start

Hear it before you write any code:

```bash
npx anecho mic your-model.anecho --seconds 10   # record, process, play both back
```

Node:

```js
const { Model, Processor, UsageReporter } = require("@anecho-official/sdk");

const reporter = new UsageReporter(process.env.ANECHO_API_KEY);
const model = Model.fromFile("your-model.anecho");
const proc = new Processor(model, await reporter.ensureLicense());
reporter.attach(proc);
reporter.start();

// float32 mono at 16 kHz in, the same number of samples out, 15 ms behind
const enhanced = proc.process(block);
```

Python:

```python
import os
from anecho import Model, Processor, UsageReporter

reporter = UsageReporter(os.environ["ANECHO_API_KEY"])
model = Model.from_file("your-model.anecho")
proc = Processor(model, reporter.ensure_license())
reporter.attach(proc)
reporter.start()

for block in blocks:                 # np.float32 mono @ 16 kHz
    enhanced = proc.process(block)
```

## The streaming contract

- Feed float32 mono at the model's rate (`model.sampleRate` / `model.sample_rate`,
  16 000 Hz). `process()` returns exactly as many samples as you fed.
- The first `model.audioDelay` / `model.audio_delay()` samples of output are the
  algorithmic delay and come back as silence — exactly as the delay says.
- The natural block is one hop: `model.blockSize` / `model.optimal_block_size()`
  (240 samples = 15 ms). Any multiple of it works identically.
- Blocks that are **not** hop multiples require
  `ProcessorConfig(..., variableBlockSize: true)`; the processor then buffers one
  extra hop, and its own `audioDelay` reports the true total (480 samples).
  Streaming output equals one-shot output for every block size — that equality is
  regression-tested.
- One `Processor` = one stream. For a new stream call `getContext().reset()` or
  make a new instance.

## Runtime API

| | Node | Python |
|---|---|---|
| Load a model | `Model.fromFile(path)` | `Model.from_file(path)` |
| Model facts | `id`, `sampleRate`, `audioDelay`, `describe()` | `model_id`, `sample_rate`, `audio_delay()`, `describe()` |
| Create | `new Processor(model, licenseToken[, config])` | `Processor(model, license_token, config=None)` |
| Process | `proc.process(Float32Array) → Float32Array` | `proc.process(ndarray) → ndarray` |
| Metering | `proc.takeProcessedMs()` | `proc.take_processed_ms()` |
| Controls | `proc.getContext()` | `proc.get_context()` |

Live controls via the context — safe mid-stream, no reconnect:

```js
const ctx = proc.getContext();
ctx.setParameter(ProcessorParameter.EnhancementLevel, 0.8); // 0..1 dry/wet
ctx.setParameter(ProcessorParameter.VoiceGain, 3.0);        // dB on the enhanced path
ctx.setParameter(ProcessorParameter.Bypass, true);          // exact passthrough
```

## Licensing and usage reporting — one mechanism

`UsageReporter` is the heartbeat: it counts processed milliseconds from every
attached processor, reports them to your account on an interval, and receives a
fresh short-lived licence token in the same round trip.

```js
const reporter = new UsageReporter(apiKey, {
  intervalMs: 300_000,          // report/refresh cadence
  stateDir: "~/.anecho",        // token cache — survives restarts
});
const token = await reporter.ensureLicense();  // cached when fresh, network when not
reporter.attach(proc);
reporter.start();               // unref'd timer; never keeps your process alive
await reporter.stop();          // final flush on shutdown
```

- **Offline grace = token TTL** (72 h). A network outage shorter than that is
  invisible; the cache at `stateDir` covers restarts.
- Failed reports are retried with idempotency keys — minutes are neither lost
  nor double-billed.
- What leaves the process: durations (ms), model id, SDK name/version, random
  idempotency keys. Never audio, never text, never anything derived from the
  signal.

## Error handling

Both runtimes throw the same taxonomy:

| Error | Meaning |
|---|---|
| `ModelInvalidError` (py) / load throw (js) | file corrupt, tampered, or HMAC stripped |
| `LicenseFormatInvalidError` | token malformed or signature does not verify |
| `LicenseExpiredError` | token outside its validity window — refresh it |
| `ProcessingNotAllowedError` | wrong product or feature not licensed |
| `AudioConfigUnsupportedError` (py) / config throw (js) | rate/block mismatch with the model |

```python
from anecho import errors
try:
    proc = Processor(model, token)
except errors.LicenseExpiredError:
    proc = Processor(model, reporter.ensure_license())
```

## CLI

```
anecho process <model.anecho> <in.wav> <out.wav> [--level 0..1] [--gain dB] [--bypass]
anecho mic     <model.anecho> [--seconds N] [--device N] [--list] [--keep DIR]
anecho inspect <model.anecho>
```

Licence token via `--license TOKEN` or the `ANECHO_LICENSE` environment variable.

## Documentation

- Quickstart: https://anecho.ai/docs
- SDK reference: https://anecho.ai/docs/sdk
- Benchmark data (CC-BY-4.0): https://anecho.ai/benchmark.json
- Dashboard, keys and model files: https://app.anecho.ai

## License

Source-available, all rights reserved — see [LICENSE](LICENSE). Using the SDK
requires an active Anecho subscription; model files are licensed per customer
and may not be redistributed.
