# Anecho SDK

Primary-speaker isolation that runs in **your** process: 16 kHz, 15 ms of
algorithmic delay, real-time on one CPU core. Audio never leaves the process —
read that invariant straight from `src/telemetry.js` / `python/anecho/telemetry.py`:
the reporter sends processed durations, a model id and an SDK version, nothing else.

## Install (from GitHub)

Node ≥ 18 — this repo root is the npm package:

    npm install github:anecho-official/anecho-sdk
    npx anecho mic your-model.anecho --seconds 10

Python ≥ 3.10 (needs numpy + torch):

    pip install "git+https://github.com/anecho-official/anecho-sdk.git#subdirectory=python"

```js
const { Model, Processor, UsageReporter } = require("@anecho-official/sdk");
```
```python
from anecho import Model, Processor, UsageReporter
```

Get an API key and your model file at https://app.anecho.ai · docs at
https://anecho.ai/docs/sdk · benchmark data at https://anecho.ai/benchmark.json
