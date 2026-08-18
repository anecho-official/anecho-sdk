# Anecho SDK

Primary-speaker isolation that runs in **your** process: 16 kHz, 15 ms of
algorithmic delay, 5.3 MB model, real-time on one CPU core.

- **`node/`** — the npm package [`@anecho-official/sdk`](https://www.npmjs.com/package/@anecho-official/sdk) (Node ≥ 18, zero dependencies, CLI `anecho` included)
- **`python/`** — the PyPI package [`anecho`](https://pypi.org/project/anecho/) (numpy + torch)

Audio never leaves the process — you can read that invariant straight from
`telemetry` in either runtime: the reporter sends processed durations, a model
id and an SDK version, nothing else. That heartbeat is what keeps your licence
token fresh; the token TTL is your offline grace.

Get an API key and your model file at https://app.anecho.ai · docs at
https://anecho.ai/docs/sdk · benchmark data at https://anecho.ai/benchmark.json
