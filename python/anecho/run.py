"""anecho-run — process a WAV file through a packed model.

    python3 -m anecho.sdk.run model.anecho in.wav out.wav [--license KEY]
                              [--level 0.8] [--gain 3] [--bypass]

Streams the file through the SDK exactly as a live caller would: block by block,
at the model's own block size, with the state carried. The output is delay-aligned
to the input, so the two files line up in an editor.

Any sample rate and channel count in, the same back out — the model's rate is used
internally and the conversion is linear interpolation, which is honest for a
listening test and not what you would ship in a real pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

from .processor import Model, Processor, ProcessorConfig, ProcessorParameter

DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}


def read_wav(path: str):
    if not os.path.exists(path):
        here = sorted(f for f in os.listdir(os.path.dirname(path) or ".")
                      if f.lower().endswith(".wav"))
        hint = ("\n  WAVs in that directory: " + ", ".join(here[:8])) if here else \
               "\n  (no .wav files in that directory — record one with anecho.sdk.mic_demo)"
        raise SystemExit(f"no such file: {path}{hint}")
    try:
        w = wave.open(path, "rb")
    except wave.Error as e:
        raise SystemExit(f"{path}: not a PCM WAV ({e}). Convert it first:\n"
                         f"  ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le out.wav")
    with w:
        if w.getsampwidth() not in DTYPES:
            raise SystemExit(f"{path}: {8*w.getsampwidth()}-bit PCM is not supported")
        dt = DTYPES[w.getsampwidth()]
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=dt)
        x = raw.reshape(-1, w.getnchannels()).astype(np.float32) / float(np.iinfo(dt).max)
        return x, w.getframerate(), w.getsampwidth()


def write_wav(path: str, x: np.ndarray, rate: int, width: int = 2) -> None:
    dt = DTYPES[width]
    peak = float(np.abs(x).max())
    if peak > 1.0:                                  # never wrap round on clipping
        print(f"  output peaked at {20*np.log10(peak):.1f} dBFS — scaling to fit")
        x = x / peak
    data = (np.clip(x, -1.0, 1.0) * np.iinfo(dt).max).astype(dt)
    with wave.open(path, "wb") as w:
        w.setnchannels(1 if data.ndim == 1 else data.shape[1])
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def align(y: np.ndarray, delay: int, length: int) -> np.ndarray:
    """Shift the output back by the algorithmic delay and fade the wind-up in.

    The first `delay` samples are the pipeline filling up; a fade of exactly that
    length removes the start transient without touching anything real.
    """
    y = np.concatenate([y[delay:], np.zeros(delay, dtype=np.float32)])[:length]
    if delay and len(y) > delay:
        y = y.copy()
        y[:delay] *= np.linspace(0.0, 1.0, delay, dtype=np.float32)
    return y


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--license", default="", help="license key, if the build enforces one")
    ap.add_argument("--level", type=float, default=1.0, help="0..1 dry/wet")
    ap.add_argument("--gain", type=float, default=0.0, help="dB on the enhanced signal")
    ap.add_argument("--bypass", action="store_true")
    args = ap.parse_args(argv)

    model = Model.from_file(args.model)
    proc = Processor(model, args.license)
    ctx = proc.get_context()
    ctx.set_parameter(ProcessorParameter.EnhancementLevel, args.level)
    ctx.set_parameter(ProcessorParameter.VoiceGain, args.gain)
    ctx.set_parameter(ProcessorParameter.Bypass, args.bypass)

    x, rate, width = read_wav(args.input)
    mono = x.mean(1)
    print(f"{args.input}: {len(mono)/rate:.1f} s, {rate} Hz, {x.shape[1]} ch")
    print(f"model {model.get_id()} at {model.sample_rate} Hz, "
          f"delay {model.audio_delay()} samples ({model.audio_delay()*1000//model.sample_rate} ms)")

    sig = resample(mono, rate, model.sample_rate)
    block = proc.config.block_size
    pad = (-len(sig)) % block
    sig = np.concatenate([sig, np.zeros(pad, dtype=np.float32)])
    out = np.concatenate([proc.process(sig[i:i + block]) for i in range(0, len(sig), block)])

    out = align(out, model.audio_delay(), len(sig) - pad)
    out = resample(out, model.sample_rate, rate)[:len(mono)]

    rms = lambda a: 20 * np.log10(float(np.sqrt((a ** 2).mean())) + 1e-12)   # noqa: E731
    print(f"level: in {rms(mono):.1f} dBFS -> out {rms(out):.1f} dBFS")
    write_wav(args.output, out, rate, width)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
