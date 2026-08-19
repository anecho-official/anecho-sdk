#!/usr/bin/env python3
"""Anecho on the input path of a Gemini Live voice agent.

    mic 16 kHz --> anecho Processor, block by block --> Gemini Live uplink
                              Gemini Live downlink, 24 kHz --> speaker

The model hears the room; the agent hears only the primary speaker. Run with
--no-filter to A/B what the agent hears without the enhancer in the path.

Setup:
    pip install google-genai sounddevice
    export ANECHO_LICENSE=...                         # app.anecho.ai
    export GOOGLE_CLOUD_PROJECT=...                   # Vertex AI project
    export GOOGLE_CLOUD_LOCATION=us-central1          # optional (this default)
    export GOOGLE_APPLICATION_CREDENTIALS=sa.json     # service-account JSON
    python examples/agent_gemini.py [model.anecho] [--no-filter]
"""
import argparse
import asyncio
import os
import sys
import threading

import numpy as np
import sounddevice as sd
import torch
from google import genai
from google.genai import types

from anecho import Model, Processor, ProcessorParameter

# One core runs a 240-sample hop in ~3 ms (5x real-time). Torch's default
# thread pool oversubscribes this tiny model and can push a hop PAST its
# 15 ms budget -- and an uplink slower than real time starves Gemini's VAD.
torch.set_num_threads(1)

GEMINI_MODEL = "gemini-live-2.5-flash-native-audio"
IN_SR = 16_000   # Gemini Live's uplink contract: 16 kHz PCM16 mono, and ...
OUT_SR = 24_000  # ... its downlink contract: 24 kHz PCM16 mono.
FRAME_MS = 100   # uplink packet size; the granularity the Live API is built around
DEFAULT_MODEL = "anecho.ai_focus_model_16khz_v4_1.anecho"


def build_processor(model_path: str, bypass: bool) -> Processor:
    model = Model.from_file(model_path)  # your per-customer .anecho file
    proc = Processor(model, os.environ.get("ANECHO_LICENSE", ""))
    # Exact passthrough for A/B listening -- same code path, no processing.
    proc.get_context().set_parameter(ProcessorParameter.Bypass, bypass)
    return proc


async def run_agent(source, sink, flush=lambda: None, *, processor: Processor) -> None:
    """The session loop. Everything audio-device-specific stays outside it.

    source     async iterator of float32 mono blocks at 16 kHz at real-time
               rate, in hop multiples (240 samples = 15 ms) -- or build the
               Processor with ProcessorConfig(..., variable_block_size=True).
    sink(b)    called with each PCM16 chunk of agent audio (24 kHz).
    flush()    called on barge-in; must drop all audio queued for playback.

    The uplink invariant: Gemini Live needs a CONTINUOUS 16 kHz stream at
    real-time rate, with pauses carried as actual silent samples. Its
    server-side VAD is the only thing that ends a turn, and it can only hear
    silence you really send -- send_realtime_input(audio_stream_end=True)
    does not end one. So never gate, skip or drop "silent" blocks; a live
    microphone satisfies this naturally.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit("set GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS for Vertex AI")
    client = genai.Client(vertexai=True, project=project,
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],  # native-audio models answer in audio
        input_audio_transcription=types.AudioTranscriptionConfig(),   # what it heard
        output_audio_transcription=types.AudioTranscriptionConfig(),  # what it said
    )
    frame_bytes = IN_SR * FRAME_MS // 1000 * 2

    async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        print("connected -- talk to it (Ctrl-C to stop)")

        async def uplink() -> None:
            pending = bytearray()
            async for block in source:
                # The whole integration is this line: the agent gets the
                # processed block instead of the raw one. The first
                # audio_delay() samples come back as silence -- that is the
                # 15 ms algorithmic delay; send it like any other audio.
                clean = processor.process(block)
                pending += (np.clip(clean, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                while len(pending) >= frame_bytes:  # coalesce to 100 ms packets
                    await session.send_realtime_input(audio=types.Blob(
                        data=bytes(pending[:frame_bytes]),
                        mime_type=f"audio/pcm;rate={IN_SR}"))
                    del pending[:frame_bytes]

        async def downlink() -> None:
            while True:  # receive() yields ONE turn, then ends; re-enter forever
                async for msg in session.receive():
                    sc = msg.server_content
                    if sc and sc.interrupted:
                        flush()  # barge-in: drop queued playback immediately
                        continue
                    if msg.data:
                        sink(msg.data)  # 24 kHz PCM16 from the agent
                    for tr, who in ((sc and sc.input_transcription, "you"),
                                    (sc and sc.output_transcription, "agent")):
                        if tr and tr.text:
                            print(f"[{who}] {tr.text}", flush=True)

        await asyncio.gather(uplink(), downlink())


async def mic_blocks(block_size: int):
    """Float32 mic blocks at 16 kHz, forever. PortAudio keeps the callback
    firing through pauses, so silence reaches Gemini as real samples."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_audio(indata, _frames, _time, _status):
        loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

    with sd.InputStream(samplerate=IN_SR, channels=1, dtype="float32",
                        blocksize=block_size, callback=on_audio):
        while True:
            yield await queue.get()


class Player:
    """Speaker playback of agent audio, with a flushable buffer for barge-in."""

    def __init__(self) -> None:
        self._buf, self._lock = bytearray(), threading.Lock()
        self._stream = sd.RawOutputStream(samplerate=OUT_SR, channels=1,
                                          dtype="int16", callback=self._pull)
        self._stream.start()

    def _pull(self, out, frames, _time, _status) -> None:
        want = 2 * frames
        with self._lock:
            chunk, self._buf = bytes(self._buf[:want]), self._buf[want:]
        out[:len(chunk)] = chunk
        out[len(chunk):] = bytes(want - len(chunk))  # underrun -> silence

    def play(self, pcm: bytes) -> None:
        with self._lock:
            self._buf += pcm

    def flush(self) -> None:
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gemini Live voice agent with Anecho on its input path")
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL, help=".anecho model file")
    ap.add_argument("--no-filter", action="store_true",
                    help="bypass the enhancer to A/B what the agent hears")
    args = ap.parse_args()

    proc = build_processor(args.model, bypass=args.no_filter)
    print(proc.model.describe(), "| filter", "OFF (bypass)" if args.no_filter else "ON")
    player = Player()
    try:
        asyncio.run(run_agent(mic_blocks(proc.model.hop),
                              player.play, player.flush, processor=proc))
    except KeyboardInterrupt:
        pass
    finally:
        player.close()


if __name__ == "__main__":
    main()
