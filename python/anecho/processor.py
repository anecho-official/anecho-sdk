"""
anecho SDK runtime API — Model / ProcessorConfig / Processor.

    from anecho.sdk import Model, Processor, ProcessorConfig
    model = Model.from_file("anecho-openspace-16khz-v1.anecho")
    cfg = ProcessorConfig.optimal(model)
    proc = Processor(model, license_key, cfg)      # license verified here
    out = proc.process(block)                      # 1-D float32, block_size samples

Runs YOUR model under YOUR license. Three things it does that the commercial SDK
this shape imitates does not:

* **The model is executed from the file.** `Model.from_file` builds an
  `Executor` over the packed graph; nothing imports the architecture. A file from
  a year ago still runs.
* **Streaming is exact, not approximate.** Each stateful node carries its own
  history (STFT tail, conv frames, GRU state, deep-filter past, overlap-add
  tail), so block-wise output equals one-shot output — a property the tests
  assert rather than hope for. The previous implementation re-ran the model over
  a sliding context window every block, which both cost more and drifted.
* **The delay is derived, not asserted.** `get_audio_delay()` computes
  `(n_fft - hop) + lookahead * hop` from the graph, so a non-causal model reports
  the latency it actually needs instead of clicking at every block boundary.

`process` also has an inspection sibling, `process_inspect`, returning every
intermediate activation by node name — the API whose absence made comparing
against a closed runtime impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .container import read_model
from .errors import AudioConfigUnsupportedError, ModelInvalidError
from .executor import Executor, State
from .graph import lookahead_frames
from .license import verify_license

# --- product identity & embedded verify key -------------------------------
# Replace VERIFY_KEY with your real public (ed25519) or secret (hmac) key at
# build time. INTEGRITY_SECRET (optional) enables HMAC model-tamper checks.
PRODUCT = "anecho-openspace"
VERIFY_KEY = bytes.fromhex("f1d3a7a30445ec60ac7d91569749ff327eeebe28671fbd976170f13145fa0bc2")            # set via keytool; empty disables license enforcement in dev
INTEGRITY_SECRET = None     # set to bytes to require/verify the model HMAC


class ProcessorParameter(Enum):
    Bypass = "bypass"                    # bool
    EnhancementLevel = "enhancement_level"   # 0.0 .. 1.0 dry/wet
    VoiceGain = "voice_gain"             # dB applied to the enhanced signal


class Model:
    """A loaded anecho model: graph + weights, integrity-checked on load."""

    def __init__(self, header: dict, tensors: dict, graph, config: dict):
        self.header = header
        self.tensors = tensors
        self.graph = graph
        self.config = config
        self.model_id = header.get("model_id", "anecho")

    @staticmethod
    def from_file(path: str, integrity_secret: bytes | None = INTEGRITY_SECRET) -> "Model":
        header, tensors, graph = read_model(path, secret=integrity_secret,
                                            require_graph=True)
        return Model(header, tensors, graph, header.get("config", {}))

    # -------------------------------------------------------------- queries
    def get_id(self) -> str:
        return self.model_id

    @property
    def sample_rate(self) -> int:
        return int(self.config.get("sr", self.graph.numerics.stft["n_fft"] // 0.03))

    @property
    def n_fft(self) -> int:
        return int(self.graph.numerics.stft["n_fft"])

    @property
    def hop(self) -> int:
        return int(self.graph.numerics.stft["hop"])

    def optimal_sample_rate(self) -> int:
        return self.sample_rate

    def optimal_block_size(self) -> int:
        return self.hop

    @property
    def lookahead(self) -> int:
        """Frames of future context the graph needs (0 = causal)."""
        return lookahead_frames(self.graph)

    def audio_delay(self) -> int:
        """Output delay in samples, derived from the graph."""
        return (self.n_fft - self.hop) + self.lookahead * self.hop

    def vad_config(self):
        """Gate parameters from the file, or the documented defaults."""
        from .graph import VadConfig
        return VadConfig.from_json((self.header.get("runtime") or {}).get("vad"))

    def describe(self) -> str:
        return self.graph.describe()


@dataclass
class ProcessorConfig:
    sample_rate: int
    block_size: int
    variable_block_size: bool = False

    @staticmethod
    def optimal(model: Model) -> "ProcessorConfig":
        return ProcessorConfig(model.sample_rate, model.optimal_block_size())


class ProcessorContext:
    """State and parameter handle, mirroring the shape of a commercial SDK."""

    def __init__(self, processor: "Processor"):
        self._p = processor

    def get_audio_delay(self) -> int:
        return self._p.model.audio_delay()

    def reset(self) -> None:
        self._p._reset()

    def set_parameter(self, parameter: ProcessorParameter, value) -> None:
        self._p._set_parameter(parameter, value)

    def get_parameter(self, parameter: ProcessorParameter):
        return self._p._params[ProcessorParameter(parameter)]

    def parameters(self) -> dict:
        return dict(self._p._params)


class Processor:
    """Streaming enhancement processor. Verifies the license on construction."""

    def __init__(self, model: Model, license_key: str,
                 config: ProcessorConfig | None = None,
                 verify_key: bytes | None = None, product: str = PRODUCT):
        vk = verify_key if verify_key is not None else VERIFY_KEY
        if vk:                                   # enforce only when a key is configured
            self.claims = verify_license(license_key, vk, product, feature="enhance")
        else:
            self.claims = {"dev": True}

        self.model = model
        self.config = config or ProcessorConfig.optimal(model)
        if self.config.sample_rate != model.sample_rate:
            raise AudioConfigUnsupportedError(
                f"model is {model.sample_rate} Hz, got {self.config.sample_rate}")
        if not self.config.variable_block_size and self.config.block_size % model.hop:
            raise AudioConfigUnsupportedError(
                f"block size {self.config.block_size} is not a multiple of the "
                f"{model.hop}-sample hop; pass variable_block_size=True to buffer instead")

        self._ex = Executor(model.graph, model.tensors)
        self._params = {ProcessorParameter.Bypass: False,
                        ProcessorParameter.EnhancementLevel: 1.0,
                        ProcessorParameter.VoiceGain: 0.0}
        self._reset()

    # ------------------------------------------------------------- lifecycle
    def initialize(self, config: ProcessorConfig | None = None) -> None:
        if config is not None:
            self.config = config
        self._reset()

    def get_context(self) -> ProcessorContext:
        return ProcessorContext(self)

    def terminate_session(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._state = State()
        self._pending = np.zeros(0, dtype=np.float32)
        # Both paths are primed with the declared delay, so after N input samples
        # exactly N are available on each and no ad-hoc zero padding is needed —
        # which is what makes the output block-size independent.
        delay = self.model.audio_delay()
        self._out = np.zeros(delay, dtype=np.float32)
        self._dry = np.zeros(delay, dtype=np.float32)

    def _set_parameter(self, parameter, value) -> None:
        p = ProcessorParameter(parameter)
        if p is ProcessorParameter.EnhancementLevel and not 0.0 <= float(value) <= 1.0:
            raise ValueError("EnhancementLevel must be in [0, 1]")
        self._params[p] = bool(value) if p is ProcessorParameter.Bypass else float(value)

    # -------------------------------------------------------------- process
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Enhance one block; returns the same number of samples.

        The first `get_audio_delay()` samples of the stream are the algorithmic
        delay and come back as silence, exactly as the delay says they will.
        """
        import torch

        block = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if self._params[ProcessorParameter.Bypass]:
            return block.copy()

        self._pending = np.concatenate([self._pending, block])
        hop = self.model.hop
        n = (len(self._pending) // hop) * hop
        if n:
            chunk = torch.from_numpy(self._pending[:n])[None]
            self._pending = self._pending[n:]
            with torch.no_grad():
                y = self._ex.run({"wav": chunk}, state=self._state)["wav"]
            self._out = np.concatenate([self._out, y[0].numpy()])

        # dry delay line, so mixing compares like with like
        self._dry = np.concatenate([self._dry, block])
        want = len(block)
        if len(self._out) < want:                      # only if a node held frames back
            self._out = np.concatenate(
                [self._out, np.zeros(want - len(self._out), dtype=np.float32)])
        wet, self._out = self._out[:want], self._out[want:]
        dry, self._dry = self._dry[:want], self._dry[want:]

        gain = 10.0 ** (self._params[ProcessorParameter.VoiceGain] / 20.0)
        level = self._params[ProcessorParameter.EnhancementLevel]
        return (level * gain * wet + (1.0 - level) * dry).astype(np.float32)

    def process_inspect(self, audio: np.ndarray, nodes=None) -> dict:
        """Like `process`, but also returns intermediate activations by node name."""
        import torch

        block = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            out = self._ex.run({"wav": torch.from_numpy(block)[None]},
                               state=self._state, capture=nodes or True)
        return {"wav": out["wav"][0].numpy(), "values": out["values"]}

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """Offline: enhance a whole 1-D signal in one pass (no streaming state)."""
        import torch

        x = np.ascontiguousarray(audio, dtype=np.float32)
        with torch.no_grad():
            y = self._ex.run({"wav": torch.from_numpy(x)[None]})["wav"]
        out = y[0].numpy()
        gain = 10.0 ** (self._params[ProcessorParameter.VoiceGain] / 20.0)
        level = self._params[ProcessorParameter.EnhancementLevel]
        if self._params[ProcessorParameter.Bypass]:
            return x.copy()
        n = min(len(out), len(x))
        return (level * gain * out[:n] + (1.0 - level) * x[:n]).astype(np.float32)


class VadParameter(Enum):
    Sensitivity = "sensitivity"                  # 0..1, 0.5 = the file's thresholds
    MinimumSpeechDuration = "attack_ms"          # ms above threshold before speech
    SpeechHoldDuration = "hold_ms"               # ms of speech held after it drops


class VadContext:
    """Read predictions and tune the gate, mirroring the commercial API's shape."""

    def __init__(self, vad: "Vad"):
        self._v = vad

    def get_probability(self) -> float:
        """Raw per-frame probability from the last processed block."""
        return self._v._probability

    def is_speech(self) -> bool:
        """Gate state: probability plus hysteresis, attack and hold."""
        return self._v._active

    def get_gain(self) -> float:
        """The gate's smoothed 0..1 gain, following `release_ms` on the way down."""
        return self._v._gain

    def set_parameter(self, parameter: VadParameter, value: float) -> None:
        self._v._set_parameter(parameter, value)

    def get_parameter(self, parameter: VadParameter) -> float:
        return self._v._params[VadParameter(parameter)]

    def reset(self) -> None:
        self._v._reset()


class Vad:
    """Voice-activity detector over any model that exposes a per-frame probability.

    The commercial SDK ships a dedicated VAD model behind the same API. Here the
    detector is not tied to one architecture: `VadConfig.probability_output` names
    a graph output, so an enhancement model that already estimates target
    presence — ours calls it `proximity` — serves as its own VAD, and a purpose-
    trained VAD model works through exactly the same class.

    The gate itself (hysteresis, attack, hold, release) comes from the model file,
    so two builds cannot disagree about what counts as speech.
    """

    def __init__(self, model: Model, license_key: str,
                 config: ProcessorConfig | None = None,
                 verify_key: bytes | None = None, product: str = PRODUCT):
        vk = verify_key if verify_key is not None else VERIFY_KEY
        self.claims = (verify_license(license_key, vk, product, feature="vad")
                       if vk else {"dev": True})
        self.model = model
        self.config = config or ProcessorConfig.optimal(model)
        self.vad_config = model.vad_config()

        out = self.vad_config.probability_output
        if out not in model.graph.outputs:
            raise ModelInvalidError(
                f"{model.get_id()} declares no output {out!r}; its outputs are "
                f"{sorted(model.graph.outputs)}. A VAD needs a per-frame "
                f"probability — repack with `vad.probability_output` set to one.")
        self._ex = Executor(model.graph, model.tensors)
        self._params = {VadParameter.Sensitivity: 0.5,
                        VadParameter.MinimumSpeechDuration: self.vad_config.attack_ms,
                        VadParameter.SpeechHoldDuration: self.vad_config.hold_ms}
        self._reset()

    # ------------------------------------------------------------- lifecycle
    def initialize(self, config: ProcessorConfig | None = None) -> None:
        if config is not None:
            self.config = config
        self._reset()

    def get_context(self) -> VadContext:
        return VadContext(self)

    def terminate_session(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._state = State()
        self._pending = np.zeros(0, dtype=np.float32)
        self._probability = 0.0
        self._active = False
        self._gain = 0.0
        self._above = 0                          # consecutive frames over threshold
        self._hold = 0                           # frames of hold left

    def _set_parameter(self, parameter, value) -> None:
        p = VadParameter(parameter)
        value = float(value)
        if p is VadParameter.Sensitivity and not 0.0 <= value <= 1.0:
            raise ValueError("Sensitivity must be in [0, 1]")
        if p is not VadParameter.Sensitivity and value < 0:
            raise ValueError(f"{p.name} must not be negative")
        self._params[p] = value

    # -------------------------------------------------------------- process
    @property
    def _frame_ms(self) -> float:
        return 1000.0 * self.model.hop / self.model.sample_rate

    def _thresholds(self) -> tuple[float, float]:
        """Sensitivity slides both thresholds: 0.5 leaves the file's values."""
        c = self.vad_config
        scale = 2.0 * (1.0 - self._params[VadParameter.Sensitivity])
        return min(1.0, c.threshold_on * scale), min(1.0, c.threshold_off * scale)

    def process(self, audio: np.ndarray) -> None:
        """Feed audio; the signal is not modified, only the prediction advances."""
        import torch

        block = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, block])
        hop = self.model.hop
        n = (len(self._pending) // hop) * hop
        if not n:
            return
        chunk = torch.from_numpy(self._pending[:n])[None]
        self._pending = self._pending[n:]
        with torch.no_grad():
            out = self._ex.run({"wav": chunk}, state=self._state)
        probs = out[self.vad_config.probability_output].reshape(-1).numpy()

        on, off = self._thresholds()
        attack = max(1, round(self._params[VadParameter.MinimumSpeechDuration]
                              / self._frame_ms))
        hold = max(0, round(self._params[VadParameter.SpeechHoldDuration] / self._frame_ms))
        decay = (0.0 if self.vad_config.release_ms <= 0
                 else float(np.exp(-self._frame_ms / self.vad_config.release_ms)))

        for p in probs:
            self._probability = float(p)
            if p >= on:
                self._above += 1
                if self._above >= attack:
                    self._active, self._hold = True, hold
            elif p < off:
                self._above = 0
                if self._active:
                    self._hold -= 1
                    if self._hold <= 0:
                        self._active = False
            self._gain = 1.0 if self._active else self._gain * decay
