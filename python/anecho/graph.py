"""Graph IR for the ``.anecho`` container — the part the vendor format lacks.

A `.aicmodel` file stores weights, shapes and constants, and nothing else: the
topology lives in the runtime's code. That single decision is what made reading
one require weeks of shape-chaining, and it means a model file is only
meaningful next to the exact build that wrote it. Everything here exists so that
never applies to our own files.

A `.anecho` v2 model carries:

* a **graph** — an ordered list of nodes, each naming its op, its inputs and
  outputs, the tensors it consumes and its attributes. A loader executes the
  graph; it does not import a hard-coded architecture, so a file written today
  still loads against tomorrow's code.
* a **numerics** block — the conventions that are invisible in the weights and
  therefore unrecoverable without an oracle. Every field in `Numerics` is one
  thing we had to reverse-engineer out of somebody else's file:

      gru_gate_order        which 1/3 slice of a GRU weight is which gate
      batchnorm_eps         hides inside `sqrt(var + eps)`; unknowable from bytes
      conv_kernel_axes      whether a flat [out, in*k] weight is (time,freq)
      conv_weight_layout    [out, in*k] vs [in, out*k]
      linear_weight_layout  the *opposite* order to conv, in the vendor's file
      feature_flatten       which slice each group of a grouped linear sees
      df_coef_layout        bin-major vs tap-major coefficient packing
      df_application        whether the deep filter replaces or mixes
      stft / erb_feature    window, synthesis, and the exact feature transform

Attributes are plain JSON. The header stays human-readable on purpose: a model
you cannot inspect with `head -c 4096` is a model you will one day have to
reverse-engineer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

FORMAT = "anecho-model/2"

#: Ops a conforming runtime must implement. Keeping this explicit means an
#: unknown op is a clear load error rather than a silent misinterpretation.
OPS = {
    "stft", "istft", "erb_analysis", "erb_synthesis", "erb_feature", "df_feature",
    "conv1d", "conv2d", "batchnorm", "activation", "grouped_linear", "gru",
    "pixel_shuffle", "flatten", "unflatten", "reshape", "transpose", "concat",
    "add", "mul", "slice", "linear", "sigmoid", "mean_norm", "apply_gains",
    "deep_filter", "mix", "gate", "pool", "limit", "agc", "agc_undo",
}


@dataclass
class Numerics:
    """Conventions that no amount of staring at the weights can recover."""

    gru_gate_order: str = "rzn"              # torch order (r, z, n)
    batchnorm_eps: float = 1e-5
    conv_kernel_axes: tuple = ("time", "freq")
    conv_weight_layout: str = "out_in_kernel"    # [out, in/groups * prod(kernel)]
    linear_weight_layout: str = "in_out"         # [in, out]
    feature_flatten: str = "freq_major"          # index = freq * channels + channel
    df_coef_layout: str = "bin_major"            # index = bin*taps*2 + tap*2 + ri
    df_application: str = "mix_alpha"            # "mix_alpha" | "replace"
    stft: dict = field(default_factory=lambda: {
        "window": "hann", "periodic": True, "center": False,
        "pad_mode": "constant",          # torch defaults to "reflect", which no
                                         # stream can reproduce: reflecting the
                                         # start of a signal needs its future
        "synthesis": "wola", "scale": 1.0,   # amplitude scaling of the analysis;
                                             # 1/n_fft is a common convention and
                                             # invisible in the weights
        "n_fft": 480, "hop": 240})
    erb_feature: dict = field(default_factory=lambda: {
        "domain": "magnitude", "log": "db20", "scale": 40.0, "mean_norm": "none"})

    def to_json(self) -> dict:
        d = asdict(self)
        d["conv_kernel_axes"] = list(self.conv_kernel_axes)
        return d

    @staticmethod
    def from_json(d: dict) -> "Numerics":
        d = dict(d or {})
        if "conv_kernel_axes" in d:
            d["conv_kernel_axes"] = tuple(d["conv_kernel_axes"])
        known = {f for f in Numerics().to_json()}
        unknown = set(d) - known
        if unknown:
            raise GraphError(f"unknown numerics fields: {sorted(unknown)}")
        return Numerics(**d)


@dataclass
class VadConfig:
    """Gate parameters for the voice-activity product, carried in the file.

    The commercial SDK keeps these in its model's runtime config — thresholds
    0.05/0.65 with 5 ms attack, 200 ms hold, 100 ms release in the model we read.
    They are product behaviour, not architecture, but leaving them in code means
    two builds disagree about what "speech" means. So they ship with the model.

    `probability_output` names the graph output to read, which is what lets a VAD
    be built on any model that exposes a per-frame probability.
    """

    probability_output: str = "proximity"
    threshold_on: float = 0.65        # hysteresis: enter speech above this
    threshold_off: float = 0.05       # leave speech below this
    attack_ms: float = 5.0            # time above threshold before declaring speech
    hold_ms: float = 200.0            # keep speech this long after it drops
    release_ms: float = 100.0         # fade the gate out over this long

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "VadConfig":
        d = dict(d or {})
        unknown = set(d) - set(VadConfig().to_json())
        if unknown:
            raise GraphError(f"unknown vad fields: {sorted(unknown)}")
        cfg = VadConfig(**d)
        if not 0.0 <= cfg.threshold_off <= cfg.threshold_on <= 1.0:
            raise GraphError(
                f"vad thresholds must satisfy 0 <= off ({cfg.threshold_off}) <= "
                f"on ({cfg.threshold_on}) <= 1")
        for field_name in ("attack_ms", "hold_ms", "release_ms"):
            if getattr(cfg, field_name) < 0:
                raise GraphError(f"vad {field_name} must not be negative")
        return cfg


@dataclass
class Node:
    name: str
    op: str
    inputs: list                 # value names produced by earlier nodes or graph inputs
    outputs: list                # value names this node produces
    params: dict = field(default_factory=dict)   # role -> tensor name in the payload
    attrs: dict = field(default_factory=dict)    # op configuration, JSON-serialisable

    def to_json(self) -> dict:
        return dict(name=self.name, op=self.op, inputs=list(self.inputs),
                    outputs=list(self.outputs), params=dict(self.params),
                    attrs=dict(self.attrs))

    @staticmethod
    def from_json(d: dict) -> "Node":
        missing = {"name", "op", "inputs", "outputs"} - set(d)
        if missing:
            raise GraphError(f"node is missing {sorted(missing)}: {d}")
        return Node(d["name"], d["op"], list(d["inputs"]), list(d["outputs"]),
                    dict(d.get("params", {})), dict(d.get("attrs", {})))


class GraphError(ValueError):
    """The graph is not executable as written."""


@dataclass
class Graph:
    inputs: list                 # value names the caller supplies
    outputs: dict                # public name -> value name
    nodes: list                  # list[Node], in execution order
    numerics: Numerics = field(default_factory=Numerics)

    # ------------------------------------------------------------- validation
    def validate(self, tensor_names: set | None = None) -> "Graph":
        """Reject anything a runtime could only guess about.

        Checks: known ops, unique node and value names, no reference to a value
        before it is produced (which also rules out cycles, since nodes are
        ordered), every declared output produced, and — when the payload's
        tensor names are supplied — every parameter present.
        """
        seen_nodes, available = set(), set(self.inputs)
        if len(self.inputs) != len(set(self.inputs)):
            raise GraphError("duplicate graph input names")

        for node in self.nodes:
            if node.op not in OPS:
                raise GraphError(f"{node.name}: unknown op {node.op!r}; "
                                 f"a runtime cannot guess it")
            if node.name in seen_nodes:
                raise GraphError(f"duplicate node name {node.name!r}")
            seen_nodes.add(node.name)
            for value in node.inputs:
                if value not in available:
                    raise GraphError(f"{node.name}: input {value!r} is not produced "
                                     f"by any earlier node or graph input")
            for value in node.outputs:
                if value in available:
                    raise GraphError(f"{node.name}: value {value!r} is written twice")
                available.add(value)
            if tensor_names is not None:
                for role, tname in node.params.items():
                    if tname not in tensor_names:
                        raise GraphError(f"{node.name}: parameter {role}={tname!r} "
                                         f"is not in the payload")

        for public, value in self.outputs.items():
            if value not in available:
                raise GraphError(f"output {public!r} refers to {value!r}, "
                                 f"which nothing produces")
        return self

    def unused_tensors(self, tensor_names: set) -> list:
        """Tensors in the payload that no node consumes — dead weight, literally."""
        used = {t for n in self.nodes for t in n.params.values()}
        return sorted(tensor_names - used)

    # ---------------------------------------------------------- serialisation
    def to_json(self) -> dict:
        return dict(inputs=list(self.inputs), outputs=dict(self.outputs),
                    nodes=[n.to_json() for n in self.nodes],
                    numerics=self.numerics.to_json())

    @staticmethod
    def from_json(d: dict) -> "Graph":
        try:
            return Graph(list(d["inputs"]), dict(d["outputs"]),
                         [Node.from_json(n) for n in d["nodes"]],
                         Numerics.from_json(d.get("numerics")))
        except KeyError as e:
            raise GraphError(f"graph is missing {e}") from None

    def describe(self) -> str:
        """One line per node — the thing you wish the other format had."""
        lines = [f"inputs: {', '.join(self.inputs)}"]
        for n in self.nodes:
            params = (" {" + ", ".join(f"{k}={v}" for k, v in n.params.items()) + "}"
                      if n.params else "")
            attrs = (" " + json.dumps(n.attrs, separators=(",", "="))
                     if n.attrs else "")
            lines.append(f"  {n.name:<24s} {n.op:<16s} "
                         f"{', '.join(n.inputs)} -> {', '.join(n.outputs)}{params}{attrs}")
        lines.append("outputs: " + ", ".join(f"{k}={v}" for k, v in self.outputs.items()))
        return "\n".join(lines)

def lookahead_frames(graph) -> int:
    """Frames of *future* context the graph needs — its unavoidable extra latency.

    Symmetric time padding buys accuracy and costs latency; a streaming runtime
    has to know which it is dealing with rather than discover it as a click at
    every block boundary. Zero means the graph is causal and block-wise output
    equals one-shot output exactly.
    """
    axes = tuple(graph.numerics.conv_kernel_axes)
    t_index = axes.index("time") if "time" in axes else 1
    total = 0
    for n in graph.nodes:
        if n.op not in ("conv1d", "conv2d") or n.attrs.get("causal"):
            continue
        pad = n.attrs.get("padding", 0)
        if isinstance(pad, (list, tuple)):        # conv2d: per-axis padding
            pad = pad[t_index] if len(pad) > t_index else 0
        total += int(pad)
    return total
