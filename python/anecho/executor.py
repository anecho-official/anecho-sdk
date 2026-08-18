"""Graph executor — runs a ``.anecho`` v2 model from the file, not from our code.

The vendor SDK exposes `process(block) -> block` and nothing else. That is why a
layer-by-layer comparison against it was impossible and why the reverse
engineering had to lean on patched model files. Our runtime is a graph
interpreter, so the same capability costs one argument:

    out = ex.run({"wav": x}, capture=True)
    out["values"]["enc_conv0"]              # any intermediate, by node name

Layout conventions, so the graph never has to guess:

    conv1d / conv2d / batchnorm   [B, C, ...]   channel-second
    gru / linear                  [B, T, C]     time-second
    spectra                       [B, F, T]     complex

`transpose` nodes make every change of layout visible in the file. Recurrent
state lives in `State`, keyed by node name, so streaming is the same graph with
the state carried across calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .errors import ModelInvalidError
from .graph import Graph

GATE_PERM = {"rzn": (0, 1, 2), "zrn": (1, 0, 2), "nrz": (1, 2, 0), "nzr": (2, 1, 0)}


@dataclass
class State:
    """Recurrent state, one entry per stateful node."""

    hidden: dict = field(default_factory=dict)      # node name -> tensor
    tail: dict = field(default_factory=dict)        # node name -> leftover samples

    def reset(self) -> None:
        self.hidden.clear()
        self.tail.clear()


class Executor:
    """Executes a validated Graph over a tensor dict."""

    def __init__(self, graph: Graph, tensors: dict, device: str = "cpu"):
        self.graph = graph
        self.numerics = graph.numerics
        self.device = device
        self.T = {k: torch.as_tensor(np.asarray(v), device=device).float()
                  for k, v in tensors.items()}
        self._gru_cache: dict = {}

        unsupported = sorted({n.op for n in graph.nodes} - set(self._ops()))
        if unsupported:
            raise ModelInvalidError(
                f"this build cannot execute {unsupported}; the model needs a newer runtime")

    # ------------------------------------------------------------------ ops
    def _ops(self) -> dict:
        return {
            "stft": self._stft, "istft": self._istft,
            "erb_feature": self._erb_feature, "apply_gains": self._apply_gains,
            "conv1d": self._conv1d, "conv2d": self._conv2d,
            "batchnorm": self._batchnorm, "activation": self._activation,
            "gru": self._gru, "linear": self._linear,
            "reshape": self._reshape, "transpose": self._transpose,
            "add": self._add, "mul": self._mul, "concat": self._concat,
            "deep_filter": self._deep_filter, "mix": self._mix,
            "slice": self._slice, "mean_norm": self._mean_norm,
            "grouped_linear": self._grouped_linear,
            "pixel_shuffle": self._pixel_shuffle,
            "df_feature": self._df_feature, "gate": self._gate,
            "pool": self._pool, "limit": self._limit,
            "agc": self._agc, "agc_undo": self._agc_undo,
        }

    # ---------------------------------------------------------------- driver
    def run(self, inputs: dict, state: State | None = None,
            capture: bool | list = False) -> dict:
        env = {}
        for name in self.graph.inputs:
            if name not in inputs:
                raise ValueError(f"missing graph input {name!r}")
            env[name] = inputs[name]
        ops = self._ops()
        wanted = set(capture) if isinstance(capture, (list, tuple, set)) else None

        for node in self.graph.nodes:
            args = [env[i] for i in node.inputs]
            result = ops[node.op](node, args, state)
            if node.op == "stft" and result.shape[-1] == 0:
                # Streaming: not enough samples for a frame yet. Emitting nothing
                # is correct — the caller's delay line covers it — and it keeps
                # every later op from meeting an empty time axis.
                empty = torch.zeros(result.shape[0], 0, device=result.device)
                out = {k: empty for k in self.graph.outputs}
                return (out | {"values": {}}) if capture else out
            outs = result if isinstance(result, tuple) else (result,)
            if len(outs) != len(node.outputs):
                raise ModelInvalidError(
                    f"{node.name}: op produced {len(outs)} value(s), "
                    f"graph declares {len(node.outputs)}")
            for name, value in zip(node.outputs, outs):
                env[name] = value

        out = {public: env[v] for public, v in self.graph.outputs.items()}
        if capture:
            out["values"] = {n.name: env[n.outputs[0]] for n in self.graph.nodes
                             if n.outputs and (wanted is None or n.name in wanted)}
        return out

    # ------------------------------------------------------------- helpers
    def _p(self, node, role: str):
        name = node.params.get(role)
        return None if name is None else self.T[name]

    def _shape(self, spec, ref: torch.Tensor) -> list:
        """Resolve a symbolic shape: ints, -1 (infer), "B" (batch), "T" (last axis)."""
        return [ref.shape[0] if s == "B" else ref.shape[-1] if s == "T" else int(s)
                for s in spec]

    def _window(self, n: int) -> torch.Tensor:
        s = self.numerics.stft
        w = torch.hann_window(n, periodic=s.get("periodic", True), device=self.device)
        return w.sqrt() if s.get("window") == "sqrt_hann" else w

    # ----------------------------------------------------------------- ops
    def _stft(self, node, args, state):
        s = self.numerics.stft
        n_fft, hop = node.attrs.get("n_fft", s["n_fft"]), node.attrs.get("hop", s["hop"])
        x = args[0]
        scale = float(s.get("scale", 1.0))
        if state is None:
            return scale * torch.stft(x, n_fft, hop, window=self._window(n_fft),
                                      center=s.get("center", False),
                                      pad_mode=s.get("pad_mode", "constant"),
                                      return_complex=True)

        # Streaming: carry the samples that the previous call could not frame.
        prev = state.tail.get(node.name)
        if prev is None:
            # Centring pads n_fft//2 at the *start* of the stream, once. Without
            # centring the first frame simply waits until n_fft samples exist.
            lead = n_fft // 2 if s.get("center", False) else 0
            prev = torch.zeros(x.shape[0], lead, dtype=x.dtype, device=x.device)
        buf = torch.cat([prev, x], dim=-1)
        n_frames = max(0, (buf.shape[-1] - n_fft) // hop + 1)
        used = n_frames * hop
        state.tail[node.name] = buf[:, used:].detach()
        if n_frames == 0:
            return torch.zeros(x.shape[0], n_fft // 2 + 1, 0,
                               dtype=torch.complex64, device=x.device)
        return scale * torch.stft(buf[:, :used + n_fft - hop], n_fft, hop,
                                  window=self._window(n_fft), center=False,
                                  return_complex=True)

    def _istft(self, node, args, state):
        """Weighted overlap-add, normalised by the true window envelope.

        `torch.istft` refuses a Hann window at 50 % overlap without centring
        (the envelope reaches zero at the edges), and a streaming runtime does
        its own overlap-add anyway.
        """
        s = self.numerics.stft
        n_fft, hop = node.attrs.get("n_fft", s["n_fft"]), node.attrs.get("hop", s["hop"])
        # A second input means "match this signal's length" — the graph says so
        # explicitly instead of the runtime guessing.
        length = args[1].shape[-1] if len(args) > 1 else node.attrs.get("length")
        frames = torch.fft.irfft(args[0].permute(0, 2, 1) / float(s.get("scale", 1.0)),
                                 n=n_fft)                                # [B,T,N]
        b, t, n = frames.shape
        w = self._window(n_fft)
        syn = w if s.get("synthesis", "wola") == "wola" else torch.ones_like(w)
        out_len = (t - 1) * hop + n
        fold = lambda x: F.fold(x, (1, out_len), (1, n), stride=(1, hop)).view(-1, out_len)
        y = fold((frames * syn).transpose(1, 2))
        env = fold((syn * w).view(1, n, 1).expand(1, n, t))

        if state is not None:
            # Streaming: add the previous call's overlap, emit only the samples
            # whose envelope is complete, and carry the rest. Signal and envelope
            # travel together so the normalisation is exact across block joins.
            skip0 = (n_fft // 2 if s.get("center", False) else 0)
            prev_y, prev_env, skip = state.tail.get(node.name, (None, None, skip0))
            if prev_y is not None:
                # The carried tail starts at the same stream position as the new
                # segment, so both are zero-extended to a common length and added.
                width = max(prev_y.shape[-1], y.shape[-1])
                pad = lambda t: F.pad(t, (0, width - t.shape[-1]))   # noqa: E731
                y, env = pad(y) + pad(prev_y), pad(env) + pad(prev_env)
            keep = max(0, y.shape[-1] - (n - hop))
            done_y, done_env = y[:, :keep], env[:, :keep]
            out = done_y / done_env.clamp_min(1e-2 * float(env.max()))
            if prev_y is None:
                # First emission: the head is as incomplete as the tail — only one
                # frame has reached it — so it is not reconstructible either. Zeroing
                # it costs the delay we already declare and removes an 8x spike in
                # the first 15 ms of every stream.
                head = min(n - hop, out.shape[-1])
                out = torch.cat([torch.zeros_like(out[:, :head]), out[:, head:]], -1)
            drop = min(skip, out.shape[-1])       # consume the leading pad gradually
            state.tail[node.name] = (y[:, keep:].detach(), env[:, keep:].detach(),
                                     skip - drop)
            return out[:, drop:]

        # The floor has to be relative to the envelope's own peak. An absolute
        # 1e-8 turns the tail — where only one frame has contributed — into a
        # division by almost nothing, which showed up as a single sample 37x
        # full scale at the end of a real recording. Relative clamping bounds
        # the edge gain instead of unleashing it.
        y = y / env.clamp_min(1e-2 * float(env.max()))
        # The last n_fft-hop samples have had only one frame contribute, so their
        # envelope is incomplete and dividing by it inflates whatever is there
        # (up to 1/floor = 100x). Streaming never emits those samples — it carries
        # them until the next block completes the envelope — so offline says the
        # same thing: unknown, hence zero. `align` shifts them out anyway.
        total = y.shape[-1]
        head, keep = min(n - hop, total), max(0, total - (n - hop))
        if head < keep:                     # zero both incomplete edges, keep length
            y = F.pad(y[:, head:keep], (head, total - keep))
        else:
            y = torch.zeros_like(y)
        if s.get("center", False):
            y = y[:, n_fft // 2:]
        if length is not None:                   # return exactly what was asked for
            y = y[:, :length] if y.shape[-1] >= length else \
                F.pad(y, (0, length - y.shape[-1]))
        return y

    def _erb_feature(self, node, args, state):
        """Complex spectrum -> band feature, in the domain `numerics` declares."""
        cfg = self.numerics.erb_feature
        mag = args[0].abs()
        x = mag ** 2 if cfg.get("domain") == "energy" else mag
        band = torch.einsum("bft,fe->bet", x, self._p(node, "matrix"))
        y = band + cfg.get("eps", 1e-10)
        kind = cfg.get("log", "db20")
        if kind == "db20":
            y = 20.0 * torch.log10(y)
        elif kind == "log10":
            y = torch.log10(y)
        elif kind == "ln":
            y = torch.log(y)
        elif kind != "linear":
            raise ModelInvalidError(f"unknown erb log {kind!r}")
        return y / cfg.get("scale", 1.0)

    def _df_feature(self, node, args, state):
        """Complex spectrum -> the deep filter's own input map [B,C,T,bins]."""
        bins = int(node.attrs["bins"])
        part = node.attrs.get("part", "real")
        low = args[0][:, :bins, :]
        if part == "real":
            x = low.real
        elif part == "imag":
            x = low.imag
        elif part == "mag":
            x = low.abs()
        else:
            raise ModelInvalidError(f"unknown df_feature part {part!r}")
        return x.transpose(1, 2).unsqueeze(1) * float(node.attrs.get("scale", 1.0))

    def _gate(self, node, args, state):
        """Per-frame speech gate: hysteresis, attack, hold, exponential release.

        A product behaviour rather than an architectural one, which is exactly why
        it belongs in the file: two builds must not disagree about when the output
        is muted. Attributes carry the thresholds and times; state carries the
        counters so streaming matches one-shot.
        """
        prob = args[0]
        a = node.attrs
        frame_ms = float(a.get("frame_ms", 15.0))
        attack = max(1, round(float(a.get("attack_ms", 5.0)) / frame_ms))
        hold = max(0, round(float(a.get("hold_ms", 200.0)) / frame_ms))
        decay = float(np.exp(-frame_ms / max(float(a.get("release_ms", 100.0)), 1e-6)))
        on, off = float(a.get("threshold_on", 0.5)), float(a.get("threshold_off", 0.5))

        carried = None if state is None else state.hidden.get(node.name)
        above, held, active, g = carried or (0, 0, False, 0.0)
        out = torch.zeros_like(prob)
        for i in range(prob.shape[0]):
            for k in range(prob.shape[-1]):
                p = float(prob[i, k])
                if p >= on:
                    above += 1
                    if above >= attack:
                        active, held = True, hold
                elif p < off:
                    above = 0
                    if active:
                        held -= 1
                        if held <= 0:
                            active = False
                g = 1.0 if active else g * decay
                out[i, k] = g
        if state is not None:
            state.hidden[node.name] = (above, held, active, g)
        return out

    def _apply_gains(self, node, args, state):
        spec, gains = args                                  # [B,F,T], [B,E,T]
        bins = torch.einsum("bet,ef->bft", gains, self._p(node, "membership"))
        return spec * bins.to(spec.dtype)

    def _time_axis(self) -> int:
        """Which spatial axis of a conv2d input is time, per the file's numerics."""
        axes = tuple(self.numerics.conv_kernel_axes)
        return 2 + (axes.index("time") if "time" in axes else 1)

    def _causal_pad(self, node, x: torch.Tensor, axis: int, k: int, state):
        """Prepend the past instead of zeros, so blocks join seamlessly.

        Offline (no state) this is a plain causal zero-pad. Streaming, the node
        keeps its own last ``k-1`` frames — which is why block-wise output equals
        one-shot output exactly rather than approximately.
        """
        if k <= 1:
            return x
        need = k - 1
        prev = None if state is None else state.tail.get(node.name)
        if prev is None:
            shape = list(x.shape)
            shape[axis] = need
            prev = torch.zeros(shape, dtype=x.dtype, device=x.device)
        x = torch.cat([prev, x], dim=axis)
        if state is not None:
            state.tail[node.name] = x.narrow(axis, x.shape[axis] - need, need).detach()
        return x

    def _conv1d(self, node, args, state):
        x, k = args[0], int(self._p(node, "weight").shape[-1])
        if node.attrs.get("causal"):
            x = self._causal_pad(node, x, 2, k, state)
            pad = 0
        else:
            pad = int(node.attrs.get("padding", 0))
        return F.conv1d(x, self._p(node, "weight"), self._p(node, "bias"),
                        stride=int(node.attrs.get("stride", 1)), padding=pad,
                        groups=int(node.attrs.get("groups", 1)))

    def _conv2d(self, node, args, state):
        pad = list(node.attrs.get("padding", [0, 0]))
        x, w = args[0], self._p(node, "weight")
        if node.attrs.get("causal"):
            t_axis = self._time_axis()
            k_t = int(w.shape[t_axis])
            other = 3 if t_axis == 2 else 2
            p_other = pad[other - 2]
            if p_other:                                     # pad the non-time axis
                spec = [0, 0, 0, 0]
                spec[0 if other == 3 else 2] = p_other
                spec[1 if other == 3 else 3] = p_other
                x = F.pad(x, spec)
            x = self._causal_pad(node, x, t_axis, k_t, state)
            pad = [0, 0]
        return F.conv2d(x, w, self._p(node, "bias"),
                        stride=tuple(node.attrs.get("stride", [1, 1])),
                        padding=tuple(pad) if pad != [0, 0] else 0,
                        groups=int(node.attrs.get("groups", 1)))

    def _batchnorm(self, node, args, state):
        return F.batch_norm(args[0], self._p(node, "running_mean"),
                            self._p(node, "running_var"), self._p(node, "weight"),
                            self._p(node, "bias"), training=False,
                            eps=self.numerics.batchnorm_eps)

    def _activation(self, node, args, state):
        kind = node.attrs.get("kind", "relu")
        fn = {"relu": torch.relu, "silu": F.silu, "sigmoid": torch.sigmoid,
              "tanh": torch.tanh, "identity": lambda t: t}.get(kind)
        if fn is None:
            raise ModelInvalidError(f"unknown activation {kind!r}")
        return fn(args[0])

    def _gru(self, node, args, state):
        """Runs a GRU with the gate order the *file* declares."""
        key = node.name
        gru = self._gru_cache.get(key)
        if gru is None:
            layers = int(node.attrs["layers"])
            hidden = int(node.attrs["hidden"])
            in_size = int(node.attrs.get("input", hidden))
            gru = torch.nn.GRU(in_size, hidden, num_layers=layers, batch_first=True)
            perm = GATE_PERM.get(self.numerics.gru_gate_order)
            if perm is None:
                raise ModelInvalidError(
                    f"unknown gru_gate_order {self.numerics.gru_gate_order!r}")

            def gates(t, axis):
                chunks = torch.chunk(t, 3, dim=axis)
                return torch.cat([chunks[i] for i in perm], dim=axis)

            with torch.no_grad():
                for layer in range(layers):
                    for kind in ("ih", "hh"):
                        w = self._p(node, f"weight_{kind}_l{layer}")
                        b = self._p(node, f"bias_{kind}_l{layer}")
                        getattr(gru, f"weight_{kind}_l{layer}").copy_(gates(w, 0))
                        if b is not None:
                            getattr(gru, f"bias_{kind}_l{layer}").copy_(gates(b, 0))
            gru.eval().to(self.device)
            self._gru_cache[key] = gru

        h0 = state.hidden.get(node.name) if state is not None else None
        with torch.no_grad() if not torch.is_grad_enabled() else _null():
            y, hn = gru(args[0], h0)
        if state is not None:
            state.hidden[node.name] = hn.detach()
        return y

    def _grouped_linear(self, node, args, state):
        """Per-group linear over the last axis: [B,T,in] -> [B,T,out].

        The weight is one matrix per group, stacked as [G, in/G, out/G]. Which
        slice each group sees is the file's `feature_flatten` convention, so the
        input is split contiguously and the header says what that means.
        """
        x, w = args[0], self._p(node, "weight")
        b, t, _ = x.shape
        g = w.shape[0]
        y = torch.einsum("btgi,gio->btgo", x.view(b, t, g, -1), w)
        bias = self._p(node, "bias")
        if bias is not None:
            y = y + bias
        return y.reshape(b, t, -1)

    def _pixel_shuffle(self, node, args, state):
        """[B, r*C, T, F] -> [B, C, T, r*F]: channels fold into frequency.

        `order` says whether the channel index runs (c, r) or (r, c); getting it
        wrong interleaves the wrong pairs, which is why it is a declared attribute
        rather than a convention.
        """
        x = args[0]
        r = int(node.attrs.get("factor", 2))
        order = node.attrs.get("order", "cr")
        b, c2, t, f = x.shape
        c = c2 // r
        if order == "cr":
            y = x.view(b, c, r, t, f)
        elif order == "rc":
            y = x.view(b, r, c, t, f).permute(0, 2, 1, 3, 4)
        else:
            raise ModelInvalidError(f"unknown pixel_shuffle order {order!r}")
        return y.permute(0, 1, 3, 4, 2).reshape(b, c, t, f * r)

    def _linear(self, node, args, state):
        w = self._p(node, "weight")
        if self.numerics.linear_weight_layout == "in_out":
            w = w.t()
        return F.linear(args[0], w, self._p(node, "bias"))

    def _reshape(self, node, args, state):
        return args[0].reshape(self._shape(node.attrs["shape"], args[0]))

    def _pool(self, node, args, state):
        """Reduce one axis away — mean or max."""
        dim = int(node.attrs.get("dim", -1))
        kind = node.attrs.get("kind", "mean")
        if kind == "mean":
            return args[0].mean(dim=dim)
        if kind == "max":
            return args[0].amax(dim=dim)
        raise ModelInvalidError(f"unknown pool kind {kind!r}")

    def _transpose(self, node, args, state):
        return args[0].permute(*node.attrs["dims"])

    def _add(self, node, args, state):
        if len(args) == 2:
            return args[0] + args[1]
        return args[0] + float(node.attrs["scalar"])

    def _mul(self, node, args, state):
        if len(args) == 2:
            return args[0] * args[1]
        return args[0] * float(node.attrs["scalar"])

    def _concat(self, node, args, state):
        return torch.cat(args, dim=int(node.attrs.get("dim", -1)))

    def _slice(self, node, args, state):
        dim = int(node.attrs.get("dim", -1))
        lo, hi = node.attrs.get("start", 0), node.attrs.get("stop")
        idx = [slice(None)] * args[0].dim()
        idx[dim] = slice(lo, hi)
        return args[0][tuple(idx)]

    def _mean_norm(self, node, args, state):
        """Causal EMA mean subtraction along the last axis.

        The running mean is state: without carrying it, every streaming block
        restarts the average from zero, and with a long time constant that is a
        different function entirely — it is what made the gated model disagree
        with itself between one-shot and block-wise processing.
        """
        alpha = float(node.attrs.get("alpha", 0.99))
        x = args[0]
        carried = None if state is None else state.hidden.get(node.name)
        acc = carried if carried is not None else torch.zeros_like(x[..., 0])
        out = torch.empty_like(x)
        for t in range(x.shape[-1]):
            acc = x[..., t] * (1.0 - alpha) + acc * alpha
            out[..., t] = x[..., t] - acc
        if state is not None:
            state.hidden[node.name] = acc.detach()
        return out

    def _deep_filter(self, node, args, state):
        """Complex filter over the low bins, applied the way `numerics` says.

        `coefs` arrives flat as [B, bins*taps*2, T]; the packing order is
        `numerics.df_coef_layout`, and whether the result replaces the low bins
        or is mixed into them is `numerics.df_application` — the two things that
        cost the most to establish in someone else's model.
        """
        enh, noisy, raw = args[0], args[1], args[2]
        bins, order = int(node.attrs["bins"]), int(node.attrs["order"])
        b, _, t = raw.shape
        if self.numerics.df_coef_layout == "bin_major":
            c = raw.reshape(b, bins, order, 2, t)
        elif self.numerics.df_coef_layout == "tap_major":
            c = raw.reshape(b, order, 2, bins, t).permute(0, 3, 1, 2, 4)
        else:
            raise ModelInvalidError(f"unknown df_coef_layout {self.numerics.df_coef_layout!r}")
        coefs = torch.view_as_complex(c.permute(0, 1, 2, 4, 3).contiguous())   # [B,bins,order,T]

        past = None if state is None else state.tail.get(node.name)
        if past is None:
            past = torch.zeros(b, bins, order - 1, dtype=noisy.dtype, device=noisy.device)
        low = torch.cat([past, noisy[:, :bins, :]], dim=-1)
        if state is not None:                    # carry the last frames as history
            state.tail[node.name] = low[:, :, low.shape[-1] - (order - 1):].detach()
        acc = torch.zeros(b, bins, t, dtype=noisy.dtype, device=noisy.device)
        for k in range(order):
            acc = acc + coefs[:, :, k, :] * low[:, :, k:k + t]

        out = enh.clone()
        if self.numerics.df_application == "replace":
            out[:, :bins, :] = acc
        elif self.numerics.df_application == "mix_alpha":
            alpha = args[3] if len(args) > 3 else None
            if alpha is None:
                raise ModelInvalidError(
                    f"{node.name}: df_application=mix_alpha needs an alpha input")
            a = alpha.reshape(b, 1, t).to(acc.dtype)
            out[:, :bins, :] = a * acc + (1.0 - a) * enh[:, :bins, :]
        else:
            raise ModelInvalidError(
                f"unknown df_application {self.numerics.df_application!r}")
        return out

    def _agc(self, node, args, state):
        """Pin the input to the level the model was trained at; undo at the output.

        The gate head's decision boundary encodes the absolute level of its
        training material (measured: the same audio gates 60% of frames at
        -42 dBFS and 0% at -22). Rather than retrain, normalise the input to that
        level and hand the reciprocal gain downstream so the caller's level
        survives end to end.

        The tracker is a per-hop RMS peak with instant attack and a slow decay,
        and the correction is **downward only**: input quieter than the training
        level already behaves, so it passes at unity rather than being boosted —
        boosting turned a quiet lead-in (background before the first word) into
        speech-level material and opened the gate on it. The caller's first loud
        word sets the reference exactly; nothing wobbles afterwards (an EMA
        tracker's convergence transient changed the gate's decisions for
        seconds). Hops below the silence floor leave the tracker untouched.
        Returns ``(normalised wav, inverse gain)``; the gain is constant within
        each hop, and hops divide the streaming block, so block-wise and
        one-shot agree exactly.
        """
        x = args[0]
        target = 10.0 ** (float(node.attrs.get("target_db", -35.0)) / 10.0)
        hop = int(node.attrs.get("hop", 240))
        decay_db_s = float(node.attrs.get("decay_db_per_s", 0.5))
        gmin = 10.0 ** (float(node.attrs.get("min_gain_db", -36.0)) / 20.0)
        floor = 10.0 ** (float(node.attrs.get("floor_db", -60.0)) / 10.0)
        sr = 16000
        decay = 10.0 ** (-decay_db_s * (hop / sr) / 10.0)   # per hop, in power

        smooth_ms = float(node.attrs.get("smooth_ms", 500.0))
        a_sm = float(np.exp(-hop / (sr * smooth_ms / 1000.0)))

        b, n = x.shape
        assert b == 1, "agc: batch is always 1 in this runtime"
        xv = x[0].detach().cpu().numpy().astype(np.float64)
        carried = None if state is None else state.hidden.get(node.name)
        sm, peak = carried if carried is not None else (None, None)
        gains = np.empty(n, dtype=np.float64)
        for h0 in range(0, n, hop):                 # a trailing partial hop is fine
            p2 = float((xv[h0:h0 + hop] ** 2).mean())
            if p2 > floor:
                # Smooth first (about half a second), then ratchet the smoothed
                # level. Raw hop peaks track the crest factor, which differs by
                # 10+ dB between microphones and clips; the loudest *sustained*
                # level is what the trained heads are calibrated to.
                sm = p2 if sm is None else p2 * (1.0 - a_sm) + sm * a_sm
                peak = sm if peak is None or sm > peak else peak * decay
            elif peak is not None:
                peak = peak * decay
            ref = peak if peak is not None else target   # silence so far: unity
            # Downward only: input quieter than the training level already
            # behaves, so it passes at unity rather than being boosted.
            gains[h0:h0 + hop] = min(max((target / max(ref, 1e-12)) ** 0.5, gmin), 1.0)
        if state is not None:
            state.hidden[node.name] = (sm, peak)
            # Feed the undo FIFO from here, not from agc_undo: on the first
            # streaming call the stft has no complete frame yet and the executor
            # returns before downstream ops run, so a FIFO filled by agc_undo
            # itself would silently start one block late and stay misaligned.
            key = f"fifo:{node.outputs[1]}"
            prev = state.tail.get(key)
            # reciprocal of the float32 gain, so the FIFO holds bit-identical
            # values to the inv tensor the offline path multiplies by
            inv = 1.0 / gains.astype(np.float32)
            state.tail[key] = inv if prev is None else np.concatenate([prev, inv])
        g = torch.from_numpy(gains.astype(np.float32))[None]
        return x * g, 1.0 / g

    def _agc_undo(self, node, args, state):
        """Undo the agc gain on the istft output, honouring the WOLA latency.

        The inverse-gain series lives on the *input* timeline; the istft output
        in *availability*: position t of the output leaves the pipeline one
        block after position t of the input arrives, but the positions map 1:1.
        A FIFO of pending gain samples absorbs that: each call pushes this
        block's gains and pops exactly as many as the output carries, which
        keeps the two streams position-aligned. Offline both have full length
        and the gain applies as-is.
        """
        y, inv = args[0], args[1]
        iv = inv[0].detach().cpu().numpy()
        if state is None:
            # Output positions map 1:1 onto input positions (the delay is
            # availability, not a shift), so offline the gain applies as-is.
            take = min(y.shape[-1], len(iv))
            d = np.concatenate([iv[:take], np.ones(y.shape[-1] - take, dtype=iv.dtype)])
        else:
            # The FIFO is filled by the agc node (see there for why); this side
            # only pops as many samples as the output carries.
            key = f"fifo:{node.inputs[1]}"
            fifo = state.tail.get(key)
            fifo = iv[:0] if fifo is None else fifo
            take = min(y.shape[-1], len(fifo))
            d = np.concatenate([fifo[:take],
                                np.ones(y.shape[-1] - take, dtype=iv.dtype)])
            state.tail[key] = fifo[take:]
        return y * torch.from_numpy(np.ascontiguousarray(d))[None].to(y.dtype)

    def _limit(self, node, args, state):
        """Cap how much a frame may be amplified relative to a reference frame.

        An enhancer attenuates noise; it has no business making a frame louder
        than it arrived. Without this the deep filter's three tanh-bounded taps can
        sum to +12 dB, and on a normal-level input that plus any output gain drives
        a third of the samples past full scale — level-dependently, so a quiet test
        file passes and a real call clips.
        """
        y, ref = args[0], args[1]
        max_db = float(node.attrs.get("max_gain_db", 12.0))
        cap = 10.0 ** (max_db / 20.0)
        e_y = y.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        e_r = ref.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        scale = torch.clamp(cap * e_r / e_y.clamp_min(1e-12), max=1.0)
        return y * scale.to(y.dtype)

    def _mix(self, node, args, state):
        a, b, alpha = args[0], args[1], args[2]
        w = alpha.reshape(alpha.shape[0], *([1] * (a.dim() - 2)), -1).to(a.dtype)
        return w * a + (1.0 - w) * b


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
