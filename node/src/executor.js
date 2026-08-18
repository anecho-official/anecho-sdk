'use strict';
// Graph executor. Layout conventions come from the file, not from here:
//   conv1d / conv2d / batchnorm   [1, C, ...]   channel-second
//   gru / linear / grouped_linear [1, T, C]     time-second
//   spectra                       [1, F, T]     complex
// Batch is always 1: this is a streaming runtime, and assuming it makes every
// index calculation below simpler and faster.
const { rfft, irfft, hann } = require('./fft');

const GATE_PERM = { rzn: [0, 1, 2], zrn: [1, 0, 2], nrz: [1, 2, 0], nzr: [2, 1, 0] };

const real = (s, fill) => ({ d: fill ? new Float32Array(prod(s)).fill(fill) : new Float32Array(prod(s)), s });
const cplx = (s) => ({ re: new Float32Array(prod(s)), im: new Float32Array(prod(s)), s });
const prod = (s) => s.reduce((a, b) => a * b, 1);
const isC = (t) => t.re !== undefined;

class State {
  constructor() { this.hidden = new Map(); this.tail = new Map(); }
  reset() { this.hidden.clear(); this.tail.clear(); }
}

class Executor {
  constructor(graph, tensors) {
    this.graph = graph;
    this.T = tensors;
    this.numerics = graph.numerics;
    this.win = hann(graph.numerics.stft.n_fft);
    const missing = [...new Set(graph.nodes.map((n) => n.op))].filter((op) => !OPS[op]);
    if (missing.length) throw new Error(`this build cannot execute ${missing.join(', ')}`);
  }

  param(node, role) {
    const name = (node.params || {})[role];
    return name === undefined ? null : this.T.get(name);
  }

  run(inputs, state = null, capture = false) {
    const env = new Map();
    for (const name of this.graph.inputs) {
      if (!(name in inputs)) throw new Error(`missing graph input '${name}'`);
      env.set(name, inputs[name]);
    }
    for (const node of this.graph.nodes) {
      const args = node.inputs.map((v) => env.get(v));
      const out = OPS[node.op].call(this, node, args, state);
      if (node.op === 'stft' && out.s[2] === 0) {          // no complete frame yet
        const empty = { d: new Float32Array(0), s: [1, 0] };
        const res = {};
        for (const k of Object.keys(this.graph.outputs)) res[k] = empty;
        return capture ? { ...res, values: new Map() } : res;
      }
      node.outputs.forEach((name, i) => env.set(name, Array.isArray(out) ? out[i] : out));
    }
    const res = {};
    for (const [pub, v] of Object.entries(this.graph.outputs)) res[pub] = env.get(v);
    if (capture) {
      res.values = new Map(this.graph.nodes.filter((n) => n.outputs.length)
        .map((n) => [n.name, env.get(n.outputs[0])]));
    }
    return res;
  }
}

const OPS = {
  stft(node, args, state) {
    const s = this.numerics.stft;
    const nFft = node.attrs.n_fft ?? s.n_fft, hop = node.attrs.hop ?? s.hop;
    const scale = s.scale ?? 1.0;
    let x = args[0].d;
    if (state) {
      let prev = state.tail.get(node.name);
      if (!prev) prev = new Float32Array(s.center ? nFft >> 1 : 0);
      const buf = new Float32Array(prev.length + x.length);
      buf.set(prev); buf.set(x, prev.length);
      const frames = Math.max(0, Math.floor((buf.length - nFft) / hop) + 1);
      state.tail.set(node.name, buf.subarray(frames * hop).slice());
      x = buf;
      return frameFft(x, nFft, hop, frames, this.win, scale);
    }
    let sig = x;
    if (s.center) {                                        // constant padding, as declared
      const pad = nFft >> 1;
      sig = new Float32Array(x.length + 2 * pad);
      sig.set(x, pad);
    }
    const frames = Math.max(0, Math.floor((sig.length - nFft) / hop) + 1);
    return frameFft(sig, nFft, hop, frames, this.win, scale);
  },

  istft(node, args, state) {
    const s = this.numerics.stft;
    const nFft = node.attrs.n_fft ?? s.n_fft, hop = node.attrs.hop ?? s.hop;
    const scale = s.scale ?? 1.0;
    const spec = args[0], F = spec.s[1], T = spec.s[2];
    const length = args.length > 1 ? args[1].s[args[1].s.length - 1] : null;
    const syn = (s.synthesis ?? 'wola') === 'wola' ? this.win : new Float32Array(nFft).fill(1);

    const outLen = (T - 1) * hop + nFft;
    const y = new Float64Array(outLen), env = new Float64Array(outLen);
    const re = new Float32Array(F), im = new Float32Array(F);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) { re[f] = spec.re[f * T + t] / scale; im[f] = spec.im[f * T + t] / scale; }
      const frame = irfft(re, im, nFft);
      const off = t * hop;
      for (let i = 0; i < nFft; i++) {
        y[off + i] += frame[i] * syn[i];
        env[off + i] += syn[i] * this.win[i];
      }
    }
    if (state) {
      const carried = state.tail.get(node.name);
      let skip = carried ? carried.skip : (s.center ? nFft >> 1 : 0);
      let py = carried ? carried.y : null, pe = carried ? carried.env : null;
      let width = outLen;
      if (py) width = Math.max(width, py.length);
      const ay = new Float64Array(width), ae = new Float64Array(width);
      ay.set(y); ae.set(env);
      if (py) { for (let i = 0; i < py.length; i++) { ay[i] += py[i]; ae[i] += pe[i]; } }
      const keep = Math.max(0, width - (nFft - hop));
      let emax = 0; for (let i = 0; i < ae.length; i++) emax = Math.max(emax, ae[i]);
      const floor = 1e-2 * emax;
      const outHead = new Float32Array(keep);
      // On the first emission the head is as incomplete as the tail — one frame has
      // reached it — so it is left at zero rather than divided by a partial envelope.
      const from = py ? 0 : Math.min(nFft - hop, keep);
      for (let i = from; i < keep; i++) outHead[i] = ay[i] / Math.max(ae[i], floor);
      state.tail.set(node.name, { y: ay.subarray(keep).slice(), env: ae.subarray(keep).slice(),
                                  skip: Math.max(0, skip - Math.min(skip, keep)) });
      const drop = Math.min(skip, keep);
      const kept = outHead.subarray(drop);
      return { d: kept.slice(), s: [1, kept.length] };
    }
    let emax = 0; for (let i = 0; i < env.length; i++) emax = Math.max(emax, env[i]);
    const floor = 1e-2 * emax;
    let out = new Float32Array(outLen);
    // Only one frame has contributed to the last nFft-hop samples, so their
    // envelope is incomplete and dividing by it inflates them by up to 1/floor.
    // Streaming carries those samples instead of emitting them; offline says the
    // same thing by leaving them at zero.
    const complete = Math.max(0, outLen - (nFft - hop));
    const head = Math.min(nFft - hop, outLen);
    for (let i = head; i < complete; i++) out[i] = y[i] / Math.max(env[i], floor);
    if (s.center) out = out.subarray(nFft >> 1);
    if (length !== null) {
      const trimmed = new Float32Array(length);
      trimmed.set(out.subarray(0, Math.min(length, out.length)));
      out = trimmed;
    }
    return { d: out.slice(), s: [1, out.length] };
  },

  erb_feature(node, args) {
    const cfg = this.numerics.erb_feature;
    const spec = args[0], F = spec.s[1], T = spec.s[2];
    const mat = this.param(node, 'matrix'), E = mat.s[1];
    const out = real([1, E, T]);
    const eps = cfg.eps ?? 1e-10, div = cfg.scale ?? 1.0;
    for (let e = 0; e < E; e++) {
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let f = 0; f < F; f++) {
          const w = mat.d[f * E + e];
          if (w === 0) continue;
          const mag = Math.hypot(spec.re[f * T + t], spec.im[f * T + t]);
          acc += w * (cfg.domain === 'energy' ? mag * mag : mag);
        }
        let v = acc + eps;
        v = cfg.log === 'db20' ? 20 * Math.log10(v)
          : cfg.log === 'log10' ? Math.log10(v)
          : cfg.log === 'ln' ? Math.log(v) : v;
        out.d[e * T + t] = v / div;
      }
    }
    return out;
  },

  df_feature(node, args) {
    const bins = node.attrs.bins, part = node.attrs.part ?? 'real';
    const spec = args[0], T = spec.s[2], sc = node.attrs.scale ?? 1.0;
    const out = real([1, 1, T, bins]);
    for (let f = 0; f < bins; f++) {
      for (let t = 0; t < T; t++) {
        const re = spec.re[f * T + t], im = spec.im[f * T + t];
        out.d[t * bins + f] = sc * (part === 'real' ? re : part === 'imag' ? im : Math.hypot(re, im));
      }
    }
    return out;
  },

  conv2d(node, args, state) {
    const x = args[0], w = this.param(node, 'weight'), b = this.param(node, 'bias');
    const [, C, H, W] = x.s;
    const [O, Cg, kh, kw] = w.s;
    const groups = node.attrs.groups ?? 1;
    const stride = node.attrs.stride ?? [1, 1];
    let padH = 0, padW = 0, xin = x, Hin = H, Win = W;
    const axes = this.numerics.conv_kernel_axes || ['time', 'freq'];
    const timeAxis = 2 + (axes.indexOf('time') >= 0 ? axes.indexOf('time') : 1);
    const pad = node.attrs.padding ?? [0, 0];

    if (node.attrs.causal) {
      // pad the non-time axis symmetrically, the time axis from carried history
      const other = timeAxis === 2 ? 3 : 2;
      const pOther = pad[other - 2] ?? 0;
      if (pOther) { xin = padAxis(xin, other, pOther, pOther); }
      const k = timeAxis === 2 ? kh : kw;
      xin = causalPad.call(this, node, xin, timeAxis, k, state);
    } else if (pad[0] || pad[1]) {
      xin = padAxis(padAxis(xin, 2, pad[0], pad[0]), 3, pad[1], pad[1]);
    }
    Hin = xin.s[2]; Win = xin.s[3];
    const outH = Math.floor((Hin - kh) / stride[0]) + 1;
    const outW = Math.floor((Win - kw) / stride[1]) + 1;
    const out = real([1, O, outH, outW]);
    const perGroupOut = O / groups, perGroupIn = C / groups;
    for (let o = 0; o < O; o++) {
      const g = Math.floor(o / perGroupOut);
      const bias = b ? b.d[o] : 0;
      for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
          let acc = bias;
          for (let c = 0; c < Cg; c++) {
            const ci = g * perGroupIn + c;
            for (let a = 0; a < kh; a++) {
              const ii = i * stride[0] + a;
              for (let bb = 0; bb < kw; bb++) {
                acc += xin.d[(ci * Hin + ii) * Win + j * stride[1] + bb] *
                       w.d[((o * Cg + c) * kh + a) * kw + bb];
              }
            }
          }
          out.d[(o * outH + i) * outW + j] = acc;
        }
      }
    }
    return out;
  },

  conv1d(node, args, state) {
    const x = args[0], w = this.param(node, 'weight'), b = this.param(node, 'bias');
    const [, C, T] = x.s, [O, Cg, k] = w.s;
    const groups = node.attrs.groups ?? 1, stride = node.attrs.stride ?? 1;
    let xin = x;
    if (node.attrs.causal) xin = causalPad.call(this, node, x, 2, k, state);
    else if (node.attrs.padding) xin = padAxis(x, 2, node.attrs.padding, node.attrs.padding);
    const Tin = xin.s[2];
    const outT = Math.floor((Tin - k) / stride) + 1;
    const out = real([1, O, outT]);
    const pgo = O / groups, pgi = C / groups;
    for (let o = 0; o < O; o++) {
      const g = Math.floor(o / pgo), bias = b ? b.d[o] : 0;
      for (let t = 0; t < outT; t++) {
        let acc = bias;
        for (let c = 0; c < Cg; c++) {
          for (let a = 0; a < k; a++) {
            acc += xin.d[(g * pgi + c) * Tin + t * stride + a] * w.d[(o * Cg + c) * k + a];
          }
        }
        out.d[o * outT + t] = acc;
      }
    }
    return out;
  },

  batchnorm(node, args) {
    const x = args[0], g = this.param(node, 'weight'), b = this.param(node, 'bias');
    const m = this.param(node, 'running_mean'), v = this.param(node, 'running_var');
    const eps = this.numerics.batchnorm_eps;
    const C = x.s[1], inner = prod(x.s) / C;
    const out = real(x.s);
    for (let c = 0; c < C; c++) {
      const sc = g.d[c] / Math.sqrt(v.d[c] + eps), sh = b.d[c] - m.d[c] * sc;
      for (let i = 0; i < inner; i++) out.d[c * inner + i] = x.d[c * inner + i] * sc + sh;
    }
    return out;
  },

  activation(node, args) {
    const kind = node.attrs.kind ?? 'relu', x = args[0], out = real(x.s);
    for (let i = 0; i < x.d.length; i++) {
      const v = x.d[i];
      out.d[i] = kind === 'relu' ? (v > 0 ? v : 0)
        : kind === 'sigmoid' ? 1 / (1 + Math.exp(-v))
        : kind === 'tanh' ? Math.tanh(v)
        : kind === 'silu' ? v / (1 + Math.exp(-v)) : v;
    }
    return out;
  },

  gru(node, args, state) {
    const x = args[0], T = x.s[1], H = node.attrs.hidden, layers = node.attrs.layers;
    const perm = GATE_PERM[this.numerics.gru_gate_order];
    if (!perm) throw new Error(`unknown gru_gate_order ${this.numerics.gru_gate_order}`);
    let input = x.d, inSize = x.s[2];
    const carried = state ? state.hidden.get(node.name) : null;
    const hs = [];
    for (let l = 0; l < layers; l++) {
      const wih = this.param(node, `weight_ih_l${l}`), whh = this.param(node, `weight_hh_l${l}`);
      const bih = this.param(node, `bias_ih_l${l}`), bhh = this.param(node, `bias_hh_l${l}`);
      const h = new Float32Array(H);
      if (carried) h.set(carried[l]);
      const outSeq = new Float32Array(T * H);
      const gi = new Float32Array(3 * H), gh = new Float32Array(3 * H);
      for (let t = 0; t < T; t++) {
        for (let r = 0; r < 3 * H; r++) {
          let a = bih ? bih.d[r] : 0, c = bhh ? bhh.d[r] : 0;
          for (let i = 0; i < inSize; i++) a += wih.d[r * inSize + i] * input[t * inSize + i];
          for (let i = 0; i < H; i++) c += whh.d[r * H + i] * h[i];
          gi[r] = a; gh[r] = c;
        }
        const [ri, zi, ni] = perm.map((p) => p * H);      // slice starts, per the file
        for (let i = 0; i < H; i++) {
          const rg = 1 / (1 + Math.exp(-(gi[ri + i] + gh[ri + i])));
          const zg = 1 / (1 + Math.exp(-(gi[zi + i] + gh[zi + i])));
          const ng = Math.tanh(gi[ni + i] + rg * gh[ni + i]);
          h[i] = (1 - zg) * ng + zg * h[i];
        }
        outSeq.set(h, t * H);
      }
      hs.push(h.slice());
      input = outSeq; inSize = H;
    }
    if (state) state.hidden.set(node.name, hs);
    return { d: input, s: [1, T, H] };
  },

  grouped_linear(node, args) {
    const x = args[0], w = this.param(node, 'weight'), b = this.param(node, 'bias');
    const [G, ipg, opg] = w.s, T = x.s[1];
    const out = real([1, T, G * opg]);
    for (let t = 0; t < T; t++) {
      for (let g = 0; g < G; g++) {
        for (let o = 0; o < opg; o++) {
          let acc = b ? b.d[g * opg + o] : 0;
          for (let i = 0; i < ipg; i++) {
            acc += x.d[t * G * ipg + g * ipg + i] * w.d[(g * ipg + i) * opg + o];
          }
          out.d[t * G * opg + g * opg + o] = acc;
        }
      }
    }
    return out;
  },

  linear(node, args) {
    const x = args[0], w = this.param(node, 'weight'), b = this.param(node, 'bias');
    const inOut = this.numerics.linear_weight_layout === 'in_out';
    const inF = inOut ? w.s[0] : w.s[1], outF = inOut ? w.s[1] : w.s[0];
    const T = x.s[1], out = real([1, T, outF]);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < outF; o++) {
        let acc = b ? b.d[o] : 0;
        for (let i = 0; i < inF; i++) {
          acc += x.d[t * inF + i] * (inOut ? w.d[i * outF + o] : w.d[o * inF + i]);
        }
        out.d[t * outF + o] = acc;
      }
    }
    return out;
  },

  reshape(node, args) {
    const x = args[0];
    const spec = node.attrs.shape.map((v) => v === 'B' ? x.s[0] : v === 'T' ? x.s[x.s.length - 1] : v);
    const total = prod(x.s), known = spec.filter((v) => v !== -1).reduce((a, b) => a * b, 1);
    const shape = spec.map((v) => v === -1 ? total / known : v);
    return isC(x) ? { re: x.re, im: x.im, s: shape } : { d: x.d, s: shape };
  },

  transpose(node, args) {
    const x = args[0], dims = node.attrs.dims, n = dims.length;
    const outShape = dims.map((d) => x.s[d]);
    const inStr = strides(x.s), outStr = strides(outShape);
    const out = isC(x) ? cplx(outShape) : real(outShape);
    const total = prod(outShape), idx = new Array(n).fill(0);
    for (let k = 0; k < total; k++) {
      let src = 0;
      for (let d = 0; d < n; d++) src += idx[d] * inStr[dims[d]];
      if (isC(x)) { out.re[k] = x.re[src]; out.im[k] = x.im[src]; } else { out.d[k] = x.d[src]; }
      for (let d = n - 1; d >= 0; d--) { if (++idx[d] < outShape[d]) break; idx[d] = 0; }
    }
    return out;
  },

  add(node, args) {
    if (args.length === 2) return elemwise(args[0], args[1], (a, b) => a + b);
    const sc = node.attrs.scalar, x = args[0], out = real(x.s);
    for (let i = 0; i < x.d.length; i++) out.d[i] = x.d[i] + sc;
    return out;
  },

  agc(node, args, state) {
    // Pin the input to the level the model was trained at; emit the inverse
    // gain so the output can undo it. The trained heads' decision boundaries
    // encode the absolute level of their training material (the same audio
    // gates 60% of frames at -42 dBFS and 0% at -22), so the input is
    // normalised DOWNWARD to that level — quieter input already behaves and
    // passes at unity. The tracker: per-hop power, smoothed over ~half a
    // second, then a ratchet with instant attack and a slow decay; hops below
    // the silence floor leave it untouched. Gains are constant within a hop
    // and hops divide the streaming block, so block-wise equals one-shot.
    const x = args[0], n = x.s[x.s.length - 1];
    const target = Math.pow(10, (node.attrs.target_db ?? -35.0) / 10);
    const hop = node.attrs.hop ?? 240;
    const decay = Math.pow(10, -(node.attrs.decay_db_per_s ?? 0.1) * (hop / 16000) / 10);
    const gmin = Math.pow(10, (node.attrs.min_gain_db ?? -36.0) / 20);
    const floor = Math.pow(10, (node.attrs.floor_db ?? -60.0) / 10);
    const aSm = Math.exp(-hop / (16000 * (node.attrs.smooth_ms ?? 500.0) / 1000));

    let sm = null, peak = null;
    const carried = state ? state.tail.get(node.name) : null;
    if (carried) { sm = carried.sm; peak = carried.peak; }
    const y = real(x.s), inv = real(x.s);
    for (let h0 = 0; h0 < n; h0 += hop) {              // a trailing partial hop is fine
      const m = Math.min(hop, n - h0);
      let p2 = 0;
      for (let i = h0; i < h0 + m; i++) p2 += x.d[i] * x.d[i];
      p2 /= m;
      if (p2 > floor) {
        sm = sm === null ? p2 : p2 * (1 - aSm) + sm * aSm;
        peak = peak === null || sm > peak ? sm : peak * decay;
      } else if (peak !== null) {
        peak = peak * decay;
      }
      const ref = peak === null ? target : peak;       // silence so far: unity
      const g = Math.min(Math.max(Math.sqrt(target / Math.max(ref, 1e-12)), gmin), 1.0);
      for (let i = h0; i < h0 + m; i++) { y.d[i] = x.d[i] * g; inv.d[i] = 1 / g; }
    }
    if (state) {
      state.tail.set(node.name, { sm, peak });
      // Fill the undo FIFO here: on the first streaming call the stft has no
      // complete frame and the run aborts before downstream ops execute, so a
      // FIFO filled by agc_undo would start one block late and stay misaligned.
      const key = 'fifo:' + node.outputs[1];
      const prev = state.tail.get(key) || new Float32Array(0);
      const joined = new Float32Array(prev.length + inv.d.length);
      joined.set(prev); joined.set(inv.d, prev.length);
      state.tail.set(key, joined);
    }
    return [y, inv];
  },

  agc_undo(node, args, state) {
    // Undo the agc gain on the istft output, honouring the WOLA latency: the
    // gain series lives on the input timeline, the output lags it by one block.
    // A FIFO of pending gains absorbs both that lag and the empty first block.
    const y = args[0], inv = args[1];
    const iv = inv.d;
    const out = real(y.s);
    const n = y.d.length;
    if (!state) {
      // positions map 1:1 — the pipeline delay is availability, not a shift
      for (let i = 0; i < n; i++) out.d[i] = y.d[i] * (i < iv.length ? iv[i] : 1.0);
      return out;
    }
    // the agc node fills this FIFO (see there for why); only pop here
    const key = 'fifo:' + node.inputs[1];
    const fifo = state.tail.get(key) || new Float32Array(0);
    const take = Math.min(n, fifo.length);
    for (let i = 0; i < n; i++) out.d[i] = y.d[i] * (i < take ? fifo[i] : 1.0);
    state.tail.set(key, fifo.slice(take));
    return out;
  },

  mul(node, args) {
    if (args.length === 2) return elemwise(args[0], args[1], (a, b) => a * b);
    const sc = node.attrs.scalar, x = args[0];
    if (isC(x)) {
      const out = cplx(x.s);
      for (let i = 0; i < x.re.length; i++) { out.re[i] = x.re[i] * sc; out.im[i] = x.im[i] * sc; }
      return out;
    }
    const out = real(x.s);
    for (let i = 0; i < x.d.length; i++) out.d[i] = x.d[i] * sc;
    return out;
  },

  concat(node, args) {
    const dim = node.attrs.dim ?? -1;
    const d = dim < 0 ? args[0].s.length + dim : dim;
    if (d !== args[0].s.length - 1) throw new Error('concat is implemented on the last axis only');
    const rows = prod(args[0].s) / args[0].s[d];
    const widths = args.map((a) => a.s[d]);
    const total = widths.reduce((a, b) => a + b, 0);
    const shape = args[0].s.slice(); shape[d] = total;
    const out = real(shape);
    for (let r = 0; r < rows; r++) {
      let off = 0;
      args.forEach((a, k) => {
        out.d.set(a.d.subarray(r * widths[k], (r + 1) * widths[k]), r * total + off);
        off += widths[k];
      });
    }
    return out;
  },

  pool(node, args) {
    const x = args[0], dim = node.attrs.dim ?? -1;
    const d = dim < 0 ? x.s.length + dim : dim;
    if (d !== x.s.length - 1) throw new Error('pool is implemented on the last axis only');
    const inner = x.s[d], rows = prod(x.s) / inner;
    const shape = x.s.slice(0, d);
    const out = real(shape);
    for (let r = 0; r < rows; r++) {
      let acc = node.attrs.kind === 'max' ? -Infinity : 0;
      for (let i = 0; i < inner; i++) {
        const v = x.d[r * inner + i];
        acc = node.attrs.kind === 'max' ? Math.max(acc, v) : acc + v;
      }
      out.d[r] = node.attrs.kind === 'max' ? acc : acc / inner;
    }
    return out;
  },

  pixel_shuffle(node, args) {
    const x = args[0], r = node.attrs.factor ?? 2, order = node.attrs.order ?? 'cr';
    const [, C2, T, F] = x.s, C = C2 / r;
    const out = real([1, C, T, F * r]);
    for (let c = 0; c < C; c++) {
      for (let k = 0; k < r; k++) {
        const src = order === 'cr' ? c * r + k : k * C + c;
        for (let t = 0; t < T; t++) {
          for (let f = 0; f < F; f++) {
            out.d[(c * T + t) * F * r + f * r + k] = x.d[(src * T + t) * F + f];
          }
        }
      }
    }
    return out;
  },

  apply_gains(node, args) {
    const spec = args[0], gains = args[1], mem = this.param(node, 'membership');
    const F = spec.s[1], T = spec.s[2], E = gains.s[1];
    const out = cplx(spec.s);
    for (let f = 0; f < F; f++) {
      for (let t = 0; t < T; t++) {
        let g = 0;
        for (let e = 0; e < E; e++) {
          const w = mem.d[e * F + f];
          if (w !== 0) g += gains.d[e * T + t] * w;
        }
        out.re[f * T + t] = spec.re[f * T + t] * g;
        out.im[f * T + t] = spec.im[f * T + t] * g;
      }
    }
    return out;
  },

  deep_filter(node, args, state) {
    const enh = args[0], noisy = args[1], raw = args[2];
    const bins = node.attrs.bins, order = node.attrs.order;
    const T = raw.s[2], F = enh.s[1];
    const layout = this.numerics.df_coef_layout;
    const hist = order - 1;

    // Streaming: the taps reach back past the block, so the previous frames of the
    // *noisy* low bins are state. Without this the first taps of every block read
    // zeros and each block starts over.
    let prevRe = state ? state.tail.get(node.name + ':re') : null;
    let prevIm = state ? state.tail.get(node.name + ':im') : null;
    if (!prevRe) { prevRe = new Float32Array(bins * hist); prevIm = new Float32Array(bins * hist); }
    const W = hist + T;
    const lowRe = new Float32Array(bins * W), lowIm = new Float32Array(bins * W);
    for (let f = 0; f < bins; f++) {
      for (let h = 0; h < hist; h++) {
        lowRe[f * W + h] = prevRe[f * hist + h]; lowIm[f * W + h] = prevIm[f * hist + h];
      }
      for (let t = 0; t < T; t++) {
        lowRe[f * W + hist + t] = noisy.re[f * T + t];
        lowIm[f * W + hist + t] = noisy.im[f * T + t];
      }
    }
    if (state) {
      const kr = new Float32Array(bins * hist), ki = new Float32Array(bins * hist);
      for (let f = 0; f < bins; f++) {
        for (let h = 0; h < hist; h++) {
          kr[f * hist + h] = lowRe[f * W + (W - hist + h)];
          ki[f * hist + h] = lowIm[f * W + (W - hist + h)];
        }
      }
      state.tail.set(node.name + ':re', kr);
      state.tail.set(node.name + ':im', ki);
    }

    const out = cplx(enh.s);
    out.re.set(enh.re); out.im.set(enh.im);
    for (let f = 0; f < bins; f++) {
      for (let t = 0; t < T; t++) {
        let ar = 0, ai = 0;
        for (let k = 0; k < order; k++) {
          const src = t + k;                       // index 0 is the oldest frame
          const base = layout === 'bin_major'
            ? ((f * order + k) * 2) * T + t
            : ((k * 2) * bins + f) * T + t;
          const cr = raw.d[base], ci = raw.d[base + T];
          const xr = lowRe[f * W + src], xi = lowIm[f * W + src];
          ar += cr * xr - ci * xi;
          ai += cr * xi + ci * xr;
        }
        out.re[f * T + t] = ar; out.im[f * T + t] = ai;
      }
    }
    return out;
  },

  mean_norm(node, args, state) {
    const alpha = node.attrs.alpha ?? 0.99, x = args[0];
    const inner = x.s[x.s.length - 1], rows = prod(x.s) / inner;
    const out = real(x.s);
    let acc = state ? state.hidden.get(node.name) : null;
    if (!acc) acc = new Float32Array(rows);
    for (let t = 0; t < inner; t++) {
      for (let r = 0; r < rows; r++) {
        const v = x.d[r * inner + t];
        acc[r] = v * (1 - alpha) + acc[r] * alpha;
        out.d[r * inner + t] = v - acc[r];
      }
    }
    if (state) state.hidden.set(node.name, acc);
    return out;
  },

  gate(node, args, state) {
    const a = node.attrs, prob = args[0], T = prob.d.length;
    const frameMs = a.frame_ms ?? 15.0;
    const attack = Math.max(1, Math.round((a.attack_ms ?? 5) / frameMs));
    const hold = Math.max(0, Math.round((a.hold_ms ?? 200) / frameMs));
    const decay = Math.exp(-frameMs / Math.max(a.release_ms ?? 100, 1e-6));
    const on = a.threshold_on ?? 0.5, off = a.threshold_off ?? 0.5;
    let st = state ? state.hidden.get(node.name) : null;
    let above = st ? st.above : 0, held = st ? st.held : 0;
    let active = st ? st.active : false, g = st ? st.g : 0;
    const out = real(prob.s);
    for (let k = 0; k < T; k++) {
      const p = prob.d[k];
      if (p >= on) { above++; if (above >= attack) { active = true; held = hold; } }
      else if (p < off) { above = 0; if (active) { held--; if (held <= 0) active = false; } }
      g = active ? 1 : g * decay;
      out.d[k] = g;
    }
    if (state) state.hidden.set(node.name, { above, held, active, g });
    return out;
  },

  limit(node, args) {
    const y = args[0], ref = args[1];
    const cap = Math.pow(10, (node.attrs.max_gain_db ?? 12) / 20);
    const F = y.s[1], T = y.s[2];
    const out = cplx(y.s);
    for (let t = 0; t < T; t++) {
      let ey = 0, er = 0;
      for (let f = 0; f < F; f++) {
        ey += y.re[f * T + t] ** 2 + y.im[f * T + t] ** 2;
        er += ref.re[f * T + t] ** 2 + ref.im[f * T + t] ** 2;
      }
      const scale = Math.min(1, (cap * Math.sqrt(er)) / Math.max(Math.sqrt(ey), 1e-12));
      for (let f = 0; f < F; f++) {
        out.re[f * T + t] = y.re[f * T + t] * scale;
        out.im[f * T + t] = y.im[f * T + t] * scale;
      }
    }
    return out;
  },
};

// ------------------------------------------------------------------ helpers
function strides(shape) {
  const st = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) st[i] = st[i + 1] * shape[i + 1];
  return st;
}

function frameFft(sig, nFft, hop, frames, win, scale) {
  const F = (nFft >> 1) + 1;
  const out = cplx([1, F, frames]);
  const buf = new Float32Array(nFft);
  for (let t = 0; t < frames; t++) {
    for (let i = 0; i < nFft; i++) buf[i] = sig[t * hop + i] * win[i];
    const S = rfft(buf, nFft);
    for (let f = 0; f < F; f++) { out.re[f * frames + t] = S.re[f] * scale; out.im[f * frames + t] = S.im[f] * scale; }
  }
  return out;
}

function padAxis(x, axis, before, after) {
  const shape = x.s.slice(); shape[axis] += before + after;
  const out = real(shape);
  const inStr = strides(x.s), outStr = strides(shape);
  const total = prod(x.s), idx = new Array(x.s.length).fill(0);
  for (let k = 0; k < total; k++) {
    let dst = 0;
    for (let d = 0; d < idx.length; d++) dst += (idx[d] + (d === axis ? before : 0)) * outStr[d];
    out.d[dst] = x.d[k];
    for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < x.s[d]) break; idx[d] = 0; }
  }
  return out;
}

/** Prepend the previous call's last k-1 frames instead of zeros, so blocks join. */
function causalPad(node, x, axis, k, state) {
  if (k <= 1) return x;
  const need = k - 1;
  let prev = state ? state.tail.get(node.name) : null;
  if (!prev) {
    const shape = x.s.slice(); shape[axis] = need;
    prev = real(shape);
  }
  const merged = concatAxis(prev, x, axis);
  if (state) state.tail.set(node.name, sliceAxis(merged, axis, merged.s[axis] - need, need));
  return merged;
}

function concatAxis(a, b, axis) {
  const shape = a.s.slice(); shape[axis] = a.s[axis] + b.s[axis];
  const out = real(shape);
  copyInto(out, a, axis, 0);
  copyInto(out, b, axis, a.s[axis]);
  return out;
}

function copyInto(dst, src, axis, offset) {
  const outStr = strides(dst.s), total = prod(src.s);
  const idx = new Array(src.s.length).fill(0);
  for (let k = 0; k < total; k++) {
    let o = 0;
    for (let d = 0; d < idx.length; d++) o += (idx[d] + (d === axis ? offset : 0)) * outStr[d];
    dst.d[o] = src.d[k];
    for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < src.s[d]) break; idx[d] = 0; }
  }
}

function sliceAxis(x, axis, start, count) {
  const shape = x.s.slice(); shape[axis] = count;
  const out = real(shape);
  const inStr = strides(x.s), total = prod(shape);
  const idx = new Array(shape.length).fill(0);
  for (let k = 0; k < total; k++) {
    let src = 0;
    for (let d = 0; d < idx.length; d++) src += (idx[d] + (d === axis ? start : 0)) * inStr[d];
    out.d[k] = x.d[src];
    for (let d = idx.length - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
  }
  return out;
}

function elemwise(a, b, fn) {
  const complex = isC(a) || isC(b);
  const shape = prod(a.s) >= prod(b.s) ? a.s : b.s;
  const n = prod(shape);
  const get = (t, i) => {
    const len = prod(t.s);
    // broadcast a [1,1,T] control over a [1,F,T] spectrum
    const j = len === n ? i : i % len;
    return isC(t) ? [t.re[j], t.im[j]] : [t.d[j], 0];
  };
  if (complex) {
    const out = cplx(shape);
    for (let i = 0; i < n; i++) {
      const [ar, ai] = get(a, i), [br, bi] = get(b, i);
      if (fn(1, 1) === 2) { out.re[i] = ar + br; out.im[i] = ai + bi; }
      else { out.re[i] = ar * br - ai * bi; out.im[i] = ar * bi + ai * br; }
    }
    return out;
  }
  const out = real(shape);
  for (let i = 0; i < n; i++) out.d[i] = fn(get(a, i)[0], get(b, i)[0]);
  return out;
}

module.exports = { Executor, State };
