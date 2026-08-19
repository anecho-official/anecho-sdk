'use strict';
// The same API shape as the Python SDK: Model / ProcessorConfig / Processor, with a
// context for delay, parameters and reset. Streaming is exact — every stateful node
// carries its own history — so block-wise output equals one-shot, which the tests
// assert against the Python runtime rather than assume.
const { readModel } = require('./container');
const { Executor, State } = require('./executor');
const { verifyLicense } = require('./license');

// Set at build time, exactly like the Python runtime's VERIFY_KEY: empty disables
// enforcement (dev), a 32-byte ed25519 public key requires a valid token. The key is
// public — embedding it gives an attacker nothing to sign with.
const PRODUCT = 'anecho-openspace';
const VERIFY_KEY = Buffer.from('f1d3a7a30445ec60ac7d91569749ff327eeebe28671fbd976170f13145fa0bc2', 'hex');

const ProcessorParameter = Object.freeze({
  Bypass: 'bypass',
  EnhancementLevel: 'enhancement_level',
  VoiceGain: 'voice_gain',
});

class Model {
  constructor(header, tensors, graph) {
    this.header = header; this.tensors = tensors; this.graph = graph;
    this.config = header.config || {};
  }

  static fromFile(path, { secret = null } = {}) {
    const { header, tensors, graph } = readModel(path, { secret, requireGraph: true });
    return new Model(header, tensors, graph);
  }

  get id() { return this.header.model_id || 'anecho'; }
  get sampleRate() { return this.config.sr || 16000; }
  get nFft() { return this.graph.numerics.stft.n_fft; }
  get hop() { return this.graph.numerics.stft.hop; }
  get blockSize() { return this.hop; }

  /** Frames of future context the graph needs; 0 means streaming is exact. */
  get lookahead() {
    const axes = this.graph.numerics.conv_kernel_axes || ['time', 'freq'];
    const ti = axes.indexOf('time') >= 0 ? axes.indexOf('time') : 1;
    let total = 0;
    for (const n of this.graph.nodes) {
      if ((n.op !== 'conv1d' && n.op !== 'conv2d') || n.attrs.causal) continue;
      const pad = n.attrs.padding ?? 0;
      total += Array.isArray(pad) ? (pad[ti] ?? 0) : pad;
    }
    return total;
  }

  /** Output delay in samples, derived from the graph rather than asserted. */
  get audioDelay() { return this.nFft - this.hop + this.lookahead * this.hop; }

  describe() {
    const lines = [`${this.id} [${this.header.format}]`,
      `  tensors ${this.tensors.size}, nodes ${this.graph.nodes.length}`,
      `  delay ${this.audioDelay} samples, block ${this.blockSize}, ${this.sampleRate} Hz`];
    for (const n of this.graph.nodes) lines.push(`    ${n.name.padEnd(16)} ${n.op}`);
    return lines.join('\n');
  }
}

class ProcessorConfig {
  constructor(sampleRate, blockSize, variableBlockSize = false) {
    this.sampleRate = sampleRate; this.blockSize = blockSize;
    this.variableBlockSize = variableBlockSize;
  }
  static optimal(model) { return new ProcessorConfig(model.sampleRate, model.blockSize); }
}

class ProcessorContext {
  constructor(p) { this._p = p; }
  getAudioDelay() { return this._p.audioDelay; }
  reset() { this._p._reset(); }
  setParameter(name, value) { this._p._setParameter(name, value); }
  getParameter(name) { return this._p._params[name]; }
  parameters() { return { ...this._p._params }; }
}

class Processor {
  constructor(model, licenseKey = '', config = null,
              { verifyKey = null, product = PRODUCT } = {}) {
    this.model = model;
    const vk = verifyKey !== null ? verifyKey : VERIFY_KEY;
    this.claims = vk.length ? verifyLicense(licenseKey, vk, product, 'enhance')
                            : { dev: true };
    this.config = config || ProcessorConfig.optimal(model);
    if (this.config.sampleRate !== model.sampleRate) {
      throw new Error(`model is ${model.sampleRate} Hz, got ${this.config.sampleRate}`);
    }
    if (!this.config.variableBlockSize && this.config.blockSize % model.hop !== 0) {
      throw new Error(`block size ${this.config.blockSize} is not a multiple of the ` +
                      `${model.hop}-sample hop`);
    }
    this._ex = new Executor(model.graph, model.tensors);
    this._params = { bypass: false, enhancement_level: 1.0, voice_gain: 0.0 };
    // Samples through process(), bypass included — bypass still holds a session
    // open. Drained by takeProcessedMs(); telemetry.js reads it on its interval.
    this._processedSamples = 0;
    this._reset();
  }

  getContext() { return new ProcessorContext(this); }
  initialize(config) { if (config) this.config = config; this._reset(); }
  terminateSession() { this._reset(); }

  /** This processor's ACTUAL output delay: the model's own, plus one hop of
   *  framing buffer for variable blocks. Without that hop a fractional block
   *  leaves the output side ahead of production and mid-stream zeros get emitted
   *  as signal — a desync that never heals, not a delay. */
  get audioDelay() {
    return this.model.audioDelay + (this.config.variableBlockSize ? this.model.hop : 0);
  }

  _reset() {
    this._state = new State();
    this._pending = new Float32Array(0);
    const delay = this.audioDelay;
    // Both paths start primed with the processor delay, so after N input samples
    // at least N are available on each; the shortfall pad below is an emergency,
    // not a code path.
    this._out = new Float32Array(delay);
    this._dry = new Float32Array(delay);
  }

  _setParameter(name, value) {
    if (!(name in this._params)) throw new Error(`unknown parameter '${name}'`);
    if (name === ProcessorParameter.EnhancementLevel && (value < 0 || value > 1)) {
      throw new Error('EnhancementLevel must be in [0, 1]');
    }
    this._params[name] = name === ProcessorParameter.Bypass ? Boolean(value) : Number(value);
  }

  /** Milliseconds processed since the last call, and reset the counter. */
  takeProcessedMs() {
    const n = this._processedSamples;
    this._processedSamples = 0;
    return Math.round((n * 1000) / this.config.sampleRate);
  }

  /** Enhance one block; returns the same number of samples. */
  process(block) {
    const x = Float32Array.from(block);
    this._processedSamples += x.length;
    if (this._params.bypass) return x;

    const merged = new Float32Array(this._pending.length + x.length);
    merged.set(this._pending); merged.set(x, this._pending.length);
    const hop = this.model.hop;
    const n = Math.floor(merged.length / hop) * hop;
    this._pending = merged.subarray(n).slice();
    if (n > 0) {
      const chunk = merged.subarray(0, n);
      const y = this._ex.run({ wav: { d: chunk, s: [1, n] } }, this._state).wav;
      const grown = new Float32Array(this._out.length + y.d.length);
      grown.set(this._out); grown.set(y.d, this._out.length);
      this._out = grown;
    }

    const dry = new Float32Array(this._dry.length + x.length);
    dry.set(this._dry); dry.set(x, this._dry.length);
    this._dry = dry;

    const want = x.length;
    if (this._out.length < want) {
      const padded = new Float32Array(want);
      padded.set(this._out);
      this._out = padded;
    }
    const wet = this._out.subarray(0, want);
    this._out = this._out.subarray(want).slice();
    const dryOut = this._dry.subarray(0, want);
    this._dry = this._dry.subarray(want).slice();

    const gain = Math.pow(10, this._params.voice_gain / 20);
    const level = this._params.enhancement_level;
    const out = new Float32Array(want);
    for (let i = 0; i < want; i++) out[i] = level * gain * wet[i] + (1 - level) * dryOut[i];
    return out;
  }

  /** Offline: the whole signal in one pass, no streaming state. */
  enhance(signal) {
    const x = Float32Array.from(signal);
    if (this._params.bypass) return x;
    const y = this._ex.run({ wav: { d: x, s: [1, x.length] } }).wav.d;
    const gain = Math.pow(10, this._params.voice_gain / 20);
    const level = this._params.enhancement_level;
    const n = Math.min(y.length, x.length);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = level * gain * y[i] + (1 - level) * x[i];
    return out;
  }

  /** Like process, but also returns intermediate activations by node name. */
  processInspect(block, nodes = null) {
    const x = Float32Array.from(block);
    const res = this._ex.run({ wav: { d: x, s: [1, x.length] } }, this._state, true);
    const values = nodes ? new Map(nodes.filter((n) => res.values.has(n)).map((n) => [n, res.values.get(n)]))
                         : res.values;
    return { wav: res.wav, values };
  }
}

/** Shift the output back by the algorithmic delay and fade the wind-up in. */
function align(y, delay, length) {
  const out = new Float32Array(length);
  const avail = Math.max(0, Math.min(length, y.length - delay));
  out.set(y.subarray(delay, delay + avail));
  for (let i = 0; i < Math.min(delay, length); i++) out[i] *= i / delay;
  return out;
}

module.exports = { Model, Processor, ProcessorConfig, ProcessorContext, ProcessorParameter, align };
