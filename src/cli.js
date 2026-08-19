#!/usr/bin/env node
'use strict';
// anecho CLI: process a file, or record from the Mac microphone and play both back.
const fs = require('fs');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');
const os = require('os');
const { Model, Processor, ProcessorParameter, align } = require('./processor');
const { readWav, writeWav } = require('./wav');

const USAGE = `anecho — voice-focus runtime

  anecho process <model.anecho> <in.wav> <out.wav> [options]
  anecho mic     <model.anecho> [--seconds 15] [--device N] [--list] [--keep DIR]
  anecho inspect <model.anecho>

options
  --level 0.8     dry/wet, 0..1        --gain -6    dB on the enhanced path
  --license TOKEN or env ANECHO_LICENSE   licence token from your dashboard
  --bypass        return the input     --no-play    mic: do not play back
`;

const TAKES_VALUE = new Set(['--level', '--gain', '--seconds', '--device', '--keep', '--license']);

/** Split flags from positionals, so `mic --list` is not read as a model path. */
function parse(argv) {
  const f = { level: 1.0, gain: 0.0, bypass: false, seconds: 15, play: true,
              keep: null, device: null, list: false, args: [],
              // A licence travels either way; the env var keeps it out of shell history.
              license: process.env.ANECHO_LICENSE || '' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) { f.args.push(a); continue; }
    const v = TAKES_VALUE.has(a) ? argv[++i] : null;
    if (TAKES_VALUE.has(a) && v === undefined) throw new Error(`${a} needs a value`);
    if (a === '--level') f.level = Number(v);
    else if (a === '--gain') f.gain = Number(v);
    else if (a === '--seconds') f.seconds = Number(v);
    else if (a === '--device') f.device = Number(v);
    else if (a === '--keep') f.keep = v;
    else if (a === '--bypass') f.bypass = true;
    else if (a === '--no-play') f.play = false;
    else if (a === '--list') f.list = true;
    else throw new Error(`unknown option ${a}`);
  }
  return f;
}

function resample(x, from, to) {
  if (from === to) return x;
  const n = Math.round((x.length * to) / from);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = (i * (x.length - 1)) / Math.max(n - 1, 1);
    const j = Math.floor(p), frac = p - j;
    out[i] = x[j] * (1 - frac) + x[Math.min(j + 1, x.length - 1)] * frac;
  }
  return out;
}

function enhanceSignal(model, proc, samples, rate) {
  const sig = resample(samples, rate, model.sampleRate);
  const B = proc.config.blockSize;
  const pad = (B - (sig.length % B)) % B;
  const padded = new Float32Array(sig.length + pad);
  padded.set(sig);
  const chunks = [];
  for (let i = 0; i < padded.length; i += B) chunks.push(proc.process(padded.subarray(i, i + B)));
  const streamed = new Float32Array(chunks.length * B);
  chunks.forEach((c, k) => streamed.set(c, k * B));
  const aligned = align(streamed, model.audioDelay, sig.length);
  return resample(aligned, model.sampleRate, rate).subarray(0, samples.length);
}

const rms = (x) => 10 * Math.log10(x.reduce((a, v) => a + v * v, 0) / Math.max(x.length, 1) + 1e-12);

function cmdProcess(argv) {
  const f = parse(argv);
  const [modelPath, inPath, outPath] = f.args;
  if (!outPath) { console.log(USAGE); process.exit(1); }
  if (!fs.existsSync(inPath)) {
    const dir = path.dirname(inPath) || '.';
    const wavs = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.toLowerCase().endsWith('.wav')) : [];
    console.error(`no such file: ${inPath}` + (wavs.length ? `\n  WAVs there: ${wavs.slice(0, 8).join(', ')}` : ''));
    process.exit(1);
  }
  const model = Model.fromFile(modelPath);
  const proc = new Processor(model, f.license);
  const ctx = proc.getContext();
  ctx.setParameter(ProcessorParameter.EnhancementLevel, f.level);
  ctx.setParameter(ProcessorParameter.VoiceGain, f.gain);
  ctx.setParameter(ProcessorParameter.Bypass, f.bypass);

  const w = readWav(inPath);
  console.log(`${inPath}: ${(w.samples.length / w.rate).toFixed(1)} s, ${w.rate} Hz, ${w.channels} ch`);
  console.log(`model ${model.id} at ${model.sampleRate} Hz, delay ${model.audioDelay} samples ` +
              `(${Math.round((model.audioDelay * 1000) / model.sampleRate)} ms)`);
  const t0 = Date.now();
  const out = enhanceSignal(model, proc, w.samples, w.rate);
  const secs = w.samples.length / w.rate;
  console.log(`level: in ${rms(w.samples).toFixed(1)} dBFS -> out ${rms(out).toFixed(1)} dBFS ` +
              `(${(secs / ((Date.now() - t0) / 1000)).toFixed(1)}x real time)`);
  writeWav(outPath, out, w.rate, w.bits === 8 ? 8 : 16);
  console.log(`written ${outPath}`);
}

function listDevices() {
  const r = spawnSync('ffmpeg', ['-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                      { encoding: 'utf8' });
  const out = [];
  let audio = false;
  for (const line of (r.stderr || '').split('\n')) {
    if (line.includes('audio devices')) { audio = true; continue; }
    if (audio && line.includes('] [')) {
      const idx = Number(line.split('] [')[1].split(']')[0]);
      const name = line.split('] ').pop().trim();
      if (!Number.isNaN(idx)) out.push([idx, name]);
    } else if (audio && line.includes('devices')) break;
  }
  return out;
}

function cmdMic(argv) {
  const f = parse(argv);
  const modelPath = f.args[0];
  const devices = listDevices();
  if (f.list || !modelPath) {
    console.log('audio inputs:');
    for (const [i, n] of devices) console.log(`  [${i}] ${n}`);
    return;
  }
  const prefer = ['микрофон', 'microphone', 'macbook', 'built-in'];
  const avoid = ['zoom', 'ndi', 'blackhole', 'loopback', 'virtual'];
  const device = f.device ?? (devices.find(([, n]) => prefer.some((k) => n.toLowerCase().includes(k)) &&
                              !avoid.some((k) => n.toLowerCase().includes(k))) || devices[0] || [0])[0];
  const dir = f.keep || fs.mkdtempSync(path.join(os.tmpdir(), 'anecho_mic_'));
  fs.mkdirSync(dir, { recursive: true });
  const raw = path.join(dir, 'recorded.wav'), enh = path.join(dir, 'enhanced.wav');

  const model = Model.fromFile(modelPath);
  console.log(`model ${model.id}  |  input [${device}] ${(devices.find(([i]) => i === device) || [, '?'])[1]}`);
  console.log(`recording ${f.seconds} s — speak now`);
  try {
    execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'avfoundation', '-i', `:${device}`,
                            '-t', String(f.seconds), '-ac', '1', '-ar', String(model.sampleRate),
                            '-c:a', 'pcm_s16le', '-y', raw]);
  } catch (e) {
    console.error('ffmpeg failed. If this mentions permissions, grant microphone access to your\n' +
                  'terminal in System Settings -> Privacy & Security -> Microphone, then run again.');
    process.exit(1);
  }
  const proc = new Processor(model, f.license);
  const ctx = proc.getContext();
  ctx.setParameter(ProcessorParameter.EnhancementLevel, f.level);
  ctx.setParameter(ProcessorParameter.VoiceGain, f.gain);
  const w = readWav(raw);
  const out = enhanceSignal(model, proc, w.samples, w.rate);
  console.log(`level: recorded ${rms(w.samples).toFixed(1)} dBFS -> enhanced ${rms(out).toFixed(1)} dBFS`);
  writeWav(enh, out, w.rate);
  if (f.play) {
    for (const [p, label] of [[raw, 'the recording'], [enh, 'the enhanced version']]) {
      console.log(`  playing ${label} …`);
      spawnSync('afplay', [p], { stdio: 'ignore' });
    }
  }
  console.log(`\nfiles: ${raw}\n       ${enh}`);
}

function cmdInspect(argv) {
  const model = Model.fromFile(parse(argv).args[0]);
  console.log(model.describe());
}

const [cmd, ...rest] = process.argv.slice(2);
try {
  if (cmd === 'process') cmdProcess(rest);
  else if (cmd === 'mic') cmdMic(rest);
  else if (cmd === 'inspect') cmdInspect(rest);
  else { console.log(USAGE); process.exit(cmd ? 1 : 0); }
} catch (e) {
  console.error(String(e.message || e));
  process.exit(1);
}
