'use strict';
// Minimal PCM WAV read/write, so the CLI needs no dependencies.
const fs = require('fs');

function readWav(path) {
  const b = fs.readFileSync(path);
  if (b.subarray(0, 4).toString('ascii') !== 'RIFF' || b.subarray(8, 12).toString('ascii') !== 'WAVE') {
    throw new Error(`${path}: not a RIFF/WAVE file`);
  }
  let pos = 12, fmt = null, data = null;
  while (pos + 8 <= b.length) {
    const id = b.subarray(pos, pos + 4).toString('ascii');
    const size = b.readUInt32LE(pos + 4);
    const body = b.subarray(pos + 8, pos + 8 + size);
    if (id === 'fmt ') fmt = { format: body.readUInt16LE(0), channels: body.readUInt16LE(2),
                               rate: body.readUInt32LE(4), bits: body.readUInt16LE(14) };
    if (id === 'data') data = body;
    pos += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`${path}: missing fmt or data chunk`);
  if (fmt.format !== 1 || ![8, 16, 32].includes(fmt.bits)) {
    throw new Error(`${path}: only PCM 8/16/32-bit is supported (got format ${fmt.format}, ${fmt.bits}-bit)`);
  }
  const bytes = fmt.bits / 8, frames = Math.floor(data.length / (bytes * fmt.channels));
  const out = new Float32Array(frames);
  const peak = 2 ** (fmt.bits - 1);
  for (let i = 0; i < frames; i++) {                     // mono-mix on the way in
    let acc = 0;
    for (let c = 0; c < fmt.channels; c++) {
      const off = (i * fmt.channels + c) * bytes;
      acc += (bytes === 1 ? data.readInt8(off) : bytes === 2 ? data.readInt16LE(off)
              : data.readInt32LE(off)) / peak;
    }
    out[i] = acc / fmt.channels;
  }
  return { samples: out, rate: fmt.rate, channels: fmt.channels, bits: fmt.bits };
}

function writeWav(path, samples, rate, bits = 16) {
  const bytes = bits / 8, peak = 2 ** (bits - 1) - 1;
  let max = 0;
  for (const v of samples) max = Math.max(max, Math.abs(v));
  const scale = max > 1 ? 1 / max : 1;                    // never wrap on clipping
  if (max > 1) console.log(`  output peaked at ${(20 * Math.log10(max)).toFixed(1)} dBFS — scaling to fit`);
  const data = Buffer.alloc(samples.length * bytes);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i] * scale)) * peak;
    if (bytes === 2) data.writeInt16LE(Math.round(v), i * 2);
    else if (bytes === 1) data.writeInt8(Math.round(v), i);
    else data.writeInt32LE(Math.round(v), i * 4);
  }
  const head = Buffer.alloc(44);
  head.write('RIFF', 0); head.writeUInt32LE(36 + data.length, 4); head.write('WAVE', 8);
  head.write('fmt ', 12); head.writeUInt32LE(16, 16); head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22); head.writeUInt32LE(rate, 24);
  head.writeUInt32LE(rate * bytes, 28); head.writeUInt16LE(bytes, 32);
  head.writeUInt16LE(bits, 34); head.write('data', 36); head.writeUInt32LE(data.length, 40);
  fs.writeFileSync(path, Buffer.concat([head, data]));
}

module.exports = { readWav, writeWav };
