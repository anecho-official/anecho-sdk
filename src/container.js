'use strict';
// Reader for the .anecho container. The format is deliberately simple enough that
// a second implementation is a day's work rather than a reverse-engineering
// project — which is the whole point of shipping the graph inside the file.
const fs = require('fs');
const crypto = require('crypto');
const zlib = require('zlib');

const MAGIC = Buffer.from('ANEC', 'ascii');
const HEAD_LEN = 10;
const SUPPORTED = [1, 2];
const DTYPES = { f32: Float32Array, f16: Uint16Array, i8: Int8Array };

/** IEEE half -> float. The format declares f16, so a loader that cannot read it
 *  is not a loader for the format; JS has no Float16Array before Node 24. */
function half(bits) {
  const s = bits >> 15 ? -1 : 1, e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;                       // subnormal, and zero
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Remove the top-level "integrity" member from the *stored* header bytes.
 *
 * The digest covers `canonical(header without integrity)`, and re-serialising the
 * parsed header to get that back is a trap: Python writes floats as `1e-05` and
 * `40.0` where JSON.stringify gives `1e-5` and `40`, so the bytes differ and every
 * file looks tampered with. The stored header already *is* the canonical form with
 * sorted keys, so deleting one member's text — key, value, and the adjacent comma —
 * yields exactly what the writer hashed, with no float formatting involved.
 */
function stripIntegrity(text) {
  const key = '"integrity":';
  let depth = 0, at = -1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (depth === 1 && c === '"' && text.startsWith(key, i)) { at = i; break; }
    if (c === '"') {                                  // skip strings wholesale
      i++;
      while (i < text.length && (text[i] !== '"' || text[i - 1] === '\\')) i++;
      continue;
    }
    if (c === '{' || c === '[') depth++;
    else if (c === '}' || c === ']') depth--;
  }
  if (at < 0) throw new Error('header carries no integrity member');
  let j = at + key.length, d = 0;
  do {
    const c = text[j];
    if (c === '"') { j++; while (j < text.length && (text[j] !== '"' || text[j - 1] === '\\')) j++; }
    else if (c === '{' || c === '[') d++;
    else if (c === '}' || c === ']') d--;
    j++;
  } while (d > 0 && j < text.length);
  let start = at, end = j;
  if (text[start - 1] === ',') start -= 1;
  else if (text[end] === ',') end += 1;
  return text.slice(0, start) + text.slice(end);
}


function readModel(path, { secret = null, requireGraph = false } = {}) {
  const buf = fs.readFileSync(path);
  if (buf.length < HEAD_LEN || !buf.subarray(0, 4).equals(MAGIC)) {
    throw new Error('bad magic — not an .anecho container');
  }
  const version = buf.readUInt16LE(4);
  if (!SUPPORTED.includes(version)) {
    throw new Error(`container version ${version}; this build reads ${SUPPORTED}`);
  }
  const hlen = buf.readUInt32LE(6);
  if (HEAD_LEN + hlen > buf.length) throw new Error('header runs past the end of the file');
  const header = JSON.parse(buf.subarray(HEAD_LEN, HEAD_LEN + hlen).toString('utf8'));
  const payload = buf.subarray(HEAD_LEN + hlen);
  if (payload.length !== header.payload_len) throw new Error('payload length mismatch');

  const integ = header.integrity || {};
  if (integ.crc32 !== undefined && (zlib.crc32 ? zlib.crc32(payload) : crc32(payload)) !== integ.crc32) {
    throw new Error('payload crc32 mismatch (file is corrupted)');
  }
  const headerText = buf.subarray(HEAD_LEN, HEAD_LEN + hlen).toString('utf8');
  const digest = crypto.createHash('sha256')
    .update(Buffer.from(stripIntegrity(headerText), 'utf8')).update(payload).digest('hex');
  if (digest !== integ.sha256) throw new Error('sha256 integrity mismatch (file was modified)');
  if (secret && integ.hmac_sha256) {
    const expect = crypto.createHmac('sha256', secret)
      .update(Buffer.from(digest, 'hex')).digest('hex');
    if (expect !== integ.hmac_sha256) throw new Error('hmac mismatch (wrong secret or tampered)');
  }

  // Tensor offsets are 64-byte aligned inside the payload, but the payload itself
  // starts at an arbitrary file offset, so a typed-array view over the file buffer
  // would fault. One copy at load time buys aligned views for every tensor.
  const store = new ArrayBuffer(payload.length);
  new Uint8Array(store).set(payload);

  const tensors = new Map();
  for (const t of header.tensors) {
    const Ctor = DTYPES[t.dtype];
    if (!Ctor) throw new Error(`tensor ${t.name}: unsupported dtype ${t.dtype}`);
    const view = new Ctor(store, t.offset, t.nbytes / Ctor.BYTES_PER_ELEMENT);
    let data = t.dtype === 'f16' ? Float32Array.from(view, half) : Float32Array.from(view);
    if (t.scale !== undefined) for (let i = 0; i < data.length; i++) data[i] *= t.scale;
    tensors.set(t.name, { d: data, s: t.shape.slice() });
  }

  const graph = header.graph || null;
  if (!graph && requireGraph) {
    throw new Error(`${path} is format ${header.format} and carries no graph`);
  }
  if (graph) validateGraph(graph, tensors);
  return { header, tensors, graph };
}

function validateGraph(graph, tensors) {
  const available = new Set(graph.inputs);
  for (const node of graph.nodes) {
    for (const v of node.inputs) {
      if (!available.has(v)) {
        throw new Error(`${node.name}: input '${v}' is not produced by any earlier node`);
      }
    }
    for (const v of node.outputs) available.add(v);
    for (const [role, name] of Object.entries(node.params || {})) {
      if (!tensors.has(name)) throw new Error(`${node.name}: parameter ${role}='${name}' missing`);
    }
  }
  for (const [pub, v] of Object.entries(graph.outputs)) {
    if (!available.has(v)) throw new Error(`output '${pub}' refers to '${v}', which nothing produces`);
  }
}

// zlib.crc32 exists from Node 20.15; keep a fallback so older runtimes still load.
let TABLE = null;
function crc32(buf) {
  if (!TABLE) {
    TABLE = new Int32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      TABLE[i] = c;
    }
  }
  let c = 0 ^ -1;
  for (let i = 0; i < buf.length; i++) c = (c >>> 8) ^ TABLE[(c ^ buf[i]) & 0xff];
  return (c ^ -1) >>> 0;
}

module.exports = { readModel, stripIntegrity };
