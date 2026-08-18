"""
anecho model container (``.anecho``) — writer, reader, and integrity check.

Layout (little-endian):
    magic   b"ANEC"            4 bytes
    version u16                container format version (1 or 2)
    hlen    u32                header length in bytes
    header  hlen bytes         UTF-8 JSON
    payload rest of file       tensor data, each tensor 64-byte aligned

Version 2 header::

    {
      "format": "anecho-model/2",
      "model_id": "anecho-openspace-16khz-v1",
      "created": "2026-08-17T...",
      "config": { ...ModelConfig... },
      "graph": { "inputs": [...], "outputs": {...},
                 "nodes": [...], "numerics": {...} },
      "payload_len": <int>,
      "tensors": [ {"name","dtype","shape","offset","nbytes"}, ... ],
      "integrity": { "crc32": <int>, "sha256": "<hex>", "hmac_sha256": "<hex>"|null }
    }

What version 2 adds, and why
----------------------------
**The graph travels with the weights.** Version 1 stored tensors plus a config
dict and left the topology in ``anecho.model`` — the same arrangement as the
vendor's ``.aicmodel``, where the runtime *is* the schema. That is what makes a
foreign model file a reverse-engineering project, and it means an old file
silently misloads against new code. A v2 file names its ops, its wiring and the
numeric conventions that weights cannot express (see ``graph.Numerics``).

**Offsets are payload-relative.** In the vendor format the tensor table holds
absolute file offsets, so nothing may change size — edits are possible only in
place. Here the header is regenerated on every write and offsets are relative to
the payload, so repacking with different shapes is ordinary.

**Integrity is layered.** ``crc32`` is a cheap corruption check that costs
nothing to verify on load; ``sha256`` detects tampering; ``hmac_sha256`` (when
packed with a product secret) detects tampering by anyone without the secret.
The vendor file's two CRCs are corruption checks only — and one of them covers
the other, a detail that silently rejects every naive edit.

**Dtypes are declared.** f32 today, f16/i8 reserved, so quantised models do not
need a new container.

Reading is backward compatible: v1 files still load, with ``graph`` set to None.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
import zlib

import numpy as np

from .errors import ModelInvalidError, ModelVersionUnsupportedError
from .graph import FORMAT, Graph, VadConfig

MAGIC = b"ANEC"
VERSION = 2
SUPPORTED = (1, 2)
ALIGN = 64
HEAD_LEN = 10                       # magic(4) + version(2) + hlen(4)

DTYPES = {"f32": np.float32, "f16": np.float16, "i8": np.int8}

#: Quantisation policies. Per-tensor dtype is what makes mixed precision cheap:
#: the weight matrices carry the size, while BatchNorm statistics and biases —
#: where 8 bits would actually hurt — stay f32.
def _i8_weights(name: str, arr: np.ndarray) -> str:
    return "i8" if (arr.ndim >= 2 and arr.size >= 1024) else "f32"


def _f16_weights(name: str, arr: np.ndarray) -> str:
    return "f16" if (arr.ndim >= 2 and arr.size >= 1024) else "f32"


POLICIES = {"none": None, "i8_weights": _i8_weights, "f16_weights": _f16_weights}


def _canonical(header_without_integrity: dict) -> bytes:
    return json.dumps(header_without_integrity, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _integrity_input(header: dict, payload: bytes) -> bytes:
    h = dict(header)
    h.pop("integrity", None)
    return _canonical(h) + payload


def _pack_payload(tensors: dict, dtype: str, policy=None) -> tuple[bytes, list]:
    payload, entries = bytearray(), []
    for name, arr in tensors.items():
        src = np.ascontiguousarray(arr, dtype=np.float32)
        dt = (policy(name, src) if policy else dtype)
        entry = {"name": name, "dtype": dt, "shape": list(src.shape)}
        if dt == "i8":
            # Symmetric per-tensor quantisation; the scale travels with the tensor
            # so a reader never has to know how it was produced.
            scale = float(np.abs(src).max()) / 127.0 or 1.0
            a = np.clip(np.rint(src / scale), -127, 127).astype(np.int8)
            entry["scale"] = scale
        else:
            a = src.astype(DTYPES[dt])
        payload.extend(b"\0" * (-len(payload) % ALIGN))
        entry["offset"] = len(payload)
        entry["nbytes"] = int(a.nbytes)
        entries.append(entry)
        payload += a.tobytes()
    return bytes(payload), entries


def write_model(path: str, tensors: dict, config: dict, model_id: str, created: str,
                secret: bytes | None = None, graph: Graph | None = None,
                dtype: str = "f32", quantize: str = "none",
                runtime: dict | None = None) -> str:
    """Pack tensors + config (+ graph) into a ``.anecho`` container.

    Passing ``graph`` writes format 2 — a self-describing model. Omitting it
    writes the legacy format 1, where the loader must already know the
    architecture; new models should always carry a graph.
    """
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; known: {sorted(DTYPES)}")
    if quantize not in POLICIES:
        raise ValueError(f"unknown quantisation policy {quantize!r}; "
                         f"known: {sorted(POLICIES)}")
    payload, entries = _pack_payload(tensors, dtype, POLICIES[quantize])

    if graph is not None:
        graph.validate(tensor_names=set(tensors))
        dead = graph.unused_tensors(set(tensors))
        if dead:
            raise ValueError(f"{len(dead)} tensor(s) no node consumes: {dead[:5]} — "
                             "either wire them up or leave them out")

    header = {
        "format": FORMAT if graph is not None else "anecho-model/1",
        "model_id": model_id,
        "created": created,
        "config": config,
        "payload_len": len(payload),
        "tensors": entries,
    }
    if graph is not None:
        header["graph"] = graph.to_json()
    if runtime:
        if "vad" in runtime:                     # validated, not free-form JSON
            runtime = dict(runtime)
            runtime["vad"] = VadConfig.from_json(runtime["vad"]).to_json()
        header["runtime"] = runtime

    digest = hashlib.sha256(_integrity_input(header, payload)).hexdigest()
    mac = hmac.new(secret, bytes.fromhex(digest), hashlib.sha256).hexdigest() if secret else None
    integrity = {"sha256": digest, "hmac_sha256": mac}
    if graph is not None:
        integrity["crc32"] = zlib.crc32(payload) & 0xFFFFFFFF
    header["integrity"] = integrity

    hb = _canonical(header)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", VERSION if graph is not None else 1))
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        f.write(payload)
    return path


def read_model(path: str, secret: bytes | None = None, require_hmac: bool = False,
               require_graph: bool = False):
    """Parse and verify a ``.anecho`` container.

    Returns ``(header, tensors, graph)`` — ``graph`` is None for v1 files.
    Raises ``ModelInvalidError`` on any integrity failure.
    """
    with open(path, "rb") as f:
        d = f.read()
    if len(d) < HEAD_LEN or d[:4] != MAGIC:
        raise ModelInvalidError("bad magic — not an .anecho container")
    ver = struct.unpack_from("<H", d, 4)[0]
    if ver not in SUPPORTED:
        raise ModelVersionUnsupportedError(
            f"container version {ver}; this build reads {SUPPORTED}")
    hlen = struct.unpack_from("<I", d, 6)[0]
    if HEAD_LEN + hlen > len(d):
        raise ModelInvalidError("header length runs past the end of the file")
    try:
        header = json.loads(d[HEAD_LEN:HEAD_LEN + hlen].decode("utf-8"))
    except Exception as e:
        raise ModelInvalidError(f"header parse: {e}")

    payload = d[HEAD_LEN + hlen:]
    if len(payload) != header.get("payload_len", -1):
        raise ModelInvalidError("payload length mismatch")

    integ = header.get("integrity") or {}
    crc = integ.get("crc32")                     # cheap check first
    if crc is not None and (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ModelInvalidError("payload crc32 mismatch (file is corrupted)")
    digest = hashlib.sha256(_integrity_input(header, payload)).hexdigest()
    if not hmac.compare_digest(digest, integ.get("sha256", "")):
        raise ModelInvalidError("sha256 integrity mismatch (file was modified)")
    stored_mac = integ.get("hmac_sha256")
    if secret is not None and stored_mac:
        expect = hmac.new(secret, bytes.fromhex(digest), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, stored_mac):
            raise ModelInvalidError("hmac integrity mismatch (wrong secret or tampered)")
    if require_hmac and not (secret is not None and stored_mac):
        raise ModelInvalidError("hmac required but not present/verified")

    tensors = {}
    for t in header["tensors"]:
        dt = DTYPES.get(t.get("dtype", "f32"))
        if dt is None:
            raise ModelInvalidError(f"tensor {t['name']}: unknown dtype {t.get('dtype')!r}")
        o, n = t["offset"], t["nbytes"]
        if o + n > len(payload):
            raise ModelInvalidError(f"tensor {t['name']} runs past the payload")
        arr = np.frombuffer(payload[o:o + n], dtype=dt).reshape(t["shape"])
        arr = np.array(arr)                      # copy out of the read-only buffer
        if "scale" in t:                         # quantised: dequantise on load
            arr = arr.astype(np.float32) * np.float32(t["scale"])
        tensors[t["name"]] = arr

    if "vad" in (header.get("runtime") or {}):   # fail on load, not at first use
        VadConfig.from_json(header["runtime"]["vad"])

    graph = None
    if "graph" in header:
        graph = Graph.from_json(header["graph"]).validate(tensor_names=set(tensors))
    elif require_graph:
        raise ModelInvalidError(
            f"{path} is format {header.get('format')} and carries no graph; "
            "repack it with a graph before loading in a graph-only runtime")
    return header, tensors, graph


def inspect(path: str) -> str:
    """Human-readable summary — a model you can read without running it."""
    header, tensors, graph = read_model(path)
    n_params = sum(int(np.prod(t["shape"])) for t in header["tensors"])
    lines = [f"{header['model_id']}  [{header['format']}]",
             f"  created  {header.get('created')}",
             f"  tensors  {len(tensors)}  ({n_params:,} parameters, "
             f"{header['payload_len']:,} bytes)",
             f"  integrity crc32={'yes' if (header.get('integrity') or {}).get('crc32') else 'no'} "
             f"sha256=yes hmac={'yes' if (header.get('integrity') or {}).get('hmac_sha256') else 'no'}"]
    if graph is not None:
        lines.append(f"  numerics {json.dumps(graph.numerics.to_json(), sort_keys=True)}")
        lines.append("  graph:")
        lines += ["    " + ln for ln in graph.describe().splitlines()]
    else:
        lines.append("  graph:   ABSENT (format 1 — the loader must know the architecture)")
    return "\n".join(lines)
