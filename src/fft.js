'use strict';
// Mixed-radix complex FFT. The model's frame is 480 = 2^5 * 3 * 5, so a
// power-of-two-only transform is not enough and a 480-point DFT would be 20x the
// work. Recursive Cooley-Tukey on the smallest prime factor covers every size the
// container can declare, and at 66 frames a second the recursion is free.

function smallestFactor(n) {
  for (let f = 2; f * f <= n; f++) if (n % f === 0) return f;
  return n;
}

/** In-place-ish complex FFT. sign = -1 forward, +1 inverse (unnormalised). */
function fft(re, im, sign) {
  const n = re.length;
  if (n === 1) return;
  const p = smallestFactor(n);
  if (p === n) { dft(re, im, sign); return; }
  const m = n / p;

  // decimate into p interleaved subsequences of length m
  const sr = [], si = [];
  for (let j = 0; j < p; j++) {
    const ar = new Float64Array(m), ai = new Float64Array(m);
    for (let k = 0; k < m; k++) { ar[k] = re[k * p + j]; ai[k] = im[k * p + j]; }
    fft(ar, ai, sign);
    sr.push(ar); si.push(ai);
  }
  // recombine with twiddles
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < p; q++) {
      let accR = 0, accI = 0;
      for (let j = 0; j < p; j++) {
        const ang = sign * 2 * Math.PI * ((j * (k + q * m)) % n) / n;
        const c = Math.cos(ang), s = Math.sin(ang);
        const xr = sr[j][k], xi = si[j][k];
        accR += xr * c - xi * s;
        accI += xr * s + xi * c;
      }
      re[k + q * m] = accR; im[k + q * m] = accI;
    }
  }
}

function dft(re, im, sign) {
  const n = re.length;
  const or_ = new Float64Array(n), oi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let ar = 0, ai = 0;
    for (let t = 0; t < n; t++) {
      const ang = sign * 2 * Math.PI * ((k * t) % n) / n;
      const c = Math.cos(ang), s = Math.sin(ang);
      ar += re[t] * c - im[t] * s;
      ai += re[t] * s + im[t] * c;
    }
    or_[k] = ar; oi[k] = ai;
  }
  re.set(or_); im.set(oi);
}

/** Real input -> the n/2+1 non-redundant bins. */
function rfft(x, n) {
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = x[i];
  fft(re, im, -1);
  const half = (n >> 1) + 1;
  return { re: re.subarray(0, half), im: im.subarray(0, half) };
}

/** The n/2+1 bins -> the real signal of length n. */
function irfft(specRe, specIm, n) {
  const re = new Float64Array(n), im = new Float64Array(n);
  const half = (n >> 1) + 1;
  for (let k = 0; k < half; k++) { re[k] = specRe[k]; im[k] = specIm[k]; }
  for (let k = half; k < n; k++) {           // conjugate-symmetric tail
    re[k] = specRe[n - k]; im[k] = -specIm[n - k];
  }
  fft(re, im, +1);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = re[i] / n;
  return out;
}

/** Periodic Hann, matching numpy's hann_window(periodic=True). */
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

module.exports = { fft, rfft, irfft, hann };
