/**
 * Display transforms shared with the GLSL display pass.
 */
export type ToneMapMode = 'aces' | 'reinhard'

/** Fitted ACES curve (Narkowicz-style rational approximation) per channel. */
export function acesFitted(x: number): number {
  const a = 2.51
  const b = 0.03
  const c = 2.43
  const d = 0.59
  const e = 0.14
  const v = (x * (a * x + b)) / (x * (c * x + d) + e)
  return v < 0 ? 0 : v > 1 ? 1 : v
}

/** Simple global Reinhard: x / (1 + x). */
export function reinhard(x: number): number {
  return x <= 0 ? 0 : x / (1 + x)
}

export function toneMap(x: number, mode: ToneMapMode): number {
  return mode === 'aces' ? acesFitted(x) : reinhard(x)
}

/** Linear -> sRGB encoding. */
export function linearToSrgb(x: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055
}

export function exposureScale(ev: number): number {
  return Math.pow(2, ev)
}

/**
 * Turn an accumulated linear RGB buffer (3 floats per pixel, sum over samples)
 * into 8-bit RGBA ready for a canvas or PNG/PPM writer.
 */
export function tonemapBuffer(
  accum: Float32Array,
  width: number,
  height: number,
  samples: number,
  exposureEv: number,
  mode: ToneMapMode,
  out?: Uint8ClampedArray,
): Uint8ClampedArray {
  const rgba = out ?? new Uint8ClampedArray(width * height * 4)
  const inv = samples > 0 ? exposureScale(exposureEv) / samples : 0
  for (let i = 0, p = 0; i < width * height; i++, p += 4) {
    const r = accum[i * 3]! * inv
    const g = accum[i * 3 + 1]! * inv
    const b = accum[i * 3 + 2]! * inv
    rgba[p] = Math.round(linearToSrgb(toneMap(r, mode)) * 255)
    rgba[p + 1] = Math.round(linearToSrgb(toneMap(g, mode)) * 255)
    rgba[p + 2] = Math.round(linearToSrgb(toneMap(b, mode)) * 255)
    rgba[p + 3] = 255
  }
  return rgba
}
