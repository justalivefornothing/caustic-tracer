/**
 * Deterministic 32-bit RNG shared (by construction, not by code) with the GLSL
 * tracer: a PCG-style permutation hash turns (pixel, frame) into a seed and an
 * xorshift32 stream produces the per-path random numbers.
 *
 * All arithmetic is kept in uint32 with Math.imul and `>>> 0`.
 */

/** PCG RXS-M-XS style output permutation over one LCG step. */
export function pcgHash(input: number): number {
  const state = (Math.imul(input >>> 0, 747796405) + 2891336453) >>> 0
  const shifted = (state >>> ((state >>> 28) + 4)) ^ state
  const word = Math.imul(shifted >>> 0, 277803737) >>> 0
  return ((word >>> 22) ^ word) >>> 0
}

/** Combine pixel index and frame index into one well-mixed uint32 seed. */
export function seedFor(pixelIndex: number, frame: number): number {
  const a = pcgHash(pixelIndex >>> 0)
  const b = pcgHash((frame >>> 0) ^ 0x9e3779b9)
  const s = pcgHash((a ^ Math.imul(b, 0x85ebca6b)) >>> 0)
  return s === 0 ? 0x1234567 : s
}

export class Rng {
  private state: number

  constructor(seed: number) {
    const s = pcgHash(seed >>> 0)
    this.state = s === 0 ? 0x9e3779b9 : s
  }

  static fromPixel(pixelIndex: number, frame: number): Rng {
    const r = new Rng(0)
    r.state = seedFor(pixelIndex, frame)
    return r
  }

  /** Next uint32 from an xorshift32 step. */
  nextU32(): number {
    let x = this.state
    x ^= (x << 13) >>> 0
    x >>>= 0
    x ^= x >>> 17
    x ^= (x << 5) >>> 0
    x >>>= 0
    this.state = x
    return x
  }

  /** Uniform float in [0, 1). */
  next(): number {
    return this.nextU32() / 4294967296
  }

  /** Uniform float in [min, max). */
  range(min: number, max: number): number {
    return min + (max - min) * this.next()
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.min(n - 1, Math.floor(this.next() * n))
  }
}
