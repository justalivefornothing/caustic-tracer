/**
 * Minimal immutable 3-vector helpers. Plain objects keep scene descriptions
 * JSON-friendly; hot loops (BVH traversal, intersection) use scalar math on
 * typed arrays instead and never touch these.
 */
export interface Vec3 {
  readonly x: number
  readonly y: number
  readonly z: number
}

export const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z })
export const ZERO: Vec3 = v3(0, 0, 0)
export const UP: Vec3 = v3(0, 1, 0)

export const add = (a: Vec3, b: Vec3): Vec3 => v3(a.x + b.x, a.y + b.y, a.z + b.z)
export const sub = (a: Vec3, b: Vec3): Vec3 => v3(a.x - b.x, a.y - b.y, a.z - b.z)
export const scale = (a: Vec3, s: number): Vec3 => v3(a.x * s, a.y * s, a.z * s)
export const mulv = (a: Vec3, b: Vec3): Vec3 => v3(a.x * b.x, a.y * b.y, a.z * b.z)
export const neg = (a: Vec3): Vec3 => v3(-a.x, -a.y, -a.z)
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z
export const cross = (a: Vec3, b: Vec3): Vec3 =>
  v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
export const length = (a: Vec3): number => Math.sqrt(dot(a, a))
export const lengthSq = (a: Vec3): number => dot(a, a)
export const distance = (a: Vec3, b: Vec3): number => length(sub(a, b))

export function normalize(a: Vec3): Vec3 {
  const len = length(a)
  return len > 0 ? scale(a, 1 / len) : v3(0, 0, 0)
}

export const lerp = (a: Vec3, b: Vec3, t: number): Vec3 =>
  v3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t)

/** Mirror `d` about unit normal `n`. */
export const reflect = (d: Vec3, n: Vec3): Vec3 => sub(d, scale(n, 2 * dot(d, n)))

/**
 * Refract unit direction `d` through a surface with unit normal `n`
 * (pointing against `d`) using the ratio eta = n_incident / n_transmitted.
 * Returns null on total internal reflection.
 */
export function refract(d: Vec3, n: Vec3, eta: number): Vec3 | null {
  const cosI = Math.min(1, Math.max(-1, -dot(d, n)))
  const sin2T = eta * eta * (1 - cosI * cosI)
  if (sin2T > 1) return null
  const cosT = Math.sqrt(1 - sin2T)
  return add(scale(d, eta), scale(n, eta * cosI - cosT))
}

export const maxComponent = (a: Vec3): number => Math.max(a.x, a.y, a.z)
export const minv = (a: Vec3, b: Vec3): Vec3 =>
  v3(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.min(a.z, b.z))
export const maxv = (a: Vec3, b: Vec3): Vec3 =>
  v3(Math.max(a.x, b.x), Math.max(a.y, b.y), Math.max(a.z, b.z))

/** Build an orthonormal basis (t, b) around unit normal n. */
export function orthonormalBasis(n: Vec3): { t: Vec3; b: Vec3 } {
  const helper = Math.abs(n.x) > 0.9 ? v3(0, 1, 0) : v3(1, 0, 0)
  const t = normalize(cross(helper, n))
  const b = cross(n, t)
  return { t, b }
}

export const isFiniteVec = (a: Vec3): boolean =>
  Number.isFinite(a.x) && Number.isFinite(a.y) && Number.isFinite(a.z)
