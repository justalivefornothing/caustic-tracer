/**
 * Material sampling with explicit PDFs. The scalar kernels (suffix `Into`)
 * write into a caller-provided Float64Array and are what the tracer's hot loop
 * calls; the Vec3 wrappers are the readable API the tests interrogate. The
 * GLSL tracer implements the same maths.
 */
import type { Rng } from './rng.ts'
import { normalize, refract, v3, type Vec3 } from './vec3.ts'

export const INV_PI = 1 / Math.PI
const TWO_PI = 2 * Math.PI

/** Scratch vector for the wrappers; kernels never keep state. */
const scratch = new Float64Array(3)

// ---------------------------------------------------------------------------
// Scalar kernels

/** Orthonormal tangent/bitangent around unit normal n -> out[0..2] = t, out[3..5] = b. */
export function basisInto(nx: number, ny: number, nz: number, out: Float64Array): void {
  // helper axis: pick the one least aligned with n
  let hx = 1, hy = 0, hz = 0
  if (Math.abs(nx) > 0.9) {
    hx = 0
    hy = 1
  }
  // t = normalize(h x n)
  let tx = hy * nz - hz * ny
  let ty = hz * nx - hx * nz
  let tz = hx * ny - hy * nx
  const inv = 1 / Math.sqrt(tx * tx + ty * ty + tz * tz)
  tx *= inv
  ty *= inv
  tz *= inv
  out[0] = tx
  out[1] = ty
  out[2] = tz
  // b = n x t
  out[3] = ny * tz - nz * ty
  out[4] = nz * tx - nx * tz
  out[5] = nx * ty - ny * tx
}

const basisScratch = new Float64Array(6)

/** Cosine-weighted hemisphere direction around unit normal n. pdf = cos(theta) / pi. */
export function cosineHemisphereInto(nx: number, ny: number, nz: number, u1: number, u2: number, out: Float64Array): void {
  const r = Math.sqrt(u1)
  const phi = TWO_PI * u2
  const x = r * Math.cos(phi)
  const y = r * Math.sin(phi)
  const z = Math.sqrt(u1 < 1 ? 1 - u1 : 0)
  basisInto(nx, ny, nz, basisScratch)
  let dx = basisScratch[0]! * x + basisScratch[3]! * y + nx * z
  let dy = basisScratch[1]! * x + basisScratch[4]! * y + ny * z
  let dz = basisScratch[2]! * x + basisScratch[5]! * y + nz * z
  const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz)
  dx *= inv
  dy *= inv
  dz *= inv
  out[0] = dx
  out[1] = dy
  out[2] = dz
}

/** Uniform unit vector -> out. */
export function unitVectorInto(u1: number, u2: number, out: Float64Array): void {
  const z = 1 - 2 * u1
  const r = Math.sqrt(z * z < 1 ? 1 - z * z : 0)
  const phi = TWO_PI * u2
  out[0] = r * Math.cos(phi)
  out[1] = r * Math.sin(phi)
  out[2] = z
}

/** Schlick's approximation given cos(theta) and the two refractive indices. */
export function schlick(cosTheta: number, etaI: number, etaT: number): number {
  const r0 = ((etaI - etaT) / (etaI + etaT)) ** 2
  const m = 1 - Math.max(0, Math.min(1, cosTheta))
  return r0 + (1 - r0) * m * m * m * m * m
}

/**
 * Fresnel reflectance of a dielectric boundary for an incident direction with
 * cos(theta_i) = cosI, going from index etaI to etaT. Uses the transmitted
 * angle when leaving the denser medium and returns 1 on total internal
 * reflection.
 */
export function dielectricReflectance(cosI: number, etaI: number, etaT: number): number {
  const eta = etaI / etaT
  const sin2T = eta * eta * (1 - cosI * cosI)
  if (sin2T >= 1) return 1
  if (etaI > etaT) return schlick(Math.sqrt(1 - sin2T), etaI, etaT)
  return schlick(cosI, etaI, etaT)
}

/**
 * Metal: mirror reflection of unit direction d about unit normal n, perturbed
 * by `roughness` times a point in the unit sphere (u1..u3 uniform). Result is
 * normalised into out.
 */
export function metalInto(
  dx: number, dy: number, dz: number,
  nx: number, ny: number, nz: number,
  roughness: number,
  u1: number, u2: number, u3: number,
  out: Float64Array,
): void {
  const k = 2 * (dx * nx + dy * ny + dz * nz)
  let rx = dx - k * nx
  let ry = dy - k * ny
  let rz = dz - k * nz
  if (roughness > 0) {
    unitVectorInto(u1, u2, out)
    const rad = Math.cbrt(u3) * roughness
    rx += out[0]! * rad
    ry += out[1]! * rad
    rz += out[2]! * rad
  }
  const inv = 1 / Math.sqrt(rx * rx + ry * ry + rz * rz)
  out[0] = rx * inv
  out[1] = ry * inv
  out[2] = rz * inv
}

/**
 * Smooth dielectric. `n` must face the incoming ray; `frontFace` says whether
 * the ray arrives from outside a medium of index `ior`. Returns true when the
 * sampled event was a reflection.
 */
export function dielectricInto(
  dx: number, dy: number, dz: number,
  nx: number, ny: number, nz: number,
  ior: number,
  frontFace: boolean,
  u: number,
  out: Float64Array,
): boolean {
  const etaI = frontFace ? 1 : ior
  const etaT = frontFace ? ior : 1
  const eta = etaI / etaT
  let cosI = -(dx * nx + dy * ny + dz * nz)
  if (cosI > 1) cosI = 1
  const reflectance = dielectricReflectance(cosI, etaI, etaT)
  const sin2T = eta * eta * (1 - cosI * cosI)
  if (u < reflectance || sin2T > 1) {
    const k = 2 * (dx * nx + dy * ny + dz * nz)
    out[0] = dx - k * nx
    out[1] = dy - k * ny
    out[2] = dz - k * nz
    return true
  }
  const cosT = Math.sqrt(1 - sin2T)
  const s = eta * cosI - cosT
  let tx = dx * eta + nx * s
  let ty = dy * eta + ny * s
  let tz = dz * eta + nz * s
  const inv = 1 / Math.sqrt(tx * tx + ty * ty + tz * tz)
  tx *= inv
  ty *= inv
  tz *= inv
  out[0] = tx
  out[1] = ty
  out[2] = tz
  return false
}

/** Power heuristic (beta = 2) MIS weight for strategy `a` against `b`. */
export function powerHeuristic(pdfA: number, pdfB: number): number {
  const a2 = pdfA * pdfA
  const b2 = pdfB * pdfB
  const denom = a2 + b2
  return denom > 0 ? a2 / denom : 0
}

/** Procedural 3D checkerboard: true selects the primary colour. */
export function checkerPrimary(px: number, py: number, pz: number, cellsPerUnit: number): boolean {
  const s = cellsPerUnit
  const sum = Math.floor(px * s + 0.5) + Math.floor(py * s + 0.5) + Math.floor(pz * s + 0.5)
  return (((sum % 2) + 2) % 2) === 0
}

// ---------------------------------------------------------------------------
// Vec3 wrappers

/** Uniformly distributed unit vector. */
export function randomUnitVector(rng: Rng): Vec3 {
  unitVectorInto(rng.next(), rng.next(), scratch)
  return v3(scratch[0]!, scratch[1]!, scratch[2]!)
}

/** Uniform point inside the unit sphere (direction times cbrt(u) radius). */
export function randomInUnitSphere(rng: Rng): Vec3 {
  unitVectorInto(rng.next(), rng.next(), scratch)
  const r = Math.cbrt(rng.next())
  return v3(scratch[0]! * r, scratch[1]! * r, scratch[2]! * r)
}

/** Uniform disk sample: r = sqrt(u), theta = 2 pi v. */
export function randomInUnitDisk(rng: Rng): { x: number; y: number } {
  const r = Math.sqrt(rng.next())
  const theta = TWO_PI * rng.next()
  return { x: r * Math.cos(theta), y: r * Math.sin(theta) }
}

/** Cosine-weighted direction on the hemisphere around unit normal `n`. */
export function sampleCosineHemisphere(n: Vec3, u1: number, u2: number): Vec3 {
  cosineHemisphereInto(n.x, n.y, n.z, u1, u2, scratch)
  return v3(scratch[0]!, scratch[1]!, scratch[2]!)
}

export function cosineHemispherePdf(n: Vec3, dir: Vec3): number {
  const c = n.x * dir.x + n.y * dir.y + n.z * dir.z
  return c > 0 ? c * INV_PI : 0
}

/** Perfect or fuzzy mirror reflection. */
export function sampleMetal(d: Vec3, n: Vec3, roughness: number, rng: Rng): Vec3 {
  const u = normalize(d)
  metalInto(u.x, u.y, u.z, n.x, n.y, n.z, roughness, rng.next(), rng.next(), rng.next(), scratch)
  return v3(scratch[0]!, scratch[1]!, scratch[2]!)
}

export interface DielectricSample {
  readonly dir: Vec3
  readonly reflected: boolean
}

/** Sample a smooth dielectric (see `dielectricInto`). */
export function sampleDielectric(d: Vec3, n: Vec3, ior: number, frontFace: boolean, u: number): DielectricSample {
  const unit = normalize(d)
  const reflected = dielectricInto(unit.x, unit.y, unit.z, n.x, n.y, n.z, ior, frontFace, u, scratch)
  return { dir: v3(scratch[0]!, scratch[1]!, scratch[2]!), reflected }
}

/** Convenience re-export so callers can compare against the analytic refraction. */
export const refractDirection = refract
