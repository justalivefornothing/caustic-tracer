/**
 * CPU reference path tracer. Walks the packed scene exactly like the GLSL
 * tracer does: stackless BVH traversal, Lambertian / metal / dielectric /
 * emissive scattering, next-event estimation with power-heuristic MIS on
 * diffuse surfaces, optional Russian roulette after the third bounce.
 *
 * The inner loop is written with scalar locals and typed-array scratch space
 * so it allocates nothing per bounce.
 */
import { cameraFrame, generateRay, type CameraFrame } from './camera.ts'
import { intersectClosest, makeHit, occluded, T_MIN, type HitRecord } from './intersect.ts'
import {
  basisInto,
  checkerPrimary,
  cosineHemisphereInto,
  dielectricInto,
  INV_PI,
  metalInto,
  powerHeuristic,
  unitVectorInto,
} from './materials.ts'
import { MAT_EMISSIVE, MAT_LAMBERTIAN, MAT_METAL, MAT_STRIDE, PRIM_SPHERE, PRIM_STRIDE, packScene, type PackedScene } from './pack.ts'
import { Rng } from './rng.ts'
import type { Scene, Sky } from './scene.ts'
import { v3, type Vec3 } from './vec3.ts'

export interface TraceOptions {
  readonly maxBounces: number
  readonly russianRoulette: boolean
  readonly nee: boolean
}

export const DEFAULT_TRACE_OPTIONS: TraceOptions = { maxBounces: 8, russianRoulette: true, nee: true }

/** Shadow rays stop just short of the light so the light itself is not an occluder. */
const SHADOW_SHORTEN = 1 - 1e-3
const TWO_PI = 2 * Math.PI

export function skyRadiance(sky: Sky, dir: Vec3): Vec3 {
  if (sky.kind === 'none') return v3(0, 0, 0)
  const t = 0.5 * (dir.y + 1)
  return v3(
    (sky.horizon.r + (sky.zenith.r - sky.horizon.r) * t) * sky.strength,
    (sky.horizon.g + (sky.zenith.g - sky.horizon.g) * t) * sky.strength,
    (sky.horizon.b + (sky.zenith.b - sky.horizon.b) * t) * sky.strength,
  )
}

// ---------------------------------------------------------------------------
// Light sampling (scalar). Results are written into `lightScratch`:
//   [0..2] unit direction, [3] distance, [4] solid-angle pdf, [5..7] emission

const lightScratch = new Float64Array(8)
const dirScratch = new Float64Array(3)
const basisScratch = new Float64Array(6)

/**
 * Solid-angle pdf with which `sampleLightInto` would generate the direction
 * from p to hitPoint on light primitive `prim` (geometric normal hn).
 */
export function lightPdfScalar(
  ps: PackedScene,
  px: number, py: number, pz: number,
  hx: number, hy: number, hz: number,
  hnx: number, hny: number, hnz: number,
  prim: number,
): number {
  const lightCount = ps.lights.length
  if (lightCount === 0) return 0
  const prims = ps.prims
  const o = prim * PRIM_STRIDE
  const tx = hx - px, ty = hy - py, tz = hz - pz
  const dist2 = tx * tx + ty * ty + tz * tz
  if (dist2 <= 0) return 0
  const inv = 1 / Math.sqrt(dist2)
  const wx = tx * inv, wy = ty * inv, wz = tz * inv
  if (prims[o + 3] === PRIM_SPHERE) {
    const cx = prims[o]! - px, cy = prims[o + 1]! - py, cz = prims[o + 2]! - pz
    const r = prims[o + 4]!
    const d2 = cx * cx + cy * cy + cz * cz
    if (d2 > r * r) {
      const cosMax = Math.sqrt(Math.max(0, 1 - (r * r) / d2))
      const solid = TWO_PI * (1 - cosMax)
      return solid > 0 ? 1 / (solid * lightCount) : 0
    }
  }
  const cosL = Math.abs(hnx * wx + hny * wy + hnz * wz)
  if (cosL < 1e-6) return 0
  return dist2 / (prims[o + 11]! * cosL * lightCount)
}

/** Pick a light uniformly and a point on it. Returns false for a degenerate sample. */
export function sampleLightInto(ps: PackedScene, px: number, py: number, pz: number, rng: Rng, out: Float64Array): boolean {
  const lightCount = ps.lights.length
  if (lightCount === 0) return false
  const prim = ps.lights[rng.int(lightCount)]!
  const prims = ps.prims
  const o = prim * PRIM_STRIDE
  const m = ps.primObject[prim]! * MAT_STRIDE
  out[5] = ps.mats[m + 12]!
  out[6] = ps.mats[m + 13]!
  out[7] = ps.mats[m + 14]!

  if (prims[o + 3] === PRIM_SPHERE) {
    const cx = prims[o]! - px, cy = prims[o + 1]! - py, cz = prims[o + 2]! - pz
    const r = prims[o + 4]!
    const d2 = cx * cx + cy * cy + cz * cz
    if (d2 > r * r) {
      // Outside: sample the cone of directions subtended by the sphere.
      const cosMax = Math.sqrt(Math.max(0, 1 - (r * r) / d2))
      const cosT = 1 - rng.next() * (1 - cosMax)
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT))
      const phi = TWO_PI * rng.next()
      const invD = 1 / Math.sqrt(d2)
      const ax = cx * invD, ay = cy * invD, az = cz * invD
      basisInto(ax, ay, az, basisScratch)
      const sx = sinT * Math.cos(phi), sy = sinT * Math.sin(phi)
      let wx = basisScratch[0]! * sx + basisScratch[3]! * sy + ax * cosT
      let wy = basisScratch[1]! * sx + basisScratch[4]! * sy + ay * cosT
      let wz = basisScratch[2]! * sx + basisScratch[5]! * sy + az * cosT
      const invW = 1 / Math.sqrt(wx * wx + wy * wy + wz * wz)
      wx *= invW
      wy *= invW
      wz *= invW
      // distance to the near side of the sphere along w
      const halfB = -(cx * wx + cy * wy + cz * wz)
      const disc = halfB * halfB - (d2 - r * r)
      if (disc < 0) return false
      const dist = -halfB - Math.sqrt(disc)
      if (dist <= 0) return false
      out[0] = wx
      out[1] = wy
      out[2] = wz
      out[3] = dist
      out[4] = 1 / (TWO_PI * (1 - cosMax) * lightCount)
      return true
    }
    // Inside: uniform area sample over the whole sphere.
    unitVectorInto(rng.next(), rng.next(), dirScratch)
    const nx = dirScratch[0]!, ny = dirScratch[1]!, nz = dirScratch[2]!
    const tx = cx + nx * r, ty = cy + ny * r, tz = cz + nz * r
    const dist2 = tx * tx + ty * ty + tz * tz
    if (dist2 <= 0) return false
    const dist = Math.sqrt(dist2)
    const wx = tx / dist, wy = ty / dist, wz = tz / dist
    const cosL = Math.abs(nx * wx + ny * wy + nz * wz)
    if (cosL < 1e-6) return false
    out[0] = wx
    out[1] = wy
    out[2] = wz
    out[3] = dist
    out[4] = dist2 / (prims[o + 11]! * cosL * lightCount)
    return true
  }

  // Triangle: uniform barycentric sample.
  let u1 = rng.next()
  let u2 = rng.next()
  if (u1 + u2 > 1) {
    u1 = 1 - u1
    u2 = 1 - u2
  }
  const ax = prims[o]!, ay = prims[o + 1]!, az = prims[o + 2]!
  const qx = ax + (prims[o + 4]! - ax) * u1 + (prims[o + 8]! - ax) * u2
  const qy = ay + (prims[o + 5]! - ay) * u1 + (prims[o + 9]! - ay) * u2
  const qz = az + (prims[o + 6]! - az) * u1 + (prims[o + 10]! - az) * u2
  const tx = qx - px, ty = qy - py, tz = qz - pz
  const dist2 = tx * tx + ty * ty + tz * tz
  if (dist2 <= 0) return false
  const dist = Math.sqrt(dist2)
  const wx = tx / dist, wy = ty / dist, wz = tz / dist
  const cosL = Math.abs(prims[o + 12]! * wx + prims[o + 13]! * wy + prims[o + 14]! * wz)
  if (cosL < 1e-6) return false
  out[0] = wx
  out[1] = wy
  out[2] = wz
  out[3] = dist
  out[4] = dist2 / (prims[o + 11]! * cosL * lightCount)
  return true
}

// ---------------------------------------------------------------------------

const hitScratch: HitRecord = makeHit()
const radianceOut = new Float64Array(3)

/**
 * Estimate incoming radiance along one ray (unit direction), writing linear
 * RGB into `out` (defaults to a shared scratch buffer).
 */
export function radianceInto(
  ps: PackedScene,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  rng: Rng,
  opts: TraceOptions,
  out: Float64Array = radianceOut,
): Float64Array {
  let Lr = 0, Lg = 0, Lb = 0
  let br = 1, bg = 1, bb = 1
  let specularBounce = true
  let prevPdf = 0
  let prevX = ox, prevY = oy, prevZ = oz
  const prims = ps.prims
  const mats = ps.mats
  const useNee = opts.nee && ps.lights.length > 0
  const sky = ps.sky

  for (let bounce = 0; bounce <= opts.maxBounces; bounce++) {
    if (!intersectClosest(ps, ox, oy, oz, dx, dy, dz, T_MIN, Infinity, hitScratch)) {
      if (sky.kind === 'gradient') {
        const t = 0.5 * (dy + 1)
        Lr += br * (sky.horizon.r + (sky.zenith.r - sky.horizon.r) * t) * sky.strength
        Lg += bg * (sky.horizon.g + (sky.zenith.g - sky.horizon.g) * t) * sky.strength
        Lb += bb * (sky.horizon.b + (sky.zenith.b - sky.horizon.b) * t) * sky.strength
      }
      break
    }
    const prim = hitScratch.prim
    const t = hitScratch.t
    const o = prim * PRIM_STRIDE
    const px = ox + dx * t, py = oy + dy * t, pz = oz + dz * t
    let nx: number, ny: number, nz: number
    if (prims[o + 3] === PRIM_SPHERE) {
      const invR = 1 / prims[o + 4]!
      nx = (px - prims[o]!) * invR
      ny = (py - prims[o + 1]!) * invR
      nz = (pz - prims[o + 2]!) * invR
    } else {
      nx = prims[o + 12]!
      ny = prims[o + 13]!
      nz = prims[o + 14]!
    }
    const frontFace = dx * nx + dy * ny + dz * nz < 0
    // shading normal faces the incoming ray
    const sx = frontFace ? nx : -nx
    const sy = frontFace ? ny : -ny
    const sz = frontFace ? nz : -nz

    const m = ps.primObject[prim]! * MAT_STRIDE
    const type = mats[m + 3]!
    const er = mats[m + 12]!, eg = mats[m + 13]!, eb = mats[m + 14]!
    if (er > 0 || eg > 0 || eb > 0) {
      let w = 1
      if (useNee && !specularBounce) {
        const pl = lightPdfScalar(ps, prevX, prevY, prevZ, px, py, pz, nx, ny, nz, prim)
        w = powerHeuristic(prevPdf, pl)
      }
      Lr += br * er * w
      Lg += bg * eg * w
      Lb += bb * eb * w
    }
    if (type === MAT_EMISSIVE) break

    let ndx: number, ndy: number, ndz: number
    if (type === MAT_LAMBERTIAN) {
      let ar = mats[m]!, ag = mats[m + 1]!, ab = mats[m + 2]!
      if (mats[m + 6]! > 0.5 && !checkerPrimary(px, py, pz, mats[m + 5]!)) {
        ar = mats[m + 8]!
        ag = mats[m + 9]!
        ab = mats[m + 10]!
      }
      if (useNee && sampleLightInto(ps, px, py, pz, rng, lightScratch)) {
        const wx = lightScratch[0]!, wy = lightScratch[1]!, wz = lightScratch[2]!
        const pdfL = lightScratch[4]!
        const cosS = sx * wx + sy * wy + sz * wz
        if (cosS > 0 && pdfL > 0 && !occluded(ps, px, py, pz, wx, wy, wz, T_MIN, lightScratch[3]! * SHADOW_SHORTEN)) {
          const pdfB = cosS * INV_PI
          const k = (cosS * INV_PI * powerHeuristic(pdfL, pdfB)) / pdfL
          Lr += br * ar * lightScratch[5]! * k
          Lg += bg * ag * lightScratch[6]! * k
          Lb += bb * ab * lightScratch[7]! * k
        }
      }
      cosineHemisphereInto(sx, sy, sz, rng.next(), rng.next(), dirScratch)
      ndx = dirScratch[0]!
      ndy = dirScratch[1]!
      ndz = dirScratch[2]!
      prevPdf = Math.max(0, sx * ndx + sy * ndy + sz * ndz) * INV_PI
      br *= ar
      bg *= ag
      bb *= ab
      specularBounce = false
    } else if (type === MAT_METAL) {
      metalInto(dx, dy, dz, sx, sy, sz, mats[m + 4]!, rng.next(), rng.next(), rng.next(), dirScratch)
      ndx = dirScratch[0]!
      ndy = dirScratch[1]!
      ndz = dirScratch[2]!
      if (ndx * sx + ndy * sy + ndz * sz <= 0) break
      br *= mats[m]!
      bg *= mats[m + 1]!
      bb *= mats[m + 2]!
      specularBounce = true
    } else {
      dielectricInto(dx, dy, dz, sx, sy, sz, mats[m + 4]!, frontFace, rng.next(), dirScratch)
      ndx = dirScratch[0]!
      ndy = dirScratch[1]!
      ndz = dirScratch[2]!
      br *= mats[m]!
      bg *= mats[m + 1]!
      bb *= mats[m + 2]!
      specularBounce = true
    }

    if (opts.russianRoulette && bounce >= 3) {
      const maxB = br > bg ? (br > bb ? br : bb) : bg > bb ? bg : bb
      const pSurvive = maxB > 1 ? 1 : maxB < 0.05 ? 0.05 : maxB
      if (rng.next() > pSurvive) break
      br /= pSurvive
      bg /= pSurvive
      bb /= pSurvive
    }

    prevX = px
    prevY = py
    prevZ = pz
    ox = px
    oy = py
    oz = pz
    dx = ndx
    dy = ndy
    dz = ndz
  }
  out[0] = Lr
  out[1] = Lg
  out[2] = Lb
  return out
}

/** Vec3 convenience wrapper around `radianceInto`. */
export function radiance(ps: PackedScene, origin: Vec3, dir: Vec3, rng: Rng, opts: TraceOptions): Vec3 {
  const len = Math.sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) || 1
  const r = radianceInto(ps, origin.x, origin.y, origin.z, dir.x / len, dir.y / len, dir.z / len, rng, opts)
  return v3(r[0]!, r[1]!, r[2]!)
}

export interface RenderOptions extends TraceOptions {
  readonly width: number
  readonly height: number
  readonly spp: number
  readonly seed?: number
}

/**
 * Render a full image; returns linear RGB (3 floats/pixel), row 0 at the top,
 * already divided by the sample count.
 */
export function renderImage(scene: Scene | PackedScene, frame: CameraFrame, opts: RenderOptions): Float32Array {
  const ps = 'primCount' in scene ? scene : packScene(scene)
  const out = new Float32Array(opts.width * opts.height * 3)
  for (let s = 0; s < opts.spp; s++) accumulatePass(ps, frame, opts.width, opts.height, s + (opts.seed ?? 0) * 7919, opts, out)
  const inv = 1 / opts.spp
  for (let i = 0; i < out.length; i++) out[i]! *= inv
  return out
}

/**
 * Add one sample per pixel into `accum` (sum, not average). This is the unit
 * of progressive work for the Web Worker fallback.
 */
export function accumulatePass(
  ps: PackedScene,
  frame: CameraFrame,
  width: number,
  height: number,
  frameIndex: number,
  opts: TraceOptions,
  accum: Float32Array,
): void {
  const useLens = frame.lensRadius > 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x
      const rng = Rng.fromPixel(pixel, frameIndex)
      const sx = (x + rng.next()) / width
      const sy = 1 - (y + rng.next()) / height
      let lx = 0, ly = 0
      if (useLens) {
        const r = Math.sqrt(rng.next())
        const theta = TWO_PI * rng.next()
        lx = r * Math.cos(theta)
        ly = r * Math.sin(theta)
      }
      const ray = generateRay(frame, sx, sy, lx, ly)
      const c = radianceInto(ps, ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z, rng, opts)
      const i = pixel * 3
      const cr = c[0]!, cg = c[1]!, cb = c[2]!
      if (Number.isFinite(cr) && Number.isFinite(cg) && Number.isFinite(cb)) {
        accum[i]! += cr
        accum[i + 1]! += cg
        accum[i + 2]! += cb
      }
    }
  }
}

/** Convenience: frame for a scene's own camera at a given aspect ratio. */
export function sceneFrame(scene: Scene, width: number, height: number): CameraFrame {
  return cameraFrame(scene.camera, width / height)
}
