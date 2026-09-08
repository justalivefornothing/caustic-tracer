/**
 * Ray/primitive intersection and stackless BVH traversal over the packed
 * scene layout. Everything here is scalar and allocation-free; it is the CPU
 * twin of the GLSL loop (same node encoding, same skip-link rule, same
 * primitive tests), so a hit found here is the hit the GPU finds.
 *
 * Kept in one module on purpose: module-boundary calls are cheap in plain
 * Node but noticeably slower under test-runner transforms, and this is the
 * hottest code in the project.
 */
import { NODE_STRIDE } from './bvh.ts'
import { PRIM_SPHERE, PRIM_STRIDE, type PackedScene } from './pack.ts'

export const T_MIN = 1e-4

/**
 * Ray/sphere intersection. Returns the nearest t in (tMin, tMax) or -1.
 * Solves |o + t d - c|^2 = r^2 with the half-b form of the quadratic.
 */
export function raySphere(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  cx: number, cy: number, cz: number, r: number,
  tMin: number, tMax: number,
): number {
  const lx = ox - cx, ly = oy - cy, lz = oz - cz
  const a = dx * dx + dy * dy + dz * dz
  const halfB = lx * dx + ly * dy + lz * dz
  const c = lx * lx + ly * ly + lz * lz - r * r
  const disc = halfB * halfB - a * c
  if (disc < 0) return -1
  const s = Math.sqrt(disc)
  let t = (-halfB - s) / a
  if (t <= tMin || t >= tMax) {
    t = (-halfB + s) / a
    if (t <= tMin || t >= tMax) return -1
  }
  return t
}

/** Output of a triangle hit: t plus barycentrics (u for b, v for c). */
export interface TriHit {
  t: number
  u: number
  v: number
}

/**
 * Two-sided Moller-Trumbore ray/triangle test. Returns true and fills `out`
 * when the ray hits inside (tMin, tMax).
 */
export function rayTriangle(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  tMin: number, tMax: number,
  out: TriHit,
): boolean {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az
  // p = d x e2
  const px = dy * e2z - dz * e2y
  const py = dz * e2x - dx * e2z
  const pz = dx * e2y - dy * e2x
  const det = e1x * px + e1y * py + e1z * pz
  if (det > -1e-12 && det < 1e-12) return false
  const invDet = 1 / det
  const tx = ox - ax, ty = oy - ay, tz = oz - az
  const u = (tx * px + ty * py + tz * pz) * invDet
  if (u < 0 || u > 1) return false
  // q = t x e1
  const qx = ty * e1z - tz * e1y
  const qy = tz * e1x - tx * e1z
  const qz = tx * e1y - ty * e1x
  const v = (dx * qx + dy * qy + dz * qz) * invDet
  if (v < 0 || u + v > 1) return false
  const t = (e2x * qx + e2y * qy + e2z * qz) * invDet
  if (t <= tMin || t >= tMax) return false
  out.t = t
  out.u = u
  out.v = v
  return true
}

/**
 * Slab test against an AABB using precomputed inverse direction. Returns the
 * entry distance (>= 0 clamped) or Infinity on a miss; hits only count when
 * the box is entered before tMax.
 */
export function rayAabb(
  ox: number, oy: number, oz: number,
  idx: number, idy: number, idz: number,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
  tMax: number,
): number {
  let t0 = (minX - ox) * idx
  let t1 = (maxX - ox) * idx
  let tNear = t0 < t1 ? t0 : t1
  let tFar = t0 < t1 ? t1 : t0
  t0 = (minY - oy) * idy
  t1 = (maxY - oy) * idy
  if (t0 < t1) {
    if (t0 > tNear) tNear = t0
    if (t1 < tFar) tFar = t1
  } else {
    if (t1 > tNear) tNear = t1
    if (t0 < tFar) tFar = t0
  }
  t0 = (minZ - oz) * idz
  t1 = (maxZ - oz) * idz
  if (t0 < t1) {
    if (t0 > tNear) tNear = t0
    if (t1 < tFar) tFar = t1
  } else {
    if (t1 > tNear) tNear = t1
    if (t0 < tFar) tFar = t0
  }
  if (tNear > tFar || tFar < 0 || tNear > tMax) return Infinity
  return tNear < 0 ? 0 : tNear
}

// ---------------------------------------------------------------------------
// BVH traversal

export interface HitRecord {
  /** Distance along the ray; Infinity when nothing was hit. */
  t: number
  /** Packed primitive index, -1 for a miss. */
  prim: number
  /** Barycentric coordinates for triangles (weight of b and c). */
  u: number
  v: number
}

export const makeHit = (): HitRecord => ({ t: Infinity, prim: -1, u: 0, v: 0 })

const triScratch: TriHit = { t: 0, u: 0, v: 0 }

const safeInv = (d: number): number => 1 / (d === 0 ? 1e-20 : d)

/** Traversal stack for the ordered CPU walk; BVH depth never gets near this. */
const STACK_SIZE = 512
const stackNode = new Int32Array(STACK_SIZE)
const stackEntry = new Float64Array(STACK_SIZE)

/**
 * Test the primitives of a leaf; updates the closest-hit state through the
 * shared `leafScratch` record. Returns the new closest t.
 */
const leafScratch: HitRecord = makeHit()

function testLeaf(
  prims: Float32Array,
  leaf: number,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  closest: number,
): number {
  const count = (leaf | 0) & 7
  const start = (leaf | 0) >> 3
  for (let p = start; p < start + count; p++) {
    const o = p * PRIM_STRIDE
    if (prims[o + 3] === PRIM_SPHERE) {
      const t = raySphere(
        ox, oy, oz, dx, dy, dz,
        prims[o]!, prims[o + 1]!, prims[o + 2]!, prims[o + 4]!,
        tMin, closest,
      )
      if (t > 0) {
        closest = t
        leafScratch.prim = p
      }
    } else if (
      rayTriangle(
        ox, oy, oz, dx, dy, dz,
        prims[o]!, prims[o + 1]!, prims[o + 2]!,
        prims[o + 4]!, prims[o + 5]!, prims[o + 6]!,
        prims[o + 8]!, prims[o + 9]!, prims[o + 10]!,
        tMin, closest, triScratch,
      )
    ) {
      closest = triScratch.t
      leafScratch.prim = p
      leafScratch.u = triScratch.u
      leafScratch.v = triScratch.v
    }
  }
  return closest
}

/**
 * Closest hit using a front-to-back walk with a small stack. Uses the same
 * flattened layout as the GPU: the left child of interior node i is i + 1 and
 * the right child is the left child's skip link. Visiting the nearer child
 * first lets the shrinking `closest` bound prune most of the far subtrees,
 * which the GPU's stackless loop cannot do.
 */
export function intersectClosest(
  ps: PackedScene,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
  hit: HitRecord,
): boolean {
  const nodes = ps.nodes
  const prims = ps.prims
  if (ps.nodeCount === 0) {
    hit.t = Infinity
    hit.prim = -1
    return false
  }
  const idx = safeInv(dx), idy = safeInv(dy), idz = safeInv(dz)
  let closest = tMax
  leafScratch.prim = -1
  leafScratch.u = 0
  leafScratch.v = 0

  if (rayAabb(ox, oy, oz, idx, idy, idz, nodes[0]!, nodes[1]!, nodes[2]!, nodes[4]!, nodes[5]!, nodes[6]!, closest) === Infinity) {
    hit.t = Infinity
    hit.prim = -1
    return false
  }

  let sp = 0
  let node = 0
  for (;;) {
    const nb = node * NODE_STRIDE
    const leaf = nodes[nb + 7]!
    if (leaf >= 0) {
      closest = testLeaf(prims, leaf, ox, oy, oz, dx, dy, dz, tMin, closest)
    } else {
      const left = node + 1
      const right = nodes[left * NODE_STRIDE + 3]! | 0
      const lb = left * NODE_STRIDE
      const rb = right * NODE_STRIDE
      const tl = rayAabb(ox, oy, oz, idx, idy, idz, nodes[lb]!, nodes[lb + 1]!, nodes[lb + 2]!, nodes[lb + 4]!, nodes[lb + 5]!, nodes[lb + 6]!, closest)
      const tr = rayAabb(ox, oy, oz, idx, idy, idz, nodes[rb]!, nodes[rb + 1]!, nodes[rb + 2]!, nodes[rb + 4]!, nodes[rb + 5]!, nodes[rb + 6]!, closest)
      if (tl !== Infinity) {
        if (tr !== Infinity) {
          if (tl <= tr) {
            stackNode[sp] = right
            stackEntry[sp++] = tr
            node = left
          } else {
            stackNode[sp] = left
            stackEntry[sp++] = tl
            node = right
          }
        } else {
          node = left
        }
        continue
      }
      if (tr !== Infinity) {
        node = right
        continue
      }
    }
    // pop, skipping entries the shrinking bound has already excluded
    do {
      if (sp === 0) {
        if (leafScratch.prim < 0) {
          hit.t = Infinity
          hit.prim = -1
          return false
        }
        hit.t = closest
        hit.prim = leafScratch.prim
        hit.u = leafScratch.u
        hit.v = leafScratch.v
        return true
      }
      sp--
    } while (stackEntry[sp]! >= closest)
    node = stackNode[sp]!
  }
}

/**
 * Closest hit using the stackless skip-link loop exactly as the GLSL tracer
 * runs it: hit the box -> step to node + 1, miss -> jump to the skip link.
 * Slower on the CPU than the ordered walk but it is the literal mirror of the
 * shader, which is why the tests exercise it too.
 */
export function intersectClosestStackless(
  ps: PackedScene,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
  hit: HitRecord,
): boolean {
  const nodes = ps.nodes
  const prims = ps.prims
  const nodeCount = ps.nodeCount
  const idx = safeInv(dx), idy = safeInv(dy), idz = safeInv(dz)
  let closest = tMax
  let bestPrim = -1
  let bestU = 0
  let bestV = 0

  let node = 0
  while (node < nodeCount) {
    const nb = node * NODE_STRIDE
    const entry = rayAabb(
      ox, oy, oz, idx, idy, idz,
      nodes[nb]!, nodes[nb + 1]!, nodes[nb + 2]!,
      nodes[nb + 4]!, nodes[nb + 5]!, nodes[nb + 6]!,
      closest,
    )
    if (entry === Infinity) {
      node = nodes[nb + 3]! | 0 // skip link: jump past this subtree
      continue
    }
    const leaf = nodes[nb + 7]!
    if (leaf >= 0) {
      const count = (leaf | 0) & 7
      const start = (leaf | 0) >> 3
      for (let p = start; p < start + count; p++) {
        const o = p * PRIM_STRIDE
        if (prims[o + 3] === PRIM_SPHERE) {
          const t = raySphere(
            ox, oy, oz, dx, dy, dz,
            prims[o]!, prims[o + 1]!, prims[o + 2]!, prims[o + 4]!,
            tMin, closest,
          )
          if (t > 0) {
            closest = t
            bestPrim = p
          }
        } else if (
          rayTriangle(
            ox, oy, oz, dx, dy, dz,
            prims[o]!, prims[o + 1]!, prims[o + 2]!,
            prims[o + 4]!, prims[o + 5]!, prims[o + 6]!,
            prims[o + 8]!, prims[o + 9]!, prims[o + 10]!,
            tMin, closest, triScratch,
          )
        ) {
          closest = triScratch.t
          bestPrim = p
          bestU = triScratch.u
          bestV = triScratch.v
        }
      }
    }
    node++
  }

  if (bestPrim < 0) {
    hit.t = Infinity
    hit.prim = -1
    return false
  }
  hit.t = closest
  hit.prim = bestPrim
  hit.u = bestU
  hit.v = bestV
  return true
}

/** True when anything blocks the ray segment (tMin, tMax). */
export function occluded(
  ps: PackedScene,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
): boolean {
  const nodes = ps.nodes
  const prims = ps.prims
  const nodeCount = ps.nodeCount
  const idx = safeInv(dx), idy = safeInv(dy), idz = safeInv(dz)
  let node = 0
  while (node < nodeCount) {
    const nb = node * NODE_STRIDE
    const entry = rayAabb(
      ox, oy, oz, idx, idy, idz,
      nodes[nb]!, nodes[nb + 1]!, nodes[nb + 2]!,
      nodes[nb + 4]!, nodes[nb + 5]!, nodes[nb + 6]!,
      tMax,
    )
    if (entry === Infinity) {
      node = nodes[nb + 3]! | 0
      continue
    }
    const leaf = nodes[nb + 7]!
    if (leaf >= 0) {
      const count = (leaf | 0) & 7
      const start = (leaf | 0) >> 3
      for (let p = start; p < start + count; p++) {
        const o = p * PRIM_STRIDE
        if (prims[o + 3] === PRIM_SPHERE) {
          if (
            raySphere(
              ox, oy, oz, dx, dy, dz,
              prims[o]!, prims[o + 1]!, prims[o + 2]!, prims[o + 4]!,
              tMin, tMax,
            ) > 0
          ) return true
        } else if (
          rayTriangle(
            ox, oy, oz, dx, dy, dz,
            prims[o]!, prims[o + 1]!, prims[o + 2]!,
            prims[o + 4]!, prims[o + 5]!, prims[o + 6]!,
            prims[o + 8]!, prims[o + 9]!, prims[o + 10]!,
            tMin, tMax, triScratch,
          )
        ) return true
      }
    }
    node++
  }
  return false
}

/** Reference O(n) scan used by tests to validate the BVH traversal. */
export function intersectBruteForce(
  ps: PackedScene,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
  hit: HitRecord,
): boolean {
  const prims = ps.prims
  let closest = tMax
  let bestPrim = -1
  let bestU = 0
  let bestV = 0
  for (let p = 0; p < ps.primCount; p++) {
    const o = p * PRIM_STRIDE
    if (prims[o + 3] === PRIM_SPHERE) {
      const t = raySphere(
        ox, oy, oz, dx, dy, dz,
        prims[o]!, prims[o + 1]!, prims[o + 2]!, prims[o + 4]!,
        tMin, closest,
      )
      if (t > 0) {
        closest = t
        bestPrim = p
      }
    } else if (
      rayTriangle(
        ox, oy, oz, dx, dy, dz,
        prims[o]!, prims[o + 1]!, prims[o + 2]!,
        prims[o + 4]!, prims[o + 5]!, prims[o + 6]!,
        prims[o + 8]!, prims[o + 9]!, prims[o + 10]!,
        tMin, closest, triScratch,
      )
    ) {
      closest = triScratch.t
      bestPrim = p
      bestU = triScratch.u
      bestV = triScratch.v
    }
  }
  hit.t = bestPrim < 0 ? Infinity : closest
  hit.prim = bestPrim
  hit.u = bestU
  hit.v = bestV
  return bestPrim >= 0
}
