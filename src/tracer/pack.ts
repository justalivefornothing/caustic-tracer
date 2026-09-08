/**
 * Flattens a Scene into the exact float arrays the GPU consumes as RGBA32F
 * textures. The CPU tracer reads these same arrays, so both renderers share
 * one data layout rather than two interpretations of the scene.
 *
 * Primitive: 16 floats = 4 texels
 *   t0: [ a.x, a.y, a.z, type ]          type 0 = sphere (a = centre), 1 = triangle
 *   t1: [ b.x, b.y, b.z, object ]        sphere stores its radius in b.x
 *   t2: [ c.x, c.y, c.z, area ]          surface area (light sampling pdf)
 *   t3: [ n.x, n.y, n.z, 0 ]             unit geometric normal (triangles)
 *
 * Material: 16 floats = 4 texels, one per scene object
 *   t0: [ r, g, b, type ]                type 0 lambertian, 1 metal, 2 dielectric, 3 emissive
 *   t1: [ param, checkerScale, checkerOn, 0 ]   param = roughness | ior | strength
 *   t2: [ r2, g2, b2, 0 ]                second checker colour
 *   t3: [ e.r, e.g, e.b, 0 ]             emitted radiance, zero for non-emitters
 */
import { allocPrimBounds, buildBvh, setPrimBounds, type Bvh } from './bvh.ts'
import {
  materialEmission,
  primitiveCount,
  shapeTriangles,
  triangleArea,
  triangleNormal,
  type Material,
  type Scene,
  type Sky,
} from './scene.ts'

export const PRIM_STRIDE = 16
export const MAT_STRIDE = 16
export const TEXELS_PER_PRIM = 4
export const TEXELS_PER_MAT = 4
export const TEXELS_PER_NODE = 2
/** Data textures are this many texels wide; index -> (i & 1023, i >> 10). */
export const TEX_WIDTH = 1024

export const PRIM_SPHERE = 0
export const PRIM_TRIANGLE = 1

export const MAT_LAMBERTIAN = 0
export const MAT_METAL = 1
export const MAT_DIELECTRIC = 2
export const MAT_EMISSIVE = 3

export interface PackedScene {
  readonly prims: Float32Array
  readonly primCount: number
  readonly nodes: Float32Array
  readonly nodeCount: number
  readonly mats: Float32Array
  readonly matCount: number
  /** Packed primitive indices of every emitting primitive. */
  readonly lights: Int32Array
  /** packed primitive index -> scene object index */
  readonly primObject: Uint32Array
  readonly sky: Sky
  readonly bvh: Bvh
}

export function materialTypeCode(m: Material): number {
  switch (m.type) {
    case 'lambertian':
      return MAT_LAMBERTIAN
    case 'metal':
      return MAT_METAL
    case 'dielectric':
      return MAT_DIELECTRIC
    case 'emissive':
      return MAT_EMISSIVE
  }
}

/** Write one material into `out` at slot `index`. */
export function writeMaterial(out: Float32Array, index: number, m: Material): void {
  const o = index * MAT_STRIDE
  out.fill(0, o, o + MAT_STRIDE)
  out[o + 3] = materialTypeCode(m)
  switch (m.type) {
    case 'lambertian':
      out[o] = m.albedo.r
      out[o + 1] = m.albedo.g
      out[o + 2] = m.albedo.b
      if (m.checker) {
        out[o + 5] = m.checker.scale
        out[o + 6] = 1
        out[o + 8] = m.checker.color2.r
        out[o + 9] = m.checker.color2.g
        out[o + 10] = m.checker.color2.b
      }
      break
    case 'metal':
      out[o] = m.albedo.r
      out[o + 1] = m.albedo.g
      out[o + 2] = m.albedo.b
      out[o + 4] = m.roughness
      break
    case 'dielectric': {
      const tint = m.tint ?? { r: 1, g: 1, b: 1 }
      out[o] = tint.r
      out[o + 1] = tint.g
      out[o + 2] = tint.b
      out[o + 4] = m.ior
      break
    }
    case 'emissive':
      out[o] = m.color.r
      out[o + 1] = m.color.g
      out[o + 2] = m.color.b
      out[o + 4] = m.strength
      break
  }
  const e = materialEmission(m)
  if (e) {
    out[o + 12] = e.r
    out[o + 13] = e.g
    out[o + 14] = e.b
  }
}

export function packMaterials(scene: Scene): Float32Array {
  const out = new Float32Array(Math.max(1, scene.objects.length) * MAT_STRIDE)
  scene.objects.forEach((obj, i) => writeMaterial(out, i, obj.material))
  return out
}

/**
 * Enumerate primitives in scene order (before BVH reordering), returning the
 * unordered primitive floats plus per-primitive bounds.
 */
function enumeratePrimitives(scene: Scene) {
  let total = 0
  for (const obj of scene.objects) total += primitiveCount(obj.shape)
  const prims = new Float32Array(Math.max(1, total) * PRIM_STRIDE)
  const primObject = new Uint32Array(Math.max(1, total))
  const bounds = allocPrimBounds(total)
  let p = 0
  scene.objects.forEach((obj, objIndex) => {
    const shape = obj.shape
    if (shape.kind === 'sphere') {
      const o = p * PRIM_STRIDE
      const c = shape.center
      const r = shape.radius
      prims[o] = c.x
      prims[o + 1] = c.y
      prims[o + 2] = c.z
      prims[o + 3] = PRIM_SPHERE
      prims[o + 4] = r
      prims[o + 7] = objIndex
      prims[o + 11] = 4 * Math.PI * r * r
      primObject[p] = objIndex
      // Tiny pad guards against float32 rounding of the node bounds.
      const rp = r + 1e-6 + Math.abs(r) * 1e-6
      setPrimBounds(bounds, p, c.x - rp, c.y - rp, c.z - rp, c.x + rp, c.y + rp, c.z + rp)
      p++
      return
    }
    for (const tri of shapeTriangles(shape)) {
      const o = p * PRIM_STRIDE
      const n = triangleNormal(tri)
      prims[o] = tri.a.x
      prims[o + 1] = tri.a.y
      prims[o + 2] = tri.a.z
      prims[o + 3] = PRIM_TRIANGLE
      prims[o + 4] = tri.b.x
      prims[o + 5] = tri.b.y
      prims[o + 6] = tri.b.z
      prims[o + 7] = objIndex
      prims[o + 8] = tri.c.x
      prims[o + 9] = tri.c.y
      prims[o + 10] = tri.c.z
      prims[o + 11] = triangleArea(tri)
      prims[o + 12] = n.x
      prims[o + 13] = n.y
      prims[o + 14] = n.z
      primObject[p] = objIndex
      const pad = 1e-6
      setPrimBounds(
        bounds,
        p,
        Math.min(tri.a.x, tri.b.x, tri.c.x) - pad,
        Math.min(tri.a.y, tri.b.y, tri.c.y) - pad,
        Math.min(tri.a.z, tri.b.z, tri.c.z) - pad,
        Math.max(tri.a.x, tri.b.x, tri.c.x) + pad,
        Math.max(tri.a.y, tri.b.y, tri.c.y) + pad,
        Math.max(tri.a.z, tri.b.z, tri.c.z) + pad,
      )
      p++
    }
  })
  return { prims, primObject, bounds, total }
}

/** Build the BVH and emit primitives in leaf order so leaves reference contiguous ranges. */
export function packScene(scene: Scene): PackedScene {
  const { prims, primObject, bounds, total } = enumeratePrimitives(scene)
  const bvh = buildBvh(bounds)
  const ordered = new Float32Array(Math.max(1, total) * PRIM_STRIDE)
  const orderedObject = new Uint32Array(Math.max(1, total))
  for (let i = 0; i < total; i++) {
    const src = bvh.order[i]!
    ordered.set(prims.subarray(src * PRIM_STRIDE, (src + 1) * PRIM_STRIDE), i * PRIM_STRIDE)
    orderedObject[i] = primObject[src]!
  }
  const mats = packMaterials(scene)
  const lights: number[] = []
  for (let i = 0; i < total; i++) {
    const obj = scene.objects[orderedObject[i]!]!
    if (materialEmission(obj.material)) lights.push(i)
  }
  return {
    prims: ordered,
    primCount: total,
    nodes: bvh.nodes,
    nodeCount: bvh.nodeCount,
    mats,
    matCount: scene.objects.length,
    lights: Int32Array.from(lights),
    primObject: orderedObject,
    sky: scene.sky,
    bvh,
  }
}

/** Number of texture rows needed for `texels` texels at TEX_WIDTH. */
export function textureRows(texels: number): number {
  return Math.max(1, Math.ceil(texels / TEX_WIDTH))
}

/** Pad a float array to a whole number of texture rows (RGBA32F, TEX_WIDTH wide). */
export function padToTexture(data: Float32Array, texelCount: number): { data: Float32Array; rows: number } {
  const rows = textureRows(texelCount)
  const needed = rows * TEX_WIDTH * 4
  if (data.length === needed) return { data, rows }
  const out = new Float32Array(needed)
  out.set(data.subarray(0, Math.min(data.length, needed)))
  return { data: out, rows }
}
