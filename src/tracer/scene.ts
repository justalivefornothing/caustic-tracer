import type { Vec3 } from './vec3.ts'
import { add, cross, normalize, scale, sub, v3 } from './vec3.ts'

/** Linear RGB colour, components nominally in [0, 1] (emission may exceed). */
export interface Color {
  readonly r: number
  readonly g: number
  readonly b: number
}
export const rgb = (r: number, g: number, b: number): Color => ({ r, g, b })
export const WHITE: Color = rgb(1, 1, 1)
export const BLACK: Color = rgb(0, 0, 0)

export interface CheckerTexture {
  /** Second checker colour; the first is the material albedo. */
  readonly color2: Color
  /** Number of checks per scene unit. */
  readonly scale: number
}

export type Material =
  | {
      readonly type: 'lambertian'
      readonly albedo: Color
      readonly checker?: CheckerTexture
      /** Optional glow on a diffuse surface (used by the white-furnace test). */
      readonly emission?: Color
    }
  | { readonly type: 'metal'; readonly albedo: Color; readonly roughness: number }
  | { readonly type: 'dielectric'; readonly ior: number; readonly tint?: Color }
  | { readonly type: 'emissive'; readonly color: Color; readonly strength: number }

export type MaterialType = Material['type']

export type Shape =
  | { readonly kind: 'sphere'; readonly center: Vec3; readonly radius: number }
  | { readonly kind: 'triangle'; readonly a: Vec3; readonly b: Vec3; readonly c: Vec3 }
  /** Parallelogram spanned by `u` and `v` from `origin`; expands to two triangles. */
  | { readonly kind: 'quad'; readonly origin: Vec3; readonly u: Vec3; readonly v: Vec3 }
  /** Indexed triangle mesh. `indices` holds 3 vertex indices per triangle. */
  | { readonly kind: 'mesh'; readonly vertices: readonly Vec3[]; readonly indices: readonly number[] }

export interface SceneObject {
  readonly id: string
  readonly name: string
  readonly shape: Shape
  readonly material: Material
}

export type Sky =
  | { readonly kind: 'none' }
  | { readonly kind: 'gradient'; readonly zenith: Color; readonly horizon: Color; readonly strength: number }

/** Orbit-style camera description; the world-space frame is derived from it. */
export interface CameraSettings {
  readonly target: Vec3
  /** Horizontal orbit angle in radians. */
  readonly yaw: number
  /** Elevation angle in radians, clamped to (-pi/2, pi/2). */
  readonly pitch: number
  readonly distance: number
  /** Vertical field of view in degrees. */
  readonly fovDeg: number
  /** Thin-lens aperture radius in scene units (0 = pinhole). */
  readonly lensRadius: number
  readonly focusDistance: number
}

export interface Scene {
  readonly name: string
  readonly objects: readonly SceneObject[]
  readonly camera: CameraSettings
  readonly sky: Sky
}

// ---------------------------------------------------------------------------
// Triangle expansion shared by the packer, picking and bounds computation.

export interface Triangle {
  readonly a: Vec3
  readonly b: Vec3
  readonly c: Vec3
}

export function quadTriangles(origin: Vec3, u: Vec3, v: Vec3): [Triangle, Triangle] {
  const p0 = origin
  const p1 = add(origin, u)
  const p2 = add(add(origin, u), v)
  const p3 = add(origin, v)
  return [
    { a: p0, b: p1, c: p2 },
    { a: p0, b: p2, c: p3 },
  ]
}

export function meshTriangles(vertices: readonly Vec3[], indices: readonly number[]): Triangle[] {
  const tris: Triangle[] = []
  for (let i = 0; i + 2 < indices.length; i += 3) {
    const a = vertices[indices[i]!]!
    const b = vertices[indices[i + 1]!]!
    const c = vertices[indices[i + 2]!]!
    tris.push({ a, b, c })
  }
  return tris
}

export function shapeTriangles(shape: Shape): Triangle[] {
  switch (shape.kind) {
    case 'sphere':
      return []
    case 'triangle':
      return [{ a: shape.a, b: shape.b, c: shape.c }]
    case 'quad':
      return quadTriangles(shape.origin, shape.u, shape.v)
    case 'mesh':
      return meshTriangles(shape.vertices, shape.indices)
  }
}

export function triangleNormal(t: Triangle): Vec3 {
  return normalize(cross(sub(t.b, t.a), sub(t.c, t.a)))
}

export function triangleArea(t: Triangle): number {
  const c = cross(sub(t.b, t.a), sub(t.c, t.a))
  return 0.5 * Math.sqrt(c.x * c.x + c.y * c.y + c.z * c.z)
}

export function primitiveCount(shape: Shape): number {
  switch (shape.kind) {
    case 'sphere':
      return 1
    case 'triangle':
      return 1
    case 'quad':
      return 2
    case 'mesh':
      return Math.floor(shape.indices.length / 3)
  }
}

/** Axis-aligned bounds of a shape. */
export function shapeBounds(shape: Shape): { min: Vec3; max: Vec3 } {
  if (shape.kind === 'sphere') {
    const r = shape.radius
    return {
      min: sub(shape.center, v3(r, r, r)),
      max: add(shape.center, v3(r, r, r)),
    }
  }
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  const pts: Vec3[] = []
  if (shape.kind === 'triangle') pts.push(shape.a, shape.b, shape.c)
  else if (shape.kind === 'quad') {
    pts.push(shape.origin, add(shape.origin, shape.u), add(shape.origin, shape.v), add(add(shape.origin, shape.u), shape.v))
  } else pts.push(...shape.vertices)
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.z < minZ) minZ = p.z
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
    if (p.z > maxZ) maxZ = p.z
  }
  return { min: v3(minX, minY, minZ), max: v3(maxX, maxY, maxZ) }
}

export function shapeCenter(shape: Shape): Vec3 {
  if (shape.kind === 'sphere') return shape.center
  const b = shapeBounds(shape)
  return scale(add(b.min, b.max), 0.5)
}

/** Total emitted radiance colour for a material, or null when it does not emit. */
export function materialEmission(m: Material): Color | null {
  if (m.type === 'emissive') {
    return rgb(m.color.r * m.strength, m.color.g * m.strength, m.color.b * m.strength)
  }
  if (m.type === 'lambertian' && m.emission) {
    const e = m.emission
    if (e.r > 0 || e.g > 0 || e.b > 0) return e
  }
  return null
}

let idCounter = 0
/** Generate a short unique id for scene objects (no DOM/crypto needed). */
export function makeId(prefix = 'obj'): string {
  idCounter = (idCounter + 1) % 1_000_000
  return `${prefix}-${Date.now().toString(36)}-${idCounter.toString(36)}`
}
