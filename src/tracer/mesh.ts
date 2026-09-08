/**
 * Procedural triangle meshes. Only what the presets need: a torus knot tube.
 */
import { add, cross, normalize, scale, sub, v3, type Vec3 } from './vec3.ts'

export interface MeshData {
  readonly vertices: Vec3[]
  readonly indices: number[]
}

export interface TorusKnotParams {
  /** Windings around the torus axis. */
  readonly p: number
  /** Windings through the hole. */
  readonly q: number
  /** Major radius of the guiding torus. */
  readonly radius: number
  /** Minor radius (distance of the curve from the torus core circle). */
  readonly ringRadius: number
  /** Radius of the swept tube. */
  readonly tube: number
  readonly tubularSegments: number
  readonly radialSegments: number
  /** World-space offset applied after generation (torus axis is +Y). */
  readonly center?: Vec3
}

/** Point on the (p, q) torus knot curve, torus axis along +Y. */
function knotPoint(t: number, p: number, q: number, R: number, r: number): { point: Vec3; core: Vec3 } {
  const cq = Math.cos(q * t)
  const sq = Math.sin(q * t)
  const cp = Math.cos(p * t)
  const sp = Math.sin(p * t)
  const ring = R + r * cq
  return {
    point: v3(ring * cp, r * sq, ring * sp),
    core: v3(R * cp, 0, R * sp),
  }
}

/**
 * Sweep a tube along a torus knot. The frame at each sample is built from the
 * curve tangent and the torus-surface normal (curve point minus core-circle
 * point), which is smooth and closes exactly, so the tube has no seam twist.
 */
export function torusKnot(params: TorusKnotParams): MeshData {
  const { p, q, radius: R, ringRadius: r, tube, tubularSegments: segs, radialSegments: rad } = params
  const center = params.center ?? v3(0, 0, 0)
  const vertices: Vec3[] = []
  const indices: number[] = []
  const dt = (2 * Math.PI) / segs

  for (let i = 0; i < segs; i++) {
    const t = i * dt
    const here = knotPoint(t, p, q, R, r)
    const next = knotPoint(t + dt * 0.01, p, q, R, r)
    const tangent = normalize(sub(next.point, here.point))
    let normal = normalize(sub(here.point, here.core))
    const binormal = normalize(cross(tangent, normal))
    normal = cross(binormal, tangent)
    for (let j = 0; j < rad; j++) {
      const a = (j / rad) * 2 * Math.PI
      const offset = add(scale(normal, Math.cos(a) * tube), scale(binormal, Math.sin(a) * tube))
      vertices.push(add(add(here.point, offset), center))
    }
  }

  for (let i = 0; i < segs; i++) {
    const i1 = (i + 1) % segs
    for (let j = 0; j < rad; j++) {
      const j1 = (j + 1) % rad
      const a = i * rad + j
      const b = i1 * rad + j
      const c = i1 * rad + j1
      const d = i * rad + j1
      indices.push(a, b, c, a, c, d)
    }
  }
  return { vertices, indices }
}
