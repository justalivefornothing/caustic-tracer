/**
 * CPU-side picking: shoot the pinhole ray through a screen position and use
 * the same BVH traversal as the tracer to find what it hits.
 */
import { primaryRay, type CameraFrame } from './camera.ts'
import { intersectClosest, makeHit, T_MIN } from './intersect.ts'
import type { PackedScene } from './pack.ts'
import { add, scale, type Vec3 } from './vec3.ts'

export interface PickResult {
  readonly objectIndex: number
  readonly prim: number
  readonly t: number
  readonly point: Vec3
}

/**
 * @param sx normalised screen x in [0, 1]
 * @param sy normalised screen y in [0, 1], 1 = top
 */
export function pick(ps: PackedScene, frame: CameraFrame, sx: number, sy: number): PickResult | null {
  const ray = primaryRay(frame, sx, sy)
  const hit = makeHit()
  if (!intersectClosest(ps, ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z, T_MIN, Infinity, hit)) {
    return null
  }
  return {
    objectIndex: ps.primObject[hit.prim]!,
    prim: hit.prim,
    t: hit.t,
    point: add(ray.origin, scale(ray.dir, hit.t)),
  }
}
