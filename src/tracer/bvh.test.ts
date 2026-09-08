import { describe, expect, it } from 'vitest'
import { MAX_LEAF_PRIMS, NODE_STRIDE, decodeLeafCount, validateBvh } from './bvh.ts'
import { packScene } from './pack.ts'
import { Rng } from './rng.ts'
import { intersectBruteForce, intersectClosest, intersectClosestStackless, makeHit, occluded } from './intersect.ts'
import type { Scene, SceneObject } from './scene.ts'
import { v3 } from './vec3.ts'

function randomScene(rng: Rng, spheres: number, triangles: number): Scene {
  const objects: SceneObject[] = []
  for (let i = 0; i < spheres; i++) {
    objects.push({
      id: `s${i}`,
      name: `s${i}`,
      shape: {
        kind: 'sphere',
        center: v3(rng.range(-10, 10), rng.range(-10, 10), rng.range(-10, 10)),
        radius: rng.range(0.05, 0.6),
      },
      material: { type: 'lambertian', albedo: { r: 0.5, g: 0.5, b: 0.5 } },
    })
  }
  for (let i = 0; i < triangles; i++) {
    const a = v3(rng.range(-10, 10), rng.range(-10, 10), rng.range(-10, 10))
    objects.push({
      id: `t${i}`,
      name: `t${i}`,
      shape: {
        kind: 'triangle',
        a,
        b: v3(a.x + rng.range(-1, 1), a.y + rng.range(-1, 1), a.z + rng.range(-1, 1)),
        c: v3(a.x + rng.range(-1, 1), a.y + rng.range(-1, 1), a.z + rng.range(-1, 1)),
      },
      material: { type: 'lambertian', albedo: { r: 0.5, g: 0.5, b: 0.5 } },
    })
  }
  return {
    name: 'random',
    objects,
    camera: { target: v3(0, 0, 0), yaw: 0, pitch: 0, distance: 30, fovDeg: 40, lensRadius: 0, focusDistance: 30 },
    sky: { kind: 'none' },
  }
}

describe('BVH', () => {
  const rng = new Rng(1234)
  const scene = randomScene(rng, 1000, 2000)
  const ps = packScene(scene)

  it('packs every primitive (1000 spheres + 2000 triangles)', () => {
    expect(ps.primCount).toBe(3000)
    expect(ps.nodeCount).toBeGreaterThan(1)
    expect(ps.nodes.length).toBe(ps.nodeCount * NODE_STRIDE)
  })

  it('keeps every leaf at or below 4 primitives', () => {
    let leaves = 0
    for (let n = 0; n < ps.nodeCount; n++) {
      const leaf = ps.nodes[n * NODE_STRIDE + 7]!
      if (leaf >= 0) {
        leaves++
        expect(decodeLeafCount(leaf)).toBeLessThanOrEqual(MAX_LEAF_PRIMS)
        expect(decodeLeafCount(leaf)).toBeGreaterThan(0)
      }
    }
    expect(leaves).toBeGreaterThanOrEqual(Math.ceil(3000 / MAX_LEAF_PRIMS))
  })

  it('round-trips the skip-link layout: every skip points exactly past its subtree', () => {
    const report = validateBvh(ps.bvh, ps.primCount)
    expect(report.errors).toEqual([])
    expect(report.ok).toBe(true)
    expect(report.maxLeafSize).toBeLessThanOrEqual(4)
    // Every node's skip index must be greater than the node itself and at most the node count.
    for (let n = 0; n < ps.nodeCount; n++) {
      const skip = ps.nodes[n * NODE_STRIDE + 3]!
      expect(skip).toBeGreaterThan(n)
      expect(skip).toBeLessThanOrEqual(ps.nodeCount)
    }
  })

  const traversals = [
    ['ordered stack walk', intersectClosest],
    ['stackless skip-link walk (GPU mirror)', intersectClosestStackless],
  ] as const

  for (const [label, traverse] of traversals) {
    it(`${label} matches brute force closest hit for 1000 random rays (id and t within 1e-6)`, () => {
      const rayRng = new Rng(99)
      const a = makeHit()
      const b = makeHit()
      let hits = 0
      for (let i = 0; i < 1000; i++) {
        const ox = rayRng.range(-14, 14), oy = rayRng.range(-14, 14), oz = rayRng.range(-14, 14)
        // aim roughly toward the cloud so most rays hit something
        const tx = rayRng.range(-8, 8), ty = rayRng.range(-8, 8), tz = rayRng.range(-8, 8)
        let dx = tx - ox, dy = ty - oy, dz = tz - oz
        const len = Math.hypot(dx, dy, dz)
        dx /= len; dy /= len; dz /= len
        const hitA = traverse(ps, ox, oy, oz, dx, dy, dz, 1e-4, Infinity, a)
        const hitB = intersectBruteForce(ps, ox, oy, oz, dx, dy, dz, 1e-4, Infinity, b)
        expect(hitA).toBe(hitB)
        if (hitA) {
          hits++
          expect(a.prim).toBe(b.prim)
          expect(Math.abs(a.t - b.t)).toBeLessThan(1e-6)
          expect(Math.abs(a.u - b.u)).toBeLessThan(1e-9)
          expect(Math.abs(a.v - b.v)).toBeLessThan(1e-9)
        }
      }
      expect(hits).toBeGreaterThan(500)
    })

    it(`${label} respects a finite tMax`, () => {
      const rayRng = new Rng(5)
      const a = makeHit()
      const b = makeHit()
      for (let i = 0; i < 300; i++) {
        const ox = rayRng.range(-12, 12), oy = rayRng.range(-12, 12), oz = rayRng.range(-12, 12)
        const dx = rayRng.range(-1, 1), dy = rayRng.range(-1, 1), dz = rayRng.range(-1, 1)
        const len = Math.hypot(dx, dy, dz) || 1
        const tMax = rayRng.range(0.5, 20)
        const hitA = traverse(ps, ox, oy, oz, dx / len, dy / len, dz / len, 1e-4, tMax, a)
        const hitB = intersectBruteForce(ps, ox, oy, oz, dx / len, dy / len, dz / len, 1e-4, tMax, b)
        expect(hitA).toBe(hitB)
        if (hitA) {
          expect(a.prim).toBe(b.prim)
          expect(a.t).toBeLessThan(tMax)
        }
      }
    })
  }

  it('agrees with brute force for occlusion queries', () => {
    const rayRng = new Rng(7)
    const b = makeHit()
    for (let i = 0; i < 500; i++) {
      const ox = rayRng.range(-12, 12), oy = rayRng.range(-12, 12), oz = rayRng.range(-12, 12)
      const dx = rayRng.range(-1, 1), dy = rayRng.range(-1, 1), dz = rayRng.range(-1, 1)
      const len = Math.hypot(dx, dy, dz) || 1
      const tMax = rayRng.range(1, 30)
      const blocked = occluded(ps, ox, oy, oz, dx / len, dy / len, dz / len, 1e-4, tMax)
      const ref = intersectBruteForce(ps, ox, oy, oz, dx / len, dy / len, dz / len, 1e-4, tMax, b)
      expect(blocked).toBe(ref)
    }
  })

  it('handles degenerate input: empty scene and coincident primitives', () => {
    const empty = packScene({ ...scene, objects: [] })
    expect(empty.primCount).toBe(0)
    expect(intersectClosest(empty, 0, 0, 0, 0, 0, 1, 1e-4, Infinity, makeHit())).toBe(false)

    const same: SceneObject[] = []
    for (let i = 0; i < 40; i++) {
      same.push({
        id: `c${i}`,
        name: 'c',
        shape: { kind: 'sphere', center: v3(1, 2, 3), radius: 0.5 },
        material: { type: 'lambertian', albedo: { r: 1, g: 1, b: 1 } },
      })
    }
    const coincident = packScene({ ...scene, objects: same })
    const report = validateBvh(coincident.bvh, coincident.primCount)
    expect(report.ok).toBe(true)
    expect(report.maxLeafSize).toBeLessThanOrEqual(4)
    const hit = makeHit()
    expect(intersectClosest(coincident, 1, 2, -5, 0, 0, 1, 1e-4, Infinity, hit)).toBe(true)
    expect(hit.t).toBeCloseTo(7.5, 5)
  })
})
