import { NODE_STRIDE } from '../src/tracer/bvh.ts'
import { rayAabb, rayTriangle } from '../src/tracer/intersect.ts'
import { packScene, PRIM_STRIDE } from '../src/tracer/pack.ts'
import { Rng } from '../src/tracer/rng.ts'
import type { Scene, SceneObject } from '../src/tracer/scene.ts'
import { v3 } from '../src/tracer/vec3.ts'

const rng = new Rng(2024)
const objects: SceneObject[] = []
const N = 100000
for (let i = 0; i < N; i++) {
  const a = v3(rng.range(-30, 30), rng.range(-30, 30), rng.range(-30, 30))
  objects.push({ id: `t${i}`, name: 't', shape: { kind: 'triangle', a, b: v3(a.x + rng.range(-0.6, 0.6), a.y + rng.range(-0.6, 0.6), a.z + rng.range(-0.6, 0.6)), c: v3(a.x + rng.range(-0.6, 0.6), a.y + rng.range(-0.6, 0.6), a.z + rng.range(-0.6, 0.6)) }, material: { type: 'lambertian', albedo: { r: 0.5, g: 0.5, b: 0.5 } } })
}
const scene: Scene = { name: 'b', objects, camera: { target: v3(0, 0, 0), yaw: 0, pitch: 0, distance: 1, fovDeg: 40, lensRadius: 0, focusDistance: 1 }, sky: { kind: 'none' } }
const ps = packScene(scene)
const M = 20000
const rays = new Float64Array(M * 6)
for (let r = 0; r < M; r++) {
  const ox = rng.range(-45, 45), oy = rng.range(-45, 45), oz = rng.range(-45, 45)
  const tx = rng.range(-25, 25), ty = rng.range(-25, 25), tz = rng.range(-25, 25)
  let dx = tx - ox, dy = ty - oy, dz = tz - oz
  const len = Math.hypot(dx, dy, dz); dx /= len; dy /= len; dz /= len
  rays.set([ox, oy, oz, dx, dy, dz], r * 6)
}
const scratch = { t: 0, u: 0, v: 0 }
let visits = 0, triTests = 0, hits = 0, leafVisits = 0
const nodes = ps.nodes, prims = ps.prims
const t0 = performance.now()
for (let r = 0; r < M; r++) {
  const o = r * 6
  const ox = rays[o]!, oy = rays[o+1]!, oz = rays[o+2]!, dx = rays[o+3]!, dy = rays[o+4]!, dz = rays[o+5]!
  const idx = 1 / dx, idy = 1 / dy, idz = 1 / dz
  let node = 0, closest = Infinity, best = -1
  while (node < ps.nodeCount) {
    visits++
    const nb = node * NODE_STRIDE
    const e = rayAabb(ox, oy, oz, idx, idy, idz, nodes[nb]!, nodes[nb + 1]!, nodes[nb + 2]!, nodes[nb + 4]!, nodes[nb + 5]!, nodes[nb + 6]!, closest)
    if (e === Infinity) { node = nodes[nb + 3]! | 0; continue }
    const leaf = nodes[nb + 7]!
    if (leaf >= 0) {
      leafVisits++
      const count = (leaf | 0) & 7, start = (leaf | 0) >> 3
      for (let p = start; p < start + count; p++) {
        triTests++
        const q = p * PRIM_STRIDE
        if (rayTriangle(ox, oy, oz, dx, dy, dz, prims[q]!, prims[q+1]!, prims[q+2]!, prims[q+4]!, prims[q+5]!, prims[q+6]!, prims[q+8]!, prims[q+9]!, prims[q+10]!, 1e-4, closest, scratch)) { closest = scratch.t; best = p }
      }
    }
    node++
  }
  if (best >= 0) hits++
}
const dt = performance.now() - t0
console.log(`${M} rays in ${dt.toFixed(0)} ms: ${(visits/M).toFixed(1)} visits/ray, ${(leafVisits/M).toFixed(1)} leaves/ray, ${(triTests/M).toFixed(1)} tri tests/ray, ${(hits/M*100).toFixed(1)}% hit, ${(dt*1000/M).toFixed(1)} us/ray`)
