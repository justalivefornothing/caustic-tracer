import { NODE_STRIDE } from '../src/tracer/bvh.ts'
import { rayAabb } from '../src/tracer/intersect.ts'
import { packScene } from '../src/tracer/pack.ts'
import { Rng } from '../src/tracer/rng.ts'
import type { Scene, SceneObject } from '../src/tracer/scene.ts'
import { v3 } from '../src/tracer/vec3.ts'

const rng = new Rng(2024)
const objects: SceneObject[] = []
const N = Number(process.argv[2] ?? 20000)
for (let i = 0; i < N; i++) {
  const a = v3(rng.range(-50, 50), rng.range(-50, 50), rng.range(-50, 50))
  objects.push({
    id: `t${i}`, name: 't',
    shape: { kind: 'triangle', a, b: v3(a.x + rng.range(-0.6, 0.6), a.y + rng.range(-0.6, 0.6), a.z + rng.range(-0.6, 0.6)), c: v3(a.x + rng.range(-0.6, 0.6), a.y + rng.range(-0.6, 0.6), a.z + rng.range(-0.6, 0.6)) },
    material: { type: 'lambertian', albedo: { r: 0.5, g: 0.5, b: 0.5 } },
  })
}
const scene: Scene = { name: 'b', objects, camera: { target: v3(0, 0, 0), yaw: 0, pitch: 0, distance: 1, fovDeg: 40, lensRadius: 0, focusDistance: 1 }, sky: { kind: 'none' } }
const ps = packScene(scene)
console.log('nodes', ps.nodeCount)
// root box
console.log('root', Array.from(ps.nodes.slice(0, 8)))
// sample a few nodes
for (const i of [1, 2, 3, 100, 1000]) console.log(i, Array.from(ps.nodes.slice(i * 8, i * 8 + 8)).map((x) => +x.toFixed(2)))

// count visits for some rays
let totalVisits = 0, totalLeaf = 0
const R = 200
for (let r = 0; r < R; r++) {
  const ox = rng.range(-70, 70), oy = rng.range(-70, 70), oz = rng.range(-70, 70)
  const tx = rng.range(-40, 40), ty = rng.range(-40, 40), tz = rng.range(-40, 40)
  let dx = tx - ox, dy = ty - oy, dz = tz - oz
  const len = Math.hypot(dx, dy, dz); dx /= len; dy /= len; dz /= len
  const idx = 1 / dx, idy = 1 / dy, idz = 1 / dz
  let node = 0, visits = 0, leaves = 0
  while (node < ps.nodeCount) {
    visits++
    const nb = node * NODE_STRIDE
    const e = rayAabb(ox, oy, oz, idx, idy, idz, ps.nodes[nb]!, ps.nodes[nb + 1]!, ps.nodes[nb + 2]!, ps.nodes[nb + 4]!, ps.nodes[nb + 5]!, ps.nodes[nb + 6]!, Infinity)
    if (e === Infinity) { node = ps.nodes[nb + 3]!; continue }
    if (ps.nodes[nb + 7]! >= 0) leaves++
    node++
  }
  totalVisits += visits; totalLeaf += leaves
}
console.log('avg visits/ray', totalVisits / R, 'avg leaves/ray', totalLeaf / R)

import { intersectClosest, makeHit } from '../src/tracer/intersect.ts'
const hit = makeHit()
const M = 100000
const t0 = performance.now()
let hits = 0
for (let r = 0; r < M; r++) {
  const ox = rng.range(-70, 70), oy = rng.range(-70, 70), oz = rng.range(-70, 70)
  const tx = rng.range(-40, 40), ty = rng.range(-40, 40), tz = rng.range(-40, 40)
  let dx = tx - ox, dy = ty - oy, dz = tz - oz
  const len = Math.hypot(dx, dy, dz); dx /= len; dy /= len; dz /= len
  if (intersectClosest(ps, ox, oy, oz, dx, dy, dz, 1e-4, Infinity, hit)) hits++
}
console.log('intersectClosest', M, 'rays', (performance.now() - t0).toFixed(0), 'ms', hits, 'hits')
