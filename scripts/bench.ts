/**
 * Benchmark: SAH BVH build over 100,000 random triangles, then 1,000,000
 * random rays traced through it with the stackless skip-link traversal.
 *
 *   npm run bench
 *
 * Scene: 100k triangles with ~1.2 unit edges scattered uniformly in a 60-unit
 * cube. Rays start on a shell around the cloud and aim at random interior
 * points, so most of them hit something and the closest-hit bound prunes the
 * traversal the way camera rays do in practice.
 *
 * Each phase runs three trials and reports every trial plus the best; the
 * acceptance budgets (build < 1500 ms, trace < 4000 ms) apply to the best.
 */
import { performance } from 'node:perf_hooks'
import { validateBvh } from '../src/tracer/bvh.ts'
import { intersectClosest, makeHit } from '../src/tracer/intersect.ts'
import { packScene, type PackedScene } from '../src/tracer/pack.ts'
import { Rng } from '../src/tracer/rng.ts'
import type { Scene, SceneObject } from '../src/tracer/scene.ts'
import { v3 } from '../src/tracer/vec3.ts'

const TRIANGLES = Number(process.env.BENCH_TRIS ?? 100_000)
const RAYS = Number(process.env.BENCH_RAYS ?? 1_000_000)
const TRIALS = Number(process.env.BENCH_TRIALS ?? 3)
const BUILD_BUDGET_MS = 1500
const TRACE_BUDGET_MS = 4000

function randomTriangleScene(count: number, seed: number): Scene {
  const rng = new Rng(seed)
  const objects: SceneObject[] = []
  const extent = 30
  const size = 0.6
  for (let i = 0; i < count; i++) {
    const a = v3(rng.range(-extent, extent), rng.range(-extent, extent), rng.range(-extent, extent))
    objects.push({
      id: `t${i}`,
      name: `t${i}`,
      shape: {
        kind: 'triangle',
        a,
        b: v3(a.x + rng.range(-size, size), a.y + rng.range(-size, size), a.z + rng.range(-size, size)),
        c: v3(a.x + rng.range(-size, size), a.y + rng.range(-size, size), a.z + rng.range(-size, size)),
      },
      material: { type: 'lambertian', albedo: { r: 0.5, g: 0.5, b: 0.5 } },
    })
  }
  return {
    name: 'bench',
    objects,
    camera: { target: v3(0, 0, 0), yaw: 0, pitch: 0, distance: 120, fovDeg: 40, lensRadius: 0, focusDistance: 120 },
    sky: { kind: 'none' },
  }
}

const fmt = (n: number) => n.toLocaleString('en-US')
const ms = (n: number) => `${n.toFixed(1)} ms`

console.log(`Caustic bench - Node ${process.version}, ${TRIALS} trials per phase`)
console.log(`  generating ${fmt(TRIANGLES)} random triangles...`)
const scene = randomTriangleScene(TRIANGLES, 2024)

// Warm the JIT on a small build so the timed trials reflect steady state.
packScene({ ...scene, objects: scene.objects.slice(0, 5000) })

let ps: PackedScene | null = null
const buildTimes: number[] = []
for (let trial = 0; trial < TRIALS; trial++) {
  const t0 = performance.now()
  ps = packScene(scene)
  buildTimes.push(performance.now() - t0)
}
const packed = ps!
const report = validateBvh(packed.bvh, packed.primCount)
const bestBuild = Math.min(...buildTimes)
console.log(`  BVH build (SAH, 12 bins, leaf <= 4): ${buildTimes.map(ms).join(' | ')}  best ${ms(bestBuild)}`)
console.log(
  `    ${fmt(packed.nodeCount)} nodes, ${fmt(report.leafCount)} leaves, depth ${report.depth}, max leaf ${report.maxLeafSize}, layout valid = ${report.ok}`,
)

// Rays: origins on a shell around the cloud aimed at random interior points.
const rng = new Rng(77)
const rays = new Float64Array(RAYS * 6)
for (let i = 0; i < RAYS; i++) {
  const ox = rng.range(-45, 45), oy = rng.range(-45, 45), oz = rng.range(-45, 45)
  const tx = rng.range(-25, 25), ty = rng.range(-25, 25), tz = rng.range(-25, 25)
  let dx = tx - ox, dy = ty - oy, dz = tz - oz
  const len = Math.hypot(dx, dy, dz) || 1
  dx /= len
  dy /= len
  dz /= len
  const o = i * 6
  rays[o] = ox; rays[o + 1] = oy; rays[o + 2] = oz
  rays[o + 3] = dx; rays[o + 4] = dy; rays[o + 5] = dz
}

const hit = makeHit()
function traceAll(): { hits: number; checksum: number } {
  let hits = 0
  let checksum = 0
  for (let i = 0; i < RAYS; i++) {
    const o = i * 6
    if (intersectClosest(packed, rays[o]!, rays[o + 1]!, rays[o + 2]!, rays[o + 3]!, rays[o + 4]!, rays[o + 5]!, 1e-4, Infinity, hit)) {
      hits++
      checksum += hit.t
    }
  }
  return { hits, checksum }
}

// warm-up on a slice
for (let i = 0; i < 20_000; i++) {
  const o = i * 6
  intersectClosest(packed, rays[o]!, rays[o + 1]!, rays[o + 2]!, rays[o + 3]!, rays[o + 4]!, rays[o + 5]!, 1e-4, Infinity, hit)
}

const traceTimes: number[] = []
let result = { hits: 0, checksum: 0 }
for (let trial = 0; trial < TRIALS; trial++) {
  const t1 = performance.now()
  result = traceAll()
  traceTimes.push(performance.now() - t1)
}
const bestTrace = Math.min(...traceTimes)
console.log(`  trace ${fmt(RAYS)} rays: ${traceTimes.map(ms).join(' | ')}  best ${ms(bestTrace)}`)
console.log(
  `    ${fmt(Math.round(RAYS / (bestTrace / 1000)))} rays/s, ${((result.hits / RAYS) * 100).toFixed(1)}% hit, checksum ${result.checksum.toFixed(2)}`,
)

const buildOk = bestBuild < BUILD_BUDGET_MS
const traceOk = bestTrace < TRACE_BUDGET_MS
console.log(`  build ${buildOk ? 'PASS' : 'FAIL'} (< ${BUILD_BUDGET_MS} ms), trace ${traceOk ? 'PASS' : 'FAIL'} (< ${TRACE_BUDGET_MS} ms)`)
if (!report.ok) {
  console.error('BVH validation failed:', report.errors.slice(0, 5))
  process.exit(1)
}
process.exit(buildOk && traceOk ? 0 : 1)
