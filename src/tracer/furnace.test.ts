import { describe, expect, it } from 'vitest'
import { cameraFrame } from './camera.ts'
import { packScene } from './pack.ts'
import type { Scene } from './scene.ts'
import { renderImage } from './tracer.ts'
import { v3 } from './vec3.ts'

/**
 * Camera inside a closed diffuse sphere (albedo 0.8) whose surface also emits
 * radiance 1.0 everywhere. Every bounce adds E and scales by albedo, so the
 * radiance is the geometric series E / (1 - albedo) = 5.0.
 */
const FURNACE: Scene = {
  name: 'furnace',
  objects: [
    {
      id: 'shell',
      name: 'shell',
      shape: { kind: 'sphere', center: v3(0, 0, 0), radius: 5 },
      material: { type: 'lambertian', albedo: { r: 0.8, g: 0.8, b: 0.8 }, emission: { r: 1, g: 1, b: 1 } },
    },
  ],
  camera: { target: v3(0, 0, -1), yaw: 0, pitch: 0, distance: 1, fovDeg: 60, lensRadius: 0, focusDistance: 1 },
  sky: { kind: 'none' },
}

describe('White furnace', () => {
  it('naive path tracing: every pixel within 5% of 1/(1-0.8) = 5.0 at 4096 spp, 64 bounces, no roulette', () => {
    const ps = packScene(FURNACE)
    const frame = cameraFrame(FURNACE.camera, 1)
    const img = renderImage(ps, frame, {
      width: 8,
      height: 8,
      spp: 4096,
      maxBounces: 64,
      russianRoulette: false,
      nee: false,
      seed: 3,
    })
    for (let i = 0; i < img.length; i++) {
      expect(Math.abs(img[i]! - 5) / 5).toBeLessThan(0.05)
    }
  }, 180_000)

  it('next-event estimation + MIS converges to the same 5.0 (unbiased)', () => {
    const ps = packScene(FURNACE)
    expect(ps.lights.length).toBe(1)
    const frame = cameraFrame(FURNACE.camera, 1)
    const img = renderImage(ps, frame, {
      width: 4,
      height: 4,
      spp: 4096,
      maxBounces: 64,
      russianRoulette: false,
      nee: true,
      seed: 9,
    })
    let mean = 0
    for (let i = 0; i < img.length; i++) mean += img[i]!
    mean /= img.length
    expect(Math.abs(mean - 5) / 5).toBeLessThan(0.03)
    for (let i = 0; i < img.length; i++) expect(Math.abs(img[i]! - 5) / 5).toBeLessThan(0.08)
  }, 180_000)

  it('Russian roulette stays unbiased (mean within 3%)', () => {
    const ps = packScene(FURNACE)
    const frame = cameraFrame(FURNACE.camera, 1)
    const img = renderImage(ps, frame, {
      width: 4,
      height: 4,
      spp: 4096,
      maxBounces: 64,
      russianRoulette: true,
      nee: false,
      seed: 5,
    })
    let mean = 0
    for (let i = 0; i < img.length; i++) mean += img[i]!
    mean /= img.length
    expect(Math.abs(mean - 5) / 5).toBeLessThan(0.03)
  }, 180_000)
})
