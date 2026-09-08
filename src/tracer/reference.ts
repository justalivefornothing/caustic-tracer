/**
 * Small end-to-end CPU renders shared by scripts and tests.
 */
import { cameraFrame } from './camera.ts'
import { packScene } from './pack.ts'
import { CORNELL_BOX } from './scenes.ts'
import { tonemapBuffer, type ToneMapMode } from './tonemap.ts'
import { renderImage, type TraceOptions } from './tracer.ts'

export interface ReferenceRender {
  readonly width: number
  readonly height: number
  /** Linear RGB, 3 floats per pixel, averaged over samples. */
  readonly linear: Float32Array
  /** 8-bit sRGB RGBA after tone mapping. */
  readonly rgba: Uint8ClampedArray
  readonly millis: number
}

export function renderCornellReference(
  width = 96,
  height = 64,
  spp = 16,
  opts: Partial<TraceOptions> = {},
  toneMap: ToneMapMode = 'aces',
  exposureEv = 0,
): ReferenceRender {
  const t0 = performance.now()
  const ps = packScene(CORNELL_BOX)
  const frame = cameraFrame(CORNELL_BOX.camera, width / height)
  const linear = renderImage(ps, frame, {
    width,
    height,
    spp,
    maxBounces: opts.maxBounces ?? 8,
    russianRoulette: opts.russianRoulette ?? true,
    nee: opts.nee ?? true,
    seed: 1,
  })
  // tonemapBuffer expects an accumulated sum; pass samples = 1 for an averaged buffer.
  const rgba = tonemapBuffer(linear, width, height, 1, exposureEv, toneMap)
  return { width, height, linear, rgba, millis: performance.now() - t0 }
}

export function linearPixel(img: ReferenceRender, x: number, y: number): { r: number; g: number; b: number } {
  const o = (y * img.width + x) * 3
  return { r: img.linear[o]!, g: img.linear[o + 1]!, b: img.linear[o + 2]! }
}
