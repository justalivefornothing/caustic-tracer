import { describe, expect, it } from 'vitest'
import { decodePPM, encodePPM, ppmPixel } from './ppm.ts'
import { linearPixel, renderCornellReference } from './reference.ts'
import { acesFitted, linearToSrgb, reinhard, tonemapBuffer } from './tonemap.ts'

describe('CPU Cornell Box render (96x64, 16 spp)', () => {
  const img = renderCornellReference(96, 64, 16)

  it('produces finite, non-black output', () => {
    let sum = 0
    for (let i = 0; i < img.linear.length; i++) {
      expect(Number.isFinite(img.linear[i])).toBe(true)
      sum += img.linear[i]!
    }
    expect(sum).toBeGreaterThan(0)
  })

  it('center pixel is brighter than the bottom-left corner pixel', () => {
    const c = linearPixel(img, 48, 32)
    const corner = linearPixel(img, 0, 63)
    const lum = (p: { r: number; g: number; b: number }) => 0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b
    expect(lum(c)).toBeGreaterThan(lum(corner))
  })

  it('red wall region has R > G', () => {
    // Average a patch on the left wall, mid-height (the back wall starts ~8% in).
    let r = 0, g = 0
    for (let y = 24; y < 40; y++) {
      for (let x = 1; x < 7; x++) {
        const p = linearPixel(img, x, y)
        r += p.r
        g += p.g
      }
    }
    expect(r).toBeGreaterThan(g * 1.5)
  })

  it('green wall region has G > R', () => {
    let r = 0, g = 0
    for (let y = 24; y < 40; y++) {
      for (let x = 89; x < 95; x++) {
        const p = linearPixel(img, x, y)
        r += p.r
        g += p.g
      }
    }
    expect(g).toBeGreaterThan(r * 1.5)
  })

  it('round-trips through the PPM encoder/decoder', () => {
    const bytes = encodePPM(img.width, img.height, img.rgba)
    const back = decodePPM(bytes)
    expect(back.width).toBe(96)
    expect(back.height).toBe(64)
    const px = ppmPixel(back, 48, 32)
    expect(px.r).toBe(img.rgba[(32 * 96 + 48) * 4])
    expect(px.g).toBe(img.rgba[(32 * 96 + 48) * 4 + 1])
  })
})

describe('Tone mapping', () => {
  it('ACES and Reinhard are monotonic and bounded to [0, 1]', () => {
    let prevA = -1, prevR = -1
    for (let x = 0; x <= 20; x += 0.1) {
      const a = acesFitted(x)
      const r = reinhard(x)
      expect(a).toBeGreaterThanOrEqual(prevA - 1e-9)
      expect(r).toBeGreaterThanOrEqual(prevR - 1e-9)
      expect(a).toBeLessThanOrEqual(1)
      expect(r).toBeLessThan(1)
      prevA = a
      prevR = r
    }
    expect(acesFitted(0)).toBe(0)
    expect(reinhard(0)).toBe(0)
    expect(reinhard(1)).toBeCloseTo(0.5, 12)
  })

  it('sRGB encoding matches known points', () => {
    expect(linearToSrgb(0)).toBe(0)
    expect(linearToSrgb(1)).toBe(1)
    expect(linearToSrgb(0.5)).toBeCloseTo(0.7354, 3)
  })

  it('tonemapBuffer divides by sample count and applies exposure', () => {
    const accum = new Float32Array([2, 2, 2])
    const a = tonemapBuffer(accum, 1, 1, 4, 0, 'reinhard') // 0.5 linear -> reinhard 1/3
    const expected = Math.round(linearToSrgb(reinhard(0.5)) * 255)
    expect(a[0]).toBe(expected)
    expect(a[3]).toBe(255)
    const b = tonemapBuffer(accum, 1, 1, 4, 1, 'reinhard') // +1 EV -> 1.0 linear
    expect(b[0]).toBe(Math.round(linearToSrgb(reinhard(1)) * 255))
  })
})
