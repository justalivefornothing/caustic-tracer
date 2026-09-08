import { describe, expect, it } from 'vitest'
import {
  cosineHemispherePdf,
  dielectricReflectance,
  powerHeuristic,
  randomInUnitDisk,
  randomUnitVector,
  sampleCosineHemisphere,
  sampleDielectric,
  sampleMetal,
  schlick,
} from './materials.ts'
import { Rng } from './rng.ts'
import { dot, length, normalize, refract, v3 } from './vec3.ts'

describe('Lambertian cosine-weighted sampling', () => {
  const rng = new Rng(42)
  const n = normalize(v3(0.3, 0.8, -0.5))
  const N = 100_000

  it('has mean cos(theta) within 2% of 2/3 over 100k samples', () => {
    let sum = 0
    for (let i = 0; i < N; i++) {
      const d = sampleCosineHemisphere(n, rng.next(), rng.next())
      expect(Math.abs(length(d) - 1)).toBeLessThan(1e-6)
      const c = dot(n, d)
      expect(c).toBeGreaterThanOrEqual(0)
      sum += c
    }
    const mean = sum / N
    expect(Math.abs(mean - 2 / 3) / (2 / 3)).toBeLessThan(0.02)
  })

  it('integrates its pdf to 1 +/- 0.02 over the hemisphere (uniform MC estimate)', () => {
    // Uniform hemisphere sampling: integral f = E[f(w)] * 2*pi
    let sum = 0
    let count = 0
    while (count < N) {
      const d = randomUnitVector(rng)
      if (dot(d, n) < 0) continue
      sum += cosineHemispherePdf(n, d)
      count++
    }
    const integral = (sum / N) * 2 * Math.PI
    expect(Math.abs(integral - 1)).toBeLessThan(0.02)
  })

  it('pdf matches the sampling density (histogram over cos theta)', () => {
    // Density in cos(theta) for pdf cos/pi over solid angle is 2*cos(theta).
    const bins = 10
    const hist = new Array<number>(bins).fill(0)
    for (let i = 0; i < N; i++) {
      const d = sampleCosineHemisphere(n, rng.next(), rng.next())
      const c = Math.min(0.999999, dot(n, d))
      hist[Math.floor(c * bins)]! += 1
    }
    for (let b = 0; b < bins; b++) {
      const lo = b / bins
      const hi = (b + 1) / bins
      const expected = (hi * hi - lo * lo) * N // integral of 2c dc
      expect(Math.abs(hist[b]! - expected) / expected).toBeLessThan(0.08)
    }
  })
})

describe('Dielectric', () => {
  it('refracts IOR 1.5 at 30 degrees incidence to within 1e-4 of Snell', () => {
    const theta = (30 * Math.PI) / 180
    const n = v3(0, 1, 0)
    const d = normalize(v3(Math.sin(theta), -Math.cos(theta), 0))
    const t = refract(d, n, 1 / 1.5)
    expect(t).not.toBeNull()
    const cosT = -dot(t!, n)
    const transmitted = Math.acos(Math.min(1, cosT))
    const snell = Math.asin(Math.sin(theta) / 1.5)
    expect(Math.abs(transmitted - snell)).toBeLessThan(1e-4)
    expect(Math.abs(length(t!) - 1)).toBeLessThan(1e-6)
    // stays in the incidence plane
    expect(Math.abs(t!.z)).toBeLessThan(1e-9)
  })

  it('Schlick at normal incidence equals ((1.5-1)/(1.5+1))^2 within 1e-6', () => {
    const expected = ((1.5 - 1) / (1.5 + 1)) ** 2
    expect(Math.abs(schlick(1, 1, 1.5) - expected)).toBeLessThan(1e-6)
    expect(Math.abs(dielectricReflectance(1, 1, 1.5) - expected)).toBeLessThan(1e-6)
    // grazing incidence approaches full reflection
    expect(schlick(0, 1, 1.5)).toBeCloseTo(1, 6)
  })

  it('reports total internal reflection beyond the critical angle', () => {
    const critical = Math.asin(1 / 1.5)
    const theta = critical + 0.05
    const d = normalize(v3(Math.sin(theta), -Math.cos(theta), 0))
    expect(refract(d, v3(0, 1, 0), 1.5)).toBeNull()
    expect(dielectricReflectance(Math.cos(theta), 1.5, 1)).toBe(1)
    const s = sampleDielectric(d, v3(0, 1, 0), 1.5, false, 0.999)
    expect(s.reflected).toBe(true)
  })

  it('sampleDielectric splits between reflection and refraction by Fresnel', () => {
    const rng = new Rng(3)
    const d = normalize(v3(0.5, -1, 0.2))
    const n = v3(0, 1, 0)
    const cosI = -dot(d, n)
    const R = dielectricReflectance(cosI, 1, 1.5)
    let reflected = 0
    const N = 50_000
    for (let i = 0; i < N; i++) if (sampleDielectric(d, n, 1.5, true, rng.next()).reflected) reflected++
    expect(Math.abs(reflected / N - R)).toBeLessThan(0.01)
  })
})

describe('Metal', () => {
  it('with roughness 0 reflects exactly the mirror direction', () => {
    const rng = new Rng(1)
    const d = normalize(v3(0.4, -0.7, 0.2))
    const n = normalize(v3(0.1, 1, 0.05))
    const r = sampleMetal(d, n, 0, rng)
    const expected = {
      x: d.x - 2 * dot(d, n) * n.x,
      y: d.y - 2 * dot(d, n) * n.y,
      z: d.z - 2 * dot(d, n) * n.z,
    }
    expect(r.x).toBeCloseTo(expected.x, 12)
    expect(r.y).toBeCloseTo(expected.y, 12)
    expect(r.z).toBeCloseTo(expected.z, 12)
    // angle of incidence equals angle of reflection
    expect(dot(r, n)).toBeCloseTo(-dot(d, n), 12)
  })

  it('with roughness > 0 scatters around the mirror direction', () => {
    const rng = new Rng(5)
    const d = normalize(v3(0, -1, 0))
    const n = v3(0, 1, 0)
    let sum = 0
    for (let i = 0; i < 1000; i++) sum += dot(sampleMetal(d, n, 0.3, rng), v3(0, 1, 0))
    expect(sum / 1000).toBeGreaterThan(0.9)
  })
})

describe('Sampling helpers', () => {
  it('unit vectors and disk samples are uniform-ish and normalised', () => {
    const rng = new Rng(11)
    let up = 0
    for (let i = 0; i < 20_000; i++) {
      const v = randomUnitVector(rng)
      expect(Math.abs(length(v) - 1)).toBeLessThan(1e-9)
      if (v.z > 0) up++
      const p = randomInUnitDisk(rng)
      expect(p.x * p.x + p.y * p.y).toBeLessThanOrEqual(1 + 1e-12)
    }
    expect(Math.abs(up / 20_000 - 0.5)).toBeLessThan(0.02)
  })

  it('power heuristic weights sum to one across the two strategies', () => {
    for (const [a, b] of [[0.3, 0.9], [2, 0.1], [1, 1]]) {
      expect(powerHeuristic(a!, b!) + powerHeuristic(b!, a!)).toBeCloseTo(1, 12)
    }
    expect(powerHeuristic(0, 0)).toBe(0)
  })
})
