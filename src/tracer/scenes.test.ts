import { describe, expect, it } from 'vitest'
import { validateBvh } from './bvh.ts'
import { cameraFrame } from './camera.ts'
import { MAT_STRIDE, packScene } from './pack.ts'
import { pick } from './picking.ts'
import { materialEmission, primitiveCount } from './scene.ts'
import { CORNELL_BOX, DEPTH_OF_FIELD, GLASS_SPHERES, MIRROR_ROOM, NIGHT_EMISSIVES, SCENE_PRESETS } from './scenes.ts'
import { parseSceneJSON, sceneToJSON } from './serialize.ts'

describe('Scene presets', () => {
  it('defines exactly five presets with unique ids', () => {
    expect(SCENE_PRESETS.length).toBe(5)
    expect(new Set(SCENE_PRESETS.map((p) => p.id)).size).toBe(5)
    expect(SCENE_PRESETS.map((p) => p.scene)).toEqual([CORNELL_BOX, GLASS_SPHERES, NIGHT_EMISSIVES, DEPTH_OF_FIELD, MIRROR_ROOM])
  })

  for (const preset of SCENE_PRESETS) {
    it(`${preset.label} builds a valid BVH`, () => {
      const ps = packScene(preset.scene)
      expect(ps.primCount).toBeGreaterThan(0)
      expect(ps.nodeCount).toBeGreaterThan(0)
      const report = validateBvh(ps.bvh, ps.primCount)
      expect(report.errors).toEqual([])
      expect(report.maxLeafSize).toBeLessThanOrEqual(4)
      expect(ps.matCount).toBe(preset.scene.objects.length)
      expect(ps.mats.length).toBe(preset.scene.objects.length * MAT_STRIDE)
      // every float uploaded to the GPU must be finite
      for (const arr of [ps.prims, ps.nodes, ps.mats]) {
        for (let i = 0; i < arr.length; i++) expect(Number.isFinite(arr[i])).toBe(true)
      }
      // ids unique
      expect(new Set(preset.scene.objects.map((o) => o.id)).size).toBe(preset.scene.objects.length)
    })

    it(`${preset.label} survives a JSON round trip`, () => {
      const json = sceneToJSON(preset.scene)
      const back = parseSceneJSON(json)
      expect(back.objects.length).toBe(preset.scene.objects.length)
      expect(back.camera).toEqual(preset.scene.camera)
      expect(packScene(back).primCount).toBe(packScene(preset.scene).primCount)
    })
  }

  it('Cornell Box contains exactly one emissive primitive (the ceiling light)', () => {
    const emitters = CORNELL_BOX.objects.filter((o) => materialEmission(o.material) !== null)
    expect(emitters.length).toBe(1)
    expect(emitters[0]!.material.type).toBe('emissive')
    const ps = packScene(CORNELL_BOX)
    const lightObjects = new Set(Array.from(ps.lights, (p) => ps.primObject[p]!))
    expect(lightObjects.size).toBe(1)
    expect(CORNELL_BOX.objects[[...lightObjects][0]!]!.name).toBe('Ceiling light')
  })

  it('Cornell Box has the classic red/green walls, a metal and a glass sphere', () => {
    const red = CORNELL_BOX.objects.find((o) => o.name === 'Red wall')!
    const green = CORNELL_BOX.objects.find((o) => o.name === 'Green wall')!
    expect(red.material.type).toBe('lambertian')
    expect(green.material.type).toBe('lambertian')
    if (red.material.type === 'lambertian') expect(red.material.albedo.r).toBeGreaterThan(red.material.albedo.g)
    if (green.material.type === 'lambertian') expect(green.material.albedo.g).toBeGreaterThan(green.material.albedo.r)
    expect(CORNELL_BOX.objects.some((o) => o.material.type === 'metal' && o.shape.kind === 'sphere')).toBe(true)
    expect(CORNELL_BOX.objects.some((o) => o.material.type === 'dielectric' && o.shape.kind === 'sphere')).toBe(true)
  })

  it('Glass Spheres holds roughly sixty small spheres on a checkerboard', () => {
    const smalls = GLASS_SPHERES.objects.filter((o) => o.shape.kind === 'sphere' && o.shape.radius < 0.5)
    expect(smalls.length).toBeGreaterThanOrEqual(50)
    expect(smalls.length).toBeLessThanOrEqual(60)
    const ground = GLASS_SPHERES.objects[0]!
    expect(ground.material.type === 'lambertian' && ground.material.checker !== undefined).toBe(true)
    expect(GLASS_SPHERES.sky.kind).toBe('gradient')
  })

  it('Night Emissives has no sky and several emitters', () => {
    expect(NIGHT_EMISSIVES.sky.kind).toBe('none')
    expect(packScene(NIGHT_EMISSIVES).lights.length).toBeGreaterThanOrEqual(5)
  })

  it('Depth of Field has a wide aperture and a receding row of spheres', () => {
    expect(DEPTH_OF_FIELD.camera.lensRadius).toBeGreaterThan(0.05)
    const spheres = DEPTH_OF_FIELD.objects.filter((o) => o.shape.kind === 'sphere')
    expect(spheres.length).toBeGreaterThanOrEqual(8)
  })

  it('Mirror Room has a ~2000 triangle mesh and metal walls', () => {
    const mesh = MIRROR_ROOM.objects.find((o) => o.shape.kind === 'mesh')!
    expect(mesh).toBeDefined()
    const tris = primitiveCount(mesh.shape)
    expect(tris).toBeGreaterThanOrEqual(1800)
    expect(tris).toBeLessThanOrEqual(2200)
    expect(MIRROR_ROOM.objects.filter((o) => o.material.type === 'metal').length).toBeGreaterThanOrEqual(4)
  })

  it('picks the glass sphere in the Cornell Box from the camera', () => {
    const ps = packScene(CORNELL_BOX)
    const frame = cameraFrame(CORNELL_BOX.camera, 1.5)
    // The glass sphere sits lower right of centre.
    const hit = pick(ps, frame, 0.74, 0.15)
    expect(hit).not.toBeNull()
    expect(CORNELL_BOX.objects[hit!.objectIndex]!.name).toBe('Glass sphere')
    expect(hit!.t).toBeGreaterThan(1)
    const miss = pick(packScene({ ...CORNELL_BOX, objects: [] }), frame, 0.5, 0.5)
    expect(miss).toBeNull()
  })
})
