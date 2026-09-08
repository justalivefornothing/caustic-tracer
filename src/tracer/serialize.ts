/**
 * Scene <-> JSON with defensive validation, so a dropped file can never put
 * NaNs into a texture.
 */
import type { CameraSettings, Color, Material, Scene, SceneObject, Shape, Sky } from './scene.ts'
import type { Vec3 } from './vec3.ts'

export const SCENE_FORMAT = 'caustic-scene'
export const SCENE_FORMAT_VERSION = 1

export class SceneParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SceneParseError'
  }
}

const fail = (msg: string): never => {
  throw new SceneParseError(msg)
}

type Json = Record<string, unknown>
const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

function num(v: unknown, path: string, min = -Infinity, max = Infinity): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) fail(`${path} must be a finite number`)
  const n = v as number
  if (n < min || n > max) fail(`${path} must be within [${min}, ${max}]`)
  return n
}

function str(v: unknown, path: string): string {
  if (typeof v !== 'string') fail(`${path} must be a string`)
  return v as string
}

function vec(v: unknown, path: string): Vec3 {
  if (!isObj(v)) fail(`${path} must be an object {x, y, z}`)
  const o = v as Json
  return { x: num(o.x, `${path}.x`), y: num(o.y, `${path}.y`), z: num(o.z, `${path}.z`) }
}

function color(v: unknown, path: string, max = 1e4): Color {
  if (!isObj(v)) fail(`${path} must be an object {r, g, b}`)
  const o = v as Json
  return { r: num(o.r, `${path}.r`, 0, max), g: num(o.g, `${path}.g`, 0, max), b: num(o.b, `${path}.b`, 0, max) }
}

function material(v: unknown, path: string): Material {
  if (!isObj(v)) fail(`${path} must be an object`)
  const o = v as Json
  switch (o.type) {
    case 'lambertian': {
      const m: { type: 'lambertian'; albedo: Color; checker?: { color2: Color; scale: number }; emission?: Color } = {
        type: 'lambertian',
        albedo: color(o.albedo, `${path}.albedo`, 1),
      }
      if (o.checker !== undefined) {
        if (!isObj(o.checker)) fail(`${path}.checker must be an object`)
        const c = o.checker as Json
        m.checker = { color2: color(c.color2, `${path}.checker.color2`, 1), scale: num(c.scale, `${path}.checker.scale`, 1e-3, 1e4) }
      }
      if (o.emission !== undefined) m.emission = color(o.emission, `${path}.emission`)
      return m
    }
    case 'metal':
      return { type: 'metal', albedo: color(o.albedo, `${path}.albedo`, 1), roughness: num(o.roughness, `${path}.roughness`, 0, 1) }
    case 'dielectric': {
      const ior = num(o.ior, `${path}.ior`, 1, 4)
      return o.tint !== undefined ? { type: 'dielectric', ior, tint: color(o.tint, `${path}.tint`, 1) } : { type: 'dielectric', ior }
    }
    case 'emissive':
      return { type: 'emissive', color: color(o.color, `${path}.color`, 1), strength: num(o.strength, `${path}.strength`, 0, 1e5) }
    default:
      return fail(`${path}.type must be lambertian | metal | dielectric | emissive`)
  }
}

function shape(v: unknown, path: string): Shape {
  if (!isObj(v)) fail(`${path} must be an object`)
  const o = v as Json
  switch (o.kind) {
    case 'sphere':
      return { kind: 'sphere', center: vec(o.center, `${path}.center`), radius: num(o.radius, `${path}.radius`, 1e-6, 1e6) }
    case 'triangle':
      return { kind: 'triangle', a: vec(o.a, `${path}.a`), b: vec(o.b, `${path}.b`), c: vec(o.c, `${path}.c`) }
    case 'quad':
      return { kind: 'quad', origin: vec(o.origin, `${path}.origin`), u: vec(o.u, `${path}.u`), v: vec(o.v, `${path}.v`) }
    case 'mesh': {
      if (!Array.isArray(o.vertices) || !Array.isArray(o.indices)) fail(`${path} mesh needs vertices[] and indices[]`)
      const verts = (o.vertices as unknown[]).map((p, i) => vec(p, `${path}.vertices[${i}]`))
      const idx = (o.indices as unknown[]).map((n, i) => num(n, `${path}.indices[${i}]`, 0, verts.length - 1))
      if (idx.length % 3 !== 0) fail(`${path}.indices length must be a multiple of 3`)
      if (idx.some((n) => !Number.isInteger(n))) fail(`${path}.indices must be integers`)
      return { kind: 'mesh', vertices: verts, indices: idx }
    }
    default:
      return fail(`${path}.kind must be sphere | triangle | quad | mesh`)
  }
}

function camera(v: unknown, path: string): CameraSettings {
  if (!isObj(v)) fail(`${path} must be an object`)
  const o = v as Json
  return {
    target: vec(o.target, `${path}.target`),
    yaw: num(o.yaw, `${path}.yaw`),
    pitch: num(o.pitch, `${path}.pitch`, -Math.PI / 2, Math.PI / 2),
    distance: num(o.distance, `${path}.distance`, 1e-3, 1e6),
    fovDeg: num(o.fovDeg, `${path}.fovDeg`, 1, 170),
    lensRadius: num(o.lensRadius, `${path}.lensRadius`, 0, 1e3),
    focusDistance: num(o.focusDistance, `${path}.focusDistance`, 1e-3, 1e6),
  }
}

function sky(v: unknown, path: string): Sky {
  if (v === undefined) return { kind: 'none' }
  if (!isObj(v)) fail(`${path} must be an object`)
  const o = v as Json
  if (o.kind === 'none') return { kind: 'none' }
  if (o.kind === 'gradient') {
    return {
      kind: 'gradient',
      zenith: color(o.zenith, `${path}.zenith`),
      horizon: color(o.horizon, `${path}.horizon`),
      strength: num(o.strength, `${path}.strength`, 0, 1e3),
    }
  }
  return fail(`${path}.kind must be none | gradient`)
}

export function sceneToJSON(scene: Scene, pretty = true): string {
  const doc = {
    format: SCENE_FORMAT,
    version: SCENE_FORMAT_VERSION,
    name: scene.name,
    camera: scene.camera,
    sky: scene.sky,
    objects: scene.objects,
  }
  return JSON.stringify(doc, null, pretty ? 2 : 0)
}

export const MAX_OBJECTS = 20000

export function parseScene(input: unknown): Scene {
  if (!isObj(input)) fail('scene must be a JSON object')
  const o = input as Json
  if (o.format !== undefined && o.format !== SCENE_FORMAT) fail(`unknown format "${String(o.format)}"`)
  const name = o.name === undefined ? 'Imported scene' : str(o.name, 'name')
  const rawObjects = o.objects
  if (!Array.isArray(rawObjects)) return fail('objects must be an array')
  if (rawObjects.length > MAX_OBJECTS) fail(`too many objects (max ${MAX_OBJECTS})`)
  const seen = new Set<string>()
  const objects: SceneObject[] = (rawObjects as unknown[]).map((raw, i) => {
    if (!isObj(raw)) fail(`objects[${i}] must be an object`)
    const r = raw as Json
    let id = r.id === undefined ? `obj-${i}` : str(r.id, `objects[${i}].id`)
    while (seen.has(id)) id = `${id}-dup`
    seen.add(id)
    return {
      id,
      name: r.name === undefined ? `Object ${i + 1}` : str(r.name, `objects[${i}].name`),
      shape: shape(r.shape, `objects[${i}].shape`),
      material: material(r.material, `objects[${i}].material`),
    }
  })
  return { name, objects, camera: camera(o.camera, 'camera'), sky: sky(o.sky, 'sky') }
}

export function parseSceneJSON(text: string): Scene {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch (e) {
    throw new SceneParseError(`invalid JSON: ${(e as Error).message}`)
  }
  return parseScene(data)
}
