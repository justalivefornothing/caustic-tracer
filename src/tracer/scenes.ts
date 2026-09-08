/**
 * The five preset scenes, defined as typed constants. Random layouts use a
 * seeded RNG so every build of the app shows the same arrangement.
 */
import { torusKnot } from './mesh.ts'
import { Rng } from './rng.ts'
import { rgb, type CameraSettings, type Color, type Material, type Scene, type SceneObject, type Shape } from './scene.ts'
import { v3, type Vec3 } from './vec3.ts'

let presetCounter = 0
function obj(name: string, shape: Shape, material: Material): SceneObject {
  presetCounter++
  return { id: `p${presetCounter.toString(36)}`, name, shape, material }
}

const sphere = (center: Vec3, radius: number): Shape => ({ kind: 'sphere', center, radius })
const quad = (origin: Vec3, u: Vec3, v: Vec3): Shape => ({ kind: 'quad', origin, u, v })

const lambert = (albedo: Color, checker?: { color2: Color; scale: number }): Material =>
  checker ? { type: 'lambertian', albedo, checker } : { type: 'lambertian', albedo }
const metal = (albedo: Color, roughness: number): Material => ({ type: 'metal', albedo, roughness })
const glass = (ior = 1.5, tint?: Color): Material => (tint ? { type: 'dielectric', ior, tint } : { type: 'dielectric', ior })
const light = (color: Color, strength: number): Material => ({ type: 'emissive', color, strength })

/** Orbit camera positioned at `eye` looking at `target`. */
function cameraAt(eye: Vec3, target: Vec3, fovDeg: number, lensRadius: number, focusDistance?: number): CameraSettings {
  const dx = eye.x - target.x
  const dy = eye.y - target.y
  const dz = eye.z - target.z
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
  return {
    target,
    yaw: Math.atan2(dx, dz),
    pitch: Math.asin(dy / distance),
    distance,
    fovDeg,
    lensRadius,
    focusDistance: focusDistance ?? distance,
  }
}

// ---------------------------------------------------------------------------
// 1. Cornell Box

function buildCornellBox(): Scene {
  const white = lambert(rgb(0.73, 0.73, 0.73))
  const red = lambert(rgb(0.65, 0.05, 0.05))
  const green = lambert(rgb(0.12, 0.45, 0.15))
  // A 3:2 box (1.5 wide, 1 tall, 1 deep) so the open front fills the mat.
  const W = 0.75
  const objects: SceneObject[] = [
    obj('Floor', quad(v3(-W, 0, -0.5), v3(2 * W, 0, 0), v3(0, 0, 1)), white),
    obj('Ceiling', quad(v3(-W, 1, -0.5), v3(2 * W, 0, 0), v3(0, 0, 1)), white),
    obj('Back wall', quad(v3(-W, 0, -0.5), v3(2 * W, 0, 0), v3(0, 1, 0)), white),
    obj('Red wall', quad(v3(-W, 0, -0.5), v3(0, 0, 1), v3(0, 1, 0)), red),
    obj('Green wall', quad(v3(W, 0, -0.5), v3(0, 0, 1), v3(0, 1, 0)), green),
    obj('Ceiling light', quad(v3(-0.2, 0.998, -0.15), v3(0.4, 0, 0), v3(0, 0, 0.3)), light(rgb(1, 0.86, 0.7), 12)),
    obj('Metal sphere', sphere(v3(-0.33, 0.24, -0.12), 0.24), metal(rgb(0.9, 0.9, 0.92), 0.02)),
    obj('Glass sphere', sphere(v3(0.3, 0.2, 0.17), 0.2), glass(1.5)),
  ]
  return {
    name: 'Cornell Box',
    objects,
    camera: cameraAt(v3(0, 0.5, 1.6), v3(0, 0.5, 0), 32, 0, 1.6),
    sky: { kind: 'none' },
  }
}

// ---------------------------------------------------------------------------
// 2. Glass Spheres on Checkerboard

function buildGlassSpheres(): Scene {
  const rng = new Rng(20240607)
  const objects: SceneObject[] = [
    obj(
      'Checker ground',
      quad(v3(-40, 0, -40), v3(80, 0, 0), v3(0, 0, 80)),
      lambert(rgb(0.82, 0.8, 0.76), { color2: rgb(0.18, 0.17, 0.16), scale: 1 }),
    ),
    obj('Big glass', sphere(v3(0, 1, 0), 1), glass(1.5)),
    obj('Big diffuse', sphere(v3(-4, 1, 0), 1), lambert(rgb(0.4, 0.2, 0.1))),
    obj('Big metal', sphere(v3(4, 1, 0), 1), metal(rgb(0.7, 0.6, 0.5), 0)),
  ]
  const bigs = [v3(0, 1, 0), v3(-4, 1, 0), v3(4, 1, 0)]
  let n = 0
  for (let a = -6; a < 6 && n < 60; a++) {
    for (let b = -4; b < 5 && n < 60; b++) {
      const center = v3(a * 1.15 + 0.7 * rng.next(), 0.2, b * 1.15 + 0.7 * rng.next())
      if (bigs.some((c) => Math.hypot(c.x - center.x, c.z - center.z) < 1.4)) continue
      const pick = rng.next()
      let material: Material
      if (pick < 0.62) {
        material = lambert(rgb(rng.next() * rng.next(), rng.next() * rng.next(), rng.next() * rng.next()))
      } else if (pick < 0.85) {
        material = metal(rgb(0.5 + 0.5 * rng.next(), 0.5 + 0.5 * rng.next(), 0.5 + 0.5 * rng.next()), 0.3 * rng.next())
      } else {
        material = glass(1.5)
      }
      objects.push(obj(`Sphere ${n + 1}`, sphere(center, 0.2), material))
      n++
    }
  }
  return {
    name: 'Glass Spheres on Checkerboard',
    objects,
    camera: cameraAt(v3(12, 2.2, 3.2), v3(0, 0.55, 0), 22, 0.06, 10.5),
    sky: { kind: 'gradient', zenith: rgb(0.5, 0.7, 1), horizon: rgb(1, 1, 1), strength: 1 },
  }
}

// ---------------------------------------------------------------------------
// 3. Night Emissives

function buildNightEmissives(): Scene {
  const glows: Array<{ pos: Vec3; r: number; color: Color; strength: number }> = [
    { pos: v3(-1.6, 0.35, -0.6), r: 0.35, color: rgb(1, 0.35, 0.1), strength: 6 },
    { pos: v3(0.2, 0.25, 0.9), r: 0.25, color: rgb(0.2, 0.55, 1), strength: 7 },
    { pos: v3(1.7, 0.45, -1.1), r: 0.45, color: rgb(0.6, 1, 0.3), strength: 4 },
    { pos: v3(-0.4, 0.15, -1.9), r: 0.15, color: rgb(1, 0.2, 0.6), strength: 10 },
    { pos: v3(1.1, 0.18, 1.6), r: 0.18, color: rgb(1, 0.85, 0.4), strength: 8 },
    { pos: v3(-2.6, 0.2, 1.4), r: 0.2, color: rgb(0.5, 0.3, 1), strength: 7 },
  ]
  const objects: SceneObject[] = [
    obj('Dark ground', quad(v3(-30, 0, -30), v3(60, 0, 0), v3(0, 0, 60)), lambert(rgb(0.09, 0.085, 0.08))),
    obj('Chrome sphere', sphere(v3(0.4, 0.55, -0.7), 0.55), metal(rgb(0.95, 0.95, 0.95), 0.0)),
    obj('Brushed sphere', sphere(v3(-1.0, 0.4, 1.1), 0.4), metal(rgb(0.9, 0.7, 0.5), 0.25)),
    obj('Glass sphere', sphere(v3(1.9, 0.4, 0.6), 0.4), glass(1.5)),
    obj('Matte sphere', sphere(v3(-2.2, 0.5, -1.5), 0.5), lambert(rgb(0.7, 0.7, 0.72))),
  ]
  glows.forEach((g, i) => objects.push(obj(`Glow ${i + 1}`, sphere(g.pos, g.r), light(g.color, g.strength))))
  return {
    name: 'Night Emissives',
    objects,
    camera: cameraAt(v3(4.2, 1.6, 4.6), v3(0, 0.35, 0), 30, 0.02, 6.2),
    sky: { kind: 'none' },
  }
}

// ---------------------------------------------------------------------------
// 4. Depth of Field Study

function buildDepthOfField(): Scene {
  const objects: SceneObject[] = [
    obj(
      'Checker ground',
      quad(v3(-30, 0, -30), v3(60, 0, 0), v3(0, 0, 60)),
      lambert(rgb(0.75, 0.74, 0.72), { color2: rgb(0.25, 0.24, 0.23), scale: 2 }),
    ),
  ]
  const palette: Material[] = [
    lambert(rgb(0.85, 0.3, 0.2)),
    metal(rgb(0.9, 0.85, 0.7), 0.05),
    glass(1.5),
    lambert(rgb(0.2, 0.45, 0.85)),
    metal(rgb(0.8, 0.8, 0.85), 0.2),
    lambert(rgb(0.95, 0.8, 0.25)),
    glass(1.5),
    lambert(rgb(0.3, 0.7, 0.4)),
    metal(rgb(0.95, 0.65, 0.5), 0),
    lambert(rgb(0.9, 0.9, 0.9)),
  ]
  for (let i = 0; i < 10; i++) {
    const z = -i * 1.1
    const x = i * 0.42 - 1.0
    objects.push(obj(`Sphere ${i + 1}`, sphere(v3(x, 0.45, z), 0.45), palette[i]!))
  }
  const eye = v3(-1.6, 1.25, 3.6)
  const focusOn = v3(-1.0 + 2 * 0.42, 0.45, -2.2)
  const focusDistance = Math.hypot(eye.x - focusOn.x, eye.y - focusOn.y, eye.z - focusOn.z)
  return {
    name: 'Depth of Field Study',
    objects,
    camera: cameraAt(eye, v3(0.2, 0.45, -1.8), 30, 0.12, focusDistance),
    sky: { kind: 'gradient', zenith: rgb(0.45, 0.6, 0.9), horizon: rgb(0.95, 0.9, 0.85), strength: 0.9 },
  }
}

// ---------------------------------------------------------------------------
// 5. Mirror Room

function buildMirrorRoom(): Scene {
  const wall = metal(rgb(0.88, 0.86, 0.82), 0.015)
  const knot = torusKnot({
    p: 2,
    q: 3,
    radius: 0.55,
    ringRadius: 0.22,
    tube: 0.11,
    tubularSegments: 100,
    radialSegments: 10,
    center: v3(0, 1.0, 0),
  })
  const objects: SceneObject[] = [
    obj('Floor', quad(v3(-2, 0, -2), v3(4, 0, 0), v3(0, 0, 4)), lambert(rgb(0.16, 0.15, 0.14))),
    obj('Ceiling', quad(v3(-2, 2.2, -2), v3(4, 0, 0), v3(0, 0, 4)), wall),
    obj('Back wall', quad(v3(-2, 0, -2), v3(4, 0, 0), v3(0, 2.2, 0)), wall),
    obj('Front wall', quad(v3(-2, 0, 2), v3(4, 0, 0), v3(0, 2.2, 0)), wall),
    obj('Left wall', quad(v3(-2, 0, -2), v3(0, 0, 4), v3(0, 2.2, 0)), wall),
    obj('Right wall', quad(v3(2, 0, -2), v3(0, 0, 4), v3(0, 2.2, 0)), wall),
    obj('Light strip A', quad(v3(-1.2, 2.19, -0.15), v3(2.4, 0, 0), v3(0, 0, 0.3)), light(rgb(1, 0.92, 0.8), 9)),
    obj('Light strip B', quad(v3(-0.15, 2.19, -1.6), v3(0.3, 0, 0), v3(0, 0, 3.2)), light(rgb(0.8, 0.9, 1), 5)),
    obj('Torus knot', { kind: 'mesh', vertices: knot.vertices, indices: knot.indices }, lambert(rgb(0.95, 0.42, 0.12))),
  ]
  return {
    name: 'Mirror Room',
    objects,
    camera: cameraAt(v3(1.15, 1.2, 1.75), v3(0, 0.95, 0), 40, 0.01, 2.4),
    sky: { kind: 'none' },
  }
}

// ---------------------------------------------------------------------------

export const CORNELL_BOX: Scene = buildCornellBox()
export const GLASS_SPHERES: Scene = buildGlassSpheres()
export const NIGHT_EMISSIVES: Scene = buildNightEmissives()
export const DEPTH_OF_FIELD: Scene = buildDepthOfField()
export const MIRROR_ROOM: Scene = buildMirrorRoom()

export interface ScenePreset {
  readonly id: string
  readonly label: string
  readonly blurb: string
  readonly scene: Scene
}

export const SCENE_PRESETS: readonly ScenePreset[] = [
  { id: 'cornell', label: 'Cornell Box', blurb: 'Colour bleeding, soft shadows, a chrome and a glass sphere under one panel light.', scene: CORNELL_BOX },
  { id: 'glass', label: 'Glass Spheres', blurb: 'Sixty seeded spheres on a checkerboard under a gradient sky.', scene: GLASS_SPHERES },
  { id: 'night', label: 'Night Emissives', blurb: 'No sky at all: only glowing spheres light the scene.', scene: NIGHT_EMISSIVES },
  { id: 'dof', label: 'Depth of Field', blurb: 'A receding row of spheres for playing with aperture and focus.', scene: DEPTH_OF_FIELD },
  { id: 'mirror', label: 'Mirror Room', blurb: 'A 2000-triangle torus knot inside five slightly rough mirrors.', scene: MIRROR_ROOM },
]

export function presetById(id: string): ScenePreset | undefined {
  return SCENE_PRESETS.find((p) => p.id === id)
}
