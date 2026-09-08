/**
 * Thin-lens camera. `CameraSettings` is the orbit description users edit;
 * `CameraFrame` is the derived world-space basis both tracers consume.
 */
import type { CameraSettings } from './scene.ts'
import { add, cross, normalize, scale, sub, v3, type Vec3 } from './vec3.ts'

export interface CameraFrame {
  readonly origin: Vec3
  /** Unit right vector. */
  readonly right: Vec3
  /** Unit up vector (orthogonal to forward). */
  readonly up: Vec3
  /** Unit view direction. */
  readonly forward: Vec3
  readonly tanHalfFov: number
  readonly aspect: number
  readonly lensRadius: number
  readonly focusDistance: number
}

export const MIN_PITCH = -Math.PI / 2 + 0.02
export const MAX_PITCH = Math.PI / 2 - 0.02

export function cameraPosition(cam: CameraSettings): Vec3 {
  const cp = Math.cos(cam.pitch)
  return add(
    cam.target,
    scale(v3(cp * Math.sin(cam.yaw), Math.sin(cam.pitch), cp * Math.cos(cam.yaw)), cam.distance),
  )
}

export function cameraFrame(cam: CameraSettings, aspect: number): CameraFrame {
  const origin = cameraPosition(cam)
  const forward = normalize(sub(cam.target, origin))
  const worldUp = v3(0, 1, 0)
  let right = cross(forward, worldUp)
  if (right.x * right.x + right.y * right.y + right.z * right.z < 1e-12) right = v3(1, 0, 0)
  right = normalize(right)
  const up = cross(right, forward)
  return {
    origin,
    right,
    up,
    forward,
    tanHalfFov: Math.tan((cam.fovDeg * Math.PI) / 360),
    aspect,
    lensRadius: cam.lensRadius,
    focusDistance: Math.max(1e-3, cam.focusDistance),
  }
}

export interface Ray {
  readonly origin: Vec3
  readonly dir: Vec3
}

/**
 * Generate a camera ray through normalised screen position (sx, sy) in [0, 1]
 * with y pointing up, jittering the lens position by a unit-disk sample.
 */
export function generateRay(frame: CameraFrame, sx: number, sy: number, lensX: number, lensY: number): Ray {
  const px = (2 * sx - 1) * frame.tanHalfFov * frame.aspect
  const py = (2 * sy - 1) * frame.tanHalfFov
  const focusPoint = add(
    frame.origin,
    scale(add(add(scale(frame.right, px), scale(frame.up, py)), frame.forward), frame.focusDistance),
  )
  const offset = add(scale(frame.right, lensX * frame.lensRadius), scale(frame.up, lensY * frame.lensRadius))
  const origin = add(frame.origin, offset)
  return { origin, dir: normalize(sub(focusPoint, origin)) }
}

/** Pinhole ray through the exact pixel centre; used for picking and focus. */
export function primaryRay(frame: CameraFrame, sx: number, sy: number): Ray {
  return generateRay(frame, sx, sy, 0, 0)
}

/** Convert a photographic f-number to a lens radius for the current field of view. */
export function fNumberToLensRadius(fNumber: number, fovDeg: number, sensorHeight = 0.24): number {
  if (!Number.isFinite(fNumber) || fNumber <= 0) return 0
  const focalLength = (0.5 * sensorHeight) / Math.tan((fovDeg * Math.PI) / 360)
  return focalLength / (2 * fNumber)
}

export function lensRadiusToFNumber(lensRadius: number, fovDeg: number, sensorHeight = 0.24): number {
  if (lensRadius <= 0) return Infinity
  const focalLength = (0.5 * sensorHeight) / Math.tan((fovDeg * Math.PI) / 360)
  return focalLength / (2 * lensRadius)
}

/** Orbit helpers. */
export function orbit(cam: CameraSettings, dYaw: number, dPitch: number): CameraSettings {
  const pitch = Math.min(MAX_PITCH, Math.max(MIN_PITCH, cam.pitch + dPitch))
  return { ...cam, yaw: cam.yaw + dYaw, pitch }
}

export function dolly(cam: CameraSettings, factor: number): CameraSettings {
  return { ...cam, distance: Math.min(1000, Math.max(0.05, cam.distance * factor)) }
}

/** Pan the target in the camera's screen plane by (dx, dy) in scene units. */
export function pan(cam: CameraSettings, dx: number, dy: number, aspect: number): CameraSettings {
  const frame = cameraFrame(cam, aspect)
  const target = add(cam.target, add(scale(frame.right, dx), scale(frame.up, dy)))
  return { ...cam, target }
}
