/**
 * Render the Cornell Box with the CPU reference tracer and write a binary PPM.
 *
 *   npm run render:cpu            -> out/cornell.ppm (96x64, 16 spp)
 *   npx tsx scripts/render-cpu.ts 192 128 64
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { encodePPM } from '../src/tracer/ppm.ts'
import { linearPixel, renderCornellReference } from '../src/tracer/reference.ts'

const width = Number(process.argv[2] ?? 96)
const height = Number(process.argv[3] ?? 64)
const spp = Number(process.argv[4] ?? 16)

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outPath = resolve(root, 'out', 'cornell.ppm')

const img = renderCornellReference(width, height, spp)
mkdirSync(dirname(outPath), { recursive: true })
writeFileSync(outPath, encodePPM(img.width, img.height, img.rgba))

const center = linearPixel(img, Math.floor(width / 2), Math.floor(height / 2))
const corner = linearPixel(img, 0, height - 1)
console.log(`wrote ${outPath} (${width}x${height}, ${spp} spp) in ${img.millis.toFixed(0)} ms`)
console.log(`  center  linear rgb = ${center.r.toFixed(3)} ${center.g.toFixed(3)} ${center.b.toFixed(3)}`)
console.log(`  corner  linear rgb = ${corner.r.toFixed(3)} ${corner.g.toFixed(3)} ${corner.b.toFixed(3)}`)
