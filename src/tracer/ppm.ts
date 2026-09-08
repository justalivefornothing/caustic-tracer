/**
 * Binary PPM (P6) encode/decode. No file system access here; scripts do that.
 */
export function encodePPM(width: number, height: number, rgba: Uint8ClampedArray | Uint8Array): Uint8Array {
  const header = `P6\n${width} ${height}\n255\n`
  const headerBytes = new TextEncoder().encode(header)
  const out = new Uint8Array(headerBytes.length + width * height * 3)
  out.set(headerBytes, 0)
  let o = headerBytes.length
  for (let i = 0; i < width * height; i++) {
    out[o++] = rgba[i * 4]!
    out[o++] = rgba[i * 4 + 1]!
    out[o++] = rgba[i * 4 + 2]!
  }
  return out
}

export interface PpmImage {
  readonly width: number
  readonly height: number
  /** 3 bytes per pixel, row 0 at the top. */
  readonly rgb: Uint8Array
}

export function decodePPM(bytes: Uint8Array): PpmImage {
  // Header tokens: magic, width, height, maxval, each separated by whitespace.
  const tokens: string[] = []
  let i = 0
  while (tokens.length < 4 && i < bytes.length) {
    while (i < bytes.length && isSpace(bytes[i]!)) i++
    if (bytes[i] === 0x23) {
      while (i < bytes.length && bytes[i] !== 0x0a) i++
      continue
    }
    let tok = ''
    while (i < bytes.length && !isSpace(bytes[i]!)) tok += String.fromCharCode(bytes[i++]!)
    tokens.push(tok)
  }
  if (tokens[0] !== 'P6') throw new Error('not a binary PPM')
  const width = Number(tokens[1])
  const height = Number(tokens[2])
  if (tokens[3] !== '255') throw new Error('only maxval 255 supported')
  i++ // single whitespace after maxval
  const rgb = bytes.slice(i, i + width * height * 3)
  if (rgb.length !== width * height * 3) throw new Error('truncated PPM data')
  return { width, height, rgb }
}

const isSpace = (b: number): boolean => b === 0x20 || b === 0x0a || b === 0x0d || b === 0x09

export function ppmPixel(img: PpmImage, x: number, y: number): { r: number; g: number; b: number } {
  const o = (y * img.width + x) * 3
  return { r: img.rgb[o]!, g: img.rgb[o + 1]!, b: img.rgb[o + 2]! }
}
