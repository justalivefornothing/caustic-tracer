/**
 * Bounding volume hierarchy built with a binned surface-area heuristic and
 * flattened into a linear, depth-first node array with skip links so a GPU
 * (or a tight CPU loop) can traverse it without a stack.
 *
 * Node layout, 8 floats (= 2 RGBA32F texels) per node:
 *
 *   texel 0: [ min.x, min.y, min.z, skip ]   skip = index of the next node after
 *                                            this node's whole subtree (DFS order)
 *   texel 1: [ max.x, max.y, max.z, leaf ]   leaf = -1 for interior nodes,
 *                                            otherwise primStart * 8 + primCount
 *
 * Traversal: start at node 0. If the ray hits the box, test primitives when it
 * is a leaf and then advance to node+1 (the first child, or for a leaf the same
 * value as skip). If the ray misses, jump to `skip`. Stop when index >= count.
 */

export const NODE_STRIDE = 8
export const MAX_LEAF_PRIMS = 4
export const SAH_BUCKETS = 12

export interface PrimBounds {
  readonly count: number
  /** 3 floats per primitive. */
  readonly min: Float64Array
  readonly max: Float64Array
  readonly centroid: Float64Array
}

export interface Bvh {
  readonly nodes: Float32Array
  readonly nodeCount: number
  /** order[i] = original primitive index stored at packed position i. */
  readonly order: Uint32Array
}

export function allocPrimBounds(count: number): PrimBounds {
  return {
    count,
    min: new Float64Array(count * 3),
    max: new Float64Array(count * 3),
    centroid: new Float64Array(count * 3),
  }
}

export function setPrimBounds(
  pb: PrimBounds,
  i: number,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): void {
  const o = i * 3
  pb.min[o] = minX; pb.min[o + 1] = minY; pb.min[o + 2] = minZ
  pb.max[o] = maxX; pb.max[o + 1] = maxY; pb.max[o + 2] = maxZ
  pb.centroid[o] = 0.5 * (minX + maxX)
  pb.centroid[o + 1] = 0.5 * (minY + maxY)
  pb.centroid[o + 2] = 0.5 * (minZ + maxZ)
}

export function encodeLeaf(start: number, count: number): number {
  return start * 8 + count
}
export function decodeLeafStart(leaf: number): number {
  return Math.floor(leaf / 8)
}
export function decodeLeafCount(leaf: number): number {
  return leaf - Math.floor(leaf / 8) * 8
}

const surfaceArea = (dx: number, dy: number, dz: number): number => 2 * (dx * dy + dy * dz + dz * dx)

class Builder {
  readonly pb: PrimBounds
  readonly order: Uint32Array
  nodes: Float32Array
  nodeCount = 0

  // scratch, reused across nodes
  private readonly bucketCount = new Int32Array(SAH_BUCKETS)
  private readonly bucketMin = new Float64Array(SAH_BUCKETS * 3)
  private readonly bucketMax = new Float64Array(SAH_BUCKETS * 3)
  private readonly rightArea = new Float64Array(SAH_BUCKETS)
  private readonly rightCount = new Int32Array(SAH_BUCKETS)

  constructor(pb: PrimBounds) {
    this.pb = pb
    this.order = new Uint32Array(pb.count)
    for (let i = 0; i < pb.count; i++) this.order[i] = i
    this.nodes = new Float32Array(Math.max(1, 2 * pb.count - 1) * NODE_STRIDE)
  }

  build(): Bvh {
    if (this.pb.count === 0) {
      // A single empty leaf so traversal has something well-formed to look at.
      const n = this.nodes
      n[0] = n[1] = n[2] = 0
      n[3] = 1
      n[4] = n[5] = n[6] = 0
      n[7] = encodeLeaf(0, 0)
      this.nodeCount = 1
    } else {
      this.buildNode(0, this.pb.count)
    }
    return {
      nodes: this.nodes.slice(0, this.nodeCount * NODE_STRIDE),
      nodeCount: this.nodeCount,
      order: this.order,
    }
  }

  private buildNode(start: number, end: number): void {
    const nodeIndex = this.nodeCount++
    const base = nodeIndex * NODE_STRIDE
    const { min, max, centroid } = this.pb
    const order = this.order
    const nodes = this.nodes

    let bMinX = Infinity, bMinY = Infinity, bMinZ = Infinity
    let bMaxX = -Infinity, bMaxY = -Infinity, bMaxZ = -Infinity
    let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity
    let cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity
    for (let i = start; i < end; i++) {
      const o = order[i]! * 3
      const x0 = min[o]!, y0 = min[o + 1]!, z0 = min[o + 2]!
      const x1 = max[o]!, y1 = max[o + 1]!, z1 = max[o + 2]!
      if (x0 < bMinX) bMinX = x0
      if (y0 < bMinY) bMinY = y0
      if (z0 < bMinZ) bMinZ = z0
      if (x1 > bMaxX) bMaxX = x1
      if (y1 > bMaxY) bMaxY = y1
      if (z1 > bMaxZ) bMaxZ = z1
      const cx = centroid[o]!, cy = centroid[o + 1]!, cz = centroid[o + 2]!
      if (cx < cMinX) cMinX = cx
      if (cy < cMinY) cMinY = cy
      if (cz < cMinZ) cMinZ = cz
      if (cx > cMaxX) cMaxX = cx
      if (cy > cMaxY) cMaxY = cy
      if (cz > cMaxZ) cMaxZ = cz
    }
    // Nudge outward by a float32 ulp-ish amount so rounding into the
    // Float32Array can never shrink a box below its contents.
    nodes[base] = bMinX - Math.abs(bMinX) * 1.2e-7 - 1e-9
    nodes[base + 1] = bMinY - Math.abs(bMinY) * 1.2e-7 - 1e-9
    nodes[base + 2] = bMinZ - Math.abs(bMinZ) * 1.2e-7 - 1e-9
    nodes[base + 4] = bMaxX + Math.abs(bMaxX) * 1.2e-7 + 1e-9
    nodes[base + 5] = bMaxY + Math.abs(bMaxY) * 1.2e-7 + 1e-9
    nodes[base + 6] = bMaxZ + Math.abs(bMaxZ) * 1.2e-7 + 1e-9

    const count = end - start
    if (count <= MAX_LEAF_PRIMS) {
      nodes[base + 3] = nodeIndex + 1
      nodes[base + 7] = encodeLeaf(start, count)
      return
    }

    let mid = this.sahSplit(start, end, cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ)
    if (mid <= start || mid >= end) {
      mid = this.medianSplit(start, end, cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ)
    }

    nodes[base + 7] = -1
    this.buildNode(start, mid)
    this.buildNode(mid, end)
    nodes[base + 3] = this.nodeCount
  }

  /** Binned SAH over 12 buckets on each axis; returns partition point or -1. */
  private sahSplit(
    start: number, end: number,
    cMinX: number, cMinY: number, cMinZ: number,
    cMaxX: number, cMaxY: number, cMaxZ: number,
  ): number {
    const { min, max, centroid } = this.pb
    const order = this.order
    const bucketCount = this.bucketCount
    const bucketMin = this.bucketMin
    const bucketMax = this.bucketMax
    const rightArea = this.rightArea
    const rightCount = this.rightCount

    let bestCost = Infinity
    let bestAxis = -1
    let bestSplit = -1
    let bestOrigin = 0
    let bestScale = 0

    for (let axis = 0; axis < 3; axis++) {
      const cmin = axis === 0 ? cMinX : axis === 1 ? cMinY : cMinZ
      const cmax = axis === 0 ? cMaxX : axis === 1 ? cMaxY : cMaxZ
      const extent = cmax - cmin
      if (!(extent > 1e-12)) continue
      const toBucket = SAH_BUCKETS / extent

      bucketCount.fill(0)
      bucketMin.fill(Infinity)
      bucketMax.fill(-Infinity)

      for (let i = start; i < end; i++) {
        const o = order[i]! * 3
        let b = Math.floor((centroid[o + axis]! - cmin) * toBucket)
        if (b >= SAH_BUCKETS) b = SAH_BUCKETS - 1
        if (b < 0) b = 0
        bucketCount[b]!++
        const bo = b * 3
        if (min[o]! < bucketMin[bo]!) bucketMin[bo] = min[o]!
        if (min[o + 1]! < bucketMin[bo + 1]!) bucketMin[bo + 1] = min[o + 1]!
        if (min[o + 2]! < bucketMin[bo + 2]!) bucketMin[bo + 2] = min[o + 2]!
        if (max[o]! > bucketMax[bo]!) bucketMax[bo] = max[o]!
        if (max[o + 1]! > bucketMax[bo + 1]!) bucketMax[bo + 1] = max[o + 1]!
        if (max[o + 2]! > bucketMax[bo + 2]!) bucketMax[bo + 2] = max[o + 2]!
      }

      // Suffix sweep: bounds/count of everything right of split s (buckets s+1..end).
      let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity
      let rx1 = -Infinity, ry1 = -Infinity, rz1 = -Infinity
      let rc = 0
      for (let s = SAH_BUCKETS - 1; s >= 1; s--) {
        const bo = s * 3
        if (bucketCount[s]! > 0) {
          if (bucketMin[bo]! < rx0) rx0 = bucketMin[bo]!
          if (bucketMin[bo + 1]! < ry0) ry0 = bucketMin[bo + 1]!
          if (bucketMin[bo + 2]! < rz0) rz0 = bucketMin[bo + 2]!
          if (bucketMax[bo]! > rx1) rx1 = bucketMax[bo]!
          if (bucketMax[bo + 1]! > ry1) ry1 = bucketMax[bo + 1]!
          if (bucketMax[bo + 2]! > rz1) rz1 = bucketMax[bo + 2]!
          rc += bucketCount[s]!
        }
        rightCount[s - 1] = rc
        rightArea[s - 1] = rc > 0 ? surfaceArea(rx1 - rx0, ry1 - ry0, rz1 - rz0) : 0
      }

      // Prefix sweep: left side includes buckets 0..s.
      let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity
      let lx1 = -Infinity, ly1 = -Infinity, lz1 = -Infinity
      let lc = 0
      for (let s = 0; s < SAH_BUCKETS - 1; s++) {
        const bo = s * 3
        if (bucketCount[s]! > 0) {
          if (bucketMin[bo]! < lx0) lx0 = bucketMin[bo]!
          if (bucketMin[bo + 1]! < ly0) ly0 = bucketMin[bo + 1]!
          if (bucketMin[bo + 2]! < lz0) lz0 = bucketMin[bo + 2]!
          if (bucketMax[bo]! > lx1) lx1 = bucketMax[bo]!
          if (bucketMax[bo + 1]! > ly1) ly1 = bucketMax[bo + 1]!
          if (bucketMax[bo + 2]! > lz1) lz1 = bucketMax[bo + 2]!
          lc += bucketCount[s]!
        }
        const rcs = rightCount[s]!
        if (lc === 0 || rcs === 0) continue
        const cost = lc * surfaceArea(lx1 - lx0, ly1 - ly0, lz1 - lz0) + rcs * rightArea[s]!
        if (cost < bestCost) {
          bestCost = cost
          bestAxis = axis
          bestSplit = s
          bestOrigin = cmin
          bestScale = toBucket
        }
      }
    }

    if (bestAxis < 0) return -1

    // In-place partition: bucket <= bestSplit goes left.
    let lo = start
    let hi = end - 1
    while (lo <= hi) {
      const o = order[lo]! * 3
      let b = Math.floor((centroid[o + bestAxis]! - bestOrigin) * bestScale)
      if (b >= SAH_BUCKETS) b = SAH_BUCKETS - 1
      if (b < 0) b = 0
      if (b <= bestSplit) {
        lo++
      } else {
        const tmp = order[lo]!
        order[lo] = order[hi]!
        order[hi] = tmp
        hi--
      }
    }
    if (lo === start || lo === end) return -1
    return lo
  }

  /** Fallback for degenerate centroid distributions: sort along the widest axis and split in half. */
  private medianSplit(
    start: number, end: number,
    cMinX: number, cMinY: number, cMinZ: number,
    cMaxX: number, cMaxY: number, cMaxZ: number,
  ): number {
    const ex = cMaxX - cMinX, ey = cMaxY - cMinY, ez = cMaxZ - cMinZ
    const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2
    const centroid = this.pb.centroid
    const slice = Array.from(this.order.subarray(start, end))
    slice.sort((a, b) => centroid[a * 3 + axis]! - centroid[b * 3 + axis]!)
    this.order.set(slice, start)
    return (start + end) >> 1
  }
}

export function buildBvh(pb: PrimBounds): Bvh {
  return new Builder(pb).build()
}

// ---------------------------------------------------------------------------
// Validation helpers (used by tests and by the JSON importer).

export interface BvhValidation {
  readonly ok: boolean
  readonly errors: readonly string[]
  readonly maxLeafSize: number
  readonly leafCount: number
  readonly depth: number
}

/**
 * Walk the flattened array recursively and check every invariant a stackless
 * traversal relies on: skip links point exactly past each subtree, children
 * are contained in parents, leaves are small and cover every primitive once.
 */
export function validateBvh(bvh: Bvh, primCount: number): BvhValidation {
  const errors: string[] = []
  const n = bvh.nodes
  let maxLeafSize = 0
  let leafCount = 0
  let depth = 0
  const covered = new Uint8Array(primCount)

  const walk = (index: number, d: number): number => {
    if (d > depth) depth = d
    if (index >= bvh.nodeCount) {
      errors.push(`node index ${index} out of range`)
      return index + 1
    }
    const base = index * NODE_STRIDE
    const skip = n[base + 3]!
    const leaf = n[base + 7]!
    if (leaf >= 0) {
      const start = decodeLeafStart(leaf)
      const count = decodeLeafCount(leaf)
      leafCount++
      if (count > maxLeafSize) maxLeafSize = count
      if (count > MAX_LEAF_PRIMS) errors.push(`leaf ${index} holds ${count} primitives`)
      for (let i = start; i < start + count; i++) {
        if (i >= bvh.order.length) {
          errors.push(`leaf ${index} references packed slot ${i} beyond range`)
          continue
        }
        const orig = bvh.order[i]!
        if (covered[orig]) errors.push(`primitive ${orig} covered twice`)
        covered[orig] = 1
      }
      if (skip !== index + 1) errors.push(`leaf ${index} skip ${skip} != ${index + 1}`)
      return index + 1
    }
    const afterLeft = walk(index + 1, d + 1)
    const afterRight = walk(afterLeft, d + 1)
    if (skip !== afterRight) errors.push(`interior ${index} skip ${skip} != subtree end ${afterRight}`)
    // children must be inside the parent box
    for (const child of [index + 1, afterLeft]) {
      const cb = child * NODE_STRIDE
      for (let k = 0; k < 3; k++) {
        if (n[cb + k]! < n[base + k]! - 1e-6 || n[cb + 4 + k]! > n[base + 4 + k]! + 1e-6) {
          errors.push(`child ${child} escapes parent ${index} on axis ${k}`)
          break
        }
      }
    }
    return afterRight
  }

  const end = walk(0, 1)
  if (end !== bvh.nodeCount) errors.push(`traversal ended at ${end}, expected ${bvh.nodeCount}`)
  if (primCount > 0) {
    for (let i = 0; i < primCount; i++) {
      if (!covered[i]) {
        errors.push(`primitive ${i} not covered by any leaf`)
        break
      }
    }
  }
  return { ok: errors.length === 0, errors, maxLeafSize, leafCount, depth }
}
