# Caustic Tracer

A physically-based progressive path tracer that runs as a WebGL2 fragment shader, with the scene and its BVH packed into float textures — backed by a TypeScript CPU reference tracer that walks the exact same packed data structures and is verified by a Node test suite.

## How it works

- **Packed scene** — spheres and triangle meshes are flattened into typed arrays (`pack.ts`); the GPU reads the same layout through float textures
- **Stackless BVH traversal** — both the GLSL tracer and the CPU reference walk the BVH without recursion (`bvh.ts`)
- **Materials** — Lambertian, metal, dielectric, and emissive, plus checker modulation (`materials.ts`)
- **Light transport** — next-event estimation with power-heuristic MIS on diffuse surfaces, optional Russian roulette after the third bounce (`tracer.ts`)
- **CPU reference tracer** (`reference.ts`) — allocation-free inner loop written with scalar locals and typed-array scratch space, runnable in Node for verification and offline rendering
- **Furnace tests** — energy-conservation checks (`furnace.test.ts`)

## Use

```bash
npm install
npm run dev          # interactive WebGL2 renderer (React UI)
npm test             # CPU tracer, BVH, materials, scenes, furnace tests
npm run render:cpu   # render a frame in Node (scripts/render-cpu.ts)
npm run bench
```

## Layout

```
src/tracer/   Core: BVH, packing, materials, intersection, camera, scenes, PPM export, tonemapping
src/App.tsx   Interactive render UI
scripts/      render-cpu, bench, debug-aabb, debug-trace
```

The interactive renderer requires WebGL2; the CPU tracer and the test suite run anywhere Node runs.

## License

none — all rights reserved.
