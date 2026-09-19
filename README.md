# Caustic Tracer

CPU path tracer in TypeScript with BVH acceleration.

## Features

- Bounding volume hierarchy (BVH) for mesh scenes
- Materials: diffuse, metal, glass, emissive
- Multiple built-in scenes and furnace tests
- Progressive render with tonemapping
- PPM export helpers and deterministic RNG

## Run

```bash
npm install
npm run dev
```

```bash
npm test
```

## Layout

| Path | Role |
|------|------|
| `src/tracer/` | Core tracer, BVH, materials, scenes, camera |
| `src/App.tsx` | Interactive render UI |

## License

MIT
