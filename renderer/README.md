# Luxo renderer

Luxo's renderer is the browser-owned body: a full-bleed Vite, TypeScript,
three.js, and Tailwind application. It renders the supplied five-DOF lamp and
accepts body state from the core, but it makes no behavioral decisions.

## Model contract

- `base_link` is held at the world origin by `lumen-base-anchor`; the URDF has
  no world joint.
- The source URDF and `assets/lamp_shade.stl` are imported with Vite `?url`
  assets. A `LoadingManager` URL modifier explicitly redirects the URDF's bare
  `assets/lamp_shade.stl` reference, so loading never depends on the process
  working directory.
- `speaker_link` is accepted as an empty semantic frame. `camera_link` and
  `light_emitter_link` are transform sources and are not assigned fake inertia.
- Body-state keys map to the five real URDF joints in `src/scene/urdf.ts`.
  There is no roll joint and the renderer does not invent one.
- `head_pitch_joint` includes a pi rotation about Z. Character front and
  azimuth zero are world -X, and positive head pitch looks down.
- The initial static state is the canonical rest pose
  `(0, 0.35, -0.75, 0, 0.25)`, never the invalid all-zero pose.

`mountRenderer(root)` in `src/app.ts` returns `applyBodyState(state)` and
`destroy()`. Its parameter is structural, so the generated protocol
`BodyStateMessage` can be passed directly without importing protocol files into
the scene foundation.

## Checks

From this directory:

```sh
npm ci
npm run typecheck
```

`npm run build` also runs the typecheck. On the isolated foundation-renderer
branch, the build's Vite phase intentionally waits for the separately owned
`src/main.ts` from the foundation-protocol packet.

## Exact browser verification

Rendering requires a browser and cannot be covered by the Node check. After the
protocol packet has supplied `src/main.ts`:

1. Run `npm ci`, then `npm run dev`, and open `http://127.0.0.1:5173`.
2. Confirm one supplied lamp appears in rest pose on the circular plinth, with
   a warm emitter, a dark studio ground, a soft shadow, and no detached parts.
3. In DevTools Network, reload and confirm both
   `dummy_lamp_5dof.urdf` and `lamp_shade.stl` return HTTP 200.
4. Drag to orbit and use the wheel/trackpad to zoom. Confirm the camera remains
   above the ground, cannot pan the anchored base away, and keeps the full lamp
   within practical debugging distance.
5. Resize the browser through portrait, square, and landscape shapes. Confirm
   the canvas fills the viewport without stretching the lamp or clipping the
   small status labels.
6. Temporarily change `REST_JOINTS` in `src/scene/urdf.ts` to
   `base_yaw: 0.45` and `head_pitch: -0.40`, then reload. Confirm the base turns
   and the head visibly looks higher. Change `head_pitch` to `0.50` and confirm
   it looks lower. Restore the canonical rest values afterward.
7. Hand a structurally compatible body state to `applyBodyState` through the
   protocol callback and repeat for all five keys. Confirm only the mapped URDF
   joint moves and that no sideways roll is synthesized.

Orbit controls are a development aid. Animation, light presets, bloom,
sensors, audio, and telemetry are separate packets and are not implemented in
this foundation.
