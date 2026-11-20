# Motion Studio (`motion serve`)

A local web page for both models, in the spirit of fastwan-rs's Prompt
Studio: `motion serve` loads the models once and serves `web/studio.html`
(compiled into the binary) with a small job API. Nothing leaves the machine.

```bash
motion serve --kimodo-dir weights/Kimodo-SOMA-RP-v1.2 --bricks-dir weights/MotionBricks/soma --embeddings prompts.safetensors
```

Then open <http://127.0.0.1:7860/>. `motion serve --synthetic` runs tiny random
models instead (the motion is noise) so the page can be tried without a
download.

## The page

- **Text to motion (Kimodo).** A prompt timeline (one prompt and length per
  segment: Kimodo's multi-prompt sequences), the prompt gallery from
  `--embeddings`, a root path drawn by clicking the ground (waypoints, times
  spread over the clip and editable), full-body keyframes pinned from any
  frame of an earlier motion, steps, text and constraint guidance, seed and
  start heading.
- **Continue with MotionBricks.** The selected motion's last four frames are
  the context; each target is a pose from any earlier motion (job and frame),
  placed on the ground with a heading. Transitions are chained.
- **Player.** A canvas skeleton (SOMA-77) with orbit, zoom, play, scrub,
  speed, follow-root and foot-contact markers; BVH and NPZ downloads.
- **History.** The engine keeps the last 24 motions; any of them can be
  played, continued or used for keyframes and targets.

Kimodo output becomes MotionBricks context directly: Kimodo's BVH rest pose is
the MotionBricks clips' rest pose.

## API

| route | |
|---|---|
| `GET /api/info` | models loaded, backend, gallery prompts, skeleton (joint names, parents, feet), fps |
| `POST /api/kimodo` | `{segments: [{prompt, seconds}], steps, cfg: [text, constraint], seed, heading, waypoints: [{t, x, z}], keyframes: [{job, frame, at}], transition}` → `{id}` |
| `POST /api/bricks` | `{context: job, targets: [{job, frame, poses, x, z, heading}], tokens: [min, max], transition_early}` → `{id}` |
| `GET /api/jobs`, `GET /api/jobs/{id}` | state (`queued`, `running`, `done`, `failed`), step / steps, segment / segments, seconds, frames |
| `GET /api/jobs/{id}/motion` | `{fps, frames, positions: [T·77·3], contacts: [T·4]}` |
| `GET /api/jobs/{id}/motion.bvh`, `/motion.npz` | downloads (NPZ for Kimodo jobs) |

POSTs must be JSON from the page's own origin (or one given with
`--allow-origin`); the server binds 127.0.0.1 unless `--host` says otherwise.
Jobs run one at a time on one worker thread.

## Verified

In the authoring container (CPU backend, released weights): a 3 s Kimodo
job with two waypoints ends 0.14 m from the last waypoint; a MotionBricks
continuation from its end reaches its target position exactly. The page was
driven in headless Chromium (path drawing, generation, pinning, a
MotionBricks continuation) with no console errors and no horizontal scroll
at phone width.
