# Module: blender-integration
<!-- Blender add-on that runs the Authoring model (path + style guidance) on Blender data -->

---

## [2026-09-23] Blender add-on and Blender<->model middleware

**Type**: `feature` + `investigation`
**Status**: `resolved` (UI modal path pending manual check, see Unverified Items)

### Context
Goal: set up path guidance and the model in Blender, run the Authoring model on
Blender data, and bake the result back. An earlier attempt failed because Blender
data and model data "were not the same". Priority: correct conversion; audit all
model inputs before designing the Blender side.

### Work Done
1. Audited all model inputs by running the real networks headless:
   Network 441 -> 16x279, PostProcessor 456 -> 64. Input layout in Blender/README.md.
2. Imported Geno with Blender's own glTF and FBX importers (bpy 4.5) and compared
   bone frames against `GLB.JointMatrices`.
3. Built the pure-NumPy middleware (conventions, rig calibration, features, exchange).
4. Built the runner (MANUAL mode, fixed dt) around `Demos/Authoring/MotionController`.
5. Built the add-on (preferences, properties, operators, UI, blender_io, pipeline).
6. Tests: 34 middleware unit tests, 8 parity tests vs ai4animation, 5 bpy end-to-end
   tests driving the real operators and runner, 5 UI smoke tests. Run on bpy 4.5.14
   and 5.0.1.

### Findings (root causes of "Blender data != model data")
Ranked by impact:
1. **Bone frame axes differ per importer.** glTF: Blender-style axes plus a rest pose
   rotated 90 degrees (lying on its back), compensated by a pose-basis rotation
   on Hips. FBX: armature scaled 0.01, different axes. Positions match; axes do not.
   The network reads per-bone Z/Y axes, so a naive conversion feeds wrong inputs.
2. **Axis convention**: Y-up/+Z-forward (model) vs Z-up/-Y-forward (Blender).
   Scene objects additionally need the frame conjugation C M C^T.
3. **Non-orthonormal model rotations**: `Rotation.Look(z, y)` does not orthogonalize
   (up to 57 degrees at start-up, 0.56 degrees mean). Without fixing this, Blender
   would get scale/shear.
4. **Quaternion layout** (w,x,y,z vs x,y,z,w) and sign continuity for F-curves.

### Other defects found in the demo code (not changed, worked around in the runner)
- `torch.load` without `map_location`: `Models/*.pt` were pickled on CUDA, so the
  demo crashes on CPU-only machines.
- `MotionController` loads `Guidances/` relative to the cwd.
- Start-up teleport: `Actor.Transforms` stays at the origin while the root starts
  at the path start.
- `LegIK` stores *views* into `Actor.Transforms`, so the initial pose must be written
  in place.

### Checklist
- [x] Input/output dims verified by loading the networks (441/279, 456/64)
- [x] `compute_root` == `RootModule.Compute` on real mocap (<= 1e-5 m)
- [x] guidance capture == `GuidanceModule.GetLegacyGuidance` (<= 1e-5 m)
- [x] shipped guidance files use the model bone order
- [x] shipped profile == live runner profile
- [x] bake -> Blender depsgraph -> model: <= 1.4e-6 m, <= 3e-5 deg (glTF and FBX)
- [x] placement invariance with a real 90 degree yaw; x2 scale correctly rejected
- [x] all three path modes and style keys/custom styles through the real operators
- [x] Blender 4.5 LTS and 5.0 (slotted actions)
- [ ] Modal (non-blocking) Generate in the interactive UI (needs a window)

### Unverified Items
> Could not be tested headless. Please run in Blender with a window:
- [ ] Sidebar > AI4A > Generate (button, i.e. INVOKE): expected progress text
      "Model: frame N / M" in the panel, ESC cancels, report "Baked ... Check vs model: 0.00 mm".
- [ ] macOS/Windows: runner subprocess with your conda python path (tested on Linux only).

### Fix / Implementation Detail
See `Blender/README.md` ("Data conventions", "Why calibration is necessary",
"Runner vs. standalone demo").

### Assumptions Made
- The user works with the Geno rig (as requested). Other rigs need retargeting.
- Scene unit scale converts Blender units to meters (`scale_length`).
- Armature object is static during the baked range.
- Default style = first guidance alphabetically ("BigSteps"), same as the demo.

---

## [2026-09-25] Joystick control input, Blender 5.2 LTS, defaults

**Type**: `feature` + `investigation`
**Status**: `resolved` (UI modal path and Windows runner still need a manual check)

### Context
User feedback: RTX 3050, Blender 4.5 LTS moving to 5.2 LTS. Mimic the original
research's joystick input (a vector) with Blender objects, e.g. a Bezier path.
Pick the less problematic bone-location mode. Default style Neutral.

### Work Done
1. Read both control schemes: Biped `Control()` (left stick velocity + right stick
   facing, `SimulationObject.Control`) and Authoring `ControlFromTarget`. Same `Network.pt`.
2. Ported Biped `Control()` into the runner, bound onto the Authoring MotionController.
3. Measured both controllers on S-curve / 90 degree corner / hairpin paths
   (tracking, foot sliding, smoothness, heading jitter, arrival), with
   moving-phase and stop-phase metrics kept separate.
4. Investigated failures and treated each as a hypothesis to falsify:
   - The pure-pursuit stick stopped short. Hypothesis "idle threshold" was falsified
     (a minimum push made it worse). Root cause: momentum overshoot makes it orbit
     the end point. Fixed with release/resume hysteresis.
   - Stuck at 1 m/s on a straight start. Isolated to a cold start below ~1 m/s: the
     model shuffles just above 0.1 m/s, synchronization pins the simulated
     trajectory to the actor, and no lag builds up. A 1 s idle warm-up did NOT fix
     0.4-0.8 m/s. A speed PID (Quadruped demo) fixed 0.8+ only. Position tracking
     (feed-forward + P on a reference) fixed all speeds.
   - Tracker overshoot at the path end: fixed with reference braking (1 m/s^2).
   - Cross-track feedback and gains 3/4 were measured worse than point tracking at
     gain 2. Removed.
5. Blender side: virtual sticks (gate circle + constrained knob), facing modes,
   Auto controller, Neutral default, helpers, UI. Middleware functions
   `stick_from_knob`, `facing_from_objects`; exchange schema v2.
6. Blender 5.2.2 LTS (Python 3.13, NumPy 2.5): full suite passes unchanged.

### Checklist
- [x] Backwards / strafe / look-at work with the network (facing error 1.8-5.7 degrees)
- [x] Speed Assist: 0.4-2.0 m/s start from standing; 1 m/s x 5 s -> 4.99 m
- [x] Auto controller: Goal for paths facing the movement (best measured), Joystick otherwise
- [x] Virtual joystick in real Blender: walks +Y at 1.00 m/s facing -Y, bake check 0.00 mm
- [x] Suites on bpy 4.5 / 5.0 / 5.2 and the model env
- [ ] Modal Generate button in an interactive Blender window
- [ ] Runner with a CUDA (RTX 3050) torch build on the user's machine

### Decisions
- Bone Locations default stays All Bones (exact): Root Only drifts feet 2-3.5 cm (sliding).
- Style default Neutral; style selection itself left as is (user: later).

### Assumptions Made
- Stick up = Blender +Y (same as the gamepad demo's world mapping); rotate the gate to change it.
- The model's slowest reliable gait is ~0.3-0.4 m/s (0.4 m/s commanded -> 0.33 m/s achieved).
