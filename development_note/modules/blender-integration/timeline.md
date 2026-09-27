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

---

## [2026-09-26] Windows: "[WinError 193] %1 is not a valid Win32 application"

**Type**: `debug`
**Status**: `resolved`

### Context
The user pressed *Test Model Environment* on Windows 64 after setting the
repository path and `Network.pt`. There is no Network.pt field: the second
preference is *Model Python*, and the add-on executes that file.

### Root Cause / Outcome
Setup: *Model Python* pointed at `Network.pt`. Windows cannot execute it and
reports WinError 193. The add-on shared some blame: it only checked
`is_file()`, the tooltip and README showed Linux paths only, and the raw OS
error reached the user.

### Fix / Implementation Detail
- `pipeline.check_model_python`: the file must exist, be named `python*`, end
  in `.exe` on Windows and be executable elsewhere. The message names the
  wrong file and gives the platform's example path.
- OS errors from starting the runner become `PipelineError("Could not start Model Python ...")`.
- `run_blocking` also uses `CREATE_NO_WINDOW` (no console flash on Windows).
- Tooltip and README show Windows and Linux/macOS paths and say the network is
  found through the repository path.

### Checklist
- [x] Tests: Network.pt rejected on win32/linux, python.exe accepted on win32,
      non-executable `python` rejected on linux, start failure is readable
- [x] Suites on bpy 4.5 / 5.0 / 5.2 (67 passed) and the model env (52 passed)
- [ ] On the user's Windows machine: the same mistake now shows the new message

---

## [2026-09-26] Stiff arms and hands

**Type**: `investigation`
**Status**: `closed` (the original publisher confirmed it is the model's true behavior; no fix)

### Context
User report: legs move naturally, upper body (arms, hands) is very stiff.
3 m/s, control strength 0.56, All Bones, Neutral. Generate reported
"Check vs model: 0.00 mm, 0.0000 deg", so Blender shows exactly the model
output. User hypothesis: something in our setup triggers a problem in the
original model; compare with the original demo.

### Method
Arm-swing metrics on the 23 model bones in root space: upper-arm swing (p95
angle from its mean direction), upper-arm angular speed, elbow range, hand
forward range, jitter (angular acceleration). References: the Geno mocap
clips in `Demos/_ASSETS_/Geno/Motions`, binned by speed. The original
Authoring program (Program.py + MotionController.py, unmodified) was stepped
headless in MANUAL mode with a stubbed speed slider.

### Ledger (one knob at a time)
| Run | Upper-arm swing | Angular speed |
|---|---|---|
| Mocap ~1 m/s | 33-54 deg | 73-130 deg/s |
| Mocap ~3 m/s | 52-73 deg | 200-330 deg/s |
| Original demo, its defaults (1 m/s, strength 2, BigSteps) | 6.3 deg | 19 deg/s |
| Original demo, user settings (3 m/s, 0.56, Neutral) | 8.9 deg | 46 deg/s |
| Our runner, user settings | 9.7 deg | 53 deg/s |

Ruled out (arm swing stayed 5-10 deg at 1-3 m/s):
- Our setup vs the demo (runner == demo, above).
- Frame rate: dt 1/30, 1/60, 1/144.
- Control strength 0.56 vs 2.0; style Neutral / BigSteps / Zombie.
- Prediction FPS 2, 3, 5, 10, 30.
- Codebook sampling: codes are already peaked (max prob 0.89 of 4 classes);
  `sample=True` or argmax codes add 1-2 deg only.
- Network iterations 0, 1, 3, 10.
- Damping in `Animate`: raw predictions inside the loop already have ~6 deg,
  the realized actor ~7 deg.
- Guidance: a static template and the true windowed guidance give the same
  prediction (teacher forcing).
- Velocity units: network velocity outputs match finite differences of its own
  positions (ratio 1.00).
- `Actor(..., True)` in the Biped demo: the flag is ignored by `Actor.Start`.

### Root Cause
The network under-predicts arm swing and the closed loop compounds it.
Teacher forcing (true mocap state, trajectory and guidance as input) gives
0.3-0.6x of the true 0.5 s upper-arm swing. Fed its own output 10 times a
second, it settles at ~15% of natural swing with nearly straight elbows.
The same happens in the original demo, so it is the model's behavior, not the
Blender conversion or our setup.

### Mitigation tried (removed, never committed)
Feeding the network arm velocities amplified x2 relative to the chest before
each prediction gives mocap-like arms (43.6 deg, 128 deg/s, elbow range 54
deg, jitter 1107 deg/s^2 vs mocap 1087-1698) in the demo, where the start turn
kicks the arms. In the runner (no start turn) the same gain, and even x3-x4 at
1 m/s, stays stiff: the loop is bistable, the gain sustains a swing but cannot
start one. At 3 m/s x3-x4 reaches 23-31 deg. Reverted: not reliable enough.

### Decision
The user checked with the original publisher: stiff arms are the model's true
result. Option 3 (accept and document) was taken. The arm-velocity experiment
was removed; the add-on applies no arm correction (README Limitations).
Options kept on record in case this is revisited: an arm-swing assist that
regulates the fed-back arm velocity toward a speed-dependent amplitude, or a
procedural secondary swing phase-locked to the legs.
Fingers are never driven (the model has 23 bones, hands end at the wrist).

### Assumptions Made
- The Geno mocap clips are valid natural references even if they are not the
  network's training data (the Biped demo says Style100).

---

## [2026-09-26] Quadruped model (Dog / Wolf), character pairing, network files

**Type**: `feature`
**Status**: `done` (awaiting user test in Blender)

### Context
User request: remove the stiff-arm experiment (confirmed model behavior), and
add UI options to pair the armature with a character and pick the networks
manually, so the add-on can switch to the quadruped model (the dog demo works
well).

### Implementation Detail
- `middleware/models.py`: registry shared by Blender and the runner. Biped =
  Demos/Authoring + Geno; Quadruped = Demos/Locomotion/Quadruped + Dog, Wolf.
  Holds default networks, root topology, iterations (3 / 1), input sizes
  (15 J + 96 / 9 J + 96), max speed, facing and goal-controller support.
- Pairing: `Calibration.character`; `pipeline.require_calibration` refuses an
  armature paired with another character. Profiles carry `model`, `character`,
  `root_topology`; shipped `dog_profile.json`, `wolf_profile.json` come from the
  runner (`ai4a_runner.py profile --model QUADRUPED --character dog|wolf`).
- Missing end sites: `*Site` leaf bones may be absent (Dog.glb has no HeadSite).
  Calibration maps them to None, `to_ai4a` returns NaN rows,
  `features.complete_pose` fills them (parent offset from the reference, or the
  rigid fit when parentless). Geno has no `*Site` bones, so it stays strict.
- Root: `compute_root` follows RootModule QUADRUPED (hips → neck, horizontal).
- Runner: boots the demo's Program in MANUAL mode with a module-local headless
  `AI4Animation.Standalone` (no camera/IO), keeps only the chosen character,
  redirects `torch.load` by lowercase name (demo asks for PostProcessor.pt, file
  is Postprocessor.pt) and to user overrides, then replaces `Control` with a
  port that reads `Command` (virtual player) and `Action` (Sit/Stand/Lie). An
  action also sets the desired speed to 0 so keyed actions trigger while
  following a path.
- Network overrides: `meta.model.network` / `postprocessor`; `check_networks`
  compares input sizes with the controller and names the problem.
- Exchange v3: `meta.model {type, character, network, postprocessor}`; missing =
  Biped/Geno (older requests stay valid). Quadruped needs controller STICK,
  facing MOVE, styles in Auto/Sit/Stand/Lie.
- Blender: Model (Biped/Quadruped) and Character enums, Networks sub-panel,
  per-character import (FBX only for Geno), "Paired with X" status, Action
  instead of Style for the quadruped, facing/controller/custom styles hidden.
  Switching model resets styles, network files and iterations.

### Measurements
- Runner, Dog, stick command 0 / 0.7 / 2.0 / 4.0 m/s: Idle / Walk / Trot /
  Canter, 0.88 / 2.12 / 4.41 m/s in the demo loop; along a path at 2 m/s:
  2.01 m/s, reached the end, then sat (hips 0.453 → 0.098 m).
- `compute_root` vs RootModule(QUADRUPED) on D1_008_KAN01_001: ≤ 1e-5 m.
- Blender 4.5, glTF import + pairing: residual Dog 0.0002 mm (HeadSite
  missing, filled), Wolf 0.007 mm. Curve at 2 m/s: 2.05 m/s for both; bake check
  ≤ 0.01 mm, ≤ 0.0003 deg. Keyed Sit lowers the hips.
- Start-up: the demo's first prediction uses Sit guidance, so the hips dip
  0.47 → 0.33 m for ~0.2 s. Kept (demo behavior), documented.

### Checklist
- [x] Middleware tests (quadruped root, missing end site, complete_pose, exchange v3)
- [x] Parity tests (RootModule quadruped, profiles, path speed Dog/Wolf, Sit, network mismatch)
- [x] Blender tests (panels, model switch, overrides, pairing, Dog/Wolf e2e) on bpy 4.5 / 5.0 / 5.2
- [ ] User test in an interactive Blender on Windows

### Assumptions Made
- "Animating network manually" = choosing the network `.pt` files manually
  (empty = the demo's). Asked-for scope did not include training.
- Keyed actions should stop the character first; in the demo the player
  releases the stick before pressing the action button.
