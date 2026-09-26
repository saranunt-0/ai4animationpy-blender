# AI4Animation for Blender

A Blender add-on that drives the **Authoring** model (`Demos/Authoring`: path
following + style guidance on the Geno rig) from Blender data and bakes the
generated motion back onto the armature.

The core of this folder is a **middleware** that converts Blender data into
exactly what the model was trained on, and converts the model output back
into Blender bone channels. It is pure NumPy and is tested against the real
AI4Animation code and against real Blender (4.5 LTS, 5.0 and 5.2 LTS).

```
Blender scene ──bpy──▶ blender_io ──▶ middleware ──▶ request.npz ──▶ runner (model env: torch)
     ▲                                                                   │  MotionController (MANUAL mode)
     └──── F-curves ◀── blender_io ◀── middleware ◀── result.npz ◀──────┘
```

* **Blender side** (`ai4animation_blender/`): UI, extraction and baking. Runs in
  Blender's bundled Python and needs only NumPy.
* **Model side** (`ai4animation_blender/runner/ai4a_runner.py`): runs in *your*
  AI4Animation Python environment (torch, ai4animation). It is started as a
  subprocess, because `setup.py` requires Python ≥ 3.12 + torch, while Blender
  ships its own Python without torch.
* **Exchange files** hold AI4Animation-space data only (Y-up, +Z forward, meters).
  No Blender conventions cross this boundary.

## Install

1. **Model environment.** Use the environment from the main
   [installation guide](https://facebookresearch.github.io/ai4animationpy/getting-started/installation/).
   Note the path of its Python executable (`which python` inside the env).
   A CUDA build of torch uses the GPU automatically (e.g. an RTX 3050); CPU works too,
   at about 20 ms per simulation step.
2. **Add-on.** Pick one:
   * *From the checkout (recommended):* symlink `Blender/ai4animation_blender` into
     Blender's add-on folder (e.g. `~/.config/blender/4.5/scripts/addons/`). The
     repository path is then detected automatically.
   * *From a zip:* `cd Blender && zip -r ai4animation_blender.zip ai4animation_blender -x "*/__pycache__/*"`,
     then *Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk*.
3. **Preferences ▸ Add-ons ▸ AI4Animation.** Set *AI4AnimationPy Repository*
   and *Model Python*, then press **Test Model Environment**. This boots the real
   MotionController once and reads bones, reference pose and styles.

## Use (View3D ▸ Sidebar ▸ AI4A)

1. **Import Geno (glb)**. Imports `Demos/_ASSETS_/Geno/Model.glb` and
   **calibrates** it immediately (see below). Using your own import of the
   same rig? Press **Calibrate** *before* posing it.
2. **Control** (see [Control design](#control-design-joystick-input)). *Create Helpers*
   builds the objects each option needs.
   * **Movement**: *Curve* (e.g. a Bezier path, resampled every *Spacing*),
     *Planner* (the demo's voxel A* from *Start* to *Goal* around the meshes in
     *Obstacles*, as axis-aligned boxes), *Target* (follow an animated object), or
     **Joystick** (keyframe a virtual left stick).
   * **Facing**: *Movement* (default), *Look At* an object, *Object Axis* (an
     object's −Y), or *Right Stick* (a second virtual stick).
3. **Style**. Guidance style (default *Neutral*) and idle style (used when
   nearly standing still). Add *style keys* to switch style at a frame, or
   **Capture** the current pose as a custom style.
4. **Generate**. *Walk Speed* can be keyframed; with a joystick it is the speed at
   full deflection. The report shows a self-check: Blender is re-evaluated after
   baking and compared to the model output.

Outputs: an action on the armature, foot contacts as animated custom properties
(`ai4a_contact_LeftFoot`, …) and helpers `AI4A_Path`, `AI4A_Goal` and `AI4A_Root`.

## Control design (joystick input)

The research demos drive the same network (`Network.pt` is identical in
`Demos/Authoring` and `Demos/Locomotion/Biped`) in two ways:

| | Goal controller (`Demos/Authoring`) | Joystick controller (`Demos/Locomotion/Biped`) |
|---|---|---|
| Input | a goal transform moving along a path | left stick → velocity (`speed × clamp(stick, 1)`), right stick → facing |
| Trajectory | `SimulationObject.ControlFromTarget(goal)` | `SimulationObject.Control(position, direction, velocity)` |
| Facing | always the walking direction | independent: strafe, walk backwards, look at |

In Blender, the joystick is mimicked in two ways:

* **Virtual sticks.** A knob empty (sphere) inside a gate empty (circle, radius 1).
  The knob's offset is the stick deflection. It is constrained to the circle, so you
  keyframe it like pushing a thumbstick. *Stick up* is the gate's +Y; rotate the
  gate to change it. The mapping equals the gamepad demo (stick up = AI4Animation −Z = Blender +Y).
* **A virtual player.** For paths and targets, each sub-step computes the stick a
  player would push. A reference point moves along the path at the keyed speed
  (braking into the end), and `stick velocity = feed-forward + 2/s × offset`.

What decided the defaults (12 s runs at 1.2 m/s: S-curve, 90° corner, 1.2 m-wide hairpin):

* Raw stick velocities (the gamepad demo as-is) **cannot start walking below
  ~1 m/s** from standing, and cannot hold speeds below ~0.6 m/s. *Speed Assist*
  (default on) tracks the position the stick implies instead: 0.4–2.0 m/s start
  reliably and hold within about 0.05 m/s; 1 m/s for 5 s → 4.99 m. Turn it off to
  get the exact demo behavior.
* For following a path while facing the walking direction, the **goal controller
  is better**: max deviation 0.28–0.29 m on every path, versus 0.34–0.41 m for the
  best joystick tracker (cross-track feedback and higher gains were worse), with
  smoother root motion. **Controller: Auto** therefore uses Goal for
  paths/targets with *Facing = Movement*, and Joystick whenever facing is
  controlled or the movement is a virtual stick.
* Joystick facing works with this network: walking backwards (facing error
  2.3°, 1.00 m/s), strafing (1.8°), looking at an object while walking past (5.7°).

## Data conventions (what the middleware converts)

| | Blender | AI4Animation |
|---|---|---|
| Up / character forward | +Z / −Y | +Y / +Z |
| Units | Blender units × `unit_settings.scale_length` | meters |
| Bone matrices | armature space (`pose_bone.matrix`), armature may be scaled | world space (`Actor.Transforms`) |
| Bone frame axes | importer specific (Y along bone, roll heuristics) | Geno joint frames (≈ world aligned at rest) |
| Quaternions | (w, x, y, z) | (x, y, z, w) |
| Keys | `matrix_basis` = rest-relative local | — |

* Points and directions: `ai4a = C @ blender`, i.e. (x, y, z) → (x, z, −y).
* Scene objects (targets, empties) change frame convention too: `C @ M @ Cᵀ`,
  so an object's −Y (Blender facing) becomes the model's +Z.
* Bones use a per-bone **calibration**: `A = C(B_world) @ Offset[bone]`.

### Why calibration is necessary

Measured on the Geno rig in Blender 4.5/5.0:

* **glTF import:** the *rest* pose lies on its back (bind pose under a rotated
  `Armature` node). The importer compensates with a 90° pose rotation on
  `Hips`. Only the *posed* skeleton matches the model, and every bone's local
  axes differ from the model's joint frames.
* **FBX import:** the armature object is scaled 0.01, rest == pose, and axes differ again.

Positions survive all importers, but bone axes do not. The network reads each
bone's forward (Z) and up (Y) axes, so wrong axes give garbage input. A calibration
stores one constant rotation per model bone, computed from a pose where
Blender and the model agree. It is placement invariant (rigid alignment) and
rejects a wrong pose, a different rig or a unit mismatch with a readable error.
It is stored on the armature (`ai4a_calibration` custom property).

## Model inputs and where each comes from

Network: 441 inputs → 16 × 279 outputs. PostProcessor (contacts): 456 → 16 × 4.

| Input (all in the current root frame) | Size | Source when driven from Blender |
|---|---|---|
| bone positions | 23×3 | start pose (rest re-rooted at the path start, or the Blender pose via calibration), then autoregressive |
| bone forward axis (Z) | 23×3 | same; needs the per-bone calibration |
| bone up axis (Y) | 23×3 | same |
| bone velocities | 23×3 | zero, or finite difference of the Blender pose (frames s−1, s) |
| future root positions, x/z | 16×2 | goal controller: path/target → `ControlFromTarget`; joystick: sticks → `Control` |
| future root directions, x/z | 16×2 | walking direction, or the right stick / look-at / object axis |
| future root velocities, x/z | 16×2 | walk speed (goal: × control strength; joystick: virtual-player command) |
| guidance positions | 23×3 | style `.npz` (bone order == model order, verified) or a captured Blender pose |

Root definition (`RootModule`, biped, ground): hip x/z at y = 0, facing
= horizontal cross product of the hip/shoulder lines with up. Reimplemented in
`middleware/features.py` and checked against `RootModule` on real mocap.

## Verification

```bash
# middleware only (any Python with NumPy)
python -m pytest Blender/tests/test_middleware.py
# parity with ai4animation + runner (model environment)
/path/to/model/python -m pytest Blender/tests/test_parity_ai4animation.py
# inside Blender's Python (pip install bpy==4.5.* | 5.0.* | 5.2.*), full pipeline
AI4A_PYTHON=/path/to/model/python python -m pytest Blender/tests
```

Measured results (float32 limits):

| Check | Result |
|---|---|
| calibration residual, glTF and FBX import | < 0.001 mm |
| bake → Blender depsgraph → model, all frames, glTF / FBX | ≤ 1.4e-6 m, ≤ 3e-5° |
| moved / rotated armature (90° yaw + offset) | same as above; a ×2 scaled parent is rejected |
| `compute_root` vs `RootModule` (walk3_subject3) | ≤ 1e-5 m |
| captured guidance vs `GuidanceModule` | ≤ 1e-5 m |
| "Root Only" bone locations | rotations exact, positions within ~1 cm (feet up to 3.5 cm) |

**Bone Locations default: All Bones (exact).** Root Only keeps a clean
rotation-only rig, but its 2–3.5 cm foot drift shows up as visible foot sliding and
wrong contacts. The location keys All Bones needs are harmless on Geno (its bones
are not connected), so exact is the setting with fewer problems.

## Runner vs. standalone demo

The runner reuses `MotionController` and `PathPlanner3D` unchanged. It differs only where
Blender needs deterministic, offline baking:

* The joystick controller is the Biped demo's `Control()` bound onto the Authoring
  `MotionController`, so `Update`, `PredictSequence` and `Animate` are unchanged.
* Fixed time step, sub-stepped to about 60 Hz (the demo runs at display rate). Blender
  scenes at 24/25/30 fps otherwise get visible prediction blend jumps.
* The pose starts **at the path start**. In the demo, `Actor.Transforms` stays at the
  origin while the root starts at (−4, 0, −4), so the first frame teleports.
  The initial pose is written *in place*, because `LegIK` keeps views into `Actor.Transforms`.
* *Stop* at the path end (the demo ping-pongs; *Ping-Pong* is still available), with a
  backward tangent at the end so the goal keeps facing along the path.
* `torch.load(..., map_location=cpu)`. The shipped `.pt` files were pickled on CUDA,
  so the demo itself fails on CPU-only machines.
* Model rotations from `Rotation.Look(z, y)` are not orthonormal (up to 57° between
  y and z during start-up, 0.56° on average). The middleware re-orthonormalizes
  them (Z kept, as the network reads it) before Blender sees them.

## Limitations

* Rig: Geno only (the model was trained on it). Another rig would need retargeting,
  not just calibration.
* Obstacles are axis-aligned boxes (the planner's limitation). Rotated meshes use their bounds.
* Bone constraints, disabled rotation inheritance and connected bones on the model
  bones are reported as warnings; the exact bake assumes default inheritance.
* The armature object is assumed static during the bake (its placement at the start frame is used).
