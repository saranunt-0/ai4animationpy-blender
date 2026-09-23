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
