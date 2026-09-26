# Architecture: AI4Animation in Blender

```
Blender (bundled Python, NumPy only)            Model environment (Python >= 3.12, torch)
------------------------------------            -----------------------------------------
ui.py / operators.py
   │
pipeline.py ── build_request ──▶ request.npz ──▶ runner/ai4a_runner.py
   │              ▲                                 AI4Animation(MANUAL) +
blender_io.py     │ middleware (pure NumPy)          Biped: Demos/Authoring MotionController
 (only bpy I/O)   │  conventions / rig / models /    Quadruped: Demos/Locomotion/Quadruped Program
   │              │  features / exchange             PathPlanner3D; fixed dt, ~60 Hz
   └── bake ◀── apply_result ◀── result.npz ◀──────┘
```

## Boundaries

| Layer | Knows about | Must not know about |
|---|---|---|
| `middleware/` | NumPy, both coordinate conventions | bpy, torch |
| `blender_io.py` | bpy, mathutils | model maths |
| `runner/` | ai4animation, torch, the demos in `models.py` | Blender conventions |
| exchange `.npz` | AI4Animation world space only | Blender |

## Key decisions

1. **Subprocess instead of in-process.** `setup.py` requires Python >= 3.12.12 and
   torch. Blender ships its own Python (3.11 in 4.x/5.0) without torch. The
   subprocess keeps both environments untouched.
2. **Calibration from a pose correspondence, not from the rest pose.** The
   glTF-imported Geno rest pose lies on its back; only the posed skeleton
   matches. Offsets are per-bone rotations, and placement is removed by
   rigid alignment (Kabsch).
3. **Blender keys are `matrix_basis`**, computed by the middleware with Blender's
   FK rule (`pose = parent_pose @ rest_rel @ basis`). After baking, Blender
   is re-evaluated and compared with the model (self-check in the operator report).
4. **Reuse the demo controller as-is.** The runner only wraps it with shims
   (CPU `map_location`, cwd for `Guidances/`, in-place initial pose).
5. **Two controllers, chosen automatically.** Goal (Authoring) for path following
   with natural facing, because it measured best. Joystick (Biped `Control()`) with a
   virtual player (reference tracking) for facing control and keyed virtual sticks.
6. **One registry for models and characters** (`middleware/models.py`), read by
   both sides: demo folder, default networks, root topology, input sizes,
   supported controls. Adding a model means a registry entry, a controller
   factory in the runner and a shipped profile.
7. **Pairing is part of calibration.** The calibration stores the character it
   was made for; Generate checks it against the selected character, so a Geno
   armature can never be driven by the Dog network (bone counts would differ
   anyway, but the message is clearer).
8. **Quadruped: the demo's Program, not a rewrite.** Its `Start`, `Predict`,
   `Animate` and IK run unchanged; only `Control` is replaced by a port that
   reads the virtual player's command instead of the gamepad.
