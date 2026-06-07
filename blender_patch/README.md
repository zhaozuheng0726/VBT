# Native Cycles Patch

This directory stores the Blender/Cycles patch used by the native VBT path.

- Base Blender commit: `940972dfaa708826f3d16709d8d54ae61d32f5a5`
- Patch file: `native_cycles_vbt.patch`

Apply with:

```powershell
git apply native_cycles_vbt.patch
```

The standalone code that this patch relies on remains in `../cycles_native_vbt/`.
