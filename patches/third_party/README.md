# Third-party compatibility patches

These patches preserve local compatibility changes without repointing the
project's submodules to unpublished commits.

Apply them from the repository root after initializing submodules:

```bash
git -C third_party/E2FGVI apply ../../patches/third_party/e2fgvi-mmcv-free-inference.patch
git -C third_party/HaWoR apply ../../patches/third_party/hawor-optional-renderer.patch
```

- `e2fgvi-mmcv-free-inference.patch` adds torchvision-based inference fallbacks
  for environments without `mmcv-full`.
- `hawor-optional-renderer.patch` keeps MANO extraction usable when the
  auxiliary PyTorch3D renderer is unavailable.
