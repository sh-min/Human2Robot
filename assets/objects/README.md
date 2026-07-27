# Object assets

Store geometry in one directory per `object_id`:

```text
assets/objects/mug/
├── visual.obj
└── collision.obj
```

Reference these files from `configs/objects/mug.yaml` with paths relative to
the config file:

```yaml
geometry:
  visual_mesh: ../../assets/objects/mug/visual.obj
  collision_mesh: ../../assets/objects/mug/collision.obj
  scale: [1.0, 1.0, 1.0]
```

Prefer a simplified watertight collision mesh for stable grasp simulation.
