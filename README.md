# skill2policy

## Cloning

This repository uses git submodules. A plain `git clone` will leave the `third_party/` directories empty.

**First-time clone:**

```bash
git clone --recurse-submodules https://github.com/<your-org>/skill2policy.git
```

**Already cloned without `--recurse-submodules`:**

```bash
git submodule update --init --recursive
```

### Third-party dependencies (`third_party/`)

| Directory | Repository |
|---|---|
| `vjepa2` | https://github.com/facebookresearch/vjepa2 |
| `HACO_RELEASE` | https://github.com/dqj5182/HACO_RELEASE |
| `HaWoR` | https://github.com/ThunderVVV/HaWoR |
| `dex-retargeting` | https://github.com/dexsuite/dex-retargeting |

### Updating submodules

To pull the latest commits from all submodules:

```bash
git submodule update --remote --recursive
```