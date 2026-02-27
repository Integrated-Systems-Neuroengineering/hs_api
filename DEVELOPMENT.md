# Development Guide

## Clean Install (Nix)

Requires [Nix](https://nixos.org/download/) with flakes enabled.

```bash
git clone https://github.com/Integrated-Systems-Neuroengineering/hs_api.git
cd hs_api
nix develop
```

This drops you into a shell with all dependencies (including `connectome_utils` and
`fxpmath` from their `dev` branches). `hs_bridge` is not included by default — see
[FPGA hardware](#fpga-hardware) below.

## Clean Install (Poetry)

Requires [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/Integrated-Systems-Neuroengineering/hs_api.git
cd hs_api
poetry install
poetry shell
```

## Local Co-Development

If you are actively modifying `connectome_utils`, `fxpmath`, or `hs_bridge` alongside
`hs_api`, you want your local changes to reflect immediately without reinstalling from
git.

Clone the repos as siblings:

```bash
git clone https://github.com/Integrated-Systems-Neuroengineering/hs_api.git
git clone https://github.com/Integrated-Systems-Neuroengineering/connectome_utils.git
git clone https://github.com/jfrank8/fxpmath.git
# optional, for FPGA work:
git clone https://github.com/Integrated-Systems-Neuroengineering/hs_bridge.git
```

### With Poetry

Install `hs_api` normally, then override the git-installed packages with local
editable installs:

```bash
cd hs_api
poetry install
poetry run pip install -e ../connectome_utils -e ../fxpmath
# optional:
poetry run pip install -e ../hs_bridge
```

Changes to the sibling repos are now immediately visible inside the `hs_api`
environment. Re-run the `pip install -e` commands if you switch branches in those
repos.

### With Nix

Use `--override-input` to point the flake at your local checkouts:

```bash
nix develop \
  --override-input connectome-utils path:../connectome_utils \
  --override-input fxpmath path:../fxpmath
```

## FPGA Hardware

`hs_bridge` is an optional dependency in the `fpga` group. To include it:

**Poetry:**
```bash
poetry install --with fpga
poetry run pip install -e ../hs_bridge   # if co-developing
```

**Nix:**
```bash
nix develop --override-input hs-bridge path:../hs_bridge
```

## Tracking `master` Instead of `dev`

The Nix flake and `pyproject.toml` both default to the `dev` branch of each
dependency. To use `master`:

**Nix** — use `--override-input` at the CLI:
```bash
nix develop \
  --override-input connectome-utils github:Integrated-Systems-Neuroengineering/connectome_utils?ref=master \
  --override-input fxpmath github:jfrank8/fxpmath?ref=master \
  --override-input hs-bridge github:Integrated-Systems-Neuroengineering/hs_bridge?ref=master
```

**Poetry** — edit the `branch` field in `pyproject.toml` and run `poetry lock`.
