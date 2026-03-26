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

Use `--override-input` to point the flake at your local checkouts:

```bash
nix develop \
  --override-input connectome-utils path:../connectome_utils \
  --override-input fxpmath path:../fxpmath
```

## FPGA Hardware

`hs_bridge` is an optional dependency in the `fpga` group. To include it:

```bash
nix develop --override-input hs-bridge path:../hs_bridge
```

## Using nix-portable on RHEL8

If you don't have Nix installed system-wide, [nix-portable](https://github.com/DavHau/nix-portable)
lets you run Nix as a regular user. Download it and place the binary somewhere on your `PATH`
(e.g. `~/nix-portable`).

### Runtime selection

nix-portable auto-detects a container runtime. On RHEL8 it may fall back to the **nix** runtime,
which tries to create a private mount namespace directly — this fails because RHEL8 does not allow
unprivileged mount namespaces without a user namespace. **bwrap** (bubblewrap) handles this
correctly by creating a user + mount namespace together, and is available on RHEL8 by default.

Force bwrap by exporting `NP_RUNTIME=bwrap` in your shell profile:

```bash
echo 'export NP_RUNTIME=bwrap' >> ~/.bashrc
source ~/.bashrc
```

### Private dependency (hs_bridge)

`hs_bridge` is a private GitHub repository. The flake fetches it over SSH, so you need a GitHub
SSH key configured. Nix does not read your normal SSH config in all environments, so pass
`GIT_SSH_COMMAND` explicitly if needed:

```bash
GIT_SSH_COMMAND="ssh -F /dev/null" nix-portable nix develop
```

(`-F /dev/null` bypasses `/etc/ssh/ssh_config.d/05-redhat.conf` which may have permission
warnings that confuse git inside Nix's environment.)

### Full invocation on RHEL8

The `fpga` group requires `hs_bridge`, which links against `libadxdma` — the userspace
interface to the FPGA PCIe DMA kernel driver. Because the flake reads this library
directly from the host (`/usr/lib64/libadxdma.so.0.12.2`), `--impure` is required:

```bash
NP_RUNTIME=bwrap GIT_SSH_COMMAND="ssh -F /dev/null" ~/nix-portable nix develop --impure
```

Without `--impure`, nix will refuse to access paths outside the store. If the adxdma
driver is not installed on the host, `hs_bridge` will still import but DMA operations
will fail at runtime.

### Flake design notes

A few decisions in `flake.nix` exist specifically to work around build issues on this nixpkgs
version and toolchain:

- **`nixpkgs` pinned to `nixos-24.11`** — `nixos-unstable` has `sphinx 9.1.0` incorrectly
  marked incompatible with Python 3.11 in its package metadata, breaking the pip build hook.
- **`preferWheels = true`** — `flit-core 3.9.0` (shipped with nixos-24.11) does not support
  the PEP 639 string license format used by `click 8.3.1`. Using pre-built wheels from the
  poetry.lock bypasses the source build entirely.
- **`docs` group** — sphinx and its dependencies are included via pre-built wheels (`preferWheels = true`), bypassing the `flit-core` / PEP 639 source-build incompatibility.
- **`nvidia-cufile-cu12` override** — disables `autoPatchelf` for this CUDA package since
  the InfiniBand RDMA libraries it links against (`libmlx5`, `librdmacm`, `libibverbs`) are
  not present on non-RDMA machines.
- **`adxdma` stub derivation** — `hs_bridge` links against `libadxdma`, a proprietary
  vendor library for the FPGA PCIe DMA interface that is not in nixpkgs. The flake
  copies it from the host (`/usr/lib64/libadxdma.so.0.12.2`) into the nix store at
  evaluation time. This is an intentionally impure operation — the flake requires
  `--impure` and the adxdma kernel driver to be installed on the host.

## Updating the Lock File After Pushing to a Dependency

Nix pins exact git revisions in `flake.lock`. Pushing new commits to a dependency branch
(e.g. `hs_bridge`) does **not** automatically update what Nix uses — the lock file must be
refreshed manually:

```bash
nix flake update hs-bridge
```

On RHEL8 with nix-portable, `nix flake update` makes HTTPS calls to the GitHub API which
will fail with an SSL error unless nix is pointed at the system cert bundle. Create
`~/.config/nix/nix.conf` with:

```
ssl-cert-file = /etc/ssl/certs/ca-bundle.crt
extra-experimental-features = nix-command flakes
```

After that, `nix flake update` works without any extra flags.

This re-resolves the `hs-bridge` input to the latest commit on its configured branch and
writes the new revision into `flake.lock`. Commit the updated lock file so others get the
same version.

To update all inputs at once:

```bash
nix flake update
```

## Tracking `master` Instead of `dev`

The Nix flake defaults to the `dev` branch of each dependency. To use `master`:

Use `--override-input` at the CLI:
```bash
nix develop \
  --override-input connectome-utils github:Integrated-Systems-Neuroengineering/connectome_utils?ref=master \
  --override-input fxpmath github:jfrank8/fxpmath?ref=master \
  --override-input hs-bridge github:Integrated-Systems-Neuroengineering/hs_bridge?ref=master
```
