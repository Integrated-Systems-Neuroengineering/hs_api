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

`hs_bridge` is an optional dependency in the `fpga` group. To include it, select the
`fpga` shell and pass `--impure` (required because the flake reads `libadxdma.so`
directly from the host):

```bash
nix develop .#fpga --impure
```

On RHEL8 with nix-portable:

```bash
./scripts/nix-develop.sh .#fpga --impure
```

## Using nix-portable on RHEL8

If you don't have Nix installed system-wide, [nix-portable](https://github.com/DavHau/nix-portable)
lets you run Nix as a regular user. Download it to `~/nix-portable`, then use the
wrapper script in this repo instead of invoking nix-portable directly:

```bash
./scripts/nix-develop.sh          # default shell
./scripts/nix-develop.sh .#fpga   # FPGA shell (requires libadxdma + --impure)
```

The script handles all RHEL8 quirks automatically:

- Forces the **bwrap** runtime (correct choice on RHEL8; the nix runtime fails without
  unprivileged mount namespaces).
- Injects your user entry into `/etc/passwd` inside the container so SSH key lookup
  works for domain accounts (AD/Winbind). Safe on plain `/etc/passwd` systems too —
  it's a no-op if your user is already there.
- Defaults `NP_LOCATION` to `/local_disk/nix-$USER` so the nix store lives on local
  disk rather than NFS (nix builds fail on NFS due to file-attribute restrictions
  inside the bwrap user namespace).

If your home directory is already on local disk, or you want to store the nix store
elsewhere, override before running:

```bash
NP_LOCATION=/some/other/path ./scripts/nix-develop.sh
```

### Manual invocation

If you need to run nix-portable directly (outside the repo, or for debugging):

```bash
export NP_RUNTIME=bwrap
export NP_LOCATION=/local_disk/nix-$USER
export NP_BWRAP=/path/to/scripts/bwrap-wrapper.sh

~/nix-portable nix develop          # default shell
~/nix-portable nix develop .#fpga --impure  # FPGA shell
```

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
