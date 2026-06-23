#!/bin/bash
# Wrapper around nix-portable nix develop that handles RHEL/bwrap quirks automatically.
#
# Usage (from repo root or anywhere):
#   ./scripts/nix-develop.sh           # default shell
#   ./scripts/nix-develop.sh .#fpga    # FPGA shell (requires libadxdma + --impure)
#
# Environment you can override before running:
#   NP_LOCATION  — where the nix store lives (default: /local_disk/nix-$USER)
#                  must be on a local filesystem, not NFS
#   NIX_PORTABLE — path to the nix-portable binary (default: searches PATH then ~)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NP_RUNTIME=bwrap
export NP_BWRAP="$SCRIPT_DIR/bwrap-wrapper.sh"
export NP_LOCATION="${NP_LOCATION:-/local_disk/nix-$USER}"

NIX_PORTABLE="${NIX_PORTABLE:-}"
if [ -z "$NIX_PORTABLE" ]; then
    if command -v nix-portable &>/dev/null; then
        NIX_PORTABLE=nix-portable
    elif [ -x "$HOME/nix-portable" ]; then
        NIX_PORTABLE="$HOME/nix-portable"
    else
        echo "error: nix-portable not found. Download it to ~/nix-portable or put it on PATH." >&2
        exit 1
    fi
fi

exec "$NIX_PORTABLE" nix develop "$@"
