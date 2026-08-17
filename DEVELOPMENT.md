# Development Guide

## Building the Documentation

The Sphinx docs (`doc/source`) build as a Nix flake package, using a poetry
environment scoped to the `main` + `docs` dependency groups (no FPGA/hardware
deps required):

```bash
nix build .#docs --impure
```

`--impure` is required due to a pre-existing pure-eval issue elsewhere in the
flake (unrelated to the docs build itself) — it is not docs-specific.

The built site is written to `result/` (`result/index.html` is the entry
point). To build to a named output instead of the default `result` symlink:

```bash
nix build .#docs --impure -o result-docs
```

The build passes `-D plot_gallery=0` to `sphinx-build`: the gallery examples
under `webexamples/` require real HiAER-Spike hardware/data to execute, so the
docs render them from source without running them, rather than regenerating
their outputs.

Note that `docs` is a separate poetry group, not included in the default
`nix develop` shell (which covers `main`/`apps`/`dev`) — `nix build .#docs` is
the supported way to build the site.
