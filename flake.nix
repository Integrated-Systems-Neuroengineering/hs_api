{
  description = "hs_api - HiAER-Spike API";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    connectome-utils = {
      url = "github:Integrated-Systems-Neuroengineering/connectome_utils?ref=marchChange";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fxpmath = {
      url = "github:Integrated-Systems-Neuroengineering/fxpmath?ref=master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hs-bridge = {
      url = "git+ssh://git@github.com/Integrated-Systems-Neuroengineering/hs_bridge?ref=marchChange";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, devshell, poetry2nix,
              connectome-utils, fxpmath, hs-bridge, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ devshell.overlays.default ]; };
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        adxdma = hs-bridge.packages.${system}.adxdma;

        overrides = p2n.defaultPoetryOverrides.extend (final: prev: {
          # For git dependencies, override src with the pre-fetched flake input
          # so nix doesn't try to fetch from git inside the sandbox.
          connectome-utils = prev.connectome-utils.overridePythonAttrs (old: {
            src = connectome-utils;
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.poetry-core ];
          });
          fxpmath = prev.fxpmath.overridePythonAttrs (old: {
            src = fxpmath;
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.poetry-core ];
          });
          hs-bridge = prev.hs-bridge.overridePythonAttrs (old: {
            src = hs-bridge;
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.poetry-core final.cython final.setuptools ];
            buildInputs = (old.buildInputs or []) ++ [ adxdma ];
          });

          fbpca = prev.fbpca.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.setuptools ];
          });

          pymetis = prev.pymetis.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.setuptools final.pybind11 ];
            buildInputs = (old.buildInputs or []) ++ [ pkgs.metis ];
          });

          absl-py = prev.absl-py.overridePythonAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ final.hatchling ];
          });

          nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overridePythonAttrs (_: {
            dontAutoPatchelf = true;
          });

          # jaal is not in nixpkgs; if it fails to build add an override here,
          # e.g. fetching it from PyPI with buildPythonPackage / fetchPypi.
        });
      in {
        packages.default = p2n.mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python311;
          inherit overrides;
        };

        devShells.default = pkgs.devshell.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python311;
              groups = [ "main" "apps" "dev" "fpga" "docs" ];
              preferWheels = true;
              inherit overrides;
            })
            pkgs.metis
            adxdma
          ];
        };
      }
    );
}
