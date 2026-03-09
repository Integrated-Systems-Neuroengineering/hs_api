{
  description = "hs_api - HiAER-Spike API";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
      url = "github:Integrated-Systems-Neuroengineering/connectome_utils?ref=dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fxpmath = {
      url = "github:Integrated-Systems-Neuroengineering/fxpmath?ref=master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hs-bridge = {
      url = "git+ssh://git@github.com/Integrated-Systems-Neuroengineering/hs_bridge?ref=dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, devshell, poetry2nix,
              connectome-utils, fxpmath, hs-bridge }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            devshell.overlays.default
            (final: prev: {
              python311 = prev.python311.override {
                packageOverrides = pyFinal: pyPrev: {
                  sphinx = pyPrev.sphinx.overridePythonAttrs (_: { disabled = false; });
                };
              };
            })
          ];
        };
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        overrides = p2n.defaultPoetryOverrides.extend (final: prev: {
          # Inject the local libraries built from their own flakes.
          # poetry2nix matches these by the normalized package name
          # (underscores replaced with hyphens, lowercased).
          connectome-utils = connectome-utils.packages.${system}.default;
          fxpmath = fxpmath.packages.${system}.default;
          hs-bridge = hs-bridge.packages.${system}.default;

          # jaal is not in nixpkgs; if it fails to build add an override here,
          # e.g. fetching it from PyPI with buildPythonPackage / fetchPypi.
        });
      in {
        packages.default = p2n.mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python311;
          inherit overrides;
          # To include optional dependency groups, e.g.:
          #   groups = [ "main" "apps" "fpga" ];
          # The "fpga" group pulls in hs-bridge (already overridden above).
        };

        devShells.default = pkgs.devshell.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python311;
              groups = [ "main" "apps" "dev" ];
              inherit overrides;
            })
          ];
        };
      }
    );
}
