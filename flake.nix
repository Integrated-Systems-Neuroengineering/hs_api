{
  description = "hs_api - HiAER-Spike API";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    connectome-utils = {
      url = "github:Integrated-Systems-Neuroengineering/connectome_utils?ref=dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    fxpmath = {
      url = "github:jfrank8/fxpmath?ref=dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hs-bridge = {
      url = "github:Integrated-Systems-Neuroengineering/hs_bridge?ref=dev";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix,
              connectome-utils, fxpmath, hs-bridge }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
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
          python = pkgs.python310;
          inherit overrides;
          # To include optional dependency groups, e.g.:
          #   groups = [ "main" "apps" ];
          # The "fpga" group pulls in hs-bridge (already overridden above).
        };

        devShells.default = pkgs.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python310;
              inherit overrides;
            })
          ];
        };
      }
    );
}
