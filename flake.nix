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

    fxpmath = {
      url = "github:Integrated-Systems-Neuroengineering/fxpmath?ref=master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Branch pinned to 1e3a114c (Christopher's working commit) + build.py/flake.nix grafted on top,
    # plus cherry-picks of f88495f and 26466f3 for L6m refractory/dual-synapse support.
    # e88e660: add packages.wheel output; fix Cython 3.x DmaMethodNormal compat.
    # 5bafaf3: fix wheel build (pip wheel --no-build-isolation; python -m build silently fails).
    hs-bridge = {
      url = "git+ssh://git@github.com/Integrated-Systems-Neuroengineering/hs_bridge?rev=5bafaf3a8d061ec68fe0b7b9fa90fd1aa2601604";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, devshell, poetry2nix,
              fxpmath, hs-bridge, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ devshell.overlays.default ]; };
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        adxdma = hs-bridge.packages.${system}.adxdma;

        # L6m-testing-suite branch: 181f8a86 + delayed synapse support (is_delayed() getter).
        connectome-utils-src = builtins.fetchGit {
          url = "https://github.com/Integrated-Systems-Neuroengineering/connectome_utils.git";
          rev = "12ec6bf1ad6f163e166418b58661739ab5a1ee01";
          allRefs = true;
        };

        nvidiaNames = [
          "nvidia-cublas-cu12" "nvidia-cuda-cupti-cu12" "nvidia-cuda-nvrtc-cu12"
          "nvidia-cuda-runtime-cu12" "nvidia-cudnn-cu12" "nvidia-cufft-cu12"
          "nvidia-cufile-cu12" "nvidia-curand-cu12" "nvidia-cusolver-cu12"
          "nvidia-cusparse-cu12" "nvidia-cusparselt-cu12" "nvidia-nccl-cu12"
          "nvidia-nvjitlink-cu12" "nvidia-nvshmem-cu12" "nvidia-nvtx-cu12"
        ];

        overrides = p2n.defaultPoetryOverrides.extend (final: prev:
          (builtins.listToAttrs (map (n: {
            name = n;
            value = prev.${n}.overridePythonAttrs (_: { dontAutoPatchelf = true; });
          }) nvidiaNames)) // {
          # For git/path dependencies, override src with the pre-fetched flake input
          # so nix doesn't try to fetch from git/path inside the sandbox.
          connectome-utils = prev.connectome-utils.overridePythonAttrs (old: {
            src = connectome-utils-src;
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

          # PyPI torch 2.10.0 is a full CUDA wheel; fetch the CPU-only build from
          # the pytorch whl index so we don't pull in the entire CUDA stack.
          torch = prev.torch.overridePythonAttrs (_: {
            src = pkgs.fetchurl {
              url = "https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl";
              hash = "sha256-ooD/rqe5yCjgwbmzvVAtm2pknclBaZe2m4RUS9Rp8hU=";
            };
          });

          torchvision = prev.torchvision.overridePythonAttrs (_: {
            src = pkgs.fetchurl {
              url = "https://download.pytorch.org/whl/cpu/torchvision-0.25.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl";
              hash = "sha256-f4USRaJod0N0IVeYjtnELG4xK5W75s/KyefQ0MKK5i8=";
            };
          });

          # jaal is not in nixpkgs; if it fails to build add an override here,
          # e.g. fetching it from PyPI with buildPythonPackage / fetchPypi.
        });
      in {
        packages.default = p2n.mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python310;
          preferWheels = true;
          inherit overrides;
        };

        packages.fpga = p2n.mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python310;
          extras = [ "fpga" ];
          preferWheels = true;
          inherit overrides;
        };

        packages.wheel = pkgs.runCommandNoCC "hs-api-wheel" {
          src = pkgs.lib.cleanSource ./.;
          nativeBuildInputs = [
            pkgs.python310
            pkgs.python310Packages.poetry-core
            pkgs.python310Packages.build
          ];
        } ''
          cp -r "$src/." .
          chmod -R +w .
          python -m build --wheel --no-isolation
          mkdir -p "$out"
          cp dist/*.whl "$out/"
        '';

        devShells.default = pkgs.devshell.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python310;
              groups = [ "main" "apps" "dev" ];
              extras = [ "fpga" ];
              preferWheels = true;
              inherit overrides;
              editablePackageSources = {
                hs-api = ./.;
                fxpmath = fxpmath;
              };
            })
            pkgs.metis
            adxdma
          ];
        };
      }
    );
}
