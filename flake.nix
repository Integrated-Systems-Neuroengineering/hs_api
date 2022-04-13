{
  description = "Flake for reproducible environments with poetry";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixpkgs-unstable;
    utils.url = github:gytis-ivaskevicius/flake-utils-plus;
  };

  outputs = inputs @ { self, nixpkgs, utils, ... }: utils.lib.mkFlake {
    inherit self inputs;

    outputsBuilder = channels: {

      devShell = channels.nixpkgs.mkShell {
        name = "poetry-flake";
        buildInputs = with channels.nixpkgs; [
          stdenv.cc.cc.lib
          #libz
          zlib
          zstd
          # Change your python version here
          (python39.withPackages (pp: with pp; [
            poetry
          ]))
          #LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
          # Add non-python packages here

        ];
        LD_LIBRARY_PATH = "${channels.nixpkgs.stdenv.cc.cc.lib}/lib:${channels.nixpkgs.zlib}/lib";#specify the location of all the .so files the python packages will need to find.
        shellHook = ''
          python -m venv .venv
          if [[ ! -f "pyproject.toml" ]]; then
            poetry init -n
          fi
          poetry env use .venv/bin/python
          poetry install --no-root
          source ./.venv/bin/activate
        '';

      };

    };
  };
}
