{
  description = "Mel-Band RoFormer vocal model (Python package)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (_final: prev: {
              buildPythonPackage =
                if prev ? buildPythonPackage
                then prev.buildPythonPackage
                else prev.python3Packages.buildPythonPackage;
            })
          ];
        };
        lib = pkgs.lib;

        pyPkgs = pkgs.python3Packages;

        melBandRoformerCheckpoint = pkgs.fetchurl {
          name = "MelBandRoformer.ckpt";
          url = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt";
          hash = "sha256-hyAfTTGvtbx5mTIw/ElEaRhCVXTbSMAcQF5E82XHVZ4=";
        };

        getPy = name:
          if builtins.hasAttr name pyPkgs
          then builtins.getAttr name pyPkgs
          else null;

        firstNonNull = xs: lib.findFirst (x: x != null) null xs;

        mlCollections = firstNonNull [
          (getPy "ml-collections")
          (getPy "ml_collections")
        ];

        rotaryEmbeddingTorch = firstNonNull [
          (getPy "rotary-embedding-torch")
          (getPy "rotary_embedding_torch")
        ];

        torch = firstNonNull [
          (getPy "torch")
          (getPy "pytorch")
        ];

        melBandRoformerVocalModel = pkgs.buildPythonPackage {
          pname = "mel-band-roformer-vocal-model";
          version = "0.1.0";
          format = "setuptools";

          src = ./.;

          nativeBuildInputs = with pyPkgs; [
            setuptools
            wheel
          ] ++ [
            pkgs.makeWrapper
          ];

          nativeCheckInputs = with pyPkgs; [
            pytestCheckHook
          ];

          doCheck = true;
          pytestFlagsArray = [ "-q" ];

          propagatedBuildInputs =
            lib.filter (x: x != null) (with pyPkgs; [
              numpy
              packaging
              tqdm
              soundfile
              librosa
              beartype
              einops
              pyyaml

              mlCollections
              rotaryEmbeddingTorch
              torch
            ]);

          postFixup = ''
            if [ -x "$out/bin/mel-band-roformer-separate" ]; then
              wrapProgram "$out/bin/mel-band-roformer-separate" \
                --set-default MEL_BAND_ROFORMER_CKPT "${melBandRoformerCheckpoint}"
            fi
          '';

          # Keep checks lightweight; importing the top-level package avoids importing torch/librosa.
          pythonImportsCheck = [ "mel_band_roformer_vocal" ];

          passthru = {
            inherit melBandRoformerCheckpoint;
          };

          meta = with lib; {
            description = "Mel-Band RoFormer vocal separation model + inference utilities";
            platforms = platforms.all;
          };
        };
      in
      {
        packages.default = melBandRoformerVocalModel;
        packages.checkpoint = melBandRoformerCheckpoint;

        apps.default = flake-utils.lib.mkApp {
          drv = melBandRoformerVocalModel;
          exePath = "/bin/mel-band-roformer-separate";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (_: [ melBandRoformerVocalModel ]))
          ];
          shellHook = ''
            export MEL_BAND_ROFORMER_CKPT="${melBandRoformerCheckpoint}"
          '';
        };
      });
}
