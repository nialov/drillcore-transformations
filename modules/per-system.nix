(
  { inputs, ... }:

  {
    perSystem =
      {
        config,
        system,
        pkgs,
        lib,
        ...
      }:
      let
        mkNixpkgs =
          nixpkgs:
          import nixpkgs {
            inherit system;
            overlays =

              [
                inputs.nix-extra.overlays.default
              ];
            config = {
              allowUnfree = true;
            };
          };

      in
      {
        _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
        devShells = {
          default = pkgs.mkShell {
            buildInputs = lib.attrValues { inherit (pkgs) fhs; };
            shellHook = config.pre-commit.installationScript;
          };

        };

        pre-commit = {
          check.enable = true;
          settings.hooks = {
            nixfmt.enable = true;
            nbstripout.enable = true;
            ruff.enable = true;
            shellcheck.enable = true;
            statix.enable = true;
            deadnix.enable = true;
          };

        };
        legacyPackages = pkgs;
      };

  }
)
