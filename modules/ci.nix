{ inputs, ... }:
let

  inherit (inputs.actions-nix.lib.steps)
    actionsCheckout
    DeterminateSystemsNixInstallerAction
    cachixCachixAction
    runNixFlakeCheck
    ;

  baseNixSteps = [
    actionsCheckout
    DeterminateSystemsNixInstallerAction
    {
      inherit (cachixCachixAction) uses;
      "with" = {
        name = "nialov";
      };
    }
  ];

in
{
  imports = [
    inputs.actions-nix.flakeModules.default
  ];
  flake.actions-nix = {
    pre-commit.enable = true;
    defaults = {
      jobs = {
        timeout-minutes = 60;
        runs-on = "ubuntu-latest";
      };
    };
    workflows = {
      ".github/workflows/main.yaml" = {
        on = {
          push = { };
          workflow_dispatch = { };
          pull_request = { };
        };
        jobs = {
          nix-flake-check = {
            steps = baseNixSteps ++ [
              runNixFlakeCheck
            ];
          };
          uv-pytest = {
            strategy.matrix.python-version = [
              "3.9"
              "3.10"
              "3.11"
              "3.12"
              "3.13"
            ];
            steps = [
              actionsCheckout
              {
                uses = "astral-sh/setup-uv@v7";
                "with".python-version = "\${{ matrix.python-version }}";
              }
              {
                name = "Run pytest";
                run = ''
                  uv run --frozen --all-extras pytest -v
                '';
              }

            ];
          };
        };
      };
    };
  };
}
