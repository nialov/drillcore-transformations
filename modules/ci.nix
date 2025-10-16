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
  flake.actions-nix =
    { config, ... }:
    {
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
            # uv-pytest = {
            #   steps = baseNixSteps ++ [
            #     {
            #       run = ''
            #         nix run .#fhs -- -c 'uv sync --all-extras'
            #       '';
            #     }
            #     {
            #       run = ''
            #         nix run .#fhs -- -c 'uv run pytest -v'
            #       '';
            #     }

            #   ];
            # };
          };
        };
      };
    };
}
