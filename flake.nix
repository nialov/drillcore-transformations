{
  description = "Description for the project";

  inputs = {
    nix-extra = {
      url = "github:nialov/nix-extra";
    };
    actions-nix = {
      url = "github:nialov/actions.nix";
    };
    import-tree = {
      url = "github:vic/import-tree";
    };
    nixpkgs.follows = "nix-extra/nixpkgs";
    flake-parts.follows = "nix-extra/flake-parts";
    git-hooks.follows = "nix-extra/git-hooks";
  };

  outputs =
    inputs:
    let
      flakePart = inputs.flake-parts.lib.mkFlake { inherit inputs; } (
        { inputs, ... }:
        {
          systems = [ "x86_64-linux" ];
          imports = [
            # ./per-system.nix
            (inputs.import-tree ./modules)
          ];
        }
      );

    in
    flakePart;

}
