{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides mkPoetryEnv;

        python = pkgs.python310;
        projectDir = ./.;
        overrides = defaultPoetryOverrides.extend
          (self: super: {
            juliacall = super.juliacall.overridePythonAttrs
            (
              old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
              }
            );
            juliapkg = super.juliapkg.overridePythonAttrs
            (
              old : {
                buildInputs = (old.buildInputs or [ ]) ++ [super.setuptools];
              }
            );
          });
          groups = ["dev" "test"];
      in
      {
        packages = {
          myapp = mkPoetryApplication {
            inherit python projectDir overrides;
          };
          default = self.packages.${system}.myapp;
        };

        /*devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.myapp ];
          packages = [ pkgs.poetry ];
        };*/
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.poetry
            (mkPoetryEnv {
              inherit python projectDir overrides groups;
              editablePackageSources = {
                compromise-wrapper = "${projectDir}";
              };
            })
          ];
        };

      });
}
