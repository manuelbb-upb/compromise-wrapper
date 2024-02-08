# Readme

This repository contains a Python module `compromise-wrapper`.
It is a very experimental wrapper around [Compromise.jl](https://github.com/manuelbb-upb/Compromise.jl) and things **will** break.

I use [poetry](https://python-poetry.org/) to manage dependencies and develop the package.
(Actually, I use `poetry2nix`, hence the .nix files.)

## Installation
There are several ways to "install" the module:

1) Install the dependencies manually (either globally or into some virtual environment) and copy the folder "compromise_wrapper" to a location where it can be imported (e.g. is visible in `sys.path`).
2) Use poetry to instantiate the virtual environment: `cd` to this folder, then `poetry install`.
   The module `compromise_wrapper` should be available within a virtual environment.
   It can be activated with `poetry shell`.
3) Use `poetry build` to package the module and install it in some other environment, e.g. using `pip`.

## Usage

```
from compromise_wrapper import MOP

def vec_objective(x):
    return x**2

mop = MOP(2)
mop.set_vec_objective(vec_objective)
```

Have a look at the "tests" folder, too!
