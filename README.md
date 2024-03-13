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

```python
from compromise_wrapper import MOP
import numpy as np

def vec_objective(x):
    x = np.array(x)
    return x**2

mop = MOP(2)
mop.set_vec_objective(vec_objective, dim_out = 2)

opt_x, opt_fx, stop_code = mop.optimize(np.array([-2.0, 1.0]))
```

`optimize` supports a few stopping criteria (`max_iter`, `stop_min_delta`, `max_num_crit_loops`).
The returned arrays can be converted to numpy arrays:
```python
x_opt = np.array(opt_x)
```

Have a look at the "tests" folder, too!

## Parallelism

The new Compromise releases feature threaded concurrent optimization.
Unfortunately, PythonCall ist not thread-safe, and I did not manage to run `mop.optimize_parallel()` without segfaults --
although I tried `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` and disabling garbage collection.
The issue most likely lies in the objective function being called from all threads, whereas only calls from masterthread are
allowed.
I don't see a simple work-around.

Threaded execution with shared RBF database on the Python side also fails, at least
```python
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=2)
pool.map(mop.optimize, list_of_vectors)
```
locks the database (or something else) forever.

It even locks with `mop.old_optimize`, which is weird.

### Matrix Evaluation

To make use of multiple workers nonetheless, you can make the objective evaluation parallel by accepting matrices:
```python
import compromise_wrapper as cw
from juliacall import ArrayValue
import numpy as np
def evaluate_fitness(x):
    x_is_mat = False
    if type(x) == ArrayValue and len(x.shape) == 2:
        x_is_mat = True

    x = np.array(x)
    if not x_is_mat:
        y = np.array([
            sum( (x + 1)**2 ),
            sum( (x - 1)**2 )
        ])
    else:
        n_x = x.shape[1]
        y = np.zeros((2, n_x))
        for i in range(n_x):
            y[0, i] = sum( (x[:, i] + 1)**2 )
            y[1, i] = sum( (x[:, i] - 1)**2 )

    return y

mop = cw.MOP(2, lb=[-4, -4], ub=[4, 4])
# `chunk_size > 1` makes batch evaluation possible.
# A finite `chunk_size` is needed to pre-allocate a scaling matrix.
# Per iteration, we usually query `num_vars` sites at most, so
# `chunk_size=3` should be okay:
mop.set_vec_objective(evaluate_fitness, dim_out=2, chunk_size=3)
mop.optimize([3.14, -2.718])
```
Now, `mop.rbf_db` holds a reference to the RBF database and this database is
reused in the next run (unless a call to `mop.old_optimize` is made):
```python
mop.set_max_func_calls(10)
mop.optimize([-3.5, 4])
```
This should tell you that there have only been 10 calls to the function, but the result looks
critical anyways, because of the database.

Note: If you set a maximum number of evaluations in `set_vec_objective` with the
`max_func_calls` keyword argument, this is used in every call to optimize, and
the counter is reset with every run.

