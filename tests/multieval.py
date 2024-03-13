from IPython import get_ipython
ipython = get_ipython()

ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

import compromise_wrapper as cw
import numpy as np
from juliacall import ArrayValue

from concurrent.futures import ThreadPoolExecutor

class Simulation():
    def __init__(self, params1 = np.array([-1.0, -1.0]), params2 = np.array([1.0, 1.0])):
        self.params1 = params1
        self.params2 = params2

    def evaluate_fitness(self, x):
        x_is_mat = False
        if (type(x) == ArrayValue or type(x) == np.ndarray) and len(x.shape) == 2:
            x_is_mat = True

        if not x_is_mat:
            y = np.array([
                sum( (x + self.params1)**2 ),
                sum( (x + self.params2)**2 )
            ])
        else:
            n_x = x.shape[1]
            y = np.zeros((2, n_x))
            for i in range(n_x):
                y[0, i] = sum( (x[:, i] + self.params1)**2 )
                y[1, i] = sum( (x[:, i] + self.params2)**2 )

        return y


if __name__ == "__main__":
    print("Running tests...")
    sim = Simulation()
    mop = cw.MOP(2, lb=[-1.0, -1.0], ub=[2.0, 2.0])
    mop.set_vec_objective(sim.evaluate_fitness, dim_out=2, chunk_size=3)

    X0 = np.random.rand(2, 10)

    for i in range(10):
        x0 = X0[:, i]
        mop.optimize(x0, max_iter=5)
