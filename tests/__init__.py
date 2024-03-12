import compromise_wrapper as cw
import numpy as np

class Simulation():
    def __init__(self, params1 = np.array([-1.0, -1.0]), params2 = np.array([1.0, 1.0])):
        self.params1 = params1
        self.params2 = params2

    def evaluate_fitness(self, x):
        y = np.array([
            sum( (x + self.params1)**2 ),
            sum( (x + self.params2)**2 )
        ])
        #print(f"Evaluating fitness for {x} gives:{y}")
        return y


if __name__ == "__main__":
    print("Running tests...")
    sim = Simulation()
    mop = cw.MOP(2, lb=[-1.0, -1.0], ub=[2.0, 2.0])
    mop.set_vec_objective(sim.evaluate_fitness, dim_out=2)
    mop.optimize([-1, 1.5])

    #X0 = np.random.rand(2, 10)
    #mop.optimize_parallel(X0)
