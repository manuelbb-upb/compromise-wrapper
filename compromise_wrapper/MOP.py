from juliacall import Main as jl
from .setup_utils import julia_utils

VEC_TYPE = jl.seval("Vector{Float64}")

class MOP():
    def __init__(self, n_vars, lb=None, ub=None):
        self.n_vars = n_vars
        self.lb = lb
        self.ub = ub

        if not jl.isdefined(jl, jl.Symbol("C")):
            julia_utils()

        self.mop = jl.C.MutableMOP(num_vars = n_vars, lb=lb, ub=ub)

    def set_vec_objective(self, func, model_cfg="rbf", dim_out=1, func_iip=False):
        if not self.mop.objectives:
            objf = jl.PyFunction(func)
            jl.C.add_objectives_b(self.mop, objf, jl.Symbol(model_cfg), dim_out=dim_out, func_iip=func_iip)
        else:
            print("Objective is already set.")
        return None

    def optimize(self, x0):
        global VEC_TYPE
        x = jl.convert(VEC_TYPE, x0)
        algo_opts = jl.C.AlgorithmOptions(stop_crit_tol_abs=-1, stop_max_crit_loops=2)
        final_vals, ret_code = jl.C.optimize(self.mop, x, algo_opts=algo_opts)

        return final_vals.ξ, final_vals.fx, ret_code
