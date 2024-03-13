from juliacall import Main as jl
from .setup_utils import julia_utils

VEC_TYPE = jl.seval("Vector{Float64}")
MAT_TYPE = jl.seval("Matrix{Float64}")

class MOP():
    def __init__(self, n_vars, lb=None, ub=None):
        self.n_vars = n_vars
        self.lb = lb
        self.ub = ub

        self.rbf_db = None

        if not jl.isdefined(jl, jl.Symbol("C")):
            julia_utils()

        self.mop = jl.C.MutableMOP(num_vars = n_vars, lb=lb, ub=ub, reset_call_counters=True)

    def set_vec_objective(self, func, model_cfg="rbf", dim_out=1, max_func_calls=300, func_iip=False, chunk_size=1):
        if not self.mop.objectives:
            objf = jl.PyFunction(func)

            self.rbf_db = jl.C.RBFModels.init_rbf_database(self.n_vars, dim_out, None, None, jl.Float64)
            cfg = jl.C.RBFConfig(database=self.rbf_db)

            jl.C.add_objectives_b(self.mop, objf, cfg, dim_out=dim_out, max_func_calls=max_func_calls, func_iip=func_iip, chunk_size=chunk_size)
        else:
            print("Objective is already set.")
        return None

    def set_max_func_calls(self, i):
        self.mop = jl.change_max_func_calls(self.mop, i)

    def set_rbf_database(self):
        if not self.rbf_db:
            dim_x = self.n_vars
            dim_y = jl.C.dim_objectives(self.mop)
            self.rbf_db = jl.C.RBFModels.init_rbf_database(dim_x, dim_y, None, None, jl.Float64)
            self.mop.mcfg_objectives.database = self.rbf_db
        return None

    def old_optimize(self, x0):
        global VEC_TYPE
        x = jl.convert(VEC_TYPE, x0)

        self.rbf_db = None
        self.mop.mcfg_objectives.database = self.rbf_db

        algo_opts = jl.C.AlgorithmOptions(stop_max_crit_loops=2, stop_delta_min=1e-5)
        ret = jl.C.optimize(self.mop, x, algo_opts=algo_opts)

        return jl.C.opt_vars(ret), jl.C.opt_objectives(ret), jl.C.opt_stop_code(ret)

    def optimize(self, x0, max_iter=100):
        global VEC_TYPE
        x = jl.convert(VEC_TYPE, x0)
        self.set_rbf_database()
        algo_opts = jl.C.AlgorithmOptions(max_iter=max_iter, stop_max_crit_loops=2, stop_delta_min=1e-5)
        ret = jl.C.optimize(self.mop, x, algo_opts=algo_opts)
        return jl.C.opt_vars(ret), jl.C.opt_objectives(ret), jl.C.opt_stop_code(ret)

    def optimize_parallel(self, x0):
        global MAT_TYPE

        print("DISABLING PYTHON GARBAGE COLLECTION")
        jl.PythonCall.GC.disable()

        x = jl.convert(MAT_TYPE, x0)
        self.set_rbf_database()
        opts = jl.C.ThreadedOuterAlgorithmOptions(inner_opts=jl.C.AlgorithmOptions(stop_max_crit_loops=2, stop_delta_min=1e-5))

        try:
            rets = jl.C.optimize_with_algo(self.mop, opts, x)
        finally:
            print("ENABLING PYTHON GARBAGE COLLECTION")
            jl.PythonCall.GC.enable()

        return rets



