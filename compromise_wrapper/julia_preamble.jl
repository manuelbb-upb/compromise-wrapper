@info "Including Preamble"

import Compromise as C

struct PyFunction <: Function
    func :: Py
end
function (pf::PyFunction)(x::AbstractVector)
    y = pf.func(x)
    return pyconvert(Vector{Float64}, y)
end
function (pf::PyFunction)(x::AbstractMatrix)
    y = pf.func(x)
    return pyconvert(Matrix{Float64}, y)
end
function (pf::PyFunction)(y, x)
    pf.func(y, x)
    return Nothing
end

import Accessors: @set
function change_max_func_calls(mop, i)
    mop_new = @set mop.objectives.wrapped_function.max_func_calls=i
    return mop_new
end
