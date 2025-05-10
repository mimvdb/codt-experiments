module QuantBnBWrapper
include("../../Quant-BnB/gen_data.jl")
include("../../Quant-BnB/Algorithms.jl")
include("../../Quant-BnB/lowerbound_middle.jl")
include("../../Quant-BnB/QuantBnB-2D.jl")
include("../../Quant-BnB/QuantBnB-3D.jl")

using PrecompileTools

export optimal_classification_2d, optimal_classification_3d, optimal_regression_2d, optimal_regression_3d, tree_eval

function optimal_classification_2d(X, Y)
    st = time()
    gre, gre_tree = greedy_tree(X, Y, 2, "C")
    opt, opt_tree = QuantBnB_2D(X, Y, 3, gre*(1+1e-6), 2, 0.2, nothing, "C", false)

    return (opt, opt_tree, time() - st)
end

function optimal_regression_2d(X, Y)
    st = time()
    gre, gre_tree = greedy_tree(X, Y, 2, "R")
    opt, opt_tree = QuantBnB_2D(X, Y, 3, gre*(1+1e-6), 2, 0.2, nothing, "R", false)

    return (opt, opt_tree, time() - st)
end

function optimal_classification_3d(X, Y, timelimit)
    st = time()
    gre, gre_tree = greedy_tree(X, Y, 3, "C")
    opt, opt_tree = QuantBnB_3D(X, Y, 3, 3, gre*(1+1e-6), 0, 0, nothing, "C", timelimit)

    return (opt, opt_tree, time() - st)
end

function optimal_regression_3d(X, Y, timelimit)
    st = time()
    gre, gre_tree = greedy_tree(X, Y, 3, "R")
    opt, opt_tree = QuantBnB_3D(X, Y, 3, 3, gre*(1+1e-6), 0, 0, nothing, "R", timelimit)

    return (opt, opt_tree, time() - st)
end

# From https://github.com/goretkin/PythonPrecompileExample.jl. Required for accurate timings that do not include compile time
py_precompile_script = """
import numpy as np
import pandas as pd
import juliacall

def python_workload(the_julia_module):
    df_qsar = pd.read_csv("./datasets/regression/qsar.csv", sep=" ", header=None)
    X_qsar = df_qsar[df_qsar.columns[1:]].to_numpy()
    y_qsar = np.array([df_qsar[0].to_numpy()]).T
    df_bank = pd.read_csv("./datasets/classification/bank.csv", sep=" ", header=None)
    X_bank = df_bank[df_bank.columns[1:]].to_numpy()
    y_bank = df_bank[0].to_numpy()
    y_quant = np.zeros((y_bank.size, y_bank.max() + 1))
    y_quant[np.arange(y_bank.size), y_bank] = 1
    the_julia_module.optimal_regression_2d(X_qsar, y_qsar)
    the_julia_module.optimal_regression_3d(X_qsar, y_qsar, 5)
    the_julia_module.optimal_classification_2d(X_bank, y_quant)
    the_julia_module.optimal_classification_3d(X_bank, y_quant, 5)
"""

@setup_workload begin
    # ENV["JULIA_CONDAPKG_BACKEND"] = "Null" # use the system installation of Python
    using PythonCall: pyexec, pyeval

    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to this package or not (on Julia 1.8 and higher)
        pyexec(py_precompile_script, Main)
        python_workload = pyeval("python_workload", Main)
        python_workload(QuantBnBWrapper)
    end
end

end