# This does not actually work, as there are problems with precompiling PythonCall.jl. Instead, the functions are precompiled with a wrapper package.
def create_julia_sysimage():
    from juliacall import Main as jl

    jl.seval('import Pkg; Pkg.add("PackageCompiler")')
    jl.seval('using PackageCompiler')
    jl.seval('PackageCompiler.create_sysimage(["QuantBnBWrapper"]; sysimage_path="sys_precompiled.so")')
