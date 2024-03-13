import juliapkg
import os
from juliacall import Main as jl
from pathlib import Path

SRC_DIR = Path(__file__).absolute().parents[0]

def install_juliapkgs(project_path=None, dev=False, offline=False, Compromise_path=None):
    if project_path:
        os.environ["PYTHON_JULIAPKG_PROJECT"] = project_path

    if offline:
        os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"

    juliapkg.require_julia("1.9")
    if dev:
       juliapkg.add("Revise", "295af30f-e4ad-537b-8983-00126c2a3abe")

    juliapkg.add("Accessors", "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697")

    Compromise_added = False
    if Compromise_path:
        if os.path.isdir(Compromise_path):
            juliapkg.add(
                "Compromise",
                uuid="254bc946-86ae-484f-a9da-8147cb79ba93",
                dev=dev,
                path=path
            )
            Compromise_added = True

    if not Compromise_added:
        if dev:
            juliapkg.add(
                "Compromise",
                uuid="254bc946-86ae-484f-a9da-8147cb79ba93",
                dev=dev,
                url="https://github.com/manuelbb-upb/Compromise.jl.git",
            )
        else:
            juliapkg.add(
                "Compromise",
                uuid="254bc946-86ae-484f-a9da-8147cb79ba93",
                dev=dev,
                url="https://github.com/manuelbb-upb/Compromise.jl.git",
                rev="613cfb1774ae78c78b32c2a847fae8db0df51b50"
            )

    juliapkg.resolve()

def julia_utils(use_Revise=False):
    global SRC_DIR
    if use_Revise:
        jl.seval("using Revise")

    jl.include(str(SRC_DIR / "julia_preamble.jl"))
