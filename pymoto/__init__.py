__version__ = "2.0.0"

import warnings

# Imports from common
from .common.domain import VoxelDomain
from .common.dyadcarrier import DyadicMatrix
from .common.mma import MMA
from .common.optimizers import Optimizer, OC, SLP

# Import solvers
from . import solvers

# Modular inports
from .core_objects import Signal, Module, Network, make_signals

# Import modules
from .modules.assembly import AssembleGeneral, AssembleStiffness, AssembleMass, AssemblePoisson
from .modules.assembly import ElementOperation, Strain, Stress, ElementAverage, NodalOperation, ThermoMechanical
from .modules.autodiff import AutoMod
from .modules.complex import MakeComplex, SplitComplex, RealPart, ImagPart, ComplexNorm, Conjugate
from .modules.filter import FilterConv, Filter, DensityFilter, OverhangFilter
from .modules.generic import MathExpression, EinSum, Concatenate, SetValue, AddMatrix
from .modules.io import FigModule, PlotDomain, PlotGraph, PlotIter, WriteToVTI, ScalarToFile, Print, SeriesToVTI
from .modules.linalg import Inverse, LinSolve, EigenSolve, SystemOfEquations, StaticCondensation
from .modules.aggregation import AggScaling, AggActiveSet, Aggregation, PNorm, SoftMinMax, KSFunction
from .modules.scaling import Scaling
from .modules.transient import TransientSolve

# Further helper routines
from .routines import finite_difference, minimize_oc, minimize_mma, minimize_slp

__all__ = [
    "Signal",
    "Module",
    "Network",
    "make_signals",
    "finite_difference",
    "minimize_oc",
    "minimize_mma",
    "minimize_slp",
    # Common
    "MMA",
    "OC",
    "SLP",
    "Optimizer",
    "DyadicMatrix",
    "VoxelDomain",
    "solvers",
    # Helpers
    "AggScaling",
    "AggActiveSet",
    # Modules
    "MathExpression",
    "EinSum",
    "Concatenate",
    "SetValue",
    "AddMatrix",
    "Inverse",
    "LinSolve",
    "EigenSolve",
    "SystemOfEquations",
    "StaticCondensation",
    "AssembleGeneral",
    "AssembleStiffness",
    "AssembleMass",
    "AssemblePoisson",
    "ElementOperation",
    "Strain",
    "Stress",
    "ElementAverage",
    "NodalOperation",
    "ThermoMechanical",
    "FilterConv",
    "Filter",
    "DensityFilter",
    "OverhangFilter",
    "FigModule",
    "PlotDomain",
    "PlotGraph",
    "PlotIter",
    "WriteToVTI",
    "ScalarToFile",
    "Print",
    "SeriesToVTI",
    "MakeComplex",
    "SplitComplex",
    "RealPart",
    "ImagPart",
    "ComplexNorm",
    "Conjugate",
    "AutoMod",
    "Aggregation",
    "PNorm",
    "SoftMinMax",
    "KSFunction",
    "Scaling",
    "TransientSolve",
]

## Deprecations (https://stackoverflow.com/a/55139609/11702471)
DEPRECATED_NAMES = [('MathGeneral', 'MathExpression', '2.2'),
                    ('VecSet', 'SetValue', '2.2'),
                    ('DomainDefinition', 'VoxelDomain', '2.2'),
                    ('DyadCarrier', 'DyadicMatrix', '2.2'),
                    ('ConcatSignal', 'Concatenate', '2.2'),
                    ]  # (old-name, new-name, removed-after-version)


def __getattr__(name):
    for old_name, new_name, v in DEPRECATED_NAMES:
        if name == old_name:
            warnings.warn(f"`{old_name}` is renamed `{new_name}` and will be removed after version {v}",
                          DeprecationWarning,
                          stacklevel=2)
            return globals()[new_name]
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + [names[0] for names in DEPRECATED_NAMES])