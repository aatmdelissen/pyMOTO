__version__ = '1.4.0'

from .common.domain import DomainDefinition

# Imports from common
from .common.dyadcarrier import DyadCarrier
from .common.mma import MMA

# Import solvers
from . import solvers

# Modular inports
from .core_objects import Signal, Module, Network, make_signals

# Import modules
from .modules.assembly import AssembleGeneral, AssembleStiffness, AssembleMass, AssemblePoisson
from .modules.assembly import ElementOperation, Strain, Stress
from .modules.autodiff import AutoMod
from .modules.complex import MakeComplex, RealPart, ImagPart, ComplexNorm
from .modules.filter import FilterConv, Filter, DensityFilter, OverhangFilter
from .modules.generic import MathGeneral, EinSum, ConcatSignal
from .modules.io import FigModule, PlotDomain, PlotGraph, PlotIter, WriteToVTI
from .modules.linalg import Inverse, LinSolve, EigenSolve, SystemOfEquations, StaticCondensation
from .modules.aggregation import AggScaling, AggActiveSet, Aggregation, PNorm, SoftMinMax, KSFunction
from .modules.scaling import Scaling

# Further helper routines
from .routines import finite_difference, minimize_oc, minimize_mma

__all__ = [
    'Signal', 'Module', 'Network', 'make_signals',
    'finite_difference', 'minimize_oc', 'minimize_mma',

    # Common
    'MMA',
    'DyadCarrier',
    'DomainDefinition',
    'solvers',

    # Helpers
    "AggScaling", "AggActiveSet",

    # Modules
    "MathGeneral", "EinSum", "ConcatSignal",
    "Inverse", "LinSolve", "EigenSolve", "SystemOfEquations", "StaticCondensation",
    "AssembleGeneral", "AssembleStiffness", "AssembleMass", "AssemblePoisson",
    "ElementOperation", "Strain", "Stress",
    "FilterConv", "Filter", "DensityFilter", "OverhangFilter",
    "FigModule", "PlotDomain", "PlotGraph", "PlotIter", "WriteToVTI",
    "MakeComplex", "RealPart", "ImagPart", "ComplexNorm",
    "AutoMod",
    "Aggregation", "PNorm", "SoftMinMax", "KSFunction",
    "Scaling",
]
