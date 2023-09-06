__version__ = '1.0.3'

from .core_objects import Signal, Module, Network, make_signals
from .routines import finite_difference, minimize_oc, minimize_mma

# Imports from common
from .common.dyadcarrier import DyadCarrier
from .common.domain import DomainDefinition
from .common.mma import MMA
from .common.solvers import matrix_is_complex, matrix_is_diagonal, matrix_is_symmetric, matrix_is_hermitian, \
    LinearSolver, LDAWrapper
from .common.solvers_dense import SolverDiagonal, SolverDenseQR, SolverDenseLU, SolverDenseCholesky, SolverDenseLDL
from .common.solvers_sparse import SolverSparsePardiso, SolverSparseLU, SolverSparseCholeskyScikit, SolverSparseCholeskyCVXOPT

# Import from modules
from .modules.generic import MathGeneral, EinSum, ConcatSignal
from .modules.linalg import Inverse, LinSolve, EigenSolve
from .modules.assembly import AssembleGeneral, AssembleStiffness, AssembleMass
from .modules.filter import FilterConv, Filter, DensityFilter, OverhangFilter
from .modules.io import PlotDomain, PlotGraph, PlotIter, WriteToVTI
from .modules.complex import MakeComplex, RealPart, ImagPart, ComplexNorm
from .modules.autodiff import AutoMod

__all__ = [
    'Signal', 'Module', 'Network', 'make_signals',
    'finite_difference', 'minimize_oc', 'minimize_mma',
    # Common
    'MMA',
    'DyadCarrier',
    'DomainDefinition',
    'matrix_is_complex', 'matrix_is_diagonal', 'matrix_is_symmetric', 'matrix_is_hermitian',
    'LinearSolver', 'LDAWrapper',
    'SolverDiagonal', 'SolverDenseQR', 'SolverDenseLU', 'SolverDenseCholesky', 'SolverDenseLDL',
    'SolverSparsePardiso', 'SolverSparseLU', 'SolverSparseCholeskyScikit', 'SolverSparseCholeskyCVXOPT',
    # Modules
    "MathGeneral", "EinSum", "ConcatSignal",
    "Inverse", "LinSolve", "EigenSolve",
    "AssembleGeneral", "AssembleStiffness", "AssembleMass",
    "FilterConv", "Filter", "DensityFilter", "OverhangFilter",
    "PlotDomain", "PlotGraph", "PlotIter", "WriteToVTI",
    "MakeComplex", "RealPart", "ImagPart", "ComplexNorm",
    "AutoMod"
]
