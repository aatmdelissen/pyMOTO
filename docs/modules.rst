pyMOTO Reference
================

Core
----
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.Signal
   pymoto.Module
   pymoto.Network

Mathematical Modules
--------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.EinSum
   pymoto.MathGeneral
   pymoto.Inverse
   pymoto.LinSolve
   pymoto.SystemOfEquations
   pymoto.StaticCondensation
   pymoto.EigenSolve
   pymoto.Scaling
   pymoto.ConcatSignal

Constraint Aggregation
------------------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.Aggregation
   pymoto.PNorm
   pymoto.SoftMinMax
   pymoto.KSFunction

Helper objects

.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.AggActiveSet
   pymoto.AggScaling

Finite Element Modules
----------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.AssembleGeneral
   pymoto.AssembleStiffness
   pymoto.AssembleMass
   pymoto.AssemblePoisson
   pymoto.ElementOperation
   pymoto.NodalOperation
   pymoto.Strain
   pymoto.Stress
   pymoto.ElementAverage
   pymoto.ThermoMechanical

Filter Modules
--------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.Filter
   pymoto.DensityFilter
   pymoto.OverhangFilter
   pymoto.FilterConv

Output Modules
--------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.FigModule
   pymoto.PlotDomain
   pymoto.PlotGraph
   pymoto.PlotIter
   pymoto.WriteToVTI
   pymoto.ScalarToFile

Complex-value Modules
---------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.MakeComplex
   pymoto.RealPart
   pymoto.ImagPart
   pymoto.ComplexNorm

Common Utilities and Routines
-----------------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.DomainDefinition
   pymoto.DyadCarrier
   pymoto.finite_difference
   pymoto.minimize_oc
   pymoto.minimize_mma

Linear Solvers
--------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.solvers.LinearSolver
   pymoto.solvers.LDAWrapper
   pymoto.solvers.SolverDiagonal
   pymoto.solvers.SolverDenseQR
   pymoto.solvers.SolverDenseLU
   pymoto.solvers.SolverDenseCholesky
   pymoto.solvers.SolverDenseLDL
   pymoto.solvers.SolverSparseLU
   pymoto.solvers.SolverSparsePardiso
   pymoto.solvers.SolverSparseCholeskyScikit
   pymoto.solvers.SolverSparseCholeskyCVXOPT

Preconditioners

.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.solvers.Preconditioner
   pymoto.solvers.DampedJacobi
   pymoto.solvers.SOR
   pymoto.solvers.ILU
   pymoto.solvers.GeometricMultigrid

Iterative solvers

.. autosummary::
   :toctree: stubs
   :nosignatures:
   pymoto.solvers.CG
