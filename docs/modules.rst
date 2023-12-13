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

Finite Element Modules
----------------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.AssembleGeneral
   pymoto.AssembleStiffness
   pymoto.AssembleMass
   pymoto.AssembleScalarMass
   pymoto.AssembleScalarField

Filter Modules
--------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.DensityFilter
   pymoto.OverhangFilter

Output Modules
--------------
.. autosummary::
   :toctree: stubs
   :nosignatures:

   pymoto.PlotDomain
   pymoto.PlotGraph
   pymoto.PlotIter
   pymoto.WriteToVTI

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

   pymoto.LDAWrapper
   pymoto.SolverDiagonal
   pymoto.SolverDenseQR
   pymoto.SolverDenseLU
   pymoto.SolverDenseCholesky
   pymoto.SolverDenseLDL
   pymoto.SolverSparseLU
   pymoto.SolverSparsePardiso
   pymoto.SolverSparseCholeskyScikit
   pymoto.SolverSparseCholeskyCVXOPT
