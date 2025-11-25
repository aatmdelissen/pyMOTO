import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym

# https://en.wikipedia.org/wiki/Test_functions_for_optimization

def opt_mma2007(x, y, **kwargs):
    pym.minimize_mma(x, y, asybound=1000, mmaversion='MMA2007', **kwargs)

def opt_mma1987(x, y, **kwargs):
    pym.minimize_mma(x, y, asybound=1000, mmaversion='MMA1987', **kwargs)

def opt_gcmma(x, y, **kwargs):
    pym.minimize_mma(x, y, mmaversion='GCMMA', **kwargs)

def opt_slp(x, y, **kwargs):
    pym.minimize_slp(x, y, asybound=1e-8, **kwargs)

optimizers = [
    opt_mma1987,
    opt_mma2007,
    opt_gcmma,
    opt_slp,
]


@pytest.mark.parametrize('n', ['float', 1, 2, 10])
@pytest.mark.parametrize('optimizer', optimizers)
def test_unconstrained(optimizer, n):
    pytest.skip("Slow test, enable when needed")
    class SphereFunction(pym.Module):
        def __call__(self, x) -> float:
            return np.sum(x**2)
        
        def _sensitivity(self, dy: float):
            x = self.get_input_states()
            return dy*2*x
        
    if n == 'float':
        x = pym.Signal('x', 1.0)
    else:
        x = pym.Signal('x', np.ones(n))
    y = SphereFunction()(x)
    pym.Print()(x)
    y.tag = 'y'

    optimizer(x, y, verbosity=4, tolx=1e-4, tolf=1e-4, maxit=100, xmin=-1, xmax=1)

    npt.assert_allclose(x.state, 0.0, atol=1e-6)


@pytest.mark.parametrize('optimizer', optimizers)
def test_constrained(optimizer):
    pytest.skip("Slow test, enable when needed")
    # Rosenbrock constrained in a circle (https://en.wikipedia.org/wiki/File:Rosenbrock_circle_constraint.svg)
    x = pym.Signal('x', -0.5)
    y = pym.Signal('y', -0.5)
    f0 = pym.MathExpression('(1-x)^2 + (y - x^2)^2')(x, y)  # We cheated here by not multiplying second term with 100
    f0.tag = 'f0'
    f1 = pym.MathExpression('x^2 + y^2 - 2')(x, y)
    f1.tag = 'f1'
    optimizer([x, y], [f0, f1], verbosity=4, tolx=1e-4, tolf=1e-4, maxit=100, xmin=-1.5, xmax=1.5)
    npt.assert_allclose(x.state, 1.0, rtol=1e-2)
    npt.assert_allclose(y.state, 1.0, rtol=1e-2)


@pytest.mark.parametrize('optimizer', optimizers)
def test_cantilever(optimizer):
    pytest.skip("Slow test, enable when needed")
    # Cantilever problem from Svanberg's 1987 MMA paper
    n = 5
    c1, c2 = 0.0624, 1.0
    xref = np.array([6.016, 5.309, 4.494, 3.502, 2.153])
    x = pym.Signal('x', 5*np.ones(n))
    xsum = pym.EinSum('i->')(x)
    f0 = pym.MathExpression(f'{c1}*inp0')(xsum)
    f0.tag = 'f0'
    xi3 = pym.MathExpression('1/inp0^3')(x)
    xi3sum = pym.EinSum('i,i->')(np.array([61, 37, 19, 7, 1]), xi3)
    f1 = pym.MathExpression(f'inp0 - {c2}')(xi3sum)
    f1.tag = 'f1'

    optimizer(x, [f0, f1], verbosity=4, tolx=1e-4, tolf=1e-4, maxit=300, xmin=0, xmax=10)
    npt.assert_allclose(x.state, xref, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
