from pymoto import Module
class Scaling(Module):
    """
    Quick module that scales to a given value on the first iteration.
    This is useful, for instance, for MMA where the objective must be scaled in a certain way for good convergence
    """
    def _prepare(self, value):
        self.value = value

    def _response(self, x):
        if not hasattr(self, 'sf'):
            self.sf = self.value/x
        return x * self.sf

    def _sensitivity(self, dy):
        return dy * self.sf
