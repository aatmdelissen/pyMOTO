# Theory on derivatives
## Backpropagation
The flow diagram of a standard compliance topology optimization is visualized as follows.
```{mermaid}
 graph LR;
      START[x]-->B[Density filter];
      B-->|xF| C[SIMP];
      C-->|sK| D[Assembly];
      D-->|K| E[Linear Solve];
      STARTF[Force f]-->E;
      E-->|u| F[Compliance u.f];
      STARTF-->F;
      F-->|c| STOP[Objective];
      B-->V[Volume];
      V-->|V| STOP1[Constraint];
      
      style START fill:#FFFFFF00, stroke:#FFFFFF00;
      style STARTF fill:#FFFFFF00, stroke:#FFFFFF00;
      style STOP  fill:#FFFFFF00, stroke:#FFFFFF00;
      style STOP1  fill:#FFFFFF00, stroke:#FFFFFF00;
```

Let us examine the module that implements the SIMP interpolation, which has input $\mathbf{x}_\text{f}$ and output $\mathbf{s}_\text{K}$, mapped as

$$
\mathbf{x}_\text{f} \rightarrow \boxed{ \begin{aligned} &\text{SIMP} \\ x_\text{min} +& (1-x_\text{min})\mathbf{x}_\text{f}^3 \end{aligned} } \rightarrow \mathbf{s}_\text{K} \rightarrow \cdots \rightarrow h \text{.}
$$

Here, $h$ is any arbitrary response, such as compliance. To calculate the design sensitivities of the SIMP 
interpolation, the chain rule is used, as depicted below

$$
\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}_\text{f}} \leftarrow \boxed{ \begin{aligned} &\quad\text{SIMP} \\ &\frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}}\frac{\mathrm{d}\mathbf{s}_\text{K}}{\mathrm{d}\mathbf{x}_\text{f}} \end{aligned} } \leftarrow \frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}} \leftarrow \cdots \leftarrow \frac{\mathrm{d}h}{\mathrm{d}h}=1 \text{.}
$$

By having the SIMP `Module` implement both the forward and the backward path, it becomes independent of anything outside itself.
It is only dependent on its $\mathbf{x}_\text{f}$ input for the forward path and on the sensitivity of its output $\frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}}$ for the backward path.


In case of the compliance optimization problem, there are multiple type of `Module` used. Each of them has its own 
forward and backward path implemented (`response()` and `sensitivity()`, respectively). This allows creating larger 
chains of different `Module` and other configurations, thus other optimization problems.



## Direct versus adjoint senstivities
In the light of the modular approach to the derivative computation, we can also implement a problem with the 
internal constraint equations $\mathbf{g}(\mathbf{x}, \mathbf{u}(\mathbf{x}))=\mathbf{0}$, as

$$
\mathbf{x} \rightarrow \boxed{ \mathbf{g}(\mathbf{x}, \mathbf{u}(\mathbf{x}))=\mathbf{0} } \rightarrow \mathbf{u} \rightarrow
\boxed{ y(\mathbf{u}) } \rightarrow y \rightarrow \cdots \rightarrow h \text{.}
$$

A system of equations is solved under the condition that $\mathbf{g}(\mathbf{x}, \mathbf{u})=\mathbf{0}$. Examples of 
such a constraint function is static equilibrium $\mathbf{K}(\mathbf{x})\mathbf{u}-\mathbf{b} = \mathbf{0}$, where 
$\mathbf{K}$ is a stiffness matrix, $\mathbf{u}$ are displacements and $\mathbf{b}$ are applied loads. Many other 
physics also involve the solution of a linear system of equations, for which the same theory applies. The state vector 
$ \mathbf{u} $ is used to calculate a response function (*e.g.*, an objective or constraint) $h(\mathbf{u})$. 
For instance, the compliance is calculated as $ h(\mathbf{u}) = \mathbf{b} \cdot \mathbf{u} $.
Summarizing, we have a response function $ h(\mathbf{u}) $ subject to $ \mathbf{g}(\mathbf{x},\mathbf{u})=\mathbf{0} $.
Of this response, we would like to know the derivatives with respect to the design variables
$\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}}$.

For this we can again use backward propagation, which is written as

$$
\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} \leftarrow \boxed{
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = -\frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
 \frac{\partial\mathbf{g}}{\partial\mathbf{x}}
 } \leftarrow \frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}} \leftarrow \boxed{ \frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}} = \frac{\mathrm{d}h}{\mathrm{d}y} \frac{\mathrm{d}y}{\mathrm{d} \mathbf{u}} }
 \leftarrow \frac{\mathrm{d} h}{\mathrm{d} y} \leftarrow \dotsc \leftarrow \frac{\mathrm{d} h}{\mathrm{d} h} = 1
$$

In the `Module` involving the constraint equation, the choice remains open to do use the *adjoint method* or the *direct 
method*. If the adjoint method is used, a second system of equations needs to be solved, for which the right-hand-side 
is the incoming sensitivity $\frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}}$. By detecting linear dependency with respect to 
earlier solutions with other right-hand-sides on the same system (self-adjointness), additional solutions might be 
prevented. This is done automatically in the `pymoto.LinSolve` module, using the LDAS framework by [Koppen *et al.* 
(2022)](https://doi.org/10.1007/s00158-022-03378-8).

### Direct sensitivities
In the direct approach the sensitivities would be calculated as

$$
\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = \frac{\mathrm{d} h}{\mathrm{d} \mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}\text{.}
$$

However, this approach needs the state sensitivities $\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$, which are
usually very costly to calculate, since it involves the solution to a system of equations for every design variable 
in $\mathbf{x}$. We can calculate it by differentiation of the constraint equation, which should be zero regardless the 
design:

$$
\frac{\mathrm{d}\mathbf{g}}{\mathrm{d}\mathbf{x}} = \frac{\partial\mathbf{g}}{\partial\mathbf{x}}
+ \frac{\partial\mathbf{g}}{\partial\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} = \mathbf{0}\mathrm{,}
$$

which we can rewrite to obtain the state sensitivities

$$
\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} = -\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
\frac{\partial\mathbf{g}}{\partial\mathbf{x}}\mathrm{.}
$$

Tho calculate the state sensitivities, a system of equations with $n$ right-hand-sides needs to be solved, where $n$ is 
the number of design variables in $\mathbf{x}$.

**Example:**
In case the constraint equations are 
$\mathbf{g}(\mathbf{x}, \mathbf{u}) = \mathbf{K}(\mathbf{x})\mathbf{u}-\mathbf{b} = \mathbf{0}$, the state sensitivities 
are calculated as

$$
\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}x_i} = -\mathbf{K}^{-1}
\left(\frac{\partial\mathbf{K}}{\partial x_i} \mathbf{u}\right) \: \forall \: i\in \{1, \dotsc, n\} \mathrm{.}
$$

This approach requires one solution to the system of equations for each design variable, which is feasible when $n$ is 
small and there are many different response functions $h_j(\mathbf{u}(\mathbf{x}))$.

### Adjoint sensitivities
The adjoint method becomes more economical if only few responses $h$ depend on a large number of design variables 
$\mathbf{x}$. In the adjoint method the state sensitivities $\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$ are not 
explicitly calculated, but substituted into the chain rule of $h$:

$$
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = \frac{\mathrm{d} h}{\mathrm{d}\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}
 =  - \frac{\mathrm{d} h}{\mathrm{d}\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
 \frac{\partial\mathbf{g}}{\partial\mathbf{x}}\mathrm{.}
$$

At this point you could choose to either calculate
$\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}\frac{\partial\mathbf{g}}{\partial\mathbf{x}}$ first, as
is done in the direct method. Or, the equations
$\frac{\mathrm{d} h}{\mathrm{d} \mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}$ can be 
solved first, which is the adjoint system of equations.
The solution to the adjoint system of equations is denoted

$$
\boldsymbol{\lambda}^T = \frac{\mathrm{d} h}{\mathrm{d} \mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
$$

Which leads to the equation for the sensitivities:

$$
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = - \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
$$

**Example:**
In case the constraint equations are 
$\mathbf{g}(\mathbf{x}, \mathbf{u}) = \mathbf{K}(\mathbf{x})\mathbf{u}-\mathbf{b} = \mathbf{0}$, the adjoint vector
is now calculated for each response function $h_j$ as

$$
\boldsymbol{\lambda}_j = \mathbf{K}^{-\text{T}} \frac{\mathrm{d} h_j}{\mathrm{d} \mathbf{u}}
 \: \forall \: j\in \{1, \dotsc, m\} \mathrm{.}
$$

and the design sensitivity is calculated as

$$
 \frac{\mathrm{d}h_j}{\mathrm{d} x_i } = - \boldsymbol{\lambda}_j^\text{T}\frac{\partial\mathbf{K}}{\partial x_i} \mathbf{u}
$$

This approach requires one solution to the system of equations for each response, which is feasible when $m$ is 
small and there are many different design variables $\mathbf{x}$.

#### Adjoint sensitivities using a Lagrangian
Formally, an adjoint calculation is derived by a Lagrangian. Lagrange multipliers are added for the constraint equation.

$$
 \mathcal{L}(\mathbf{x}, \mathbf{u}) = h(\mathbf{x}, \mathbf{u}) + \boldsymbol{\lambda}^T\mathbf{g}(\mathbf{x}, \mathbf{u})
$$

Now the sensitivity of the Lagrangian becomes

$$
\begin{aligned}
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} &= \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{x}} \\
 &= \frac{\partial h}{\partial\mathbf{x}} + \frac{\partial h}{\partial\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}
     + \frac{\mathrm{d}\boldsymbol{\lambda}^T}{\mathrm{d}\mathbf{x}}\mathbf{g}(\mathbf{x}, \mathbf{u})
     + \boldsymbol{\lambda}^T\left(\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
     + \frac{\partial\mathbf{g}}{\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} \right) \\
 &= \frac{\partial h}{\partial\mathbf{x}} + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
     + \left(\frac{\partial h}{\partial\mathbf{u}}
     + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{u}} \right)\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} \\
\end{aligned}
$$

By choosing the adjoint vector $\boldsymbol{\lambda}$ correctly, we can cause the state sensitivities
$\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$ to drop out and end up with the same equation:

$$
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = \frac{\partial h}{\partial\mathbf{x}}
 + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}} \quad\text{subject to } \frac{\partial h}{\partial \mathbf{u}}
 + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{u}}=\mathbf{0}
$$



## Complex-valued functions
Derivatives with respect to complex values are also supported by the pyMOTO framework. However, they behave a bit differently
than "normal" derivatives. A complex value ($z = x+iy$) can be seen as having two independent variables
(*e.g.*  $x$ and $y$, or, $z$ and its conjugate $z^*$).

Definition of partial derivatives (Wirtinger derivatives) for the complex value $z = x+iy$ are defined as

$$ \begin{aligned}
    \frac{\partial}{\partial z} &= \frac{1}{2}\left( \frac{\partial}{\partial x} - i \frac{\partial}{\partial y}\right) \leftarrow \text{Stored in }\texttt{Signal.sensitivity}\\
    \frac{\partial}{\partial z^*} &= \frac{1}{2}\left( \frac{\partial}{\partial x} + i \frac{\partial}{\partial y}\right) \text{.}
\end{aligned} $$

Throughout the pyMOTO package, if `Signal.state` represents $z$, the derivative with respect to any response
function $f\in\mathbb{R}$ that is stored in `Signal.sensitivity` *always* equals
$\frac{\partial f}{\partial z}$. Note that this may be different in other packages that implement complex
derivatives (*e.g.*, [PyTorch](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers),
JAX, or TensorFlow).

Useful identities in case $z\in\mathbb{C}$ and $s\in\mathbb{C}$ are

$$ \begin{aligned}
\frac{\partial s^*}{\partial z^*} &= \left( \frac{\partial s}{\partial z} \right)^* \\
\frac{\partial s^*}{\partial z} &= \left( \frac{\partial s}{\partial z^*} \right)^*
\end{aligned} $$

from which in case $f\in\mathbb{R}$, it can be seen that

$$\frac{\partial f}{\partial z^*} = \left( \frac{\partial f}{\partial z} \right)^*\text{.}$$

The chain rule for a mapping from $z\in\mathbb{C}\rightarrow s\in\mathbb{C}\rightarrow f$ can be interpreted as the
contributions of two independent variables (here, $s$ and $s^*$):

$$\frac{\partial f}{\partial z} = \frac{\partial f}{\partial s}\frac{\partial s}{\partial z} + \frac{\partial f}{\partial s^*}\frac{\partial s^*}{\partial z}\text{.}$$

In case the intermediate variable is real, thus $z\in\mathbb{C}\rightarrow r\in\mathbb{R}\rightarrow f$, the
chain rule reduces to

$$\frac{\partial f}{\partial z} = 2 \frac{\partial f}{\partial r}\frac{\partial r}{\partial z}\text{,}$$

which may seem counter-intuitive, but compensates for the initial factor of $1/2$.

For a mapping from real to complex, thus $r\in\mathbb{R}\rightarrow z\in\mathbb{C}\rightarrow f$, the chain rule
becomes

$$\frac{\partial f}{\partial z} = 2 \text{Re}\left( \frac{\partial f}{\partial s}\frac{\partial s}{\partial r} \right)\text{.}$$

**References and further reading**
  - [Wirtinger derivatives, Wikipedia](https://en.wikipedia.org/wiki/Wirtinger_derivatives)
  - Sarason (2007). *Complex function theory*. American Mathematical Society.
  - [Delgado (2009). *The complex gradient operator and the CR-calculus*](https://arxiv.org/pdf/0906.4835.pdf)
  - [Cauchy-Riemann equations](https://mathworld.wolfram.com/Cauchy-RiemannEquations.html)
  - [Pytorch AutoGrad](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers)

