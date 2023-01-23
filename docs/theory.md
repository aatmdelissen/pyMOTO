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
\mathbf{x}_\text{f} \rightarrow \boxed{ \begin{aligned} &\text{SIMP} \\ x_\text{min} +& (1-x_\text{min})\mathbf{x}_\text{f}^3 \end{aligned} } \rightarrow \mathbf{s}_\text{K} \rightarrow \dotsc \rightarrow h \text{.}
$$

Here, $h$ is any arbitrary response, such as compliance. To calculate the design sensitivities of the SIMP 
interpolation, the chain rule is used, as depicted below

$$
\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}_\text{f}} \leftarrow \boxed{ \begin{aligned} &\quad\text{SIMP} \\ &\frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}}\frac{\mathrm{d}\mathbf{s}_\text{K}}{\mathrm{d}\mathbf{x}_\text{f}} \end{aligned} } \leftarrow \frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}} \leftarrow \dotsc \leftarrow \frac{\mathrm{d}h}{\mathrm{d}h}=1 \text{.}
$$

By having the SIMP `Module` implement both the forward and the backward path, it becomes independent of anything outside itself.
It is only dependent on its $\mathbf{x}_\text{f}$ input for the forward path and on the sensitivity of its output $\frac{\mathrm{d}h}{\mathrm{d} \mathbf{s}_\text{K}}$ for the backward path.


In case of the compliance optimization problem, there are multiple type of `Module` used. Each of them has its own 
forward and backward path implemented (`response()` and `sensitivity()`, respectively). This allows creating larger 
chains of different `Module` and other configurations, thus other optimization problems.






For any module with an input $\mathbf{x}$ and output $\mathbf{y}$, which eventually results in response $h$ ...........




In the light of modular programming, we can write the problem as

$$
\mathbf{x} \rightarrow \boxed{ \mathbf{g}(\mathbf{x}, \mathbf{u}(\mathbf{x}))=\mathbf{0} } \rightarrow \mathbf{u} \rightarrow
\boxed{ \mathbf{f}(\mathbf{u}) } \rightarrow y \rightarrow \dotsc \rightarrow h \text{.}
$$

The backward operation is written as

$$
\frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} \leftarrow \boxed{
 \frac{\mathrm{d}h}{\mathrm{d}\mathbf{x}} = -\frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
 \frac{\partial\mathbf{g}}{\partial\mathbf{x}}
 } \leftarrow \frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}} \leftarrow \boxed{ \frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}} = \frac{\mathrm{d}h}{\mathrm{d}y} \frac{\mathrm{d}y}{\mathrm{d} \mathbf{u}} }
 \leftarrow \frac{\mathrm{d} h}{\mathrm{d} y} \leftarrow \dotsc \leftarrow \frac{\mathrm{d} h}{\mathrm{d} h} = 1
$$

In the module involving the constraint equation, the choice remains free to do an adjoint solve or a direct method.
If an adjoint is done, the pseudo-forces are the incoming sensitivities. By looking for linear dependency with
respect to earlier solves with other force vectors on the same system, some solves might be prevented.




## Direct versus adjoint senstivities

A system is solved under the condition that $\mathbf{g}(\mathbf{x}, \mathbf{u})=\mathbf{0}$. Examples of such a function is static compliance
$ \mathbf{K}(\mathbf{x})\mathbf{u}-\mathbf{b} = \mathbf{0} $ or any other function that involves solving an inverse, in order
to satisfy the constraint condition. The state vector $ \mathbf{u} $ is then used to calculate a response
$ f(\mathbf{x}, \mathbf{u})$. For instance $ f=\mathbf{c} \cdot \mathbf{u} $.
Summarizing, we have a response function $ f(\mathbf{x},\mathbf{u}) $ subject to $ \mathbf{g}(\mathbf{x},\mathbf{u})=\mathbf{0} $.
Of this response, we would like to know the sensitivities with respect to the design variables
$\frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}}$.

### Direct sensitivities
By the direct approach the sensitivities would be calculated as

$$
\frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} = \frac{\partial f}{\partial\mathbf{x}}
+ \frac{\partial f}{\partial \mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}\mathrm{.}
$$

However, this approach needs the state sensitivities $\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$, which are
usually very expensive to calculate, since it involves a matrix inverse for every design variable in $\mathbf{x}$.
We can calculate it using the constraint equation, which should be zero regardless the design:

$$
\frac{\mathrm{d}\mathbf{g}}{\mathrm{d}\mathbf{x}} = \frac{\partial\mathbf{g}}{\partial\mathbf{x}}
+ \frac{\partial\mathbf{g}}{\partial\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} = \mathbf{0}\mathrm{,}
$$

which we can rewrite to obtain the state sensitivities

$$
\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} = -\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
\frac{\partial\mathbf{g}}{\partial\mathbf{x}}\mathrm{.}
$$

### Adjoint sensitivities
The adjoint method becomes more economical if only one or a few responses require sensitivities. Here the state
sensitivities $\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$ are not explicitly calculated, but substituted
by the direct sensitivity relation:

$$
 \frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} = \frac{\partial f}{\partial\mathbf{x}}
 + \frac{\partial f}{\partial\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}
 = \frac{\partial f}{\partial\mathbf{x}}
 - \frac{\partial f}{\partial\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
 \frac{\partial\mathbf{g}}{\partial\mathbf{x}}\mathrm{.}
$$

At this point you could choose to either calculate
$\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}\frac{\partial\mathbf{g}}{\partial\mathbf{x}}$ first
(direct method) or
$\frac{\partial f}{\partial\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}$ first
(adjoint method).
Here we rename the adjoint solve:

$$
\boldsymbol{\lambda}^T = -\frac{\partial f}{\partial\mathbf{u}}\left(\frac{\partial\mathbf{g}}{\partial\mathbf{u}}\right)^{-1}
$$

Which leads to the equation for the sensitivities:

$$
 \frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} = \frac{\partial f}{\partial\mathbf{x}}
 + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
$$

Formally, an adjoint calculation is derived by a Lagrangian. Lagrange multipliers are added for the constraint equation.

$$
 \mathcal{L}(\mathbf{x}, \mathbf{u}) = f(\mathbf{x}, \mathbf{u}) + \boldsymbol{\lambda}^T\mathbf{g}(\mathbf{x}, \mathbf{u})
$$

Now the sensitivity of the Lagrangian becomes

$$
\begin{aligned}
 \frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} &=& \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{x}} \\
 &=& \frac{\partial f}{\partial\mathbf{x}} + \frac{\partial f}{\partial\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}
     + \frac{\mathrm{d}\boldsymbol{\lambda}^T}{\mathrm{d}\mathbf{x}}\mathbf{g}(\mathbf{x}, \mathbf{u})
     + \boldsymbol{\lambda}^T\left(\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
     + \frac{\partial\mathbf{g}}{\mathbf{u}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} \right) \\
 &=& \frac{\partial f}{\partial\mathbf{x}} + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
     + \left(\frac{\partial f}{\partial\mathbf{u}}
     + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{u}} \right)\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} \\
\end{aligned}
$$

By choosing the adjoint vector $\boldsymbol{\lambda}$ correctly, we can cause the state sensitivities
$\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}$ to drop out and end up with the same equation:

$$
 \frac{\mathrm{d}f}{\mathrm{d}\mathbf{x}} = \frac{\partial f}{\partial\mathbf{x}}
 + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{x}} \quad\text{subject to } \frac{\partial f}{\partial \mathbf{u}}
 + \boldsymbol{\lambda}^T\frac{\partial\mathbf{g}}{\partial\mathbf{u}}=\mathbf{0}
$$



## Complex valued-functions
For a function which maps complex to complex

$$
\mathbf{u}\in\mathbb{C} \rightarrow \boxed{ \mathbf{f}(\mathbf{u}) } \rightarrow \mathbf{y}\in\mathbb{C}
\rightarrow \dotsc \rightarrow h \in\mathbb{R}
$$

$$
h_\mathbf{u}\leftarrow\boxed{h_\mathbf{u}=...
} \leftarrow h_\mathbf{y} \leftarrow \cdots \leftarrow h_h=1
$$

We can write $\mathbf{u} = \mathbf{u}_{\Re} + i\mathbf{u}_{\Im}  $, and its derivative using the defition of complex derivatives:

$$
\begin{aligned}
h_\mathbf{u} &=
\frac{\partial h}{\partial\mathbf{u}_{\Re}} + i\frac{\partial h}{\partial\mathbf{u}_{\Im}} \\
&= \frac{\mathrm{d}h}{\mathrm{d}\mathbf{y}_{\Re}}\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}_{\Re}}
 + \frac{\mathrm{d}h}{\mathrm{d}\mathbf{f}_{\Im}}\frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathrm{d}\mathbf{u}_{\Re}}
 + i\left(\frac{\mathrm{d}h}{\mathbf{y}_{\Re}}\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathbf{u}_{\Im}}
        + \frac{\mathrm{d}h}{\mathbf{y}_{\Im}}\frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathbf{u}_{\Im}}\right) \\
&= \frac{\mathrm{d}h}{\mathrm{d}\mathbf{y}_{\Re}}\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}} +
   \frac{\mathrm{d}h}{\mathrm{d}\mathbf{y}_{\Im}}\frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathrm{d}\mathbf{u}}
\end{aligned}
$$

For a holomorphic/analytical function, the <A HREF="https://mathworld.wolfram.com/Cauchy-RiemannEquations.html" target="_blank" rel="noopener noreferrer">Cauchy-Riemann equations</A> are valid.

$$
\left.
\begin{matrix}
\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}_{\Re}} =  \frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathrm{d}\mathbf{u}_{\Im}} \\
\frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathrm{d}\mathbf{u}_{\Re}} = -\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}_{\Im}}
\end{matrix}\right\} \frac{\mathrm{d}\mathbf{y}_{\Im}}{\mathrm{d}\mathbf{u}} = i \frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}}
$$

Which can be used to rewrite the sensitivities as

$$
\begin{aligned}
h_\mathbf{u} &= \left(\frac{\mathrm{d}h}{\mathrm{d}\mathbf{y}_{\Re}}
+ i\frac{\mathrm{d}h}{\mathrm{d}\mathbf{y}_{\Im}}  \right)\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}} \\
&= h_\mathbf{y}\frac{\mathrm{d}\mathbf{y}_{\Re}}{\mathrm{d}\mathbf{u}}
\end{aligned}
$$

### Real to complex functions
Assume we now have a complex value in the loop. Although, the initial and final variables are real.

$$
\mathbf{x}\in\mathbb{R}^n \rightarrow \boxed{ \mathbf{u}(\mathbf{x}) } \rightarrow \mathbf{u}\in\mathbb{C}^m
\rightarrow \boxed{ f(\mathbf{u}) } \rightarrow y \in\mathbb{R}^1\rightarrow \dotsc \rightarrow h \text{.}
$$

$$
h_\mathbf{x} \leftarrow
\boxed{
      h_\mathbf{x} = \Re\left(h_\mathbf{u}^* \frac{\mathrm{d}\mathbf{u}}{\mathrm{d} \mathbf{x}}\right)
} \leftarrow h_\mathbf{u} \leftarrow
\boxed{
      h_\mathbf{u} = h_y \frac{\mathrm{d} y}{\mathrm{d} \mathbf{u}}
} \leftarrow h_y \leftarrow \dotsc \leftarrow h_h = 1
$$

For notation, the following derivative operator can be used:

$$
\begin{aligned}
\frac{\mathrm{d} y}{\mathrm{d} \mathbf{u}}&=\frac{\partial y}{\partial \mathbf{u}_{\Re}} + i\frac{\partial y}{\partial\mathbf{u}_{\Im}}\\
\frac{\mathrm{d}\mathbf{u}}{\mathrm{d} \mathbf{x}}&=\frac{\mathrm{d}\mathbf{u}_{\Re}}{\mathrm{d}\mathbf{x}} + i\frac{\mathrm{d}\mathbf{u}_{\Im}}{\mathrm{d} \mathbf{x}}
\end{aligned}
$$

Following the explanation in [van der Veen, 2015], the complex variable can be decomposed into real and imaginary
parts $\mathbf{u}(\mathbf{x}) = \mathbf{u}_{\Re}(\mathbf{x})+i\mathbf{u}_{\Im}(\mathbf{x})$. Now, according to the chain rule:

$$
\frac{\mathrm{d} y}{\mathrm{d} \mathbf{x}}=\frac{\partial y}{\partial\mathbf{u}_{\Re}}\frac{\mathrm{d}\mathbf{u}_{\Re}}{\mathrm{d}\mathbf{x}}
+ \frac{\partial y}{\partial\mathbf{u}_{\Im}}\frac{\mathrm{d}\mathbf{u}_{\Im}}{\mathrm{d} \mathbf{x}} \text{.}
$$

With the complex derivative definitions, we end up with

$$
\begin{aligned}
\frac{\mathrm{d}y}{\mathrm{d}\mathbf{x}}
&=\Re\left(\frac{\mathrm{d} y}{\mathrm{d} \mathbf{u}^*}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} \right)\\
&=\Re\left(\frac{\partial y}{\partial \mathbf{u}_{\Re}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}
- i \frac{\partial y}{\partial \mathbf{u}_{\Im}}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}}  \right)\\
&=\Re\left(\frac{\partial y}{\partial \mathbf{u}_{\Re}}\frac{\mathrm{d}\mathbf{u}_{\Re}}{\mathrm{d}\mathbf{x}}
+ \frac{\partial y}{\partial\mathbf{u}_{\Im}}\frac{\mathrm{d}\mathbf{u}_{\Im}}{\mathrm{d} \mathbf{x}}
+ i \left(\frac{\partial y}{\partial \mathbf{u}_{\Re}}\frac{\mathrm{d}\mathbf{u}_{\Im}}{\mathrm{d}\mathbf{x}}
- \frac{\partial y}{\partial \mathbf{u}_{\Im}}\frac{\mathrm{d}\mathbf{u}_{\Re}}{\mathrm{d}\mathbf{x}}\right)\right)
\end{aligned}
\text{,}
$$

with $\bullet^*$ the conjugate transpose.



### Adjoint sensitivities with complex values
The adjoint case is also described in [van der Veen, 2015]. The constraint condition
$\mathbf{g}(\mathbf{x}, \mathbf{u})=0$ is now a complex valued function.

$$
\mathbf{x}\in\mathbb{R}^n \rightarrow \boxed{ \mathbf{g}(\mathbf{x}, \mathbf{u})=\mathbf{0} } \rightarrow
\mathbf{u}\in\mathbb{C}^m \rightarrow \boxed{ \mathbf{f}(\mathbf{u}) } \rightarrow f\in\mathbb{R}^1 \rightarrow \dotsc \rightarrow h
$$

The complex function $\mathbf{g}(\mathbf{x}, \mathbf{u})=\mathbf{0}$ can be written as two separate real
functions of $\Re(\mathbf{g}) = \mathbf{0}$ and $\Im(\mathbf{g}) = \mathbf{0}$. Of these we can calculate
the derivatives as

$$
\begin{aligned}
\frac{\mathrm{d}\mathbf{g}}{\mathrm{d}\mathbf{x}} =& \frac{\mathrm{d}\Re(\mathbf{g})}{\mathrm{d}\mathbf{x}}
+ i\frac{\mathrm{d}\Im(\mathbf{g})}{\mathrm{d}\mathbf{x}} =\\
&\frac{\partial\Re(\mathbf{g})}{\mathrm{d}\mathbf{x}} + \frac{\partial\Re(\mathbf{g})}{\partial\Re(\mathbf{u})}
\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}} + \frac{\partial\Re(\mathbf{g})}{\partial\Im(\mathbf{u})}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}} \\
&+ i\left(\frac{\partial\Re(\mathbf{g})}{\partial\mathbf{x}}
+ \frac{\partial\Re(\mathbf{g})}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}} +
\frac{\partial\Re(\mathbf{g})}{\partial\Im(\mathbf{u})}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}}\right) \\
=&\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
+ \frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}}
+ \frac{\partial\mathbf{g}}{\partial\Im(\mathbf{u})}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}}
= \mathbf{0}
\end{aligned}
$$

If the function is holomorphic (i.e. analytical), the Cauchy-Riemann differential equations hold:
$$
\begin{aligned}
\frac{\partial\Re(\mathbf{g})}{\partial\Re(\mathbf{x})} &= \frac{\partial\Im(\mathbf{g})}{\partial\Im(\mathbf{x})} \\
\frac{\partial\Im(\mathbf{g})}{\partial\Re(\mathbf{x})} &= -\frac{\partial\Re(\mathbf{g})}{\partial\Im(\mathbf{x})}
\end{aligned}
$$

Using these, we can rewrite:

$$
\frac{\partial\mathbf{g}}{\partial\Im(\mathbf{u})}
=\frac{\partial\Re(\mathbf{g})}{\partial\Im(\mathbf{u})}
+i\frac{\partial\Im(\mathbf{g})}{\partial\Im(\mathbf{u})}
=-\frac{\partial\Im(\mathbf{g})}{\partial\Re(\mathbf{u})}
+i\frac{\partial\Re(\mathbf{g})}{\partial\Re(\mathbf{u})}=i\frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}
$$

Using this condition into the derivatives equation:

$$
\begin{aligned}
 & \frac{\partial\mathbf{g}}{\partial\mathbf{x}} + \frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}}+ \frac{\partial\mathbf{g}}{\partial\Im(\mathbf{u})}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}} \\
=& \frac{\partial\mathbf{g}}{\partial\mathbf{x}} + \frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}}+i \frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}} \\
=& \frac{\partial\mathbf{g}}{\partial\mathbf{x}} + \frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} =\mathbf{0}
\end{aligned}
$$

Or in a block scheme:

$$
\frac{\partial h}{\partial\mathbf{x}} \leftarrow \boxed{
	-\frac{\partial h}{\partial\mathbf{u}^*}\left(\frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\right)^{-1}\frac{\partial\mathbf{g}}{\partial\mathbf{x}}
} \leftarrow \frac{\partial h}{\partial\mathbf{u}} \leftarrow
\boxed{ \frac{\partial h}{\partial f}\frac{\partial f}{\partial\mathbf{u}} }
\leftarrow \frac{\partial h}{\partial f} \leftarrow \dotsc
$$

For a (dynamic) compliance equation, which is holomorphic: $\mathbf{g}=\mathbf{Z}\mathbf{u}-\mathbf{b}=\mathbf{0}$, using
the complex derivative, this leads to:

$$
\frac{\partial\mathbf{g}}{\partial\mathbf{x}} + \mathbf{Z}\frac{\mathrm{d}\Re(\mathbf{u})}{\mathrm{d}\mathbf{x}}
+ i\mathbf{Z}\frac{\mathrm{d}\Im(\mathbf{u})}{\mathrm{d}\mathbf{x}}
= \mathbf{Z}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} + \frac{\partial\mathbf{g}}{\partial\mathbf{x}} = \mathbf{0}
$$

When using the holomorphic formulation, it leads to the same answer:

$$
\mathbf{Z}\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}\mathbf{x}} + \frac{\partial\mathbf{g}}{\partial\mathbf{x}} =\mathbf{0}
$$

We can solve the adjoint vector as:

$$
\boldsymbol{\lambda}^T = -\frac{\partial h}{\partial\mathbf{u}^*}\left(\frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}\right)^{-1}
$$


When the input values of the adjoint equation are also complex, an extra term gets added:

$$
\begin{aligned}
\frac{\partial h}{\partial\mathbf{x}} &= -(1+i)\frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}}
\left(\frac{\partial\Re(\mathbf{g})}{\partial\mathbf{u}}\right)^{-1}\frac{\partial\Re(\mathbf{g})}{\partial\mathbf{x}} \\
&=  -(1+i)\frac{\mathrm{d}h}{\mathrm{d}\mathbf{u}} \left(\bar{\frac{\partial\mathbf{g}}{\partial\Re(\mathbf{u})}}\right)^{-1}
\frac{\partial\Re(\mathbf{g})}{\partial\mathbf{x}}
\end{aligned}
$$
