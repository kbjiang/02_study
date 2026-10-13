---
title: "Support Vector Machines"
course: "CIS 520: Machine Learning — Fall 2009"
source: "https://web.archive.org/web/20100712105012/http://alliance.seas.upenn.edu/~cis520/wiki/index.php?n=Lectures.SVMs"
---

# Support Vector Machines

Historically, Support Vector Machines (SVMs) were motivated by the notion of the maximum margin separating hyperplane. Before we explore this motivation, which is a bit of a [MacGuffin](https://en.wikipedia.org/wiki/MacGuffin), let's relate it to the other linear classification algorithms: we have seen that Logistic Regression and Boosting minimize a convex upper-bound on the 0/1 loss for binary classification: logistic in one case and exponential in the other.

Support Vector Machines minimize the hinge loss — also an upper-bound on the 0/1 loss.

$$
\ell_{\text{hinge}}(y,f(\mathbf{x})) = \max(0, 1-yf(\mathbf{x}))
$$

A few things to note about the hinge loss.

- It's convex like the exponential and logistic loss: optimization is fairly easy.
- It grows slower below zero: the penalty for misclassification is more mild.
- It is zero after 1: once correct and confident enough, there is no bonus for driving \(yf(\mathbf{x})\) higher.
- It's non-differentiable at the hinge point: simple gradient doesn't work.

Why all the excitement about this loss? First, it has a nice geometric interpretation (at least in the linearly separable case) as leading to a maximum margin hyperplane separating positive from negative examples. Second, more importantly, it fits particularly well with kernels because many examples will not be a part of the solution, leading to a much more efficient dual representation and algorithms.

## Hyperplanes and Margins

Consider the case of linearly separable data. In the toy 2D example below, there are many hyperplanes (lines) that separate the negative examples (blue) from positive (green). If we define the margin of a hyperplane as the distance of the closest point to the decision boundary, the maximum margin hyperplane is shown below.

> Original page figure: a 2D linearly separable dataset and its maximum-margin hyperplane. The figure itself is not embedded in this Markdown file.

We will denote our linear classifier as (note the constant term \(b\) is separated out):

$$
h(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^{\top}\mathbf{x} + b)
$$

Since rescaling the classifier by multiplying it by a positive constant does not change the decision boundary,

$$
\operatorname{sign}(\mathbf{w}^{\top}\mathbf{x} + b)
=
\operatorname{sign}(\gamma(\mathbf{w}^{\top}\mathbf{x} + b)),
$$

we will set the scale so that at the points closest to the boundary,

$$
\mathbf{w}^{\top}\mathbf{x} + b = \pm 1,
$$

as shown in the original figure.

Now we will compute the margin, which is the distance of the points closest to the boundary, using one point from each class, \(\mathbf{x}_1\) and \(\mathbf{x}_2\).

$$
\mathbf{w}^{\top}\mathbf{x}_1 + b = -1
\qquad\text{and}\qquad
\mathbf{w}^{\top}\mathbf{x}_2 + b = 1
$$

Hence,

$$
\mathbf{w}^{\top}(\mathbf{x}_2-\mathbf{x}_1) = 2
\quad\Longrightarrow\quad
\frac{\mathbf{w}^{\top}}{2\|\mathbf{w}\|_2}
(\mathbf{x}_2-\mathbf{x}_1) = \frac{1}{\|\mathbf{w}\|_2}
$$

Note that \(\frac{\mathbf{w}^{\top}}{2\|\mathbf{w}\|_2}(\mathbf{x}_2-\mathbf{x}_1)\) is precisely the distance to the hyperplane: the vector \(\frac{1}{2}(\mathbf{x}_2-\mathbf{x}_1)\) projected on the unit vector \(\frac{\mathbf{w}}{\|\mathbf{w}\|_2}\). Let's restate this more precisely. A rescaled hyperplane classifier satisfies:

$$
y_i(\mathbf{w}^{\top}\mathbf{x}_i + b) \ge 1,
\qquad i=1,\ldots,n
$$

(with equality for the closest points) and has margin \(\frac{1}{\|\mathbf{w}\|_2}\).

To find the maximal margin hyperplane, we can simply maximize \(\frac{1}{\|\mathbf{w}\|_2}\), which is equivalent to minimizing \(\frac{1}{2}\mathbf{w}^{\top}\mathbf{w}\) subject to the (scaled) correct classification constraints. The SVM optimization problem (in the separable case) is:

**Separable SVM primal:**

$$
\begin{aligned}
\min_{\mathbf{w},b}\quad & \frac{1}{2}\mathbf{w}^{\top}\mathbf{w}\\
\text{subject to}\quad & y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\ge 1,
\quad i=1,\ldots,n
\end{aligned}
$$

Note on notation: I will use \(\min/\max\) instead of \(\inf/\sup\) since the continuous nature of the optimization will be obvious from context. This optimization problem has a quadratic objective and linear inequality constraints. Note that only the points closest to the boundary participate in defining the SVM hyperplane. These points are called the support vectors, and we will see that the solution can be expressed only using these points.

## The Dual View

In order to get more intuition about the problem, we can look at it from the dual perspective. Lagrangian duality, reviewed in recitation, provides the right tool here. We formulate the Lagrangian by introducing a non-negative multiplier \(\alpha_i\) for each inequality constraint:

$$
L(\mathbf{w},b,\boldsymbol{\alpha})
=
\frac{1}{2}\mathbf{w}^{\top}\mathbf{w}
+
\sum_i \alpha_i
\left(1-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\right)
$$

Note that if \(\mathbf{w},b\) is feasible (i.e. satisfy all the constraints), then

$$
\max_{\boldsymbol{\alpha}\ge 0}
L(\mathbf{w},b,\boldsymbol{\alpha})
=
\frac{1}{2}\mathbf{w}^{\top}\mathbf{w}
$$

since all the coefficients \(1-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\) are negative or zero. When \(\mathbf{w},b\) are not feasible,

$$
\max_{\boldsymbol{\alpha}\ge 0}L(\mathbf{w},b,\boldsymbol{\alpha})=\infty.
$$

Assuming that feasible \(\mathbf{w},b\) exist, \(\min_{\mathbf{w},b}(\max_{\boldsymbol{\alpha}\ge0}L(\mathbf{w},b,\boldsymbol{\alpha}))\) solves the original problem, and since the original objective is convex, we can change the order of \(\min/\max\) to obtain a dual problem that has the same optimal value:

$$
\max_{\boldsymbol{\alpha}\ge0}
\left(\min_{\mathbf{w},b}L(\mathbf{w},b,\boldsymbol{\alpha})\right).
$$

Taking derivatives with respect to \(\mathbf{w}\) and \(b\) and setting them to zero, we get:

$$
\frac{\partial L}{\partial b}=0
\quad\Longrightarrow\quad
\sum_i \alpha_i y_i=0
$$

and

$$
\frac{\partial L}{\partial \mathbf{w}}=0
\quad\Longrightarrow\quad
\mathbf{w}=\sum_i \alpha_i y_i\mathbf{x}_i.
$$

Note the dual representation of \(\mathbf{w}\): it is a linear combination of the examples, weighted by dual variables \(\alpha_i\). Plugging in the expression for \(\mathbf{w}\), we get

$$
\begin{aligned}
L(b,\boldsymbol{\alpha})
&= \frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_i^\top\mathbf{x}_j\\
&\quad+\sum_i\alpha_i
\left(1-y_i\left(\sum_j\alpha_jy_j\mathbf{x}_j^\top\mathbf{x}_i+b\right)\right)\\
&= -\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_j^\top\mathbf{x}_i
+\sum_i\alpha_i-\sum_i\alpha_i y_i b.
\end{aligned}
$$

Since \(\sum_i\alpha_i y_i=0\), the dual problem is:

**Separable SVM dual:**

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}\ge0}\quad&
\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_i^\top\mathbf{x}_j\\
\text{subject to}\quad&\sum_i\alpha_i y_i=0
\end{aligned}
$$

This dual optimization problem, as the primal, has a quadratic objective and a single linear constraint. Because of the KKT optimality conditions, when \(\mathbf{x}_i\) is not the closest to the boundary, \(\alpha_i\) is zero. So only the points that support the hyperplane, called support vectors, participate in the solution. Given a solution \(\boldsymbol{\alpha}\) to the problem, we can recover the primal via:

$$
\mathbf{w}=\sum_i\alpha_i y_i\mathbf{x}_i
$$

To recover \(b\), note that for any support vector \(i\), where \(\alpha_i>0\), we have \(y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)=1\), so

$$
b=y_i-\mathbf{w}^{\top}\mathbf{x}_i,
\qquad \forall i\text{ such that }\alpha_i>0.
$$

## Kernels

We can use the kernel trick again, by assuming a feature map \(\phi(\mathbf{x})\) such that

$$
k(\mathbf{x}_i,\mathbf{x}_j)
=
\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j).
$$

The kernelized dual optimization problem is:

**Kernelized separable dual:**

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}\ge0}\quad&
\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
k(\mathbf{x}_i,\mathbf{x}_j)\\
\text{subject to}\quad&\sum_i\alpha_i y_i=0
\end{aligned}
$$

Given a solution \(\boldsymbol{\alpha}\) to the problem, we can recover

$$
b=y_i-\sum_j\alpha_jy_j k(\mathbf{x}_j,\mathbf{x}_i),
\qquad \forall i\text{ such that }\alpha_i>0
$$

and the prediction:

$$
\mathbf{w}^{\top}\mathbf{x}+b
=
\sum_i\alpha_i y_i k(\mathbf{x}_i,\mathbf{x})+b.
$$

## Non-separable case

When the data is not separable, the above optimization will fail. To handle the case of points on the wrong side of the fence, SVMs introduce a penalty. Although we might perhaps like to minimize the number of misclassified points, this is an NP-hard problem. Instead, SVMs penalize points where

$$
y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)<1
$$

using a linear cost,

$$
\max(0,1-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)),
$$

the hinge function.

In the primal optimization problem above, this can be accomplished by adding a slack variable, \(\xi_i\), for each example and penalizing their sum, weighted by the positive slack penalty constant \(C\):

**Hinge primal:**

$$
\begin{aligned}
\min_{\mathbf{w},b,\boldsymbol{\xi}\ge0}\quad&
\frac{1}{2}\mathbf{w}^{\top}\mathbf{w}+C\sum_i\xi_i\\
\text{subject to}\quad&
y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\ge1-\xi_i,
\quad i=1,\ldots,n
\end{aligned}
$$

For any given \(\mathbf{w},b\), since we're minimizing over \(\boldsymbol{\xi}\), we will have

$$
\xi_i=\max(0,1-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b))
$$

for each \(i\).

Taking the dual, we get:

**Hinge dual:**

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}\ge0}\quad&
\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_i^\top\mathbf{x}_j\\
\text{subject to}\quad&
\sum_i\alpha_i y_i=0,\qquad \alpha_i\le C
\end{aligned}
$$

Here's a short derivation. In addition to the \(\alpha_i\)s, we introduce a non-negative Lagrange multiplier \(\lambda_i\) for each \(\xi_i\ge0\) constraint.

$$
\begin{aligned}
L(\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\alpha},\boldsymbol{\lambda})
&=\frac{1}{2}\mathbf{w}^{\top}\mathbf{w}+C\sum_i\xi_i\\
&\quad+\sum_i\alpha_i
(1-\xi_i-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b))
-\sum_i\lambda_i\xi_i
\end{aligned}
$$

$$
\frac{\partial L}{\partial b}=0
\quad\Longrightarrow\quad
\sum_i\alpha_i y_i=0
$$

$$
\frac{\partial L}{\partial \mathbf{w}}=0
\quad\Longrightarrow\quad
\mathbf{w}=\sum_i\alpha_i y_i\mathbf{x}_i
$$

$$
\frac{\partial L}{\partial \xi_i}=0
\quad\Longrightarrow\quad
C-\alpha_i-\lambda_i=0
$$

Plugging in the expression for \(\mathbf{w}\) and using the condition \(\sum_i\alpha_i y_i=0\) as before, we have

$$
L(\boldsymbol{\xi},\boldsymbol{\alpha},\boldsymbol{\lambda})
=
\sum_i\alpha_i
-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_i^\top\mathbf{x}_j
+\sum_i\xi_i(C-\alpha_i-\lambda_i).
$$

Now using the condition \(C-\alpha_i-\lambda_i=0\), the last sum vanishes, so we are left with the dual:

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}\ge0,\boldsymbol{\lambda}\ge0}\quad&
\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
\mathbf{x}_i^\top\mathbf{x}_j\\
\text{subject to}\quad&
\sum_i\alpha_i y_i=0,\qquad C-\alpha_i-\lambda_i=0
\end{aligned}
$$

We can get rid of the \(\lambda_i\)s by noting that they only appear in the constraint \(C-\alpha_i-\lambda_i=0\), and since they must be non-negative, we can just replace the constraint with \(\alpha_i\le C\), as in the dual stated above.

Note that the same dual representation properties hold for this version:

$$
\mathbf{w}=\sum_i\alpha_i y_i\mathbf{x}_i
$$

and the kernelized version is just:

**Hinge kernelized dual:**

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}\ge0}\quad&
\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_j
k(\mathbf{x}_i,\mathbf{x}_j)\\
\text{subject to}\quad&
\sum_i\alpha_i y_i=0,\qquad \alpha_i\le C
\end{aligned}
$$

Given a solution \(\boldsymbol{\alpha}\) to the problem, we can recover

$$
b=y_i-\sum_j\alpha_j y_j k(\mathbf{x}_j,\mathbf{x}_i),
\qquad \forall i\text{ such that }0<\alpha_i<C
$$

and the prediction:

$$
\mathbf{w}^{\top}\mathbf{x}+b
=
\sum_i\alpha_i y_i k(\mathbf{x}_i,\mathbf{x})+b.
$$

## Solving SVMs

We will not spend too much time on how to optimize quadratic programs with linear constraints. There is a very large literature on the topic of general convex optimization and SVM optimization in particular, and many off-the-shelf efficient tools have been developed for both. In recent years, simple algorithms (e.g., [SMO](https://en.wikipedia.org/wiki/Sequential_minimal_optimization), [subgradient methods](https://en.wikipedia.org/wiki/Subgradient_method)) that find approximate answers to the SVM problem have been shown to work quite well. Here is [an applet for kernel SVMs in 2D](https://web.archive.org/web/20100712105012/http://alliance.seas.upenn.edu/~cis520/wiki/index.php?n=Lectures.SVMs).

---

## Notes on this Markdown copy

- Source page: [Wayback Machine capture from July 12, 2010](https://web.archive.org/web/20100712105012/http://alliance.seas.upenn.edu/~cis520/wiki/index.php?n=Lectures.SVMs).
- The lecture text and mathematical expressions have been transcribed into Markdown and LaTeX delimiters for local rendering.
- The original page's figures and interactive applet are not embedded here.
- A few display equations have been laid out for readability; this is a transcription, not a byte-for-byte HTML dump.
