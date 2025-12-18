# Multivariate Uncertainty Quantification with Tomographic Quantile Forests

# Author
Takuya Kanazawa  

# Abstract
Quantifying predictive uncertainty is essential for safe and trustworthy
real-world AI deployment. Yet, fully nonparametric estimation of conditional
distributions remains challenging for multivariate targets. We propose
Tomographic Quantile Forests (TQF), a nonparametric, uncertainty-aware,
tree-based regression model for multivariate targets. TQF learns conditional
quantiles of directional projections $\mathbf{n}^{\top}\mathbf{y}$ as functions
of the input $\mathbf{x}$ and the unit direction $\mathbf{n}$. At inference, it
aggregates quantiles across many directions and reconstructs the multivariate
conditional distribution by minimizing the sliced Wasserstein distance via an
efficient alternating scheme with convex subproblems. Unlike classical
directional-quantile approaches that typically produce only convex quantile
regions and require training separate models for different directions, TQF
covers all directions with a single model without imposing convexity
restrictions. We evaluate TQF on synthetic and real-world datasets, and release
the source code on GitHub.

# Status
Submitted to arXiv
