
# HierarchicalBayesZoo

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> HierarchicalBayesZoo

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "HierarchicalBayesZoo"
```
which auto-activate the project and enable local path handling from DrWatson.

## Todo List
- [] Deep exponential families (Ranganath, *et al.*, 2015)
- [x] Dirichlet-exponential non-negative matrix factorization (Kucukelbir, *et al.*, 2017)
- [] Gamma-gamma non-negative matrix factorization (Canny, 2004; Gopalan, *et al.*, 2015)
- [] Probabilistic principle component analysis (Tipping & Bishop, 1999)
- [] Non-conjugate Gaussian-Bernoulli matrix factorization (Agrawal, *et al.*, 2021)
- [] Gaussian mixture models
- [] Latent Dirichlet allocation
- [] (maybe) state space model?

## References
- Agrawal, A., & Domke, J. (2021). Amortized variational inference for simple hierarchical models. Advances in Neural Information Processing Systems, 34, 21388-21399.
- Canny, J. (2004, July). GaP: a factor model for discrete data. In Proceedings of the 27th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 122-129).
- Gopalan, P., Hofman, J. M., & Blei, D. M. (2015, July). Scalable Recommendation with Hierarchical Poisson Factorization. In UAI (pp. 326-335).
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research.
- Ranganath, R., Tang, L., Charlin, L., & Blei, D. (2015, February). Deep exponential families. In Artificial Intelligence and Statistics (pp. 762-771). PMLR.
- Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. Journal of the Royal Statistical Society Series B: Statistical Methodology, 61(3), 611-622.
