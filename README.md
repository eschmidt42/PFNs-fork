# PFNs-fork
> Fork of [this repo](https://github.com/SamuelGabriel/PFNs).

## Install with uv

The project requires Python 3.10 or newer. `uv sync` installs the project in editable mode along with
the default development dependencies. Use a PyTorch-compatible Python version, as PyTorch does not
always support the latest Python release.
```bash
git clone https://github.com/eschmidt42/PFNs-fork.git
cd PFNs-fork
uv sync
```

Optional dependencies are available for notebooks, Bayesian optimization benchmarks, and priors:
```bash
uv sync --group notebooks
uv sync --extra benchmarks
uv sync --extra priors
```

## Experimenting

The [getting started notebook](./notebooks/getting-started.ipynb) contains the essential pieces. It is a somewhat cleaned up version of the collab notebook the authors provided.

## References

- Blog post series about tabular foundation models:
  - https://mindfulmodeler.substack.com/p/tabular-ml-is-about-to-get-weird
  - https://mindfulmodeler.substack.com/p/how-pfns-make-tabular-foundation
  - https://mindfulmodeler.substack.com/p/the-architecture-behind-tabpfn
- `nanoTabPFN` [repo](https://github.com/automl/nanoTabPFN) / [article](https://arxiv.org/pdf/2511.03634) - condensed version of `TabPFN`
