# personalitygen

![personalitygen social preview](https://raw.githubusercontent.com/btfranklin/personalitygen/main/.github/social%20preview/personalitygen_social_preview.jpg "personalitygen")

`personalitygen` generates simulated character personalities for games,
storytelling, simulations, and tests. The project models conventional Big Five
(OCEAN) profiles and Adaptive Bifurcated Big Five (ABBF) signed vectors.

## Implementations

The repository is organized around first-class language packages:

- [Python](packages/python/README.md): the current dependency-free Python
  3.11+ implementation, published on
  [PyPI](https://pypi.org/project/personalitygen/).
- TypeScript: planned as an equal implementation of the same behavioral
  contract.

## Repository layout

```text
packages/
  python/        Python package, tests, and examples
docs/            Architecture, quality, usage, and decisions
```

## Python development

```bash
pdm install -p packages/python --group dev
pdm run -p packages/python test
pdm run -p packages/python lint
```

See the [documentation index](docs/README.md) for architecture, quality, and
maintenance guidance.
