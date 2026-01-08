# Citation

If you use Tempest in your research, please cite the following papers.

## Papers

### Primary Method Paper

```bibtex
@article{karamanis2022accelerating,
    title={Accelerating astronomical and cosmological inference with preconditioned Monte Carlo},
    author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David and Seljak, Uro{\v{s}}},
    journal={Monthly Notices of the Royal Astronomical Society},
    volume={516},
    number={2},
    pages={1644--1653},
    year={2022},
    publisher={Oxford University Press}
}
```

**Abstract**: We introduce the Persistent Sampling (PS) algorithm, a novel method for accelerating Bayesian inference in high-dimensional parameter spaces. PS combines the advantages of Sequential Monte Carlo and Markov Chain Monte Carlo methods by using a normalizing flow to condition the target distribution.

---

### Software Paper

```bibtex
@article{karamanis2022pocomc,
    title={pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology},
    author={Karamanis, Minas and Nabergoj, David and Beutler, Florian and Peacock, John A and Seljak, Uros},
    journal={arXiv preprint arXiv:2207.05660},
    year={2022}
}
```

**Abstract**: We present Tempest (formerly pocoMC), a Python package for accelerated Bayesian inference using the Persistent Sampling algorithm. The package provides efficient posterior sampling and evidence estimation for problems in astronomy, cosmology, and beyond.

---

## Links

- **arXiv (Method Paper)**: [2112.05909](https://arxiv.org/abs/2112.05909)
- **MNRAS (Method Paper)**: [DOI](https://doi.org/10.1093/mnras/stac2272)
- **arXiv (Software Paper)**: [2207.05660](https://arxiv.org/abs/2207.05660)
- **GitHub**: [github.com/minaskar/tempest](https://github.com/minaskar/tempest)

---

## Acknowledgements

Tempest (formerly pocoMC) was developed by:

- **Minas Karamanis** (University of Edinburgh)
- **David Nabergoj** (University of Ljubljana)
- **Florian Beutler** (University of Edinburgh)
- **John A. Peacock** (University of Edinburgh)
- **Uroš Seljak** (UC Berkeley / Lawrence Berkeley National Laboratory)

---

## License

Tempest is free software made available under the **GPL-3.0 License**.

```
Copyright 2022-2026 Minas Karamanis and contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

See the [LICENSE](https://github.com/minaskar/tempest/blob/master/LICENCE) file for full details.
