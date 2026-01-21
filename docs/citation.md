# Citation

If you use Tempest in your research, please cite the following papers.

## Papers

### Primary Method Paper

```bibtex
@article{karamanis2025persistent,
  title={Persistent {S}ampling: {E}nhancing the {E}fficiency of {S}equential {M}onte {C}arlo},
  author={Karamanis, Minas and Seljak, Uro{\v{s}}},
  journal={Statistics and Computing},
  volume={35},
  number={5},
  pages={1--22},
  year={2025},
  publisher={Springer},
  doi={10.1007/s11222-025-10682-y},
  eprint={2407.20722},
  archiveprefix={arXiv}
}
```

**arXiv**: [arXiv:2407.20722](https://arxiv.org/abs/2407.20722)

**DOI**: https://doi.org/10.1007/s11222-025-10682-y

**Abstract**: Sequential Monte Carlo (SMC) samplers are powerful tools for Bayesian inference but suffer from high computational costs due to their reliance on large particle ensembles for accurate estimates. We introduce persistent sampling (PS), an extension of SMC that systematically retains and reuses particles from all prior iterations to construct a growing, weighted ensemble. By leveraging multiple importance sampling and resampling from a mixture of historical distributions, PS mitigates the need for excessively large particle counts, directly addressing key limitations of SMC such as particle impoverishment and mode collapse. Crucially, PS achieves this without additional likelihood evaluations—weights for persistent particles are computed using cached likelihood values. This framework not only yields more accurate posterior approximations but also produces marginal likelihood estimates with significantly lower variance, enhancing reliability in model comparison. Furthermore, the persistent ensemble enables efficient adaptation of transition kernels by leveraging a larger, decorrelated particle pool. Experiments on high-dimensional Gaussian mixtures, hierarchical models, and non-convex targets demonstrate that PS consistently outperforms standard SMC and related variants, including recycled and waste-free SMC, achieving substantial reductions in mean squared error for posterior expectations and evidence estimates, all at reduced computational cost. PS thus establishes itself as a robust, scalable, and efficient alternative for complex Bayesian inference tasks.


---

## Links

- **GitHub**: [github.com/minaskar/tempest](https://github.com/minaskar/tempest)

---

## Acknowledgements

Tempest was developed by:

- **Minas Karamanis** (University of Edinburgh)
- **Uroš Seljak** (UC Berkeley / Lawrence Berkeley National Laboratory)

---

## License

Tempest is free software made available under the **MIT License**.

```
Copyright (c) 2026 Minas Karamanis and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See the [LICENCE](https://github.com/minaskar/tempest/blob/master/LICENCE) file for full details.
