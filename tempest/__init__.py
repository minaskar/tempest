__bibtex__ = """
@article{karamanis2025persistent,
  title={Persistent Sampling: Enhancing the Efficiency of Sequential Monte Carlo},
  author={Karamanis, Minas and Seljak, Uro{\v{s}}},
  journal={Statistics and Computing},
  volume={35},
  number={5},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
"""
__url__ = "https://tempest-sampler.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "MIT"
__description__ = "A Python implementation of Persistent Sampling for accelerated Bayesian Computation"
__version__ = "0.1.2"

from .sampler import Sampler

__all__ = ["Sampler"]
