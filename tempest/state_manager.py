from pathlib import Path
from typing import Union, Optional
import os
import dill
import numpy as np

CURRENT_STATE_KEYS = frozenset(
    {
        "u",
        "x",
        "logl",
        "assignments",
        "blobs",
        "acceptance",
        "steps",
        "efficiency",
        "ess",
        "beta",
        "logz",
        "calls",
        "iter",
    }
)

HISTORY_STATE_KEYS = frozenset(
    {
        "u",
        "x",
        "logl",
        "blobs",
        "iter",
        "logz",
        "calls",
        "steps",
        "efficiency",
        "ess",
        "acceptance",
        "beta",
    }
)


class StateManager:
    """
    Unified state management for particle sampling operations.

    Manages both current particle state and historical iterations with a
    fluent API. Handles state persistence, weight computation, and evidence
    estimation.

    Parameters
    ----------
    n_dim : int
        Dimension of the parameter space.

    Attributes
    ----------
    n_dim : int
        Dimension of the parameter space.
    _current : dict
        Dictionary storing current state values.
    _history : dict
        Dictionary storing historical state values (list per key).
    """

    def __init__(self, n_dim: int):
        self.n_dim = n_dim

        self._current = dict.fromkeys(CURRENT_STATE_KEYS, None)
        self._history = {key: [] for key in HISTORY_STATE_KEYS}
        self._results_dict = None

    def get_current(self, key: Optional[str] = None):
        """
        Get current state values.

        Parameters
        ----------
        key : str, optional
            Specific state key to retrieve. If None, returns all current state.

        Returns
        -------
        value : any or dict
            Value for specified key or dict of all current state values.
            Returns a copy to prevent unintended mutations.

        Raises
        ------
        KeyError
            If key is not a valid current state key.
        """
        if key is None:
            return {k: self._ensure_copy(v) for k, v in self._current.items()}
        else:
            self._validate_current_key(key)
            value = self._current[key]
            return self._ensure_copy(value)

    def set_current(self, key: str, value):
        """
        Set a single current state value.

        Parameters
        ----------
        key : str
            State key to set.
        value : any
            Value to assign.

        Raises
        ------
        KeyError
            If key is not a valid current state key.
        """
        self._validate_current_key(key)
        self._current[key] = value
        self._results_dict = None

    def update_current(self, data_dict: dict):
        """
        Update multiple current state values at once.

        Parameters
        ----------
        data_dict : dict
            Dictionary of state key-value pairs to update.

        Raises
        ------
        KeyError
            If any key in data_dict is not a valid current state key.
        """
        for key, value in data_dict.items():
            self._validate_current_key(key)
            self._current[key] = value
        self._results_dict = None

    def get_history(self, key: str, index: Optional[int] = None, flat: bool = False):
        """
        Get historical state values.

        Parameters
        ----------
        key : str
            State key to retrieve.
        index : int, optional
            Specific iteration index to retrieve. If None, returns all iterations.
        flat : bool, optional
            If True and index is None, returns flattened array across all iterations.
            If False, returns list of arrays (one per iteration).

        Returns
        -------
        value : numpy.ndarray or list of numpy.ndarray
            Historical state values.

        Raises
        ------
        KeyError
            If key is not a valid history state key.
        IndexError
            If index is out of range.

        Examples
        --------
        >>> state = StateManager(n_dim=2)
        >>> state.set_current('u', np.random.randn(10, 2))
        >>> state.commit_current_to_history()
        >>> u = state.get_history('u')
        >>> u.shape
        (1, 10, 2)
        >>> u_flat = state.get_history('u', flat=True)
        >>> u_flat.shape
        (20,)
        """
        self._validate_history_key(key)

        if index is None:
            if flat:
                return np.concatenate(self._history[key])
            else:
                return np.asarray(self._history[key])
        else:
            if index >= len(self._history[key]) or index < 0:
                raise IndexError(f"Index {index} out of range for history key '{key}'")
            return self._ensure_copy(self._history[key][index])

    def commit_current_to_history(self):
        """
        Commit current state to history.

        Appends current state values to history lists. This should be called
        once per iteration to record the complete state.

        Notes
        -----
        Values of None are stored as is. Keys not in history are skipped.
        """
        for current_key in CURRENT_STATE_KEYS:
            if current_key in HISTORY_STATE_KEYS:
                value = self._current[current_key]
                if value is not None:
                    self._history[current_key].append(self._ensure_copy(value))
        self._results_dict = None

    def compute_logw_and_logz(self, beta_final: float = 1.0, normalize: bool = True):
        """
        Compute importance log-weights for all collected samples.

        Targets a final inverse temperature ``beta_final`` and computes the
        corresponding log-evidence estimate.

        This implementation is robust to variable numbers of active particles
        across iterations (adaptive n_active). It treats the overall proposal as
        a mixture over the per-iteration tempered distributions, weighted by the
        number of particles drawn from each (balance heuristic for MIS).

        Let T be the number of iterations for which particles were stored, with
        per-iteration parameters (beta_t, logZ_t, n_t) where n_t is the number
        of particles in iteration t and N = sum_t n_t. For each collected sample
        s with log-likelihood logl_s, the (unnormalized) importance weight is

            logw_s = beta_final * logl_s - log( sum_t (n_t/N) * exp(beta_t * logl_s - logZ_t) ).

        Parameters
        ----------
        beta_final : float, optional
            Target inverse temperature. Defaults to 1.0 (posterior).
        normalize : bool, optional
            If True, return log-weights normalized to sum to 1 in linear space.

        Returns
        -------
        logw : numpy.ndarray, shape (N,)
            Per-sample log-weights, optionally normalized so that exp(logw).sum() == 1.
        logz_new : float
            Log-evidence estimate for the target at ``beta_final`` based on
            the (unnormalized) log-weights.
        """
        beta = np.asarray(self.get_history("beta"))

        if beta.size == 0:
            return np.array([]), -np.inf

        logz_iter = np.asarray(self.get_history("logz"))
        logl_all = self.get_history("logl", flat=True)

        A = logl_all * beta_final

        logl_per_iter = self._history.get("logl")
        n_per_iter = np.array([len(logl_per_iter[t]) for t in range(len(beta))])
        N_total = n_per_iter.sum()

        b = logl_all[:, None] * beta[None, :] - logz_iter[None, :]

        log_mixture_weights = np.log(n_per_iter) - np.log(N_total)
        b_weighted = b + log_mixture_weights[None, :]

        B = np.logaddexp.reduce(b_weighted, axis=1)

        logw = A - B

        logz_new = np.logaddexp.reduce(logw) - np.log(logw.size)

        if normalize and logw.size:
            logw = logw - np.logaddexp.reduce(logw)

        return logw, logz_new

    def compute_results(self) -> dict:
        """
        Compute complete results dictionary from historical state.

        Returns
        -------
        results_dict : dict
            Dictionary containing all historical state and computed log-weights.

        Notes
        -----
        Results are cached after first computation to avoid recomputation.
        """
        if self._results_dict is None:
            self._results_dict = dict()
            for key in self._history.keys():
                self._results_dict[key] = self.get_history(key)

            logw, _ = self.compute_logw_and_logz(1.0)
            self._results_dict["logw"] = logw

        return self._results_dict

    def save_state(self, path: Union[str, Path], exclude: Optional[list] = None):
        """
        Save current state to file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save state file.
        exclude : list, optional
            List of keys to exclude from saving. Defaults to ['pbar', 'pool', 'distribute'].

        Notes
        -----
        Uses atomic write pattern with temporary file to ensure data integrity.
        """
        if exclude is None:
            exclude = ["pbar", "pool", "distribute"]

        print(f"Saving state to {path}")
        Path(path).parent.mkdir(exist_ok=True)
        temp_path = Path(path).with_suffix(".temp")

        state_dict = {
            "_current": self._current,
            "_history": self._history,
            "n_dim": self.n_dim,
        }

        for key in exclude:
            state_dict.pop(key, None)

        with open(temp_path, "wb") as f:
            dill.dump(file=f, obj=state_dict)
            f.flush()
            os.fsync(f.fileno())

        os.rename(temp_path, path)

    def load_state(self, path: Union[str, Path]):
        """
        Load state from file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to load state file from.

        Notes
        -----
        Merges loaded state with existing state. Loaded values override
        current values where keys overlap.
        """
        with open(path, "rb") as f:
            state_dict = dill.load(file=f)

        if "_current" in state_dict:
            self._current.update(state_dict["_current"])
        if "_history" in state_dict:
            self._history.update(state_dict["_history"])
        if "n_dim" in state_dict:
            self.n_dim = state_dict["n_dim"]

        self._results_dict = None

    def _validate_current_key(self, key: str):
        """Validate key is a valid current state key."""
        if key not in CURRENT_STATE_KEYS:
            raise ValueError(
                f"Invalid current state key '{key}'. Valid keys: {sorted(CURRENT_STATE_KEYS)}"
            )

    def _validate_history_key(self, key: str):
        """Validate key is a valid history state key."""
        if key not in HISTORY_STATE_KEYS:
            raise ValueError(
                f"Invalid history state key '{key}'. Valid keys: {sorted(HISTORY_STATE_KEYS)}"
            )

    def _ensure_copy(self, value):
        """Return copy of value to prevent unintended mutations."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy()
        return value
