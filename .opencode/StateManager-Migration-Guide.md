# StateManager API Migration Guide

This guide helps users migrate to the improved StateManager API introduced in Phases 1-6 of the refactoring effort.

## Overview

The StateManager improvements focus on:
- **Better encapsulation** - No more direct access to private attributes
- **Cleaner APIs** - Serialization, validation, and copy control
- **Better performance** - Eliminated redundant copying
- **Enhanced safety** - Optional strict validation mode

## All Changes Are Backward Compatible

**Good news:** All existing code should continue to work without modifications. The improvements are additive and follow backward-compatible design principles.

---

## Phase 1: New Accessor Methods

### Before (Deprecated Pattern)
```python
# Direct access to private attributes (discouraged)
beta_hist = sampler.state._history["beta"]
last_beta = beta_hist[-1] if len(beta_hist) > 0 else 0
n_iters = len(sampler.state._history["beta"])
```

### After (Recommended)
```python
# Use public accessor methods
last_beta = sampler.state.get_last_history("beta", default=0)
n_iters = sampler.state.get_history_length()
beta_hist = sampler.state.get_history("beta")
```

**Benefits:**
- No reliance on private implementation details
- Built-in default value handling
- Automatic copy semantics (prevents accidental mutation)

---

## Phase 2: Serialization API

### Before (Deprecated Pattern)
```python
# Manual dictionary construction
state_dict = {
    "_current": sampler.state._current,
    "_history": sampler.state._history,
    "n_dim": sampler.state.n_dim,
}

# Manual dictionary unpacking
state._current.update(state_dict["_current"])
state._history.update(state_dict["_history"])
state.n_dim = state_dict["n_dim"]
state._results_dict = None  # Easy to forget!
```

### After (Recommended)
```python
# Clean serialization API
state_dict = sampler.state.to_dict()

# Three ways to deserialize:
# 1. Create new instance
new_state = StateManager.from_dict(state_dict)

# 2. Update existing instance
state.update_from_dict(state_dict)

# 3. Load from file (uses update_from_dict internally)
state.load_state("checkpoint.pkl")
```

**Benefits:**
- Cache invalidation handled automatically
- Cleaner, more maintainable code
- Centralized serialization logic

---

## Phase 3: Copy Semantics Control

### Before (Implicit Behavior)
```python
# Unclear whether data is copied or shared
state.set_current("beta", my_array)
state.update_current({"logl": my_logl_array})
```

### After (Explicit Control)
```python
# Default: no copy (performance)
state.set_current("beta", my_array)  # Fast, but don't mutate my_array later!

# Explicit copy (safety)
state.set_current("beta", my_array, copy=True)  # Safe to mutate my_array

# Batch update with copy
state.update_current({"beta": 0.5, "logl": my_logl}, copy=True)

# Get operations always return copies (unchanged)
beta = state.get_current("beta")  # Always safe to modify beta
```

**Benefits:**
- Explicit performance/safety trade-off
- Clear documentation of copy behavior
- Get operations always safe (return copies)

---

## Phase 4: Centralized Cache Invalidation

**No user-facing changes** - Internal improvement only.

This phase replaced scattered `self._results_dict = None` statements with a centralized `_invalidate_cache()` method. Reduces risk of bugs in future maintenance.

---

## Phase 5: Removed Redundant Copying

**No user-facing changes** - Performance improvement only.

This phase eliminated double-copying in `Sampler.sample()` return values since `StateManager.get_current()` already returns copies.

**Expected impact:** ~50% reduction in array copying for `sample()` return values.

---

## Phase 6: Strict Validation Mode

### Before (Lenient)
```python
# Commits succeed even with missing required keys
state.set_current("beta", None)
state.commit_current_to_history()  # Silently succeeds
```

### After (Optional Strict Mode)
```python
# Default: lenient (backward compatible)
state.commit_current_to_history()  # Still works as before

# Strict mode: catch bugs early
state.set_current("beta", 0.5)
state.set_current("logl", my_logl_array)
state.commit_current_to_history(strict=True)  # OK

# Missing required keys
state.set_current("beta", None)
state.commit_current_to_history(strict=True)  # ValueError!
```

**Required Keys for Strict Mode:**
- `beta` - Required for iteration tracking
- `logl` - Required for likelihood-based operations

**Benefits:**
- Opt-in validation for development/debugging
- Backward compatible (default is `strict=False`)
- Clear error messages identifying missing keys

---

## Migration Checklist

For users with existing codebases:

1. **Search for direct attribute access:**
   ```bash
   grep -r "\.state\._history\|\.state\._current" .
   ```
   Replace with public accessor methods.

2. **Update serialization code:**
   - Replace manual dict construction with `to_dict()`
   - Replace manual unpacking with `from_dict()` or `update_from_dict()`

3. **Review copy semantics:**
   - Add `copy=True` to setters if you mutate data after setting
   - Remember: getters always return copies (no change needed)

4. **Consider strict mode:**
   - Use `strict=True` during development
   - Remove or keep `strict=False` in production (your choice)

5. **Run tests:**
   ```bash
   python -m unittest discover tests
   ```

---

## Performance Considerations

### Improved Performance
- Phase 5 eliminated ~50% of redundant copying in `sample()` returns
- No performance regressions in any phase

### Copy Control Trade-offs
```python
# Fast but unsafe (don't mutate after!)
state.set_current("logl", large_array, copy=False)  # Default

# Safe but slower (copies data)
state.set_current("logl", large_array, copy=True)
```

**Recommendation:**
- Use `copy=False` (default) when you won't modify data afterward
- Use `copy=True` when safety is more important than performance
- Remember: get operations always copy (for safety)

---

## Common Patterns

### Pattern 1: Progress Bar Updates
```python
# Before
beta_hist = self.state._history["beta"]
calls_hist = self.state._history["calls"]
pbar.update_stats({
    "beta": beta_hist[-1] if len(beta_hist) > 0 else 0,
    "calls": calls_hist[-1] if len(calls_hist) > 0 else 0,
})

# After
pbar.update_stats({
    "beta": self.state.get_last_history("beta", default=0),
    "calls": self.state.get_last_history("calls", default=0),
})
```

### Pattern 2: State Checkpointing
```python
# Before
with open(path, "wb") as f:
    dill.dump({
        "_current": self.state._current,
        "_history": self.state._history,
        "n_dim": self.state.n_dim,
    }, f)

# After
state_dict = self.state.to_dict()
with open(path, "wb") as f:
    dill.dump(state_dict, f)
```

### Pattern 3: Iteration Counting
```python
# Before
n_iterations = len(self.state._history["beta"])

# After
n_iterations = self.state.get_history_length()
```

---

## Questions?

If you encounter issues during migration:

1. Check that all tests pass: `python -m unittest discover tests`
2. Review the StateManager class docstring for examples
3. Consult the improvement plan: `.opencode/plans/state-manager-improvements.md`

All improvements are backward compatible, so migration can be done incrementally.
