# Development Guide

This guide covers the development workflow and best practices for maintaining Tempest.

## Changelog Maintenance

The changelog follows the [Keep a Changelog](https://keepachangelog.com) format and documents all notable changes to the project.

### When to Update

Update `CHANGELOG.md` when:
- Adding new features
- Changing existing functionality
- Fixing bugs
- Deprecating or removing features
- Addressing security vulnerabilities

### Adding Entries

1. Add entries to the `[Unreleased]` section at the top of `CHANGELOG.md`
2. Use appropriate category:
   - **Added**: New features
   - **Changed**: Changes in existing functionality
   - **Deprecated**: Soon-to-be removed features
   - **Removed**: Now removed features
   - **Fixed**: Bug fixes
   - **Security**: Security vulnerability fixes

3. Provide clear, user-friendly descriptions:
   ```markdown
   ### Added
   - Support for custom proposal distributions
   - New method for computing posterior summaries

   ### Fixed
   - Fixed memory leak in parallel MCMC sampling
   - Corrected handling of NaN values in likelihood
   ```

### Release Process

1. Update version in `tempest/__init__.py`
2. Move `[Unreleased]` entries to new version section:
   ```markdown
   ## [1.1.0] - 2026-01-15

   ### Added
   - (moved from Unreleased)
   ```
3. Add release date in ISO 8601 format (YYYY-MM-DD)
4. Create git tag:
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
5. Release manually to PyPI when ready:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   twine check --strict dist/*
   twine upload dist/*
   ```
6. Create new `[Unreleased]` section for next version

### Version Links

After creating a release tag, update the version links at the bottom of `CHANGELOG.md`:
```markdown
[Unreleased]: https://github.com/minaskar/tempest/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/minaskar/tempest/releases/tag/v1.1.0
[1.0.0]: https://github.com/minaskar/tempest/releases/tag/v1.0.0
```

## Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_sampler

# Run with pytest (if installed)
pytest tests/
```

### Test Coverage

Ensure new features include corresponding tests. Test files are in the `tests/` directory.

## Code Style

Tempest uses the following conventions:
- Type hints where appropriate (Python 3.8+)
- Docstrings following NumPy style
- PEP 8 formatting

## Dependencies

Core dependencies are defined in:
- `pyproject.toml` (project dependencies)
- `requirements.txt` (simplified list)

When adding new dependencies:
1. Update both `pyproject.toml` and `requirements.txt`
2. Consider impact on existing users
3. Document in changelog

## Documentation

Documentation is built using MkDocs. To build locally:

```bash
# Install docs requirements
pip install -r docs/requirements.txt

# Serve docs
mkdocs serve

# Build docs
mkdocs build
```

## Pull Requests

When submitting a PR:
1. Update CHANGELOG.md with relevant entries
2. Add or update tests
3. Ensure all tests pass
4. Update documentation if needed
5. Reference any related issues

## Release Checklist

Before creating a release:
- [ ] Update version number in `tempest/__init__.py`
- [ ] Update CHANGELOG.md with all changes
- [ ] Ensure all tests pass
- [ ] Update documentation if needed
- [ ] Tag release with `git tag v{version}`
- [ ] Build and upload to PyPI manually (see Release Process)
- [ ] Verify release on PyPI
- [ ] Create GitHub Release with release notes
