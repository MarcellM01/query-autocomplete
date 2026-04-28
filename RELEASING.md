# Releasing

Use this checklist to publish a new version to GitHub, PyPI, and ReadTheDocs.

## 1. Update the package version

Edit `python-package/pyproject.toml`:

```toml
version = "0.1.2"
```

## 2. Run tests

From the repository root:

```bash
PYTHONPATH=core/src ./.venv/bin/python -m pytest
```

## 3. Commit the release changes

```bash
git status --short
git add README.md docs/index.md python-package/README.md python-package/pyproject.toml core/src/query_autocomplete tests
git commit -m "Prepare 0.1.2 release"
```

Adjust the commit message and file list for the actual release.

## 4. Push `main`

```bash
git push origin main
```

Pushing `main` updates the public GitHub README. ReadTheDocs should rebuild from the pushed docs if the project webhook is connected.

## 5. Tag the release

Tag the exact commit that is on `main`:

```bash
git tag -a v0.1.2 -m "v0.1.2"
git push origin v0.1.2
```

The PyPI release workflow runs when a tag matching `v*.*.*` is pushed. It runs tests, builds `python-package`, and publishes the package to PyPI.

## 6. Check the release

After the GitHub Action finishes:

- GitHub: confirm the repository README shows the new docs.
- PyPI: confirm the new version is published at https://pypi.org/project/query-autocomplete/
- ReadTheDocs: confirm the latest docs rebuilt at https://query-autocomplete.readthedocs.io/en/latest/

