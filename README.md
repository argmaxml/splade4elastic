# PyPi Cookie Cutter

## Folder Structure
1. `pkg/` - the package to be deployed.
2. `docs/` - template for readthedocs documentation.
3. `test/` - unit tests for the package.
4. `notebooks/` - throw away jupyter notebooks.

## Automation
Any update to the master branch should trigger a github action to push latest version to `pypi`
