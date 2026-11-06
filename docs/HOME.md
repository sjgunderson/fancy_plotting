# fancy_plotting

`fancy_plotting` is a user contributed package for advanced histogram
operations on PHA-like datasets for use in [Sherpa](https://github.com/sherpa/sherpa),
the recommended X-ray modeling package for the Chandra X-ray Observatory.
Operations include grid matching, combining datasets to use a shared grid
for model fitting, and options for plotting PHA-like datasets as sums/averages
in run-time instead of combining data on disk.

The methods used in this package are based on the (open source)
Interactive Spectral Interpretation Software ([ISIS](https://github.com/houckj/isis);
[Houck & Denicola 2000](https://scixplorer.org/abs/2000ASPC..216..591H/abstract))
and ISIS package [fancy_plots](https://www.sternwarte.uni-erlangen.de/gitlab/remeis/isisscripts/-/blob/86676048abef1374e6ed5e09bcdf6d69632185b4/src/Mike/isis_fancy_plots.sl) by
Michael Nowak.

## Install

```bash
python -m pip install .
```

Install Matplotlib support for plotting:

```bash
python -m pip install '.[plotting]'
```

Sherpa/CIAO is required for the fitting adapters that integrate with Sherpa.
The numerical grid and rebinning tools can be used without Sherpa.

## Quick start

```python
from fancy_plotting import match_dataset_grids, combine_datasets

group = match_dataset_grids([1, 2])
group = combine_datasets(group)
group.set_source(source)
fit = group.make_fit().fit()
```

By default, `combine_datasets` prepares a joint fit over the original PHA
datasets, preserving their individual responses and Sherpa background
handling. Use `combine_datasets(group, mode="sum")` for an explicitly summed
spectrum; this mode does not support background-bearing or subtracted inputs.

See `docs/GRATING_MATCHING.md`, `docs/GROUPING.md`, `docs/PLOTTING.md`, and
the other files in `docs/` for detailed APIs and examples.

## Tests

```bash
python -m pip install '.[test]'
python -m pytest
```

The test suite includes Sherpa/CIAO integration tests, which require an
environment where Sherpa is available.
