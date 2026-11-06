# Grid Matching

This section describes the process of matching the grids of PHA-like data.
**The use of grid matching is intended for advanced users.** Users are
encouraged to read this documentation before using the provided functions.
The separately provided `plot_*` commands can be used independent of grid
matching but will have limited functionality. See [PLOTTING.md](PLOTTING.md)
for details.

Grid matching prepares a shared source-evaluation grid and registers an
observed-bin mapping used by plotting and combined fitting. See
[PLOTTING.md](PLOTTING.md) for plotting normalization. The shared grid
does not change the associated data files, similar
to how the Sherpa `ui.group_*` commands change the grids only in memory.

The simplest call for matching dataset grids is

```python
import sherpa.astro.ui as ui
import fancy_plotting as fp

ui.load_data(1, `path/to/pha1`)
ui.load_data(2, `path/to/pha2`)

matched = match_dataset_grids([1,2])
```

which creates a new object `matched` that can be referenced for model fitting.
The first dataset is the default reference if the reference options is omitted.
The matching operator is independent of set_analysis and instrument headers.
Matching uses wavelength internally, as ISIS does; there is no unit argument
or conversion for the user to manage.

A reference ID can also be given if the data order is not appropriate

```python
ids = [1,2,3,4]
matched = match_dataset_grids(ids, reference=3)
```

## Regridding to Courser or Finer Grids

For two datasets with bins $x_i$ and $x_j$ over the same energy range $[E_1, E_N]$
and no gaps, the processes of matching grids is equivalent to using Sherpa's
`group_*` commands such that $\Delta x_j = \Delta x_i$. However, if datasets
have two different energy ranges $[E_{i1}, E_{iN}]$ and $[E_{j1}, E_{jN}]$,
then `match_dataset_grids` will additionally append zero count bins to the
datasets such that $[E'_{j1},E'_{jN}] = [E_{i1}, E_{iN}]$.

The `match_dataset_grids` command is written generally to work with any histogram
grids. For example, this function works with high resolution data, such as
regridding the Chandra  High Energy Grating (HEG) onto the Medium Energy Grating
(MEG). Or, if one were to want, they could also regrid the XMM MOS1 and 2 spectra
onto the PN Camera. Using `match_dataset_grids` to regrid a high resolution data
set to a lower resolution one is the primary and intended use.

Opposite operations are also supported but are **heavily** discouraged. Matching
a courser grid to a finer grid will produce fractional counts. While these may
appear visually to be fine, model fitting will not be as accurate and dubious
results will result. Unless data is explicitly loaded in the order of Courser
$\rightarrow$ Finer, users are encouraged to use the reference keyword in
`match_dataset_grids`.

## Regridding and grouping

Native PHA counts and response files remain unchanged. A registered matched
view maps each selected native bin onto the reference grid. Therefore raw
arrays obtained from ui.get_data still have their original channel counts;
plot_data and combine_datasets use the aligned view.

The reference's current grouping and filtering define the common output
bins. Other members' native grouping is removed on working copies before
regridding, preserving native resolution; their noticed/quality channel
selections are retained. This implements the intended operation order:
native bins $\rightarrow$ matched reference bins $\rightarrow$ reference
grouping. Grouping all members with group_width is allowed, but the
reference determines the common view even when HEG/MEG native channel widths
differ.

**Non-uniform grouping commands will not retain a matched grid.** Due to
data-dependent computations of grouping commands like `group_snr` and
`group_counts`, a previously computed matched grid will no longer be accurate.

No extrapolation is performed. Counts outside the reference's selected
range are intentionally excluded from that view. Empty coverage contributes
zero. Pick a reference with the desired wavelength coverage, then select
plot limits as usual. Neither sum nor mean is an unfolded flux estimate.

Overlap weights are W[j,i] = length(target_j intersect source_i) /
width(source_i), measured in the fixed internal coordinate. Counts and folded
model predictions use the identical W. This assumes constant density within
an input bin; matching does not recover information finer than the native
resolution. Bin boundaries are not made artificially equal using loose tolerances.

## Errors and covariance

For covariance $C$ on native bins, propagation is $C_\mathrm{matched} = W C W^T$.
Off-diagonal terms are retained when a native bin is split. Later plot
rebinning applies its grouping matrix to the full covariance, so errors are
not underestimated by treating fractional pieces as independent counts.
PlotCurve.covariance exposes the displayed sparse covariance; PlotCurve.yerr
is the square root of its diagonal. Background subtraction/error propagation
happens on native bins through Sherpa before plotting regrids them.

## Combined fits

By default, `combine_datasets` follows the ISIS combination model: it returns
the matched `DatasetGroup`, and Sherpa fits the original PHA datasets jointly.
Each spectrum retains its own response, observed bins, and background
likelihood. The shared source grid avoids repeated source evaluation without
collapsing the member datasets into a single Poisson spectrum.

```python
from fancy_plotting import combine_datasets

group = combine_datasets(matched)
group.set_source(source)
result = group.make_fit().fit()
```

For UI fit-result bookkeeping, use `ui.fit(*group.datasets)` after assigning
the source in the session. Session-backed fits use Sherpa's configured
statistic and include associated background-region likelihoods. Session-free
fits with backgrounds or subtracted data require an explicit compatible
statistic, such as `WStat()` for unsubtracted Poisson backgrounds or a
chi-square statistic after subtraction. For raw background-free data, the
default remains `CStat`.

The method of combining datasets can also be used with arbitrary combinations.

```python
import sherpa.astro.ui as ui
import fancy_plotting as fp

ui.load_data(1, `path/to/pha1`)
ui.load_data(2, `path/to/pha2`)
ui.load_data(3, `path/to/pha3`)
ui.load_data(4, `path/to/pha4`)
ui.load_data(5, `path/to/pha5`)

match12 = match_dataset_grids([1,2])
match34 = match_dataset_grids([3,4])

pair12 = combine_datasets(matched12)
pair34 = combine_datasets(matched34)


pair12.set_source(source)
pair34.set_source(source)
ui.set_source(5, source)


ui.fit()
```
All loaded datasets will then be used for minimization, but the model will
only be computed on the grids of match12, match34, and 5.


### Advanced Usage

To create a single summed `Data1DInt` spectrum instead, opt in explicitly:

```python
summed = combine_datasets(matched, mode="sum")
summed.set_source(source)
result = summed.make_fit().fit()
```

This sum mode preserves the per-response folding and observed-grid
reprojection behavior described below. An explicit `grid=` or `name=` is
available only in this mode. It is restricted to background-free,
unsubtracted raw counts; it cannot preserve the member-level background
likelihoods. For fractionally rebinned inputs, it uses covariance-aware
Gaussian fitting and is approximate, particularly at low counts.

In sum mode, fractional rebinning creates correlated, noninteger counts. In
that case `make_fit` defaults to `CovarianceChi2`, a Gaussian statistic using the FULL
propagated covariance, with a banded Cholesky implementation suitable for
large grating grids. It is an approximation for count data, particularly at
low counts; it is not an exact Poisson likelihood. Native zero-count bins
use a one-count variance floor for this approximation. Singular covariance
from excessive upsampling requires a coarser fitting reference grid.

Stored statistical/systematic errors are propagated when available. Pure
whole-bin sums of raw Poisson counts retain the CStat default. Explicit
Poisson statistics are rejected by sum mode for fractional data.

If you register a fractionally rebinned combination in the UI:

```python
summed.register('sum')
ui.fit('sum')
```

Sum-mode registration selects the covariance statistic for that dataset. Do
not subsequently replace it with CStat. Fit that summed dataset alone.
Additional notice/ignore selections on it are reflected in its covariance.

Sum-mode combinations snapshot their reference grid and native selections;
rebuild one after regrouping or changing selections. Joint mode uses the
original datasets and therefore does not snapshot or combine their observed
counts. Combination weights and non-DataPHA datasets are not supported.

## Validation

Synthetic shifted, differently sized grating bins now reproduce the former
HEG/MEG failure and pass. Tests cover counts, average/sum normalization,
matching before/after unit conversion, covariance through later grouping,
large sparse covariances, and identical data/model regrid operators.

Real Sherpa regression tests are included for HETG-like PHA/RMF data,
reference grouping, background plotting, combined predictions, covariance
statistics, and UI registration. Sherpa and your real FITS files are not
available in the development workspace, so those tests remain unexecuted.
This does not claim validation on your observation or complete statistical
parity with every ISIS implementation.
