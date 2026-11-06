# Plotting Model Fits and Residuals

The provided `plot_*` commands automatically overplot data, available
response-folded models, and optional residual panels. Matplotlib style
names and Sherpa's analysis settings still control presentation.
No separate `ui.plot_fit` is required. For specific details on `plot_*`
options, see [COUNTS_PLOTTING.md](COUNTS_PLOTTING.md) and [UNFOLDING.md](UNFOLDING.md)


```python
from sherpa.astro import ui
from fancy_plotting import match_dataset_grids, plot_data

# PHA/ARF/RMF data are already loaded. Matching requires no model or units.
matched = match_dataset_grids([1, 2, 3, 4])
matched.set_source(source)
ui.set_stat('cstat')
ui.set_analysis('wavelength')

result = plot_data(
    [[1, 2], [3, 4]],
    res=4,
    color=['tab:blue', 'tab:orange'],
    marker='o', linestyle='none', capsize=2,
    model_styles={'linestyle': 'solid', 'linewidth': 1.5},
    model_plot='line',
    label=['Group A', 'Group B'],
    ylog=True,
)
result.residual_ax.set_ylim(-4, 4)
result.figure.savefig('fit.png', dpi=200)
```

Existing models assigned through normal Sherpa calls work too; using
`matched.set_source` is what enables shared source evaluation. Plotting
never assigns a source, changes parameters, or runs a fit. However, if
a user changes a parameter manually or a fit-specific global variable
(e.g., the AtomDB library for `apec`), the model will be re-evaluated.
`plot_*` calls without changing models will use a cached evaluation.

## Default behavior and controls

- `show_model='auto'` overlays an available model for each displayed curve.
  A combined curve needs models for all its members; partial model sums are
  never drawn. An incomplete auto-model group has None in model_curves.
- `show_model=True` requires a model for every member; missing IDs produce
  an error before axes are cleared. `show_model=False` hides the model line.
- `show_model` can be a per-curve list or an entry in per-curve `styles`.
- `res=0` is the single-panel default. A nonzero res automatically creates
  or reuses a lower panel, with a shared X axis and linear Y axis.
- `show_model=False, res=0` performs no model lookup or evaluation. With
  res>0, a model is still evaluated to calculate residuals even if hidden above.
- `residual_stat=None` reads the selected Sherpa statistic. An explicit
  residual_stat selects a plotting diagnostic only, without changing the fit.

## Residual modes

| res | Diagnostic | Calculation scope |
| --- | --- | --- |
| 0 | No residual panel | Model/data above only |
| 1 | Standardized residual | Each input dataset |
| 2 | Signed squared standardized residual | Each input dataset |
| 3 | Data/model ratio | Each input dataset |
| 4 | Standardized residual | Each displayed group |
| 5 | Signed squared standardized residual | Each displayed group |
| 6 | Data/model ratio | Each displayed group |

For `[[1, 2], [3, 4]]`, modes 1-3 produce four residual curves; modes 4-6
produce two. Their order is outer group first, then member order. Individual
residuals retain the matched physical view when matching was prepared.

Model predictions go through their own ARF/RMF/exposure/AREASCAL and the same
noticed-channel selection, reference matching, and plot-only rebinning as
the data. The upper model curve uses the same exposure combination, density,
plot factor, scale, and offset as its data curve. Gaps remain gaps.

Residuals are calculated after whole-bin plot rebinning, from raw count sums,
without exposure weights, display scale/offset, density, or plot-factor weights.
This follows Michael Nowak's `fancy_plots` resplot, which calls make_data with
unit scalings. Consequently, a combined count diagnostic can be zero even when
a sum of unequal-exposure rates differs from its upper model curve. The distinction
is intentional; it is not an average of individual standardized residuals.

## Statistics and uncertainties

For raw observed counts D and raw predicted source-aperture counts M:

- Data-error chi is `(D-M)/sigma`, with sigma from propagated measurement
  covariance, including background uncertainty. Modes 2/5 show
  `sign(chi)*chi**2`, not the unsigned contribution to chi-square.
- Chi2Gehrels selects the Gehrels error estimator unless error_method was
  supplied explicitly. Explicit stored errors/covariance remain authoritative.
- Chi2ModVar uses the propagated total-model count variance. For a native
  Poisson count model this starts as diag(M); fractional rebinning propagates
  it with the same overlap operator, rather than resetting its diagonal.
- Cash and CStat select `sign(D-M)*sqrt(C_i)` in modes 1/4, or
  `sign(D-M)*C_i` in modes 2/5, where
  `C_i = 2*(M-D+D*log(D/M))`. These are deviance diagnostics relative to the
  saturated model, rather than Sherpa Cash's raw per-bin expression.
- An explicit `residual_stat='ml'` reproduces the supplied ISISscripts ML
  branch, using `2*sign(D-M)*(M + log(D+1) - D*log(M))` for M>0 in modes 2/5,
  and the signed square root of that term in modes 1/4. For M=0 the D*log(M)
  term is omitted, as in the source. The uncentered ML expression can itself
  be negative, so its displayed sign need not equal sign(D-M).
  This is an explicit legacy diagnostic, not a
  claim of a generic mapping for all Sherpa likelihood statistics.

Gaussian standardized residuals have unit marginal error bars. Signed-square
and Cash/ML diagnostics are histograms without error bars. The lower panel's
reference line is zero, or one for ratio modes.

For ratios, data-error sigma gives symmetric errors `sigma/(M-B)` for a net
plot, where B is the measured background subtracted from both numerator and
denominator. Model-error sigma gives asymmetric bounds with denominators
`M-B+sigma` and `M-B-sigma`; bins requiring a nonpositive denominator are
masked. With backgrounds included for display, the ratio uses D/M instead.

Undefined residuals are NaN with valid=False in the returned ResidualCurve.
They are omitted only from lower-panel artists; upper data remain visible.
Cash's zero-count limit is C_i=2M, and an exact D=M match has residual zero,
including D=M=0. Positive data with zero model are not drawn as finite points.
Nearly equal large counts use a stable expansion to avoid cancellation.

WStat and other unsupported statistic-to-residual mappings are not silently
treated as Cash or ordinary chi. Choose res=3/6 for a ratio, or explicitly
choose residual_stat='data' for a measurement-error diagnostic. This does
not change the fitting statistic. Stored-error Chi2 and covariance chi-square
use the measurement-error branch; Chi2XspecVar/constant-variance/least-squares
variants do not currently have automatic standardized-residual mappings.

The covariance-aware implementation retains cross-bin correlations when
coarsening. Pointwise plotted chi values use marginal errors, not a whitened
residual vector; the displayed squared values therefore need not sum to a
correlated Gaussian fit statistic. Cash/ML functions applied to fractional
matched counts are display diagnostics, not a valid independent-Poisson
likelihood for those correlated counts. Plotting does not change any fitting
statistic or remove the previously documented limits of fractional combination.

## Backgrounds

The displayed background selection is controlled by subtract_background.
Counts are handled in raw source-aperture space before display conversion:

- Standard source expressions without a background model use the measured,
  appropriately scaled background to form total predicted counts, as ISIS does.
- Defined background source models are included using Sherpa's own response
  construction and scaling. The same measured B is subtracted from both D and
  M for a net display. With a modeled background, this is deliberately a
  background-subtracted total-model display, not just the source component.
- Standardized/deviance diagnostics compare raw D and total M. Measurement
  variance includes background uncertainty regardless of the display selection.
- A source `set_full_model` expression follows its PHA's subtraction convention:
  an unsubtracted PHA's expression predicts raw totals; a subtracted PHA's
  expression predicts net counts and measured B is added for raw diagnostics.
- Separate background full models without a background source expression are
  not automatically interpreted. Use a source full model containing the raw
  total prediction, or supply explicit total model arrays with models=.

PHA arrays, subtraction flags, grouping, filters, units, and model assignments
are preserved. Background response construction uses copies. Model startup/
teardown may update normal evaluation caches, as ordinary plotting does.
All response models are started before evaluation, so matched shared-source
wrappers reuse one source evaluation per parameter vector across datasets.

## Matplotlib styles and axes

`model_styles` accepts one common Matplotlib style dictionary, or one per
displayed group. Its default is a dashed histogram in the data curve's color.
Model styles describe lines/markers; model-parameter error bars are not inferred.
`residual_styles` accepts a common dictionary or one per residual curve;
its length thus depends on res. These dictionaries accept style controls,
not rebin/combine/scale or panel settings. Use ordinary Matplotlib options:

```python
result = plot_data(
    [[1, 2]], res=1,
    model_styles={'color': 'black', 'linestyle': 'solid'},
    residual_styles=[{'color': 'navy', 'marker': 'o'},
                     {'color': 'orange', 'marker': 's'}],
)
```

You can supply `ax=(data_ax, residual_ax)` or separate ax and residual_ax
arguments. With one supplied data Axes, the lower panel is added inside that
subplot's area; unrelated subplots remain intact. With no axes, both panels
are created automatically. The upper X labels are hidden and the lower panel
receives the analysis-unit label. The lower Y axis is always linear.

`overplot=True` reuses the linked lower panel and its reference line. Incompatible
analysis units or residual families/statistics are rejected before existing
artists are changed. Replacing an automatic two-panel plot with res=0 removes
the automatically created lower axes and restores the original data area.
For supplied two-axes layouts, res=0 clears stale residual artists but retains
the user's axes. Use result.residual_ax.set_ylim or other native Axes methods
for lower-panel customization.

## Arrays and return values

For offline use, `models={id: array}` provides deterministic folded predicted
counts in each input dataset's native bins/channel order, before plotting
grouping, selection, density, exposure normalization, or background subtraction.
These are counts, not source fluxes or rate densities. All arrays must be
finite and match their input counts. SpectrumData already represents selected
input bins; DataPHA arrays must cover the full original channel array.

PlotResult retains ax, figure, curves, lines, errorbars, and artists, and adds:

- model_curves and model_artists, one per upper data curve (None where absent).
- residual_ax, residuals, and residual_artists.
- reference_line for the lower panel.

ModelCurve contains members, lo/hi, y, and label. ResidualCurve contains members,
lo/hi, y, yerr (a 2-by-n asymmetric-error array or None), valid, mode,
statistic, group_index, and label. These arrays describe diagnostics before
any Matplotlib axis clipping; original data and parameter values stay unchanged.

## Examples and validation

`python examples/fit_plotting_demo.py` runs without Sherpa and saves an actual
Matplotlib comparison of individual, combined, and ratio residual panels.
`examples/ciao_fit_plotting_demo.py` prepares synthetic PHA/ARF/RMF datasets,
attaches a shared model, and displays all six modes under CStat in CIAO.

The numerical/rendering tests exercise the actual normalization, regrid,
covariance, diagnostic, and Matplotlib code. Separate integration tests cover
Sherpa model responses, grouping/filtering, unmodeled and modeled backgrounds,
source full models, shared source-call counts, analysis switches, and preservation
of data/parameters. Those integration tests are included but unrun in this
environment because Sherpa is unavailable. Full ISIS runtime parity is not claimed.

