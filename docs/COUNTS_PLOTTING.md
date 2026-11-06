# Plotting Counts

`plot_counts` displays integrated counts per bin, existing folded models,
and optional lower-panel residuals.

```python
from sherpa.astro import ui
from fancy_plotting import plot_counts

ui.set_analysis('energy')
result = plot_counts(
    [[1, 2], [3, 4]],
    res=4,
    color=['tab:blue', 'tab:orange'],
    marker='o', linestyle='none', capsize=2,
    label=['Group A', 'Group B'],
    model_styles={'linestyle': 'solid', 'linewidth': 1.5},
    ylog=True,
)
result.residual_ax.set_ylim(-4, 4)
result.figure.savefig('counts.png', dpi=200)
```

The outer sequence gives displayed curves; each inner sequence combines
datasets. `[1, 2]` displays two individual spectra; `[[1, 2]]` sums them into
one. Matching, if needed, is prepared separately with
`match_dataset_grids(ids)` and requires no unit argument or model.

## Counts and analysis settings

The default `power=1` shows the count sum, with statistical errors and full
covariance propagated through matching and plot-only rebinning. No exposure
or bin-width division is applied. `combine='mean'` instead displays the mean
counts and scales covariance by the square of the number of members.
Count plotting does not require exposure metadata unless background scaling
itself needs it. Models still include their physical response/exposure
normalization; their displayed predictions are folded counts per bin.

The X axis follows the first displayed dataset's current `set_analysis`
coordinate, or the selected reference. `set_analysis('wavelength')` reverses
the spectral coordinate and bin order; integrated counts and covariance
are preserved with that permutation. Matching is unchanged. No units need
to be passed to either matching or plotting in a Sherpa workflow.

`plot_counts` ignores Sherpa's rate and positive `plot_fac` settings. Its
explicit `power` controls count weighting. Requests for `rate=True` or
`density=True` raise an error directing you to `plot_data`. `plot_data`
continues to follow its existing Sherpa normalization defaults.

## Count weighting

`power` is one integer for the entire call:

| power | Count multiplier | Default Y label in energy analysis |
| --- | --- | --- |
| 0 | Representative 1/X | Counts / bin / keV |
| 1 (default) | 1 | Counts / bin |
| 2 | Representative X | Counts × keV / bin |
| 3 | Representative X² | Counts × keV² / bin |

X is the displayed coordinate. For example, `power=2` in wavelength
analysis weights by wavelength, not energy.

```python
plot_counts(1, power=2, power_scale='geometric', res=6,
            color='navy', model_styles={'color': 'orange'})
```

Let l and h be the final displayed bin boundaries:

| power_scale | power=0 | power=1 | power=2 | power=3 |
| --- | --- | --- | --- | --- |
| `'arithmetic'` (default; ISIS 1) | 2/(l+h) | 1 | (l+h)/2 | ((l+h)/2)² |
| `'geometric'` (ISIS 2) | 1/√(lh) | 1 | √(lh) | lh |
| `'powerlaw'` (ISIS 3) | (1+r+r²)/(3l), r=l/h | 1 | lh log(h/l)/(h−l) | lh |

Integer settings 1, 2, and 3 are also accepted. The powerlaw `power=0`
expression deliberately preserves the literal supplied formula.
Factoring this expression and using `log1p` for `power=2` avoid cancellation
for narrow bins. Reciprocal weighting needs positive centers or bounds;
invalid or nonfinite weights are rejected before existing axes are cleared.

Matching, count combination, and whole-bin `rebin` happen before display
conversion and weighting. The same weight multiplies data and folded model.
Errors multiply by that weight; covariance transforms as D C D, where D is
the diagonal weight matrix. `scale` multiplies afterward algebraically, and
`offset` is added last. Neither weights nor scale/offset enter residuals.

## Background views

The modern spelling of ISIS `bkg` option is `background`:

| background | ISIS bkg | Upper data | Upper model |
| --- | --- | --- | --- |
| `'subtract'` (default) | 0 | Source − scaled measured background | Total model − same measured background |
| `'include'` | 1 | Source including background | Total model |
| `'only'` | −1 | Scaled measured background only | Hidden |

Background-only plots show measured background scaled to the source counts
using Sherpa's count scaling, including exposure and area factors. Its
variance uses the square of that scaling. Native grouping and noticed
channels are then applied. With no associated background the background
view is zero; an offline `SpectrumData` has no attached background.

```python
plot_counts([1, 1], background=['include', 'only'],
            color=['black', 'tab:green'],
            label=['Source aperture', 'Scaled background'])
```

`background` broadcasts per displayed curve and can be overridden through
`styles`. The older `subtract_background=True/False` still selects subtract
or include. Do not supply both spellings in the same option dictionary.
Loaded source/background arrays, subtraction flags, filters, and grouping
are preserved.

## Models, residuals, and plotting controls

An available folded model is overlaid automatically (`show_model='auto'`).
Use `show_model=False` to hide it. No fit is run. Background-only curves
always hide the top model; without residuals they do not look up a model.

`res=0` is the single-panel default. Modes 1/2/3 calculate individual
standardized, signed-square, or data/model residuals. Modes 4/5/6 calculate
the corresponding diagnostic from the summed counts of each displayed
group. A nonzero mode requires models and creates/reuses a shared lower
panel. Cash/CStat use signed Poisson deviance diagnostics. Residuals use
the same count-based definitions as `plot_data`, independently of display
power, scale, offset, or count combination mean normalization.

For ratios, `'subtract'` and `'only'` use net source/model counts; `'include'`
uses raw source/model counts. Background-only residuals still diagnose the
source fit. Other residual families are independent of the background view.
See [FIT_PLOTTING.md](FIT_PLOTTING.md) for formulas, model-background handling,
zero-count limits, and the statistical limits of fractional rebinning.

Matplotlib styling, `ax`, `overplot`, `clearwindow`, `styles`,
`model_styles`, `residual_styles`, and `PlotResult` match `plot_data`.
Overplotting requires compatible units, normalization, and count weights.
See [PLOTTING.md](PLOTTING.md). Velocity/redshift transforms and full legacy
ISISscripts qualifier parity are not implemented.

## Validation and examples

`tests/test_counts_plotting.py` contains 27 executed numerical/rendering
tests for counts, weight formulas, narrow-bin limits, covariance, rebinning,
models/residuals, background scaling, styles, and invalid-option handling.
Six additional real-Sherpa tests cover PHA analysis coordinates,
response-folded weighting, source-aperture backgrounds, grouping/filtering,
model suppression, and missing exposure. They are included but unrun here
because Sherpa is unavailable. This is not a full ISIS runtime comparison.

```bash
python -m unittest discover -s tests -v
python examples/counts_plotting_demo.py       # executed here, no Sherpa needed
python examples/ciao_counts_plotting_demo.py  # included for CIAO, unrun here
```
