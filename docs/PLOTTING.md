# Fancy Plotting Functions

Three wrappers for advanced plotting operations are provided in this package: `plot_data`, `plot_counts`, and `plot_unfold`. This section is for discussion of the general behavior and documentation of the operations. See specific API documentation for [COUNTS_PLOTTING](COUNTS_PLOTTING.md) and [UNFOLDING](UNFOLDING.md).

All three `plot_*` functions support independent data and model presentation:

```python
result = plot_data(
    [[1, 2], [3, 4]],
    color=['navy', 'black'], ecolor='lightgray',
    model_plot='line',
    model_styles=[
        {'color': 'crimson', 'ls': '-', 'lw': 2, 'label': 'Model A'},
        {'color': 'darkorange', 'ls': '--', 'lw': 1.5, 'label': 'Model B'},
    ],
)
```

`model_styles` accepts one dictionary shared by all model curves or one
dictionary per displayed group. It controls color, line style/width, markers,
transparency, zorder, and legend labels, independently of data/error bars.
`model_plot='histogram'` is the default and follows bin boundaries;
`model_plot='line'` joins bin centers. A list such as `['line', 'histogram']`
selects presentation per displayed group. Gaps remain disconnected in both
forms. These controls affect the rendering of folded and intrinsic models;
they do not change predictions or residuals. **A line through bin centers is
a display choice, not additional spectral resolution.**

Error bars and their caps now have a lower zorder than the separate data
histogram, including when no markers are drawn. `barsabove=True` explicitly
places them above it. The same rule applies to residuals. An explicit
`zorder` sets the curve's layer; bars are placed immediately below/above it.

Default axis labels use Matplotlib Mathtext: upright units, superscript
powers, Å for Angstrom, and μm for microns. No external LaTeX installation is
required. `xlabel=` and `ylabel=` still accept arbitrary user text/Mathtext.
See `examples/model_styling_demo.py` for a rendered example of all three
plotting functions.

`show_model=False` requests data only. See [FIT_PLOTTING.md](FIT_PLOTTING.md)
for model/residual options. Matplotlib 3.6+ is required for rendering.

## Typical Sherpa workflow

```python
import matplotlib.pyplot as plt
from sherpa.astro import ui
from fancy_plotting import match_dataset_grids, plot_data

# The spectra and responses have already been loaded.
matched = match_dataset_grids([1, 2, 3, 4])
ui.set_analysis('wavelength')

fig, ax = plt.subplots(figsize=(9, 5))
result = plot_data(
    [[1, 2], [3, 4]],
    ax=ax,
    combine='exposure',
    color=['tab:blue', 'tab:orange'],
    marker=['o', 's'],
    markerfacecolor='none',
    linestyle='none',
    capsize=2,
    elinewidth=0.8,
    label=['Observation A', 'Observation B'],
)
ax.set(yscale='log', title='Combined spectra')
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```

Use `ui.set_analysis` to switch energy/wavelength/channel analysis. No units
are required by matching, and no rematching is needed after an analysis
change. Rate/counts and the positive plot factor come from the first displayed
PHA (or an explicit plotting reference); see [UNITS.md](UNITS.md).

## Dataset and group syntax
Example options for combining 4 spectra loaded into Sherpa are below. All three `plot_*`
functions accept an arbitrary number of spectra and combinations, including repeated
calls of the same ID.
| Input | Display |
| --- | --- |
| `plot_data()` | Sherpa's default dataset ID |
| `plot_data(1)` or `plot_data(id=1)` | One spectrum |
| `plot_data([1, 2, 3, 4])` | Four individual spectra |
| `plot_data([[1, 2], [3, 4]])` | Two separately combined spectra |
| `plot_data([1, [2, 3]])` | One individual and one combined spectrum |
| `plot_data([1, 2, [1, 2]])` | Spectra 1 and 2 plotted separated and combined |

Integer and string IDs work. Lists/tuples are ordered; Python sets are not
used. Combining different physical detector bins requires a registered match,
or `align=True` with an optional reference. All matched plotting continues
through the registered physical reference bins and propagates covariance.

When `ids` is omitted with offline `datasets`, the first mapping key is used.
The plotting/aggregation core supports `SpectrumData` inputs without Sherpa.
See `examples/plotting_demo.py` for an executable offline example.

## Matplotlib style options

A scalar style is shared by all displayed curves. A sequence normally gives
one value per displayed curve, after grouping. Two groups need two styles,
even if they contain four datasets.

| Options | Meaning |
| --- | --- |
| `color` / `c` | A Matplotlib color: name, hex, `C0`, grayscale string, RGB, or RGBA |
| `marker` | Matplotlib symbols, math-text symbols, marker tuples, or None |
| `linestyle` / `ls` | Matplotlib names such as `'solid'`, `'dotted'`, `'dashdot'`, `'none'`, or a dash tuple |
| `linewidth` / `lw` | Histogram-line width; default follows Matplotlib rcParams or Sherpa preferences |
| `markersize` / `ms` | Marker size, using the same defaults |
| `markerfacecolor` / `mfc`, `markeredgecolor` / `mec` | Marker face and edge colors; `'none'` gives a hollow face |
| `markeredgewidth` / `mew`, `fillstyle` | Marker edge thickness and Matplotlib fill style |
| `ecolor`, `elinewidth` | Independent error-bar color and thickness; default to curve color/width |
| `capsize`, `capthick` | Error-bar cap size and thickness |
| `errorevery` | Positive stride, or `(start, stride)`, for displaying fewer error bars |
| `barsabove` | Draw error bars/caps above the histogram and markers; default is behind |
| `xerrorbars`, `yerrorbars` | Sherpa-style switches for bin half-widths and statistical error bars |
| `alpha`, `zorder`, `rasterized` | Transparency, stacking order, and export rendering |
| `drawstyle`, `dashes` | Matplotlib line draw style and dash pattern |
| `label` | Legend label for each displayed curve |
| `show_model` | `'auto'` overlays a complete available model; True requires one; False hides it |

RGB/RGBA colors, custom marker tuples, custom dash tuples, and `errorevery`
tuples are single style values. For example, `color=(0.1, 0.2, 0.8)` colors
every curve blue; it is not three per-curve color entries. Lists of valid
colors, such as `['navy', 'orange']`, select per-curve colors.

```python
result = plot_data(
    [[1, 2], [3, 4]],
    color=['navy', 'orange'],
    ls=['solid', 'dashed'], lw=1.5, ms=4,
    marker='o', mfc='none',
    ecolor='gray', capsize=3, elinewidth=0.7,
)
```

Per-curve dictionaries make more complex choices explicit:

```python
result = plot_data(
    [[1, 2], [3, 4]],
    color='gray',
    styles=[
        {'color': 'navy', 'combine': 'sum', 'marker': 'o'},
        {'color': 'orange', 'combine': 'mean',
         'linestyle': (0, (5, 2)), 'rebin': 4},
    ],
)
```

Spectral lines follow actual bin boundaries, including nonuniform bins and
gaps, as in Sherpa's histogram plots. Markers and error bars lie at bin centers.
`drawstyle` applies to that histogram path. Error bars show marginal standard
deviations; numerical covariance remains available on the returned curves.

## Sherpa preferences and precedence

For UI/session calls, the module reads recognized fields from
`session.get_data_plot_prefs(id)` without changing those dictionaries. Each
curve starts with the preferences of its first member. Panel log settings
come from the first curve. Explicit keywords override these preferences;
per-curve `styles` override the shared keywords. Matplotlib rcParams provide
unspecified sizes and the axes color cycle.

```python
prefs = ui.get_data_plot_prefs(1)
prefs['marker'] = 'o'
prefs['capsize'] = 2
plot_data(1)  # inherits these settings
plot_data(1, marker='s')  # overrides the marker for this call
```

No new global style registry is introduced. Use `plt.style.context`,
`plt.rc_context`, or the returned Matplotlib artists for further customization.
The implementation renders with Matplotlib, even if a different Sherpa
plotting backend has been selected; recognized preferences are the common
Matplotlib/Sherpa subset documented here.

## Axes, subplots, and overplotting

| Option | Meaning |
| --- | --- |
| `ax` | A specific Matplotlib Axes to draw into |
| `overplot=True` | Add spectra to existing axes without clearing them |
| `clearwindow=False` | Reuse the current/supplied axes and preserve existing artists and subplot layout |
| `xlog`, `ylog` | Sherpa-style boolean log/linear settings |
| `xscale`, `yscale` | Matplotlib axis scale names; explicit values override inherited log preferences |
| `xlim`, `ylim` | Matplotlib limits, including reversed limits and None endpoints |
| `xlabel`, `ylabel`, `title` | Panel labels/title; normal defaults use the analysis coordinate |
| `legend` | True, False, or a dict passed to `ax.legend` |

With no `ax`, a normal call creates new axes. `overplot=True` or
`clearwindow=False` uses `plt.gca()`. With a supplied Axes and default
`overplot=False, clearwindow=True`, only that Axes is cleared; other subplots
are untouched. This function never clears the entire figure. Use the
returned `ax`/`figure` for normal Matplotlib customization and saving.

```python
fig, axes = plt.subplots(2, 1, sharex=True)
plot_data([[1, 2]], ax=axes[0], label='A')
plot_data([[3, 4]], ax=axes[1], label='B', color='orange')
plot_data(5, ax=axes[0], overplot=True, color='black')
axes[0].set_yscale('log')
fig.savefig('spectra.pdf', bbox_inches='tight')
```

Overplotted curves must have the same analysis unit, rate/counts, density,
and plot-factor settings. Style changes do not change the data. Axis limits
and offsets use displayed units. For direct rendering on a logarithmic Y axis,
nonpositive values are hidden in artists; returned arrays retain those values.

## Numerical options

The numerical definitions are unchanged. With counts C_i, exposure T_i,
and display bin width dx, rate plotting uses:

| `combine` | Value before optional bin-width normalization |
| --- | --- |
| `'exposure'` (rate default) | sum(C_i) / sum(T_i) |
| `'sum'` | sum(C_i / T_i) |
| `'mean'` / `'average'` | mean(C_i / T_i) |
| `'mean_exposure'` | sum(C_i) / mean(T_i) |

For `rate=False`, use `'sum'` or `'mean'` on integrated counts. `density=True`
then divides by displayed widths. `plot_data` defaults to Sherpa's rate/counts
and grouped-bin density setting. `plot_counts` fixes `rate=False,
density=False` and ignores the session's `plot_fac`. Its default is counts
per bin; use its explicit `power` option for count weighting. Its styling,
model/residual controls, and group syntax are identical. See
[COUNTS_PLOTTING.md](COUNTS_PLOTTING.md).

`plot_unfold` uses the same styles, nested IDs, axes, backgrounds, and
residual controls for response-corrected flux. Its normalization pools
counts and response weights, ignoring the session's rate/plot_fac settings.
Use `flux_unit='photons'` or `'ergs'`, and
`model_display='convolved'` or `'source'`. See [UNFOLDING.md](UNFOLDING.md).

Per-curve `rebin` performs whole-bin plot-only grouping. It never bridges
physical gaps or changes native grouping. `scale` and `offset` multiply and
shift the displayed curve; uncertainties scale with `scale`. Independent
spectra combine covariances in quadrature; correlated bins retain their full
covariance through rebinning. `errorevery` only reduces displayed error bars;
it does not group data or change error values.

`subtract_background`, `error_method` (`'sqrt'`, `'gehrels'`, `'stored'`), and
`exposure_source` (`'arf'`, `'pha'`) retain their existing meanings. Display
exposure defaults to positive ARF exposure when available, otherwise PHA
exposure. Sherpa handles the background scaling on copies. For corrected/net
offline counts, provide their absolute uncertainties or covariance.
The named `background='subtract'`, `'include'`, or `'only'` setting is also
available for both plotting commands, per curve. Background-only curves
show measured background scaled into the source aperture and hide the
upper model. Choose `background` or the older `subtract_background` in a
single option dictionary; per-curve styles can override either global form.
`align` and `reference` control whether to use the matched physical view.
The legacy optional plot `unit` override affects display only; normally use
Sherpa's `set_analysis` and omit it.

## Return value

`PlotResult` contains:

- `ax` and `figure`: the actual Matplotlib objects.
- `curves`: plotted bin edges, values, uncertainties, memberships, normalization,
  and sparse covariance.
- `lines`: one histogram Line2D per displayed curve.
- `errorbars`: one Matplotlib ErrorbarContainer per curve.
- `artists`: the existing `(line, errorbar_container)` pairs, for compatibility.
- `model_curves`, `model_artists`, `residual_ax`, `residuals`,
  `residual_artists`, and `reference_line`: model/residual components when
  requested; see [FIT_PLOTTING.md](FIT_PLOTTING.md).

```python
result = plot_data([[1, 2]], color='navy')
result.lines[0].set_linewidth(2)
result.figure.savefig('spectrum.png', dpi=200)
```

Plotting preserves original PHA arrays, analysis settings, grouping, filters,
subtraction flags, model assignments, and parameter values. Available models
are evaluated for display, without fitting. With `show_model=False, res=0`,
no model lookup/evaluation is performed. Matched plotting still uses a
registered view rather than modifying Sherpa's native data.

## Migration from the earlier interface

For users migrating from ISIS, compatibility with Michael Nowak's `fancy_plots`
keywords is supported. Old keywords work through a deprecated compatibility layer
and issue `DeprecationWarning`. Update scripts using these mappings:

| Earlier keyword | Modern spelling |
| --- | --- |
| `dcol` | `color` |
| `dsym` | `marker` |
| `dstyle` | `linestyle` |
| `decol`, `error_color` | `ecolor` |
| `d_width` | `linewidth` |
| `oplt` | `overplot` |
| `xrange`, `yrange` | `xlim`, `ylim` |
| `sum_exp=True` | `combine='exposure'` |
| `sum_exp=False` | `combine='mean_exposure'` |

Old numeric PGPLOT codes are translated only under the old names.
For `plot_counts`, the legacy sum_exp flag remains a no-op, as before.
`res` is now an active ISISscripts-compatible plotting option: 0 is the
single-panel default and 1 through 6 add the corresponding residual panel.
