# Unfolding Spectra

**NOTE:** The discussion below focuses on the *model-independent* method
of unfolding originally defined in the ISIS program. This method is
different than the methods used in XSPEC. Users are encouraged and to
read the theoretical background of the ISIS-style of unfolding in
[Gunderson & Huenemoerder (2025)](https://scixplorer.org/abs/2025AJ....169....2G/abstract).

`plot_unfold` implements ISIS-style flux correction for a general
spectral response. The unfolding computes

$$
S_{i}(E')=\frac{C_i(E')}{t_i \int_{\Delta E(E')} A_i(E)R_i(E',E)dE}.
$$

Here $C$ is the spectrum counts, $A$ is effective area (ARF), and $R$ is
redistribution matrix (RMF), $E$ is the photon energy, $E'$ is the channel
energy, t is the exposure time, and the integral is over all energies $E$
that can be read as channel $E'$ by the detector.

```python
from sherpa.astro import ui
from fancy_plotting import plot_unfold

# Data and their responses are already loaded; source models are optional.

ui.load_data(1, `path/to/pha1`)
plot_unfold(1)
```

`plot_unfold` also supports combining arbitrary sets of data with different
grids and plotting options.


```python
from sherpa.astro import ui
from fancy_plotting import plot_unfold

# Data and their responses are already loaded; source models are optional.

ui.load_data(1, `path/to/pha1`)
ui.load_data(2, `path/to/pha2`)
ui.load_data(3, `path/to/pha3`)
ui.load_data(4, `path/to/pha4`)
ui.load_data(5, `path/to/pha4`)

match12 = match_dataset_grids([1,2])
match34 = match_dataset_grids([3,4])


plot_unfold(
    [[1, 2], [3, 4, 5]],
    color=['tab:blue', 'tab:orange'],
)
```

## Numerical definition

For dataset $i$, response component $r$, and native detector channel $h$, compute

$$
W_{ih}=a_{ih}\sum_r t_{ir}\sum_j R_{ir,hj}
             \int_{\lambda\in j} A_{ir}(\lambda)\,d\lambda.
$$

Here $a$ is Sherpa AREASCAL, $t$ is exposure, $A$ is effective area, and $R$ is
redistribution from incident bin $j$ into channel $h$. $W$ is the predicted
count vector for unit photon flux density per Angstrom.

Effective area is piecewise constant within each tabulated ARF bin. Its
wavelength integral is mapped into the RMF incident bins before
redistribution. This supports different ARF/RMF bin boundaries, including
different boundaries with the same number of bins. Full incident grids
contribute even when only some detector channels are noticed. The RMF's
channel offset maps its output to PHA channel labels.

All assigned linear response components contribute. ARF-only responses use
the diagonal-response limit. RMF-only inputs are interpreted as full RSP
matrices already containing effective area; a probability-only RMF cannot
supply an absolute $\mathrm{cm}^2$ calibration by itself.

By default a positive ARF exposure is preferred for each component, with
PHA exposure as fallback, matching our existing ISIS-style plotting
convention. `exposure_source='pha'` uses PHA exposure explicitly. Missing
positive exposures or invalid calibration values produce errors. Folded
model counts follow the assigned Sherpa model's exposure convention; if
ARF and PHA exposure metadata disagree, the selected normalization should
be considered when comparing to an unsmeared source curve.

Let $L_i$ represent the operation of matching into a common physical
reference, and $G$ the operation of whole-bin plot grouping. For the
default background-subtracted view:

$$
N=\sum_i GL_i(C_i-B_i),\qquad
W=\sum_i GL_iW_i,\qquad n_\lambda=N/W.
$$

Background arrays $B$ use Sherpa's source-aperture count scaling. Both counts
and weights are summed before division. `combine='sum'` is the supported
combination spelling; it means pooled counts/weights, not a sum of individual
flux estimates. Arithmetic mean and exposure-rate combination options are
rejected. The deprecated `sum_exp` alias remains a no-op for this count-based
calculation.

For displayed coordinate $x$, the photon density is

$$
n_x=n_\lambda\frac{\Delta\lambda}{\Delta x}.
$$

This bin-integral transformation preserves photon flux and covariance under
energy, wavelength, frequency, and scaled-unit changes. Plot-only grouping
always occurs in the same internal wavelength coordinate, so switching
display units does not move an odd partial group. A gap in either response
grid contributes zero outside the tabulated bins. Detector channels with
zero pooled response sensitivity are masked. A second dataset can supply
sensitivity to such a channel when spectra are pooled. Existing Sherpa
grouping is respected, including groups the user has explicitly made across
a detector gap.

## Detector gaps and zero-valued real-data plots

The response integral uses the tabulated ARF and RMF incident intervals;
there is no assumption that adjacent intervals touch. RMF redistribution
may have detector channels with no contributing matrix entries. AREASCAL=0
also makes a channel insensitive. Such bins have `valid=False`, `y=NaN`,
and `weights=0`. A full RSP (such as the public Sherpa RGS RSPMAT fixture)
belongs in the RMF position with no separate ARF; multiplying by a second
ARF would double-count its effective area. Sherpa's RMF loading and any
OGIP-format warnings occur before `plot_unfold`; the ISIS
`Rmf_OGIP_Compliance` setting is not a Sherpa option.

If all *valid* unfolded bins are exactly zero using default `scale=1` and
`offset=0`, inspect the selected count numerator: the code computes
`curve.y = selected_counts * curve.factor`, with `curve.factor > 0` on
usable bins. Background subtraction can cancel source counts; selecting
`background='only'` with no attached background intentionally yields
zero. A logarithmic y-axis also hides zero and negative values.

## Unfolded Flux Units

`flux_unit='photons'` is the default; `'ergs'` selects energy
flux conventions. Singular `'photon'` and `'erg'` are accepted aliases.

| Display | Default power | Default quantity |
| --- | --- | --- |
| Photons, any physical coordinate | 1 | Photon density per displayed unit |
| Ergs, energy/frequency | 3 | E² n_E or its frequency equivalent, in erg cm⁻² s⁻¹ |
| Ergs, wavelength | 1 | λ F_λ, in erg cm⁻² s⁻¹ |
| Ergs, calibrated channel | 1 | Energy-flux density per channel |

For photons, explicit `power=0/1/2/3` multiplies $n_x$ by a representative
$1/x$, 1, $x$, or $x^2$. `power_scale='arithmetic'`, `'geometric'`, or `'powerlaw'`
uses the same supplied ISIS weighting formulas as `plot_counts`.

For energy coordinates, the erg conversion multiplies photon density by
the energy-unit-to-erg constant before power weighting. For wavelength, it
uses $hc$ divided by the wavelength-unit scale. Consequently explicit power
has the ISIS meaning: for example, ergs with energy `power=2` gives energy
flux density per energy unit, while wavelength `power=0` gives energy flux
density per wavelength unit. See the returned axis label for dimensions.
Do not interpret every combination as the same physical quantity.

The default arithmetic bin centers reproduce ISIS convention. For
finite bins $E^2n_E$ and $λF_λ$ differ by $E_mid^2/(E_lo E_hi)$. With
`power_scale='geometric'`, that factor is exactly one for corresponding
bins. This is a bin-weighting convention, not a change in the response or
the measured counts.

```python
plot_unfold(1, flux_unit='photons')
plot_unfold(1, flux_unit='ergs', power_scale='geometric')
ui.set_analysis('wavelength')
plot_unfold(1, flux_unit='ergs', power_scale='geometric')
```

Sherpa rate and positive `plot_fac` settings are ignored for flux
normalization. Omit `rate` and `density`; this command defines its own
normalization. `scale` multiplies the displayed result and `offset` is added
last. Model curves receive the same presentation scaling.

## Model and residual displays

`model_display='convolved'` is the default: response-folded model counts go
through the same matching, grouping, background convention, and flux
correction as the data. Thus data and model retain the same instrumental
blurring. This follows ISISscripts' default `con_mod=1`.

`model_display='source'` evaluates the unsmeared source expression on the
union of its ARF/RMF incident boundaries, usually finer than the detector
grid. Multiplicative
absorption already present in that expression remains present. This does
not automatically produce a deabsorbed spectrum. For combined curves,
source integrals are rebinned onto the first member's source grid and
arithmetically averaged, matching the supplied ISISscripts implementation.
This average differs from the response-weighted folded-model display.

Source-model bins must lie within a contiguous run of valid noticed data
bins to be drawn. Gaps and zero-sensitivity intervals are preserved.
`show_model=False` hides either display. Missing intrinsic expressions for
full/count-only models produce a clear error when a source curve is
requested; use the convolved display in that case.

`res=1...6` uses the existing count-space formulas, independently of flux
units, powers, scale/offset, or which upper model display was chosen. The
unsmeared source curve can differ visibly from the flux-corrected data
while the folded prediction fits well. See [FIT_PLOTTING.md](FIT_PLOTTING.md)
for diagnostics, Cash/CStat, background conventions, and fractional-count
statistical limits. No fit or source assignment is performed by plotting.

The same `background='subtract'`, `'include'`, and `'only'` views are
available. A background-only upper curve hides the model; requested lower
diagnostics still describe the source fit, as in `plot_counts`.

## Channel display

`set_analysis('channel')` is mathematically usable when the PHA response
provides physical calibration. The physical matching and normalization
still use wavelength. Bin-integrated photon flux is divided by the
corresponding channel width and labeled per channel.

For ergs, the representative photon energy comes from the physical
calibrated bin boundaries, never from the numerical channel label. The
channel-axis default is energy-flux density per channel. Additional channel
powers weight by channel number; they are not automatically $\nu F_\nu$.
An unsmeared source curve on a channel axis is rebinned onto the reference's
calibrated display bins because an arbitrary fine incident grid has no
unique channel labeling. Offline uncalibrated channel arrays are rejected.

## Uncertainties and returned values

Count/background uncertainties and full matching-induced covariance are
propagated through grouping. With the diagonal count-to-flux transform D:

$$
\mathrm{Cov}(F)=D\,\mathrm{Cov}(N)\,D^T.
$$

Presentation scale is included in the returned covariance; offsets do not
affect errors. Calibration/response uncertainty and fitted-parameter bands
are not estimated. Our covariance convention remains distinct from ISIS's
recreated errors after fractional rebinning.

The returned `PlotResult` has the same axes and artists as the other
commands. Each data curve is an `UnfoldedCurve` with these extra arrays:

| Field | Meaning |
| --- | --- |
| `weights` | Summed, matched, grouped W values in displayed order |
| `factor` | Count-to-plotted-flux factor, including units/power before scale/offset |
| `valid` | Mask of usable response normalization |

Bins with zero sensitivity have NaN values/errors and zero covariance
rows/columns; use `valid` when analyzing the arrays. An entirely insensitive
plotted range raises an error before clearing existing axes.

## Offline calculations

Ordinary Sherpa calls compute $W$ automatically. Numerical/offline callers
can supply `datasets={id: SpectrumData(...)}` and
`response_weights={id: native_W}`. Each W array must contain reference counts
for unit photon density per Angstrom, already including exposure and area,
in exactly the native input count bins. Exposure metadata is then optional.

Optional `source_spectra={id: SourceSpectrum(lo, hi, flux, unit='keV')}`
provides unsmeared source integrals for `model_display='source'`. `flux` is
photons cm⁻² s⁻¹ integrated over each source bin, never a density.
`models={id: native_predicted_counts}` supplies folded predictions for the
convolved model display and count-space residuals.

## Validation and mathematical limits

The numerical/rendering suite covers analytic diagonal and broad responses,
gapped ARF/RMF grids, full RSP, multiple responses, exposure/AREASCAL,
channel offsets, zero sensitivity and plot-only grouping at its boundaries,
matching covariance, photon/erg conversions,
source-model independence, both model displays, and all residual modes.
`examples/unfold_plotting_demo.py` renders a synthetic broad-response
comparison without Sherpa. Real DataPHA integration tests and
`examples/ciao_unfold_plotting_demo.py` cover the Sherpa interface. An
additional integration test uses Sherpa's published XMM RGS PHA and RSPMAT
files if `SHERPA_TEST_DATA` points to the downloaded sherpa-test-data tree.
The CI Sherpa jobs supply those fixtures; see [TESTING.md](TESTING.md) for
reports and local commands.

The reference calculation was checked against the supplied
`make_flux`/`plot_unfold`/`power_scale` formulas and the upstream
[`isis_unit_source_response` routine](https://github.com/houckj/isis/blob/master/src/histogram.c).
Sherpa's [`DataRMF.notice` and `apply_rmf`](https://github.com/sherpa/sherpa/blob/4.18.0/sherpa/astro/data.py)
interfaces were inspected. This is not execution of the full ISIS runtime
or validation against an actual instrument observation.

The source-independent formula requires a linear response. Known Sherpa
pileup models are rejected because their response depends on source
intensity; they do not admit this normalization. User-supplied full models
must also be consistent with the PHA response used for correction. This
routine performs flux correction and retains response blurring; it does
not recover information lost to redistribution. mJy, Watts, custom ISIS bin
correction hooks, and velocity/redshift transforms are not implemented.

```bash
python -m unittest discover -s tests -v
python examples/unfold_plotting_demo.py
python examples/ciao_unfold_plotting_demo.py  # in CIAO; unrun here
```
