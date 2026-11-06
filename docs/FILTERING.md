# Select one spectral x-axis range with xnotice

A helper function is provided that wraps the Sherpa commands `ignore`
and `notice` to provide a single line, ease of use command for choosing
the range of a spectrum for operations. This function is called `xnotice`
in ISIS and follows a similar heritage and name here. `xnotice` applies
an exclusive filter to loaded Sherpa PHA data. It first ignores the
selected dataset, then notices the specified range using Sherpa's
native `ignore_id` and `notice_id`. The selected range affects fits,
`plot_data`, `plot_counts`, and `plot_unfold`.

```python
from sherpa.astro import ui
from fancy_plotting import xnotice

xnotice(lo=0.8, hi=2.0)    # current default Sherpa dataset

xnotice([1,...,N], 0.8, 2.0)  # An arbitrary list of dataset IDs can be passed

ui.set_analysis('wavelength')
xnotice(1, 7.5, 15.0)      # The call is independent of analysis settings

ui.set_analysis('energy')
xnotice(1, 0.8, 2.0)       # same dataset, now keV

ui.set_analysis('channel')
xnotice(1, 100, 600)       # same dataset, now detector channels
```

Pass `session=my_session` when using a separate `sherpa.astro.ui.utils.Session`.
The helper accepts one dataset ID, a list/tuple of IDs, or the default ID.
It requires two finite, increasing bounds and validates all IDs before
changing filters. It never changes `set_analysis` or the data's response.
The x coordinate is exactly Sherpa's current analysis coordinate for each
dataset: keV for energy, Angstrom for wavelength, or channel numbers. Sherpa
decides which integrated bins touch the range and how the upper edge is
handled. For several datasets with different settings, the *same numeric*
bounds are interpreted separately in each dataset's analysis coordinate.

This changes the noticed data for both fitting and plotting. **Matplotlib's
`xlim` changes only the visible axis limits.** To restore the entire dataset
later, use `ui.notice_id(1)`. To add an interval to an existing filter, use
Sherpa's `ui.notice_id(1, lo, hi)` directly. `xnotice` can not add non-sequential
ranges of data noticing. For advanced operations, users are encouraged to use the
default `ui.ignore` and `ui.notice` commands in Sherpa.

Local tests check argument validation and exclusive call ordering. Real
Sherpa integration tests exercise energy, wavelength, channel, mixed
analysis, filter replacement, and plotting, but require a CIAO/Sherpa
installation; the validation manifest distinguishes executed from skipped
tests.
