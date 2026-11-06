"""Run inside an existing Sherpa session to inspect a zero-valued unfold plot.

    from examples.diagnose_unfold import diagnose_unfold
    diagnose_unfold(1)

The routine reads the loaded spectrum and makes temporary Matplotlib figures;
it does not change Sherpa's filters, grouping, fit, response, or analysis.
"""
import numpy as np

from fancy_plotting import plot_unfold


def _summary(name, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    print(f"{name}: bins={values.size}, finite={finite.size}, "
          f"nonzero={np.count_nonzero(finite)}, sum={finite.sum():.8g}, "
          f"range=({finite.min():.8g}, {finite.max():.8g})" if finite.size else
          f"{name}: bins={values.size}, no finite values")


def diagnose_unfold(dataset_id=1):
    """Compare native data, included/background-subtracted views, and W."""
    from matplotlib import pyplot as plt
    from sherpa.astro import ui

    pha = ui.get_data(dataset_id)
    print(f"dataset {dataset_id}: analysis={pha.units}, grouped={pha.grouped}, "
          f"subtracted={pha.subtracted}, exposure={pha.exposure}, "
          f"responses={tuple(pha.response_ids)}")
    _summary("native PHA counts", pha.counts)
    _summary("selected PHA counts", pha.get_dep(filter=True))
    for response_id in pha.response_ids:
        arf, rmf = pha.get_response(response_id)
        print(f"response {response_id}: ARF={'yes' if arf is not None else 'none'}, "
              f"RMF={'yes' if rmf is not None else 'none'}")
    for view in ("include", "subtract"):
        result = None
        try:
            result = plot_unfold(dataset_id, show_model=False, background=view,
                                 rebin=1, flux_unit="photons", ylog=False)
            curve = result.curves[0]
            print(f"background={view}: valid={int(curve.valid.sum())}/{curve.valid.size}")
            _summary("  response weights W", curve.weights)
            _summary("  unfolded photon density", curve.y)
            _summary("  count-to-flux factor", curve.factor)
            # With default scale=1 and offset=0, invert the plotted flux
            # only on valid bins to expose the selected count numerator.
            counts = np.divide(curve.y, curve.factor,
                               out=np.full_like(curve.y, np.nan),
                               where=curve.valid & (curve.factor > 0))
            _summary("  selected counts (unfold numerator)", counts)
        except Exception as exc:
            print(f"background={view}: {type(exc).__name__}: {exc}")
        finally:
            if result is not None:
                plt.close(result.figure)
