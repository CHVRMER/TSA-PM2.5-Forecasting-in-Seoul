"""(2,1,2) did not converge within maxiter=200 in a_sarima.py. Refit every order with maxiter=1000 and report
convergence, AIC and AR/MA roots, so the order choice uses properly converged likelihoods.
Usage: python check_order_convergence.py
"""
import json
import time
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from a_sarimax import INDEX, ORDERS, TRAIN, fourier
from config import RESULTS
from data_access import seoul_pm25

z = np.log1p(seoul_pm25().reindex(INDEX))
F = fourier(INDEX)
out = {}
for o in ORDERS:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = SARIMAX(z[TRAIN], exog=F[TRAIN], order=o).fit(disp=False, maxiter=1000)
    out[str(o)] = {"aic": float(r.aic), "converged": bool(r.mle_retvals.get("converged")),
                   "iterations": r.mle_retvals.get("iterations"), "seconds": round(time.time() - t0, 1),
                   "ar_root_moduli": [float(abs(x)) for x in r.arroots], "ma_root_moduli": [float(abs(x)) for x in r.maroots]}
    print(o, out[str(o)], flush=True)
(RESULTS / "a_sarima_order_check_maxiter1000.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
