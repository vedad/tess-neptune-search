#!/usr/bin/env python

import numpy as np
from astropy.constants import G
from astropy.stats import sigma_clip
import astropy.units as u
from transitleastsquares import transitleastsquares, transit_mask
from wotan import flatten
# from scipy.signal import medfilt, savgol_filter
# import wotan

class Star:
    def __init__(self, tic, lightcurves):

        self.tic = tic
        self.lc = lightcurves

        # stellar parameters
        self.teff    = self.lc[0].header["TEFF"] * u.K
        self.radius  = self.lc[0].header["RADIUS"] * u.R_sun
        self.logg    = self.lc[0].header["LOGG"]
        self.MH      = self.lc[0].header["MH"]
        self.Tmag    = self.lc[0].header["TESSMAG"]
        self.rho     = self._calculate_stellar_density()
        self.mass    = self._calculate_stellar_mass()
        print('star mass', self.mass)
        print('star radius', self.radius)

        # combined data across all sectors
        self.time = [x.time for x in self.lc]
        self.flux = [x.flux for x in self.lc]
        self.flux_err = [x.flux_err for x in self.lc]

        self.x = np.concatenate(self.time)
        self.y = np.concatenate(self.flux)
        self.y_err = np.concatenate(self.flux_err)
        print('span', self.x[-1] - self.x[0])


    def _calculate_stellar_density(self):
        G_cgs = G.to(u.cm**3 / (u.g * u.s**2))
        R_cgs = self.radius.to(u.cm)
        g_cgs = 10**self.logg * u.cm / u.s**2
        return 3 * g_cgs / (4 * np.pi * G_cgs * self.radius.to(u.cm))
    
    def _calculate_stellar_mass(self):
        G_cgs = G.to(u.cm**3 / (u.g * u.s**2))
        R_cgs = self.radius.to(u.cm)
        g_cgs = 10**self.logg * u.cm / u.s**2
        return (g_cgs * R_cgs**2 / G_cgs).to(u.M_sun)
     
    def _create_candidate_id(self, index):
        return ".".join([self.tic, f"{index:02}"])
    
    def detrend_data(self, x, y, **kwargs):

        # # parameter stuff
        # if edge_cutoff is None:
        #     edge_cutoff = 0.5 * window_length

        # break_tolerance explicitly set for ``pspline`` method since it cannot be a function of ``window_length`` (the parameter is not used for this method)
        # if method == "pspline" and break_tolerance is None:
        #     break_tolerance = 0.5

        # _kwargs = dict(
        #     method = method,
        #     window_length = window_length,
        #     cval = cval,
        #     break_tolerance = break_tolerance,
        #     edge_cutoff = edge_cutoff,
        # )

        flatten_y, trend = flatten(x, y, **kwargs)

        # if method == "biweight":
        #     flatten_y, trend_y = flatten(x, y, **_kwargs)
        # elif method == "pspline":
        #     _kwargs = _validate_kwargs(_kwargs)
        #     _kwargs["max_splines"] = 1000
        #     _kwargs["stdev_cut"] = 2
        #     _kwargs.pop("window_length")
        #     _kwargs.pop("cval")
        #     flatten_y, trend_y = flatten(x, y, method="pspline", **_kwargs)

        return flatten_y, trend
    
    # def _validate_tls_kwargs(d):
            
    #         for key,val in d.iteritems():
    #             if key not in d:
    #                 d[key] = val
    #         return d
    
    def search(self, wotan_kwargs=None, tls_kwargs=None, truth=None):



        # y, trend, m = self.detrend_data(self.y, window_size)
        y, trend = self.detrend_data(self.x, self.y, **wotan_kwargs)
        m = np.isfinite(y)
        # m = np.ones_like(y, dtype=bool)
        x = self.x
        # data_span = self.x.max() - self.x.min()

        def _run_search(x, y):
             model = transitleastsquares(x, y)
             return model.power(
                # period_min=0.3,
                R_star=self.radius.value,
                M_star=self.mass.value,
                # M_star_max=1.5,
                **tls_kwargs
                )
        
        results_multi = []
        results = _run_search(x[m], y[m])
        
        # add data: adding sector only, beware
        results["data_x"] = self.time
        results["data_y"] = self.flux
        # results["mask"] = m
        results["detrending"] = trend

        # add stellar parameters
        results["Tmag"] = self.Tmag
        results["radius"] = self.radius
        results["teff"] = self.teff

        results["cid"] = self._create_candidate_id(1)

        # print('odd even', results["odd_even_mismatch"])
        # print('transit count', results["transit_count"])
        # print('distinct transit count', results["distinct_transit_count"])
        # print('empty transit count', results["empty_transit_count"])

        results_multi.append(results)

        current_significance = results["SDE"]
        iterations = 1
        # m = np.ones_like(self.x, dtype=bool)

        while current_significance > 7:
            # if previous candidate was significant, search for another signal
            if iterations > 2:
                break
            iterations += 1

            m_tr = transit_mask(x, results.period, 2*results.duration, results.T0)
            m[m_tr] = False
            # m[m_tr] = False
            # m = m | mask
            results = _run_search(x[m], y[m])

            # add data
            results["data_x"] = self.time
            results["data_y"] = self.flux
            results["mask"] = m
            results["detrending"] = trend

            if truth is not None:
                results["sim"] = self.sim
                results["period_true"] = truth["period"]
                results["t0_true"] = truth["t0"]
                results["duration_true"] = truth["duration"]
                results["depth_true"] = truth["depth"]

            # add stellar parameters
            results["Tmag"] = self.Tmag
            results["radius"] = self.radius
            results["teff"] = self.teff

            results["cid"] = self._create_candidate_id(iterations)

            results_multi.append(results)
            current_significance = results["SDE"]

            # add a new candidate ID to the new results

        return results_multi
