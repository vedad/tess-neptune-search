#!/usr/bin/env python

import numpy as np
from astropy.constants import G
import astropy.units as u
from transitleastsquares import transitleastsquares, transit_mask

class Star:
    def __init__(self, tic, lightcurves):

        self.tic = tic
        self.lc = lightcurves
        print(self.tic)

        # stellar parameters
        self.teff    = self.lc[0].header["TEFF"] * u.K
        self.radius  = self.lc[0].header["RADIUS"] * u.R_sun
        self.logg    = self.lc[0].header["LOGG"]
        self.MH      = self.lc[0].header["MH"]
        self.Tmag    = self.lc[0].header["TESSMAG"]
        self.rho     = self._calculate_stellar_density()

        self.time = [x.time for x in self.lc]
        self.flux = [x.flux for x in self.lc]
        self.flux_err = [x.flux_err for x in self.lc]

        self.x = np.concatenate(self.time)
        self.y = np.concatenate(self.flux)
        self.y_err = np.concatenate(self.flux_err)


    def _calculate_stellar_density(self):
        G_cgs = G.to(u.cm**3 / (u.g * u.s**2))
        R_cgs = self.radius.to(u.cm)
        g_cgs = 10**self.logg * u.cm / u.s**2
        return 3 * g_cgs / (4 * np.pi * G_cgs * self.radius.to(u.cm))
    
    def _create_candidate_id(self, index):
        return ".".join([self.tic, f"{index:02}"])
    
    def search(self, method='tls'):

        def _run_search(x, y):
             model = transitleastsquares(x, y)
             return model.power(oversampling_factor=5, duration_grid_step=1.02)
        
        results_multi = []
        results = _run_search(self.x, self.y)
        
        # add data
        results["data_x"] = self.time
        results["data_y"] = self.flux

        # add stellar parameters
        results["Tmag"] = self.Tmag
        results["radius"] = self.radius
        results["teff"] = self.teff

        results["cid"] = self._create_candidate_id(1)
        print(results["cid"])

        results_multi.append(results)

        current_significance = results["SDE"]
        iterations = 1
        m = np.ones_like(self.x, dtype=bool)

        while current_significance > 7:
            # if previous candidate was significant, search for another signal
            iterations += 1

            m_tr = transit_mask(self.x, results.period, 2*results.duration, results.T0)
            m[m_tr] = False
            results = _run_search(self.x[m], self.y[m])

            # add data
            results["data_x"] = self.time
            results["data_y"] = self.flux

            # add stellar parameters
            results["Tmag"] = self.Tmag
            results["radius"] = self.radius
            results["teff"] = self.teff

            results["cid"] = self._create_candidate_id(iterations)

            results_multi.append(results)
            current_significance = results["SDE"]

            # add a new candidate ID to the new results


        return results_multi

if __name__ == "__main__":
    pass
