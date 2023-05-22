#!/usr/bin/env python

import numpy as np
from astropy.constants import G
import astropy.units as u

class Star:
    def __init__(self, tic, lightcurves):

        self.tic = tic
        self.data_dir = data_dir
        self.lc = lightcurves

        # stellar parameters
        self.teff    = self.lc[0]["TEFF"] * u.K
        self.radius  = self.lc[0]["RADIUS"] * u.R_sun
        self.logg    = self.lc[0]["LOGG"]
        self.MH      = self.lc[0]["MH"]
        self.Tmag    = self.lc[0]["TESSMAG"]
        self.rho     = self._calculate_stellar_density()

        self.time = [x.time for x in self.lc]
        self.flux = [x.flux for x in self.lc]
        self.flux_err = [x.flux_err for x in self.lc]


    def _calculate_stellar_density(self):
        G_cgs = G.to(u.cm**3 / (u.g * u.s**2))
        R_cgs = self.radius.to(u.cm)
        g_cgs = 10**self.logg * u.cm / u.s**2
        return 3 * g_cgs / (4 * np.pi * G_cgs * R.to(u.cm))  

if __name__ == "__main__":
    pass
    # star = Star()
    # plt.errorbar(lc.time, lc.flux, yerr=lc.flux_err, capsize=0, fmt='.')
    # plt.show()