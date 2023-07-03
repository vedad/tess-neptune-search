#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import astropy.units as u

from star import Star
from data import Lightcurve

__all__ = ["LightcurveInjected", "StarInjected", "_split_injected_lightcurve"]

class LightcurveInjected(Lightcurve):
    def _read_data(self, data):

        scc = data[0].split("-")
        self.sector = scc[0].lstrip("S0")
        self.camera = scc[1]
        self.ccd = scc[2]

        self.time, self.flux = data[1]
        self.flux_err = None
        self.elapsed_time = self.time[-1] - self.time[0]
    
        # masking invalid values
        # m_nan = np.isfinite(time) & np.isfinite(flux)
        # m_zero = np.isclose(flux, 0)
        # m = m_nan & ~m_zero

        # self.time = time[m]
        # self.flux = flux[m]

class StarInjected(Star):
    def __init__(self, tic, lightcurves, truth=None):

        self.tic = tic
        self.lc = lightcurves
        self.truth = truth

        # stellar parameters
        self.teff    = None
        self.radius  = 1 * u.R_sun
        self.logg    = None
        self.MH      = None
        self.Tmag    = None
        self.rho     = None
        self.mass    = 1 * u.M_sun

        # combined data across all sectors
        self.time = [x.time for x in self.lc]
        self.flux = [x.flux for x in self.lc]
        self.flux_err = None

        self.x = np.concatenate(self.time)
        self.y = np.concatenate(self.flux)
        self.y_err = None

        self.elapsed_time = self.x[-1] - self.x[0]

def _split_injected_lightcurve(filepath):
        hdu = fits.open(filepath)

        truth = dict(
                    period = hdu[0].header["P"],
                    t0 = hdu[0].header["T0"],
                    duration = hdu[0].header["TDUR"],
                    depth = hdu[0].header["DEPTH"]
                    )

        time = hdu[1].data["TIME"]
        flux = hdu[1].data["FLUX"]

        seclen = [int(x) for x in hdu[0].header['SECLEN'].split(' ')]
        scc = hdu[0].header['SCC'].split(' ')

        sec_lcs = {"TIC" : str(hdu[0].header["TICID"])}
        s = 0
        e = 0
        for i, length in enumerate(seclen):
            e += length
            sec_lcs[scc[i]] = (time[s:e], flux[s:e])
            s += length

        return sec_lcs, truth