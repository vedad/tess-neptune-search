#!/usr/bin/env python

from pathlib import Path
from astropy.io import fits
import numpy as np
from astropy.constants import G
import astropy.units as u
# from transitleastsquares import transitleastsquares, transit_mask
from wotan import flatten
from pathlib import Path

__all__ = [
    "LightcurveInjected", "StarInjected", "_split_injected_lightcurve", "_create_savedir"]

def _create_padded_id(input, output_length):
        # adds leading zeros to `input` until `output_lenght` is reached
        return f'{int(input):0{output_length}}'

def _create_savedir(basedir, candidate, tic, sim, scc, prefix=""):

    tic_id = _create_padded_id(tic, 16)
    sector_string = scc.split()
    sector_string = [x.split("-")[0] for x in sector_string]
    sector_string = [x.lstrip("S") for x in sector_string]
    sector_string = [_create_padded_id(x, 2) for x in sector_string]
    sector_string = "-".join([f"s{x}" for x in sector_string])

    return Path(
        basedir, 
        f"{prefix}_{sim}_{tic_id}.0{candidate}-{sector_string}.pickle"
        )

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

        sec_lcs = {"TIC" : str(hdu[0].header["TICID"]),
                   "SIM" : hdu[0].header["SIM"],
                   "SCC" : hdu[0].header["SCC"]
                   }
        s = 0
        e = 0
        for i, length in enumerate(seclen):
            e += length
            sec_lcs[scc[i]] = (time[s:e], flux[s:e])
            s += length

        return sec_lcs, truth

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

        # combined data across all sectors
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
    
    def _calculate_stellar_mass(self):
        G_cgs = G.to(u.cm**3 / (u.g * u.s**2))
        R_cgs = self.radius.to(u.cm)
        g_cgs = 10**self.logg * u.cm / u.s**2
        return (g_cgs * R_cgs**2 / G_cgs).to(u.M_sun)
     
    def _create_candidate_id(self, index):
        return ".".join([self.tic, f"{index:02}"])


class StarInjected(Star):
    def __init__(self, tic, lightcurves, sim, truth=None):

        self.tic = tic
        self.sim = sim
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




class Lightcurve:
    """
    Class for holding the light curve data for a single TESS-SPOC `lightcurve` file
    """
    def __init__(self, filepath, **kwargs):
        self._read_data(filepath, **kwargs)
        
    def _read_data(self, filepath, quality_flag=0):
        hdu = fits.open(filepath)

        self.header = hdu[0].header
    
        # self.tic     = hdu[0].header["TICID"]
        self.sector  = hdu[0].header["SECTOR"]
        self.ccd     = hdu[0].header["CCD"]
        self.camera  = hdu[0].header["CAMERA"]
        self.elapsed_time = hdu[1].header["TELAPSE"]

        # select data by quality and normalize
        m_quality = hdu[1].data["QUALITY"] == quality_flag
        m_nan = (
                np.isfinite(hdu[1].data["PDCSAP_FLUX_ERR"]) &
                np.isfinite(hdu[1].data["PDCSAP_FLUX"]) &
                np.isfinite(hdu[1].data["TIME"])
        )
        m_zero = (np.isclose(hdu[1].data["PDCSAP_FLUX_ERR"], 0) |
                  np.isclose(hdu[1].data["PDCSAP_FLUX"], 0)
        )
        m = m_quality & m_nan & ~m_zero
        self.time = hdu[1].data["TIME"][m]
        flux = hdu[1].data["PDCSAP_FLUX"][m]

        self.flux_err = hdu[1].data["PDCSAP_FLUX_ERR"][m] / np.median(flux)
        self.flux = flux / np.median(flux)

class LightcurveInjected(Lightcurve):
    def _read_data(self, data):

        scc = data[0].split("-")
        self.sector = scc[0].lstrip("S0")
        self.camera = scc[1]
        self.ccd = scc[2]

        self.time, self.flux = data[1]
        self.flux_err = None
        self.elapsed_time = self.time[-1] - self.time[0]