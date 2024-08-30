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
    "LightcurveInjected", "StarInjected", "Star", "Lightcurve", "_split_injected_lightcurve", "_create_savedir", "_create_savedir_injected" ,"_create_padded_id", "_create_lightcurve_filenames",
    "_create_fits_file"]

def _create_fits_file(d, filename):

    # create record array of data
    recarray_data = np.rec.array(
        [
        d["periods"],
        d["power"]
        ],
        dtype=[
            ("periods", "<f8"),
            ("power", "<f8")
        ]
    )

    # create record array of solutions
    d_peaks = d["peaks"]
    recarray_sols = np.rec.array(
        [
        d_peaks["SDE"],
        d_peaks["period"],
        d_peaks["period_err"],
        d_peaks["t0"],
        d_peaks["dur"],
        d_peaks["phi"],
        d_peaks["q"],
        d_peaks["power"]
        ],
        dtype=[
            ("SDE", "<f8"),
            ("period", "<f8"),
            ("period_err", "<f8"),
            ("t0", "<f8"),
            ("dur", "<f8"),
            ("phi", "<f8"),
            ("q", "<f8"),
            ("power", "<f8")
        ]
    )

    # create primary header
    hdr = fits.Header()
    keys = ["tic", "teff", "logg", "rstar", "mstar", "rho", "Tmag"]
    comments = ["TIC identifier", "effective temperature", "surface gravity", "radius", "mass", "density", "TESS magnitude"]
    for key, comment in zip(keys, comments):
        hdr[key] = (d[key], comment)

    hdu_pri = fits.PrimaryHDU(header=hdr)

    ## create data extension
    hdr_data = fits.Header()
    keys = tuple(d["frequency_grid_args"].keys())
    comments = ("number of frequency samples", "parameter to calculate grid", "parameter to calculate grid", "minimum frequency", "maximum frequency")

    for key, comment in zip(keys, comments):
        hdr_data[key] = (d["frequency_grid_args"][key], comment)

    hdu_data = fits.BinTableHDU(
        data=recarray_data,
        header=hdr_data,
        name="SPECTRUM"
    )

    # create solutions extension
    hdr_sols = fits.Header()

    if "sim" in d:
        keys = ["sim"] + list(d["truth"].keys())
        comments = ("injection identifier", "true period", "true transit midpoint", "true transit duration", "true transit depth")

        _d = {**dict(sim = d["sim"]), **d["truth"]}
        for key, comment in zip(keys, comments):
            hdr_sols[key] = (_d[key], comment)

    hdu_sols = fits.BinTableHDU(data=recarray_sols, header=hdr_sols, name="SOLUTIONS")

    hdul = fits.HDUList([hdu_pri, hdu_data, hdu_sols])
    hdul.writeto(filename, overwrite=True)

    return

def _create_lightcurve_filenames(tic, sectors):
    tic_16d = f"{int(tic):016d}"
    prefix = "hlsp_tess-spoc_tess_phot_"
    suffix = "_tess_v1_lc.fits"
    return [prefix + f"{tic_16d}-s00{s}" + suffix for s in sectors]

def _create_padded_id(input, output_length):
        # adds leading zeros to `input` until `output_lenght` is reached
        return f'{int(input):0{output_length}}'

def _create_savedir_injected(basedir, candidate, tic, sim, scc, prefix=""):

    tic_id = _create_padded_id(tic, 16)
    sector_string = scc.split()
    sector_string = [x.split("-")[0] for x in sector_string]
    sector_string = [x.lstrip("S") for x in sector_string]
    sector_string = [_create_padded_id(x, 2) for x in sector_string]
    sector_string = "-".join([f"s{x}" for x in sector_string])

    return Path(
        basedir, 
        f"{prefix}_{sim}_{tic_id}.0{candidate}_{sector_string}.fits"
        )

def _create_savedir(basedir, candidate, tic, sectors, prefix=""):

    tic_id = _create_padded_id(tic, 16)
    sector_string = [_create_padded_id(x, 2) for x in sectors]
    sector_string = "-".join([f"s{x}" for x in sectors])

    return Path(
        basedir, 
        f"{prefix}_{tic_id}.0{candidate}_{sector_string}.fits"
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

        self._validate_parameter("teff",
                                 self.lc[0].header["TEFF"],
                                 u.K,
                                 default_value=-1
        )
        self._validate_parameter("radius",
                                 self.lc[0].header["RADIUS"],
                                 u.R_sun
        )
        self._validate_parameter("logg",
                                 self.lc[0].header["LOGG"],
                                 u.dimensionless_unscaled,
                                 default_value=-1
        )
        self._validate_parameter("Tmag",
                                 self.lc[0].header["TESSMAG"],
                                 u.dimensionless_unscaled,
                                 default_value=-1
        )
        if self.logg > 0:
            self.rho = self._calculate_stellar_density()
            self.mass = self._calculate_stellar_mass()
        else:
            self.mass = 1 * u.M_sun
            self.rho = self._calculate_stellar_density_mass()
            # self.radius = 1 * u.R_sun

        self.teff = int(np.round(self.teff.value, 0)) * u.K
        self.radius = np.round(self.radius.value, 2) * u.R_sun
        self.mass = np.round(self.mass.value, 2) * u.M_sun
        self.rho = np.round(self.rho.value, 2) * u.dimensionless_unscaled
        self.logg = np.round(self.logg.value, 1) * u.dimensionless_unscaled
        self.Tmag = np.round(self.Tmag.value, 1) * u.dimensionless_unscaled

        # combined data across all sectors
        self.time = [x.time for x in self.lc]
        self.flux = [x.flux for x in self.lc]
        self.flux_err = [x.flux_err for x in self.lc]

        self.x = np.concatenate(self.time)
        self.y = np.concatenate(self.flux)
        self.y_err = np.concatenate(self.flux_err)

    def _validate_parameter(self, param, value, unit, default_value=1):
        try:
            setattr(self, param, value * unit)
        except TypeError:
            setattr(self, param, default_value * unit)

    def _calculate_stellar_density_mass(self):
        V = 4 / 3 * np.pi * self.radius.to(u.cm)**3
        return self.mass.to(u.g) / V

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
        self.teff    = 5778 * u.K
        self.radius  = 1 * u.R_sun
        self.logg    = 4.4374 * u.dimensionless_unscaled
        self.Tmag    = -1 * u.dimensionless_unscaled
        self.mass    = 1 * u.M_sun
        self.rho     = self._calculate_stellar_density()


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