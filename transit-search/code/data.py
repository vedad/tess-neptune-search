#!/usr/bin/env python

from pathlib import Path
from astropy.io import fits

# class LightcurveFile:
#     """
#     Class for holding the path to a single TESS-SPOC `lightcurve` file. 
#     """
#     def __init__(self, data_dir, tic=None, sector=None, provenance="tess-spoc", product="hlsp", mission="tess", version="v1", instrument="phot"):

#         if tic is None:
#             raise ValueError("`tic` number needs to be specified")
#         if sector is None:
#             raise ValueError("TESS `sector` needs to be specified")

#         self.path = data_dir

#         self.full_tic = _create_full_id(tic, 16)
#         self.tic = tic

#         self.full_sector = _create_full_id(sector, 4)
#         self.sector = sector

#         base_filename = f"{product}_{provenance}_{mission}_{instrument}_{self.full_tic}-s{self.full_sector}_{mission}_{version}"

#         self.lc_filename = f"{base_filename}_lc.fits"
#         self.lc_filepath = Path(self.path, "_".join([base_filename, "tp"]), self.lc_filename).as_posix()

#         # let's not worry about TP files for now
#         # self.tp_filename = f"{base_filename}_tp.fits"
#         # self.tp_filepath = Path(self.path, "_".join([base_filename, "tp"]), self.tp_filename).as_posix()

class Lightcurve:
    """
    Class for holding the light curve data for a single TESS-SPOC `lightcurve` file
    """
    def __init__(self, filepath=None):

        hdu = _read_fits(filepath)
        
        self.tic     = hdu[0].header["TICID"]
        self.sector  = hdu[0].header["SECTOR"]
        self.ccd     = hdu[0].header["CCD"]
        self.camera  = hdu[0].header["CAMERA"]
        self.elapsed_time = hdu[1].header["TELAPSE"]

        # stellar parameters
        self.teff    = hdu[0].header["TEFF"]
        self.radius  = hdu[0].header["RADIUS"]
        self.logg    = hdu[0].header["LOGG"]
        self.MH      = hdu[0].header["MH"]
        self.Tmag    = hdu[0].header["TESSMAG"]

        m = hdu[1].data["QUALITY"] == 0
        self.time = hdu[1].data["TIME"][m]
        self.flux = hdu[1].data["PDCSAP_FLUX"][m]
        self.flux_err = hdu[1].data["PDCSAP_FLUX_ERR"][m]
        
def _get_filepath(data_path, tic, sector, provenance="tess-spoc", product="hlsp", mission="tess", version="v1", instrument="phot"):

    # if tic is None:
    #     raise ValueError("`tic` number needs to be specified")
    # if sector is None:
    #     raise ValueError("TESS `sector` needs to be specified")

    data_path = data_path

    full_tic = _create_full_id(tic, 16)
    full_sector = _create_full_id(sector, 4)

    base_filename = f"{product}_{provenance}_{mission}_{instrument}_{full_tic}-s{full_sector}_{mission}_{version}"

    lc_filename = f"{base_filename}_lc.fits"
    lc_filepath = Path(data_path, "_".join([base_filename, "tp"]), lc_filename).as_posix()

    return lc_filepath

def _create_full_id(input, output_length):
    # adds leading zeros to `input` until `output_lenght` is reached
    return f'{int(input):0{output_length}}'

def _read_fits(filename):
    return fits.open(filename)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dir = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data"

    lcpath = _get_filepath(data_dir, tic=21371, sector=11)

    lc = Lightcurve(lcpath)

    plt.errorbar(lc.time, lc.flux, yerr=lc.flux_err, capsize=0, fmt='.')
    plt.show()
