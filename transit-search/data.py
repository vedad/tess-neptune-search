#!/usr/bin/env python

from pathlib import Path
from astropy.io import fits
import numpy as np
import re

__all__ = ["Lightcurve", "Pathfinder"]

class Pathfinder:
    """
    Class for holding the path to a single TESS-SPOC `lightcurve` file. 
    """
    def __init__(self, data_path, tic=None, sector=None, provenance="tess-spoc", product="hlsp", mission="tess", version="v1", instrument="phot"):

        if tic is None:
            raise ValueError("`tic` number needs to be specified")

        self.data_path = data_path
        if isinstance(tic, list):
            tic = tic[0]
        self.tic_id = Pathfinder._create_full_id(tic, 16)
        self.tic = tic
        self.provenance = provenance
        self.product = product
        self.mission = mission
        self.version = version
        self.instrument = instrument

        if sector is not None:
            self.sectors = sector
        else:
            self.sectors = self._get_all_sectors()

        self.filepaths = [self._get_sector_filepath(s) for s in self.sectors]
    
    def _get_sector_filename(self, sector) -> str:
        sector_id = Pathfinder._create_full_id(sector, 4)
        base_filename = (
            f"{self.product}_{self.provenance}_{self.mission}_{self.instrument}_{self.tic_id}-s{sector_id}_{self.mission}_{self.version}"
        )

        return f"{base_filename}_lc.fits"

    def _get_sector_filepath(self, sector):
        sector_id = Pathfinder._create_full_id(sector, 4)
        base_filename = (
            f"{self.product}_{self.provenance}_{self.mission}_{self.instrument}_{self.tic_id}-s{sector_id}_{self.mission}_{self.version}"
        )
        filename = self._get_sector_filename(sector)
        filepath = Path(
            self.data_path, "_".join([base_filename, "tp"]), filename
                        ).as_posix()

        return filepath

    def _get_all_sectors(self):
        filepaths = sorted(
            list(
                Path(self.data_path).glob(
                    f"*{self.tic_id}*"
                    )
                )
            )
        matches = [
            re.search("s00[0-9][0-9]", x.name).group() for x in filepaths
            ]
        sectors = [x.lstrip("s00") for x in matches]
        return sectors
    
    @staticmethod
    def _create_full_id(input, output_length):
        # adds leading zeros to `input` until `output_lenght` is reached
        return f'{int(input):0{output_length}}'

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
        m = hdu[1].data["QUALITY"] == quality_flag
        self.time = hdu[1].data["TIME"][m]
        flux = hdu[1].data["PDCSAP_FLUX"][m]
        
        self.flux_err = hdu[1].data["PDCSAP_FLUX_ERR"][m] / np.median(flux)
        self.flux = flux / np.median(flux)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data_dir = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data"

    # lcpath = _get_filepath(data_dir, tic=21371, sector=11)

    lc = Pathfinder(data_dir, tic=21371)
    print(lc.tic_id)
    print(lc.sectors)
    print(lc.filepaths)

    # plt.errorbar(lc.time, lc.flux, yerr=lc.flux_err, capsize=0, fmt='.')
    plt.show()
