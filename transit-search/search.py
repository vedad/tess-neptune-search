#!/usr/bin/env python

import argparse
import pickle
import matplotlib.pyplot as plt

from data import Lightcurve, Pathfinder
from star import Star


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
                                     Run a transit search on TESS-SPOC light curves using the TIC identifier.
                                     """)

    parser.add_argument("tic", nargs='+', type=str, default=[],
                        help="""
            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            """)

    parser.add_argument('--sector', nargs='+', type=int, default=None,
                        help='which sectors to show')
    
    # data location
    parser.add_argument("--data_dir", type=str, action="store", 
                        help="path to directory where TESS data is located",
                        default="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data")

    # save data to file
    parser.add_argument('--save', type=str, action="store", 
                        help='path to directory where to save output files',
                        default="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data_results")
    parser.add_argument('--overwrite', action='store_true',
                        help='flag to overwrite already existing file')


    args = parser.parse_args()

    p = Pathfinder(args.data_dir, tic=args.tic, sector=args.sector)
    lightcurve_files = p.filepaths
    print(f"Found {len(lightcurve_files)} data files for sectors {p.sectors}")

    lightcurves = [Lightcurve(x) for x in lightcurve_files]

    if len(args.tic) < 2:
        tic = args.tic[0]
    star = Star(tic, lightcurves)

    results = star.search()

    savedirs = [p._create_savedir(args.save, results[i].SDE, i+1) 
                for i in range(len(results))]

    for i,savedir in enumerate(savedirs):
        with open(savedir, "wb") as handle:
            pickle.dump(results[i], handle)


# questions:

# maximum period to search?
# search combined light curves (if observed in multiple sectors?) or
# sector by sector
