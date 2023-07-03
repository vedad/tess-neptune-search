#!/usr/bin/env python

import argparse
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import mpi4py.MPI

from data import *
from star import *
from injected import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
                                     Run a transit search on TESS-SPOC light curves
                                     """)

    parser.add_argument("--tic", nargs='+', type=str, default=[],
                        help="""
            The TIC ID of a star to search. Will fetch the correct data located in `--data-dir`
            """)
    
    parser.add_argument("--target-list", type=str, default=None,
                        help="""
            Path to a file containing TIC IDs, one per line
            """)
    
    # parser.add_argument("--target-list-path", type=str, default=None,
    #                     help="""
    #         Path to a file containing paths to FITS files, one per line
    #         """)
    parser.add_argument("--injected", action='store_true',
                        help="""
            Path to a file containing paths to FITS files, one per line
            """)
    
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="""
            Multiprocessing threads to use
            """)
    
    # save data to file
    parser.add_argument('--mpi', action="store_true", 
                        help='Flag whether to to use mpi4py to parallelize',
                        )

    parser.add_argument('--sector', nargs='+', type=int, default=None,
                        help='which sectors to show')
    
    # data location
    parser.add_argument("--data-dir", type=str, action="store", 
                        help="path to directory where TESS data is located",
                        default="/Users/u2271802/Data/tess/tess-spoc_ffi")

    # save data to file
    parser.add_argument('--save', type=str, action="store", 
                        help='path to directory where to save output files',
                        default="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data_results")
    parser.add_argument('--overwrite', action='store_true',
                        help='flag to overwrite already existing file')


    args = parser.parse_args()

    if (args.target_list is None 
        # and args.target_list_path is None 
        and len(args.tic) == 0
    ):
        raise ValueError("Target(s) must be specified by `--tic`, `--target-list` or `--target-list-path")
    elif len(args.tic) > 0:
        tic = args.tic
    elif args.target_list is not None and not args.injected:
        tic = np.loadtxt(args.target_list, format=str)



    if args.injected:
        lightcurve_files = np.loadtxt(args.target_list, dtype=str)
        # special stuff for injected light curves since each FITS file contains data from all sectors

        #find out which number processor this particular instance is,
        #and how many there are in total
        # SOURCE: https://gist.github.com/joezuntz/7c590a8652a1da6dc4c9
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()

        with tqdm(total=len(lightcurve_files)) as pbar:
            for i,f in enumerate(lightcurve_files):
        # for i,f in enumerate(lightcurve_files):

                if i%size!=rank: continue

                # print(f"Task number {i} ({f}) is being done by processor {rank} of {size}")

                data, truth = _split_injected_lightcurve(f)
                _tic = data["TIC"]
                data.pop("TIC")
                lightcurves = [LightcurveInjected(d) for d in data.items()]
                star = StarInjected(_tic, lightcurves)
                results = star.search(use_threads=args.threads,
                                      truth=truth,
                                      verbose=False,
                                      show_progress_bar=False)

                pbar.update()

            # savedirs = [p._create_savedir(args.save, results[i].SDE, i+1) 
            #     for i in range(len(results))]

            # for i,savedir in enumerate(savedirs):
            #     with open(savedir, "wb") as handle:
            #         pickle.dump(results[i], handle)
    else:
        for _tic in tic:
            p = Pathfinder(tic=_tic,
                sector=args.sector, data_path=args.data_dir)
            lightcurve_files = p.filepaths
            lightcurves = [Lightcurve(x) for x in lightcurve_files]
            star = Star(_tic, lightcurves)
            results = star.search()



    # if single object
    # if len(args.tic) == 1:
    #     p = Pathfinder(tic=args.tic[0],
    #                sector=args.sector, data_path=args.data_dir)
    #     lightcurve_files = p.filepaths
    #     lightcurves = [Lightcurve(x) for x in lightcurve_files]
    #     star = Star(tic, lightcurves)
    #     results = star.search()
    # multiple objects -> parallelize

    # print(f"Found {len(lightcurve_files)} data files for sectors {p.sectors}")

    # if len(args.tic) < 2:
    #     tic = args.tic[0]
    # star = Star(tic, lightcurves)

    # print('star mass', star.mass)

    # results = star.search(use_threads=1)

    # savedirs = [p._create_savedir(args.save, results[i].SDE, i+1) 
    #             for i in range(len(results))]

    # for i,savedir in enumerate(savedirs):
    #     with open(savedir, "wb") as handle:
    #         pickle.dump(results[i], handle)


    # # start 4 worker processes
    # with Pool(processes=4) as pool:

    #     # print "[0, 1, 4,..., 81]"
    #     print(pool.map(f, range(10)))

    #     # print same numbers in arbitrary order
    #     for i in pool.imap_unordered(f, range(10)):
    #         print(i)

    #     # evaluate "f(20)" asynchronously
    #     res = pool.apply_async(f, (20,))      # runs in *only* one process
    #     print(res.get(timeout=1))             # prints "400"

    #     # evaluate "os.getpid()" asynchronously
    #     res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    #     print(res.get(timeout=1))             # prints the PID of that process

    #     # launching multiple evaluations asynchronously *may* use more processes
    #     multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    #     print([res.get(timeout=1) for res in multiple_results])

    #     # make a single worker sleep for 10 seconds
    #     res = pool.apply_async(time.sleep, (10,))
    #     try:
    #         print(res.get(timeout=1))
    #     except TimeoutError:
    #         print("We lacked patience and got a multiprocessing.TimeoutError")

    #     print("For the moment, the pool remains available for more work")

    # # exiting the 'with'-block has stopped the pool
    # print("Now the pool is closed and no longer available")


# questions:

# maximum period to search?
# search combined light curves (if observed in multiple sectors?) or
# sector by sector
