#!/usr/bin/env python

import argparse
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import mpi4py.MPI
from pathlib import Path

from data import *
from star import *
from injected import *
from validate import validate_wotan_args


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
    
    # # save data to file
    # parser.add_argument('--mpi', action="store_true", 
    #                     help='Flag whether to to use mpi4py to parallelize',
    #                     )

    parser.add_argument('--sector', nargs='+', type=int, default=None,
                        help='which sectors to show')
    
    # data location
    parser.add_argument("--data-dir", type=str, action="store", 
                        help="path to directory where TESS data is located",
                        default="/Users/u2271802/Data/tess/tess-spoc_ffi")

    # save data to file
    parser.add_argument('--save', type=str, action="store", 
                        help='path to directory where to save output files',
                        default="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data_results"
                        )
    parser.add_argument('--overwrite', action='store_true',
                        help='flag to overwrite already existing file')
    
    # TLS args
    parser.add_argument("--oversampling", type=int, default=5,
                        help="""
            Factor to oversample period grid
            """)
    parser.add_argument("--duration-step", type=float, default=1.05,
                        help="""
            Duration grid step of transit search
            """)
    parser.add_argument("--period-max", type=float, default=11,
                        help="""
            Maximum period to search
            """)
    parser.add_argument("--verbose", action="store_true",
                        help="""
            Show progress
            """)
    parser.add_argument("--show-progress-bar", action="store_true",
                        help="""
            Show progress
            """)
    
    # Wotan args
    parser.add_argument("--window-length", type=float, default=0.5,
                        help="""
            Sliding window size for detrending
            """)
    parser.add_argument("--detrending-method", type=str, default="biweight",
                        help="""
            ``Wotan`` method to use for detrending the light curve
            """)

    args = parser.parse_args()

    wotan_kwargs = validate_wotan_args(
                        window_length=args.window_length,
                        method = args.detrending_method
                        )
    tls_kwargs = dict(oversampling_factor = args.oversampling,
                      duration_grid_step = args.duration_step,
                      period_max = args.period_max,
                      use_threads = args.threads,
                      verbose = args.verbose,
                      show_progress_bar = args.show_progress_bar)

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

        # rudimentary progress bar (per target)
        
        with tqdm(total=len(lightcurve_files)) as pbar:
            for i,f in enumerate(lightcurve_files):

                if i % size != rank:
                    continue

                data, truth = _split_injected_lightcurve(f)
                _tic, _sim, _scc = data["TIC"], data["SIM"], data["SCC"]
                data.pop("TIC")
                data.pop("SIM")
                data.pop("SCC")

                # skip it file exists
                file_pattern = Path(args.save).glob(f"*{_sim}*{_tic}*.pickle")
                file_pattern_exists = len(list(file_pattern)) > 0
                if file_pattern_exists:
                    print(f"skipped {_tic}, results found")
                    pbar.update()
                    continue

                lightcurves = [LightcurveInjected(d) for d in data.items()]
                star = StarInjected(_tic, lightcurves, _sim)
                results = star.search(truth=truth, tls_kwargs=tls_kwargs, wotan_kwargs=wotan_kwargs)
                savedirs = [_create_savedir(args.save, results[i].SDE, i+1, _tic, _sim, _scc)
                    for i in range(len(results))]

                for i,savedir in enumerate(savedirs):
                    with open(savedir, "wb") as handle:
                        pickle.dump(results[i], handle)

                pbar.update(1)
    else:

        # copy directory structure for data into results
        # https://stackoverflow.com/questions/4073969/copy-folder-structure-without-files-from-one-location-to-another
        for _tic in tic:
            p = Pathfinder(tic=_tic,
                sector=args.sector, data_path=args.data_dir)
            lightcurve_files = p.filepaths
            lightcurves = [Lightcurve(x) for x in lightcurve_files]
            star = Star(_tic, lightcurves)
            results = star.search(
                    tls_kwargs=tls_kwargs,
                    wotan_kwargs=wotan_kwargs
                                  )

            savedirs = [p._create_savedir2(args.save, results[i].SDE, i+1) 
                    for i in range(len(results))]

            for i,savedir in enumerate(savedirs):
                with open(savedir, "wb") as handle:
                    pickle.dump(results[i], handle)



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
