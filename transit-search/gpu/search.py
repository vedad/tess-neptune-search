#!/usr/bin/env python

import numpy as np
import time, datetime, argparse, logging
import dill as pickle
from pathlib import Path
from wotan import flatten
from tqdm import tqdm
import logging
# import cuvarbase.bls as bls

from data import *
from validate import *
from stats import spectra, period_uncertainty
from grid import frequency_grid

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="""
                                     Run a transit search on TESS-SPOC light curves
                                     """)

    parser.add_argument("-f", "--filename", type=str, default=None,
                        help="""
            The path to a file with an injected transit signal
            """)
    
    parser.add_argument("--target-list", type=str, default=None,
                        help="""
            Path to a file containing TIC IDs, one per line
            """)
    
    # BLS args
    parser.add_argument("--q_min", type=float, default=3e-3,
                        help="""
            Minimum transit phase duration (dur/period) to search
            """)
    parser.add_argument("--q_max", type=float, default=0.15,
                        help="""
            Maximum transit phase duration (dur/period) to search
            """)
    parser.add_argument("--dlogq", type=float, default=0.05,
                        help="""
            Logarithmic spacing of the transit phase duration
            """)
    parser.add_argument("--period_min", type=float, default=0.5,
                        help="""
            Maximum period to search
            """)
    parser.add_argument("--period_max", type=float, default=12,
                        help="""
            Maximum period to search
            """)
    parser.add_argument("--oversampling", type=int, default=5,
                        help="""
            Factor to oversample frequency grid
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
    
    # save data to file
    parser.add_argument('--save', type=str, action="store", 
                        help='path to directory where to save output files',
                        default=None
                        )

    args = parser.parse_args()

    do_logging = False

    if args.save is not None:
        do_logging = True

        # Create and configure logger
        logging.basicConfig(filename=Path(args.save, "LOG"),
                            format='%(asctime)s|%(message)s',
                            filemode='w')

        # Creating an object
        logger = logging.getLogger()

        # Setting the threshold of logger to DEBUG
        logger.setLevel(logging.DEBUG)



    wotan_kwargs = validate_wotan_args(
                        window_length=args.window_length,
                        method = args.detrending_method
                        )
    
    search_params = validate_search_args(
                    qmin=args.q_min,
                    qmax=args.q_max,
                    dlogq=args.dlogq,
                    ignore_negative_delta_sols=True
                    )
    
    lightcurve_files = np.loadtxt(args.target_list, dtype=str)

    # N = len(lightcurve_files)
    N = 3

    start = time.time()

    for i in tqdm(range(N)):

        try:
            data, _ = _split_injected_lightcurve(lightcurve_files[i])

            _tic, _sim, _scc = data["TIC"], data["SIM"], data["SCC"]
            data.pop("TIC")
            data.pop("SIM")
            data.pop("SCC")

        except FileNotFoundError as err:
            if do_logging:
                logger.error(err)
            continue

        try:

            lightcurves = [LightcurveInjected(d) for d in data.items()]
            star = StarInjected(_tic, lightcurves, _sim)

            y, trend = flatten(star.x, star.y, **wotan_kwargs)
            m = np.isfinite(y)
            x, y, yerr = star.x[m], y[m], np.ones_like(y[m])

            # frequency_grid_kwargs = dict(
            #     R_star = star.radius.value,
            #     M_star = star.mass.value,
            #     time_span = x.max() - x.min(),
            #     period_min = args.period_min,
            #     period_max = args.period_max,
            #     oversampling_factor = args.oversampling
            # )

            freqs, frequency_grid_args = frequency_grid(
                                star.radius.value,
                                star.mass.value,
                                np.max(x) - np.min(x),
                                period_min=args.period_min,
                                period_max=args.period_max,
                                oversampling_factor=args.oversampling
                                )


            # bls_power = bls.eebls_transit_gpu(x, y, yerr, freqs, fast=True,
                                        # **search_params)
            bls_power = np.ones_like(freqs)
            message = f"TIC {_tic:<12} ----- PERIODOGRAM COMPLETE"
            
            # save power
            # if args.save is not None:
            #     savedir = _create_savedir(
            #         args.save, 1, _tic, _sim, _scc, prefix="bls_power"
            #         )
            #     freq_power = np.array(
            #         list(
            #             zip(freqs, bls_power)
            #         ), dtype=[('frequency', '<f8'), ('power', '<f8')]
            #     )
            #     with open(savedir, 'wb') as handle:
            #         pickle.dump(freq_power, handle)
                # message += " & SAVED"

            # run again in vicinity of maximum power to get best-fit solution since # the ``fast`` version doesn't keep the best q and phi values
            index = np.argmax(bls_power)
            best_freq = freqs[index]

            # best_power, ( (best_q, best_phi), ) = bls.eebls_transit_gpu(
                # x, y, yerr, freqs=np.array([best_freq]), use_fast=False, **search_params
                # )

            best_power, best_q, best_phi = np.ones(3)
            message += " ----- SOLUTION COMPLETE"
            
            best_period = 1/best_freq
            best_t0 = best_phi * best_period + x[0]
            best_dur = best_period * best_q
            period_err = period_uncertainty(1/freqs, bls_power)
            # _, SDE, _, _ = spectra(bls_power, args.oversampling)
            SDE=1

            # parameters to save
            names = ["power", "SDE", "best_freq", "best_power", "q", "phi", "period", "period_err", "t0", "dur", "frequency_grid_args"
                ]
            values = [bls_power, SDE, best_freq, best_power, best_q, best_phi,
                    best_period, period_err, best_t0, best_dur, frequency_grid_args]
            
            if args.save is not None:
                savedir = _create_savedir(
                    args.save, 1, _tic, _sim, _scc, prefix=f"sol_SDE={SDE:.1f}"
                    )
                with open(savedir, 'wb') as handle:
                    pickle.dump(
                        dict(
                            zip(names, values)
                            ), handle)
                message += f" & SAVED" 

            if do_logging:
               logger.info(message) # log completion of power spectrum calculation

        except:
            if do_logging:
                logger.exception(f"TIC {_tic:<12} ----- SEARCH FAILED")
            continue

    if do_logging:
        logger.info("TARGETS COMPLETE")

    # end = time.time()
    # elapsed_time = end - start
    # hours = elapsed_time // 60 // 60
    # mins = elapsed_time - hours * 60
    # secs = elapsed_time - hours * 60 - mins * 60
    time_str = str(datetime.timedelta(seconds=time.time() - start))
    print(f"runtime: {time_str}")



