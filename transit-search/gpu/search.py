#!/usr/bin/env python

import numpy as np
import time, datetime, argparse, pickle
from pathlib import Path
from wotan import flatten
from tqdm import tqdm
from scipy.signal import find_peaks
from astropy.timeseries import BoxLeastSquares


from data import *
from validate import *
from stats import *#spectra, period_uncertainty, estimate_trend, find_top_peaks, compute_SDE
from grid import frequency_grid
from loggers import get_logger

if __name__ == '__main__':

    GPU = True

    if GPU:
        import cuvarbase.bls as bls

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

    parser.add_argument("--index-file", type=str, default=None,
                        help="""
            Path to a file containing TIC IDs and the sectors it was observed in
            """)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="""
            Path to directory containing all TESS LC files
            """)
    parser.add_argument("--inverted", action="store_true",
                        help="""
            Flag to invert light curves
            """)
    # BLS args
    # parser.add_argument("--qmin_fac", type=float, default=3e-3,
    #                     help="""
    #         Minimum transit phase duration (dur/period) to search
    #         """)
    # parser.add_argument("--qmax_fac", type=float, default=0.15,
    #                     help="""
    #         Maximum transit phase duration (dur/period) to search
    #         """)
    parser.add_argument("--qmin-fac", type=float, default=0.5,
                        help="""
            Lower transit phase duration factor compared to Keplerian assumption
            """)
    parser.add_argument("--qmax-fac", type=float, default=2,
                        help="""
            Upper transit phase duration factor compared to Keplerian assumption
            """)
    parser.add_argument("--dlogq", type=float, default=0.05,
                        help="""
            Logarithmic spacing of the transit phase duration
            """)
    parser.add_argument("--period-min", type=float, default=0.5,
                        help="""
            Maximum period to search
            """)
    parser.add_argument("--period-max", type=float, default=12,
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

        logger_log = get_logger(
                            name="LOG",
                            filename=Path(args.save, "LOG"),
                            )
        logger_sde = get_logger(
                            name="SDE",
                            filename=Path(args.save, "SDE"),
                            format="%(message)s",
                            level="INFO",
                            )

    wotan_kwargs = validate_wotan_args(
                        window_length=args.window_length,
                        method = args.detrending_method
                        )
    
    search_params = validate_search_args(
                    qmin_fac=args.qmin_fac,
                    qmax_fac=args.qmax_fac,
                    dlogq=args.dlogq,
                    ignore_negative_delta_sols=True,
                    functions=bls.compile_bls()
                    )

    if args.index_file is None:
        lightcurve_files = np.loadtxt(args.target_list, dtype=str)
        N = len(lightcurve_files)
    else:
        # if spoc-ffi list then these are tic numbers, not filepaths as above
        tic = np.loadtxt(args.target_list, dtype=str)
        N = len(tic)
        with open(args.index_file, "rb") as handle:
            index_file = pickle.load(handle)

    start = time.time()
    success = 0
    fail = 0

    for i in range(N):

        # if running on injected data
        try:
            if args.index_file is None:
                data, truth = _split_injected_lightcurve(lightcurve_files[i])
                _tic, _sim, _scc = data["TIC"], data["SIM"], data["SCC"]
                data.pop("TIC")
                data.pop("SIM")
                data.pop("SCC")
            else:
                _tic = tic[i]
                filenames = (
                    _create_lightcurve_filenames(
                        _tic,
                        index_file[_tic]
                    )
                )
                lightcurve_files = (
                    [Path(args.data_dir, x).as_posix() for x in filenames]
                )
                lightcurves = [Lightcurve(x) for x in lightcurve_files]
        except TypeError as err: # catch "buffer is too small for requested array" error when reading corrupted files
            if do_logging:
                logger_log.error(err)
                logger_sde.info(f"{_tic},-1")
            fail += 1
            continue
        except FileNotFoundError as err:
            if do_logging:
                logger_log.error(err)
                logger_sde.info(f"{_tic},-1")
            fail += 1
            continue
        except PermissionError as err:
            if do_logging:
                logger_log.error(err)
                logger_sde.info(f"{_tic},-1")
            fail += 1
            continue

        try:
            if args.index_file is None:
                lightcurves = [LightcurveInjected(d) for d in data.items()]
                star = StarInjected(_tic, lightcurves, _sim)
                yerr = np.ones_like(star.y)
            else:
                star = Star(_tic, lightcurves)
                yerr = star.y_err

            try:
                y, trend = flatten(star.x, star.y, **wotan_kwargs)
            except ValueError:
                flag = -4
                raise

            m = np.isfinite(y)
            x, y, yerr = star.x[m], y[m], yerr[m]

            # invert light curve
            if args.inverted:
                diff = y - 1
                y -= 2*diff

            try:
                freqs, frequency_grid_args, flag = frequency_grid(
                                star.radius.value,
                                star.mass.value,
                                np.max(x) - np.min(x),
                                period_min = args.period_min,
                                period_max = args.period_max,
                                oversampling_factor = args.oversampling
                                )
            except ValueError:
                flag = -1
                raise
            
            if flag != 0:
                raise

            if GPU:
                _, bls_power = bls.eebls_transit_gpu(
                    x, y, yerr, freqs=freqs, use_fast=True, **search_params
                    )
            else:
                bls_power = np.ones_like(freqs)

            message = f"TIC {_tic:<12} ----- PERIODOGRAM COMPLETE"

            power_trend = estimate_trend(freqs, bls_power, args.oversampling)
            top_freqs, top_power, top_index = find_top_peaks(freqs, bls_power-power_trend)
            top_periods = 1/top_freqs
            top_SDE = compute_SDE(
                bls_power-power_trend, args.oversampling, peaks=top_power
            )
            SDE = top_SDE[0]

            SDE = SDE if np.isfinite(SDE) else -1

            if GPU:
                _, _, sols = bls.eebls_transit_gpu(
                    x, y, yerr, freqs=top_freqs, use_fast=False, **search_params
                    )
                q, phi = np.array(sols).T

                # find mid-transit time
                phi_mid = phi + 0.5 * q # phi is the phase of the start of transit
                m = phi_mid > 1
                phi_mid[m] -= 1
                n_epochs = np.floor(x[0] / top_periods)
                t0 = top_periods * (phi_mid + n_epochs)

                top_solutions = dict(
                    period=top_periods,
                    period_err=np.array([
                        period_uncertainty(1/freqs, bls_power-power_trend, j)
                        for j in top_index
                        ]),
                    SDE=top_SDE,
                    power=top_power,
                    q=q,
                    phi=phi,
                    t0=t0,
                    dur=top_periods * q
                    )
            else:
                top_solutions = dict()

            message += " ----- SOLUTION COMPLETE"

            # parameters to save
            names = ["tic",
                "periods", "power", "peaks",
                "frequency_grid_args",
                "teff", "logg", "rstar", "mstar", "rho", "Tmag"
                ]
            values = [_tic,
                1/freqs, bls_power, top_solutions,
                frequency_grid_args,
                star.teff.value, star.logg.value, star.radius.value, star.mass.value, star.rho.value, star.Tmag.value
                ]
            
            if args.index_file is None:
                names += ["sim", "truth"]
                values += [_sim, truth]

            if args.save is not None:
                if args.index_file is None:
                    savedir = _create_savedir_injected(
                        args.save, 1, _tic,
                        _sim,
                        _scc,
                        prefix=f"sol_SDE-{SDE:06.1f}"
                        )
                else:
                    savedir = _create_savedir(
                        args.save, 1, _tic,
                        index_file[_tic],
                        prefix=f"sol_SDE-{SDE:06.1f}"
                        )
                    
                _create_fits_file(dict(zip(names, values)), savedir)
                # with open(savedir, 'wb') as handle:
                #     pickle.dump(
                #         dict(
                #             zip(names, values)
                #             ), handle)
                message += f" & SAVED"

            if do_logging:
                logger_log.info(message) # log completion of power spectrum calculation
                if args.index_file is None:
                    logger_sde.info(f"{_tic},{SDE:.2f},{_sim}")
                else:
                    logger_sde.info(f"{_tic},{SDE:.2f}")

            success += 1

        except:
            if do_logging:
                logger_log.exception(f"TIC {_tic:<12} ----- SEARCH FAILED")
                SDE = flag
                if args.index_file is None:
                    logger_sde.info(f"{_tic},{SDE:.0f},{_sim}")
                else:
                    logger_sde.info(f"{_tic},{SDE:.0f}")
            fail += 1
            continue

    time_str = str(datetime.timedelta(seconds=time.time() - start))
    print(f"runtime: {time_str}")

    if do_logging:
        logger_log.info(f"{success}/{N} TARGETS COMPLETE ({fail} FAILED) in {time_str}")

    # end = time.time()
    # elapsed_time = end - start
    # hours = elapsed_time // 60 // 60
    # mins = elapsed_time - hours * 60
    # secs = elapsed_time - hours * 60 - mins * 60




