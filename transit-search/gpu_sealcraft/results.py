#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u
from pathlib import Path
import argparse
from multiprocessing import Pool
from wotan import flatten
from astropy.io import fits
from functools import partial
from scipy.signal import find_peaks

from grid import _frequency_grid_NAC
from stats import spectra
from validate import validate_wotan_args

# data_file = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/injection/results/20230808_biweight/sol_SDE-0009.3_PLA-6-8363_0000000305940013.01_s39.pickle"



def get_bls_filename(data_dir, target):
    p = Path(data_dir)
    filepath = p.glob(f"*{target}*.pickle")
    return filepath

# def get_bls_filenames(data_dir, targets=None, sde_min=None):

#     if targets is None:
#         p = Path(data_dir)
#         filepaths = p.glob("*.pickle")
#     else:
#             filepaths = (get_bls_filename(
#                 data_dir, target) for target in targets)

#     return filepaths

    
def get_data_filenames_per_target(data_dir, target):

    p = Path(data_dir)
    filepath = p.glob(f"*{target}*.fits")
    return filepath

def plot_periodogram(results, ax):
    ax.axvline(results["period"], alpha=0.2, lw=3)
    ax.set_ylim(0, results["SDE_ary"].max()*1.1)
    for n in [0.5, 2, 3, 4]:
        if n * results["period"] < np.max(results["periods"]):
            ax.axvline(
                n * results["period"],
                alpha=0.2, lw=1, linestyle="dashed"
                )
    ax.set_ylabel("SDE")
    ax.set_xlabel("period (days)")
    ax.plot(results["periods"], results["SDE_ary"], color='black', lw=0.5)
    ax.set_xlim(0, max(results["periods"]))

    return ax

def plot_folded(results, ax):

    X, Y = (np.concatenate(results["model_folded_phase"]),
            np.concatenate(results["model_folded_model"])
            )

    ax.plot(
        X,
        Y,
        color='C0')

    X, Y = (np.concatenate(results["phase"]),
            np.concatenate(results["flux"])
            )
    ax.errorbar(
        X,
        Y,
        color='k',
        fmt='.',
        mew=0,
        alpha=0.5,
        zorder=2)
    
    m = (X > -1.5 * results["q"]) & (X < 1.5 * results["q"])
    window_x = [X[m].min(), X[m].max()]
    window_y = [np.percentile(Y[m], 5), np.percentile(Y[m], 95)]

    ax.set_xlim(*window_x)
    ax.set_ylim(*window_y)
    ax.set_xlabel('phase')
    ax.set_ylabel('relative flux')

    return ax

def plot_model(results, ax):
    X, Y = np.concatenate(results["time"]), np.concatenate(results["flux"])
    residuals = Y - np.concatenate(results["trend"])

    ax.errorbar(
        X,
        residuals + 1,
        color='k',
        fmt='.',
        mew=0,
        alpha=0.5,
        zorder=2)
    
    ax.set_xlim(X.min()-1, X.max()+1)
    ax.set_xlabel('time (TJD)')
    ax.set_ylabel('flux')

    X, Y = np.concatenate(results["time"]), np.concatenate(results["model"])
    ax.plot(X, Y,
            "C1", lw=2)
    
    return ax

def plot_data(results, ax):

    X, Y = np.concatenate(results["time"]), np.concatenate(results["flux"])
    trend = np.concatenate(results["trend"])

    ax.errorbar(
        X,
        Y,
        color='k',
        fmt='.',
        mew=0,
        alpha=0.5,
        zorder=2)
    ax.plot(X, trend, lw=1, color="C1", zorder=3)
    ax.set_xlim(X.min()-1, X.max()+1)
    ax.set_xlabel('time (TJD)')
    ax.set_ylabel('raw flux')

    return ax

def plot_residuals(results, ax):

    X, Y = np.concatenate(results["time"]), np.concatenate(results["flux"])
    residuals = Y - np.concatenate(results["trend"])

    ax.errorbar(
        X,
        residuals,
        color='k',
        fmt='.',
        mew=0,
        alpha=0.5,
        zorder=2)
    
    ax.set_xlim(X.min()-1, X.max()+1)
    ax.set_xlabel('time (TJD)')
    ax.set_ylabel('flux')

    return ax


# def plot_sector(results, ax):

#     """
#     probably want on separate page
#     """

#     x, y = results.data_x, results.data_y

#     ax.errorbar(
#         X,
#         Y,
#         color='k',
#         fmt='.',
#         alpha=0.5,
#         zorder=2)
    
#     ax.set_xlim(x.min()-1, x.max()+1)
#     ax.set_xlabel('time (TJD)')
#     ax.set_ylabel('relative flux')

#     return ax

def plot_information(results, ax):

    # Rp = np.sqrt(results["depth"]/1e6) * results["rstar"].to(u.R_earth)

    # if np.isfinite(results.FAP):
    #     fap_str = "{:<5} = {:.3f}\n\n".format("FAP", results.FAP)
    # else:
    #     fap_str = "{:<5} = >0.1\n\n".format("FAP")

    info = ("\n\n"
        "{:<5} = {:.1f}\n".format("SDE", results["SDE"])+
        # "{:<5} = {:.1f}\n".format("S/N", results.snr)+
        # "{:<5} = {:.3f}\n\n".format("FAP", results.FAP)+
        # fap_str+
#
        "{:<5} = {:.5f} d\n".format("P", results["period"])+
        "{:<5}+- {:.5f} d\n".format("", results["period_err"])+
        "{:<5} = {:.3f}\n".format("r/R", results["ror"])+
        "{:<5} = {:.3f} h\n".format("dur", results["dur"]*24)+
        "{:<5} = {:.4f} ppm\n".format("depth", results["depth"]*1e6)
        # "{:<5} = {:.1f}\n\n".format("Rp", Rp)+
        #
        # "{:<5} = {:.1f}\n".format("Tmag", results["Tmag"]) +
        # "{:<5} = {:.0f}\n".format("Teff", results["teff"]) +
        # "{:<5} = {:.2f}\n\n".format("Rs", results["rstar"]) +
        # "{:<5} = {:.2f}\n\n".format("rho", results["rho"])
        #
        # "{:} = {:.1f}\n".format("odd/even mismatch", results.odd_even_mismatch)+
        # "{:}/{:} transits with data".format(results.distinct_transit_count, results.transit_count)
    )

    text_kwargs = dict(multialignment="left", va="top", ha="left",
                       transform=ax.transAxes, fontfamily="monospace")

    # tic id
    # ax.text(0, 1, f"TIC {results.cid}",fontweight="bold", **text_kwargs)

    # # SDE colour coded
    # sde = results.SDE
    # if sde < 6.98: # >1% FAP
    #     sde_colour = "red"
    # elif 6.98 <= sde < 8.319: # between 0.1% and 1% FAP
    #     sde_colour = "orange"
    # else: # <0.1% FAP
    #     sde_colour = "green"
    sde_str = "\n{:<5} = {:.1f}".format("SDE", results["SDE"])
    ax.text(0, 1, sde_str, **text_kwargs)

    # all other info
    ax.text(0, 1, info, **text_kwargs)

    ax.axis("off")

    return ax
    

def plot_sheet(results, savedir=None):
    # X, Y = np.concatenate(results["time"]), np.concatenate(results["flux"])

    fig, ax = plt.subplot_mosaic(
        """
        AAAt
        aaat
        BBBb
        CCCC
        DDDD
        """,
    figsize=(8,10), height_ratios=[3,2,3,2,2])

    # A: raw data
    ax["A"] = plot_data(results, ax["A"])
    ax["a"] = plot_residuals(results, ax["a"])
    ax["t"] = plot_information(results, ax["t"])
    ax["B"] = plot_folded(results, ax["B"])
    ax["C"] = plot_model(results, ax["C"])
    ax["D"] = plot_periodogram(results, ax["D"])

    if savedir is not None:
        savedir = _filepath(results, savedir)
        # savedir += f"/{results.cid}_SDE={int(np.round(results.SDE)):.0f}.pdf"
        fig.savefig(savedir, bbox_inches="tight")

    plt.show()
    return



def _read_data_injected(filename):
    hdu = fits.open(filename)

    x = hdu[1].data["TIME"]
    y = hdu[1].data["FLUX"]
    print('y', np.any(np.isnan(y)))
    yerr = np.ones_like(x)

    wotan_kwargs = validate_wotan_args(
                        window_length=0.5,
                        method = "biweight"
                        )

    y_flatten, trend = flatten(x, y, **wotan_kwargs)
    print('trend', trend, 'x', x, 'y', y)
    m = np.isfinite(y)

    truth = dict(
                period = hdu[0].header["P"],
                t0 = hdu[0].header["T0"],
                duration = hdu[0].header["TDUR"],
                depth = hdu[0].header["DEPTH"]
                )
    
    return x[m], y[m], yerr[m], trend[m], truth

def approx_lte(x, y):
    return x < y or np.isclose(x, y)

def approx_gte(x, y):
    return x >= y or np.isclose(x, y)

def _set_data(target, data_dir=None, results_dir=None):
    # create BLS model at right freq/duration/phase/depth

    try:
        filename = list( # generator is returned from glob
            Path(results_dir).glob(f"*{target}*.pickle")
        )[0]
    except IndexError:
        print(f"could not find pickle file for {target}")
        return
    # print(filename)
    # filename = next(filename)
    # print(filename)
    with open(filename, "rb") as handle:
        result = pickle.load(handle)

    # should add oversamplign to results when creating file
    result["oversampling"] = 5

    
    filename = Path(data_dir).glob(f"{target}.fits")
 
    for f in filename:
        truth = _read_data_injected(f.as_posix())[-1]
    result["truth"] = truth

    return result

def get_peaks(freqs, power, oversampling=5, npeaks=5):
    w = int(oversampling * 10)
    peaks_idx, props = find_peaks(power, distance=w, height=5)
    idx_sorted_by_height = peaks_idx[
        np.argsort(props["peak_heights"])[::-1][:npeaks]
    ]

    peaks = np.column_stack(
        (freqs[idx_sorted_by_height], power[idx_sorted_by_height])
    )
    return peaks

def recreate_freq(filename):

    with open(filename, "rb") as handle:
        result = pickle.load(handle)

    # get frequencies for periodogram
    freq_grid_args = [
        result["frequency_grid_args"][s] for s in ["N_opt", "A", "C"]
    ]
    freqs = _frequency_grid_NAC(*freq_grid_args)

    m_min = freqs >= result["frequency_grid_args"]["f_min"]
    first_true, = np.where(m_min)
    if np.any(first_true):
        i = first_true[0]
        m_min[i] = approx_gte(freqs[i], result["frequency_grid_args"]["f_min"])
        m_min[i-1] = approx_gte(freqs[i-1],
                                result["frequency_grid_args"]["f_min"])
        m_min[i+1] = approx_gte(freqs[i+1],
                                result["frequency_grid_args"]["f_min"])

    m_max = freqs < result["frequency_grid_args"]["f_max"]
    first_true, = np.where(m_max)
    if np.any(first_true):
        i = first_true[-1]
        m_max[i] = approx_lte(freqs[i], result["frequency_grid_args"]["f_max"])
        m_max[i-1] = approx_lte(freqs[i-1],
                                result["frequency_grid_args"]["f_max"])
        if (i+1) < len(m_max):
            m_max[i+1] = approx_lte(freqs[i+1],
                                    result["frequency_grid_args"]["f_max"])

    m = m_min & m_max

    freqs = freqs[m]
    result["periods"] = 1/freqs

    return result

def _set_data_extended(target, data_dir=None, results_dir=None):
    # create BLS model at right freq/duration/phase/depth

    try:
        filename = list( # generator is returned from glob
            Path(results_dir).glob(f"*{target}*.pickle")
        )[0]
    except IndexError:
        print(f"could not find pickle file for {target}")
        return

    result = recreate_freq(filename)

    # should add oversamplign to results when creating file
    result["oversampling"] = 5

    # get SDE array for periodogram
    sde = spectra(1/result["periods"],
                  result["power"],
                  result["oversampling"])[0]
    result["SDE_ary"] = sde

    result["peaks"] = get_peaks(1/result["periods"], sde)

    # tic_16 = _create_padded_id(target, 16)
    # filename = Path(data_dir).glob(f"*{target}*.fits")
    filename = Path(data_dir).glob(f"{target}.fits")
    # print(list(filename))

    # `*{target}*.fits` if TESS data
    # for filename in filenames:

    result["time"] = []
    result["flux"] = []
    result["flux_err"] = []
    result["phase"] = []
    result["model"] = []
    result["model_folded_phase"] = []
    result["model_folded_model"] = []
    result["trend"] = []
    for f in filename:
        x, y, yerr, trend, truth  = _read_data_injected(f.as_posix())
        result["time"].append(x)
        result["flux"].append(y)
        result["flux_err"].append(yerr)
        result['trend'].append(trend)
        print(trend)
        result['truth'] = truth
        # print('truth', truth)
        w = np.power(yerr, -2)
        w /= np.sum(w)

        phase = (x - result["t0"]) % result["period"] / result["period"]
        phase[phase > 0.5] -= 1
        result["phase"].append(phase)

        def ybar(mask):
            return np.dot(w[mask], (y-trend)[mask]) / sum(w[mask])

        transit = (phase > -0.5*result["q"]) & (phase < 0.5 * result["q"])
        y0 = ybar(~transit)
        delta = y0 - ybar(transit)
        result["depth"] = delta
        result["ror"] = np.sqrt(delta)

        mod = np.ones_like(x)
        mod[transit] -= delta
        result["model"].append(mod)

        sorted_idx = np.argsort(phase)
        phase = phase[sorted_idx]
        transit = (phase > -0.5*result["q"]) & (phase < 0.5 * result["q"])
        result["model_folded_phase"].append(phase)
        mod = np.ones_like(x)
        mod[transit] -= delta
        result["model_folded_model"].append(mod)

    return result

def _pool_fun(target, data_dir=None, results_dir=None, savedir=None):

    result = _set_data(target, data_dir=data_dir, results_dir=results_dir)

    plot_sheet(result, savedir=savedir)

    return None

def _matching_value(measurement, target, wriggle=1):
    lo, hi = (100 - wriggle) / 100, (100 + wriggle) / 100
    return (measurement < ( hi * target) ) & (measurement > (lo * target) )

def recovery(target, data_dir=None, results_dir=None):

    result = _set_data_extended(target, data_dir=data_dir, results_dir=results_dir)
    if result is None:
        return False

    periods = 1/result["peaks"][:,0]
    # print(periods)
    matching = [_matching_value(p, result["truth"]["period"]) for p in periods]
    # return _matching_value(result["period"], result["truth"]["period"])
    return any(matching)
    
def _filepath(results, savedir):
    savedir = Path(savedir,
                   f"{results.cid}_sde={np.round(results.SDE):.1f}.pdf"
                   )
    return savedir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
                                     Run a transit search on TESS-SPOC light curves 
                                     """)

    # parser.add_argument("--tic", nargs='+', type=str, default=[],
    #                     help="""
    #         The TIC ID of a star to search. Will fetch the correct data located in `--data-dir`
    #         """)

    parser.add_argument("--sde-min", type=float, default=None,
                        help="""
                        SDE threshold below which result sheets are not computed
                        """)
    
    parser.add_argument("--ncpu", type=int, default=1,
                        help="""
                        number of CPUs to use
                        """)
    
    parser.add_argument("--target-list", type=str, default=None,
                        help="""
            Path to a file containing TIC IDs, one per line
            """)
    
    parser.add_argument("--results-dir", type=str, default=None,
                        help="""
                        Directory where the BLS power spectra can be found
                        """)
    
    parser.add_argument("--data-dir", type=str, default=None,
                        help="""
                        Directory where the light curves can be found
                        """)
 

    # save data to file
    parser.add_argument('--save', type=str, action="store", 
                        help='path to directory where to save output files',
                        default=None
                        )
    # parser.add_argument('--overwrite', action='store_true',
                        # help='flag to overwrite already existing file')
    
    args = parser.parse_args()

    # tic, sde = np.genfromtxt(args.target_list, delimiter=',', unpack=True, 
    #                             dtype=None)

    # tic_key, sim_key = np.genfromtxt("/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/injection/data/data1000/key.txt", delimiter=',', encoding="UTF-8", dtype=str, unpack=True)
    # # print(key)

    # tic = np.atleast_1d(tic)
    # sde = np.atleast_1d(sde)
    # sim = []
    # for i, _tic in enumerate(tic):
    #     sim.append(sim_key[tic_key == str(_tic)][0])
    # # sim = np.atleast_1d(sim)
    # # sim = [x.decode("UTF-8") for x in sim]
    # sim = np.atleast_1d(sim)

    # if args.sde_min is not None:
    #     m = sde > args.sde_min
    #     tic = tic[m]
    #     sim = sim[m]

    # N = len(tic)

    # with Pool(processes=args.ncpu) as pool:  # Create a multiprocessing Pool
    #     matches = pool.map(
    #                         partial(
    #                             recovery,
    #                             data_dir=args.data_dir,
    #                             results_dir=args.results_dir
    #                             ),
    #                             sim
    #                             )
        
    # print(f"Recovery: {np.count_nonzero(matches) / N * 100:.2f}%")

# with Pool(args.ncpu) as pool:  # Create a multiprocessing Pool
#     # pool.map(plot_sheet, data_inputs)
#     pool.map(
#         partial(
#             _pool_fun, data_dir=args.data_dir, results_dir=args.results_dir,
#             savedir=False
#                 ),
#             targets
#             )

# filepaths = Path(args.results_dir).glob("**/*.pickle")
# for f in filepaths:
#     with open(f, 'rb') as handle:
#         res = pickle.load(handle)
#     savedir = f.parent
#     filepath = _filepath(res, savedir)
#     if not filepath.is_file():
#         plot_sheet(res, savedir=savedir)

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/injection/results/20230808_biweight/sol_SDE-0158.7_PLA-8-3578_0000000064134387.01_s07-s33-s34.pickle"

    results = _set_data_extended("PLA-8-3578", data_dir="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/injection/data/data1000", results_dir="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/injection/results/20230808_biweight")
    w = int(5 * 50)
    peaks, props = find_peaks(results["SDE_ary"], distance=w, height=5)

    idx_sorted_by_height = peaks[
        np.argsort(props["peak_heights"])[::-1][:5]
    ]

    peaks5 = np.column_stack(
        (results["periods"][idx_sorted_by_height], results["SDE_ary"][idx_sorted_by_height])
    )

    plt.plot(results["periods"], results["SDE_ary"])
    plt.plot(results["periods"][peaks], results["SDE_ary"][peaks], "x")
    for p in peaks5:
        plt.scatter(p[0], p[1], marker="o", color="C4")
    # plt.plot()
    print(results)
    # fig4 = plot_sheet(results)
    plt.show()