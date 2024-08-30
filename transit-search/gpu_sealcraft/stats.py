#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from astropy.stats import sigma_clip
from scipy.signal import find_peaks, medfilt
from scipy.ndimage import gaussian_filter1d, median_filter


__all__ = ["period_uncertainty", "estimate_trend", "find_top_peaks", "compute_SDE"]

def estimate_trend(freq, power, oversampling_factor):

    # calculating SDE with the median and MAD for different bins with interpolation. Has the advantage of accounting for higher scatter at lower frequencies

    # ``power`` is the signal residue (SR) output from cuvarbase.bls

    os = oversampling_factor

    N = 1 + 3.322 * np.log10(len(power))
    N = int(np.round(N))

    index_ary = np.arange(len(power))
    index_subary = np.array_split(index_ary, N) # splitting by index, not frequency. Maybe this is a problem?
    os = oversampling_factor
    # index_subary = get_bins(freq, power, os)

    def _get_fill_values(x0, func, *args, **kwargs):
        first_two = slice(1)
        last_two = slice(-1,N)
        _y1 = np.concatenate([x0[x] for x in index_subary[first_two]])
        _y2 = np.concatenate([x0[x] for x in index_subary[last_two]])

        y1 = func(_y1[::os], *args, **kwargs)
        y2 = func(_y2[::os], *args, **kwargs)

        return (y1, y2)

    X = [np.mean(freq[x][::os]) for x in index_subary]
    Y = [np.percentile(power[x][::os], 5) for x in index_subary]
    # Y = [np.min(power[x]) for x in index_subary]

    # interpolate oversampled grid to get the median trend of the power
    f = interp1d(X, Y,
                #  kind="cubic",
                 kind="linear",
                 assume_sorted=True, 
                 fill_value=_get_fill_values(power, np.percentile, 5),
                #  fill_value=_get_fill_values(power, np.min),
                 bounds_error=False
                 )

    return f(freq)

def _remove_nearby_peaks(freq, peak_positions, rel_tol=1.02, 
                         rel_tol_alias=1.01,
                         aliases=[2, 1/2, 1/3, 2/3]):
    periods = 1/freq
    p = peak_positions[0] # top peak

    top_period = periods[p]

    # get only primary peak
    tol = (rel_tol - 1) * top_period
    lo, hi = top_period - tol, top_period + tol
    m = ( 
        (periods[peak_positions] > hi) | 
        (periods[peak_positions] < lo)
    )

    # remove aliases from further search
    aliases = 1 / np.array(aliases) # aliases in period space
    for alias in aliases:
        tol = (rel_tol_alias - 1) * top_period * alias
        lo, hi = top_period * alias - tol, top_period * alias + tol
        m &= ( 
            (periods[peak_positions] > hi) | 
            (periods[peak_positions] < lo)
        )
    # print(m)

    return peak_positions[0], peak_positions[m]
    
def _get_top_peaks(freq, peak_positions, npeaks=5):
    peaks = []
    n = 0
    remaining_peaks = peak_positions.copy()
    while n <= npeaks:
        top_peak, remaining_peaks = _remove_nearby_peaks(
            freq, remaining_peaks
        )
        peaks.append(top_peak)
        n += 1

    return peaks

def find_top_peaks(freq, power, height=0, distance=None, npeaks=5):
    peaks_idx, props = find_peaks(power, distance=distance, height=height)

    peak_positions_sorted_by_height = peaks_idx[
                np.argsort(props["peak_heights"])[::-1]
            ]
    peaks = _get_top_peaks(freq, peak_positions_sorted_by_height, npeaks=npeaks)
    return freq[peaks], power[peaks], peaks

def compute_SDE(power, oversampling_factor, peaks=None):
    os = oversampling_factor

    median = np.median(power[::os])
    mad = median_abs_deviation(
                    power[::os], scale="normal"
                    )
    if peaks is not None:
        return (peaks - median) / mad
    return (power - median) / mad

# def spectra(freq, power, oversampling_factor):

#     # calculating SDE with the median and MAD for different bins with interpolation. Has the advantage of accounting for higher scatter at lower frequencies

#     # ``power`` is the signal residue (SR) output from cuvarbase.bls

#     os = oversampling_factor

#     N = 1 + 3.322 * np.log10(len(power))
#     N = int(np.round(N))

#     # power = power[::-1]

#     index_ary = np.arange(len(power))
#     index_subary = np.array_split(index_ary, N) # splitting by index, not frequency. Maybe this is a problem?

#     def _get_fill_values(x0, func, *args, **kwargs):
#         first_two = slice(1)
#         last_two = slice(-1,N)
#         _y1 = np.concatenate([x0[x] for x in index_subary[first_two]])
#         _y2 = np.concatenate([x0[x] for x in index_subary[last_two]])

#         y1 = func(_y1[::os], *args, **kwargs)
#         y2 = func(_y2[::os], *args, **kwargs)

#         return (y1, y2)

#     # periods = 1/freq[::-1]
#     # get power spectrum trend
#     # frequency bins and power bins for interpolation
#     # X = [np.mean(freq[x][::os]) for x in index_subary]
#     X = [int(np.mean(x[::os])) for x in index_subary]
#     # X = [np.mean(periods[x][::os]) for x in index_subary]
#     Y = [np.median(power[x][::os]) for x in index_subary]
#     Y_lo = [np.percentile(power[x][::os], 5) for x in index_subary]

#     # interpolate oversampled grid to get the median trend of the power
#     f = interp1d(freq[X], Y_lo,
#                 #  kind="cubic",
#                  kind="linear",
#                  assume_sorted=True, 
#                  fill_value=_get_fill_values(power, np.percentile, 5),
#                  bounds_error=False
#                  )
#     power_trend = f(freq)

#     # get power spectrum uncertainty trend
#     Y_err = [median_abs_deviation(
#                     power[x][::os], scale="normal"
#                     )
#              for x in index_subary]
    
#     power_median_in_bins = np.concatenate(
#         [
#         np.ones_like(x)*y for x,y in zip(index_subary, Y)
#         ]
#     )
#     power_uncertainty_in_bins = np.concatenate(
#         [
#         np.ones_like(x)*y for x,y in zip(index_subary, Y_err)
#         ]
#     )

#     f = interp1d(freq[X], Y_err,
#                 #  kind="cubic",
#                  kind="linear",
#                  assume_sorted=True,
#                  fill_value=_get_fill_values(
#                                 power, median_abs_deviation, scale="normal"
#                                 ),
#                 bounds_error=False)
#     power_uncertainty_trend = f(freq)

#     # SDE as defined in Kovacs et al. (2002)
#     SDE = (power - power_trend) / power_uncertainty_trend

#     SDE_naive = power - power_trend
#     SDE_naive = (SDE_naive - power_median_in_bins) / power_uncertainty_in_bins
#     # SDE_naive = (power - power_median_in_bins) / power_uncertainty_in_bins
#     # SDE_naive -= power_trend

#     return SDE, SDE[np.argmax(SDE)], power_trend, power_uncertainty_trend, SDE_naive, (X, Y)


def period_uncertainty(periods, power, index_peak):

    # Determine estimate for uncertainty in period
    # Method: Full width at half maximum
    try:
        # Upper limit
        # index_highest_power = np.argmax(power)
        idx = index_peak
        while True:
            idx += 1
            if power[idx] <= 0.5 * power[index_peak]:
                idx_upper = idx
                break
        # Lower limit
        # idx = index_highest_power
        while True:
            idx -= 1
            if power[idx] <= 0.5 * power[index_peak]:
                idx_lower = idx
                break
        period_fwhm = periods[idx_upper] - periods[idx_lower]
        # TLS version, doesn't quite make sense to me (should be + if taking arithmetic average?)
        # period_uncertainty = 0.5 * (periods[idx_upper] - periods[idx_lower])
        # assume normal distribution of peak to get std
        period_uncertainty = np.abs( period_fwhm / (2 * np.sqrt(2 * np.log(2))) ) # absolute value because period will be in reverse order so uncertainty is negative
    except:
        period_uncertainty = float("inf")
    return period_uncertainty


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import pickle
    from astropy.constants import R_sun, M_sun, G
    # from grid import _get_frequency_grid
    # from grid import frequency_grid

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-1-5473_0000000341553254.01-s07-s08-s10-s34-s35-s36-s37.pickle"

    filename = "/Users/u2271802/Downloads/sol_SDE-0098.2_PLA-7-10992_0000000171544091.01_s12-s39.pickle"

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-1-11941_0000000219324481.01-s27.pickle"

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-3-9136_0000000097514673.01-s27.pickle"

    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    
    freq = 1/data["periods"]
    power = data["power"]
    # A = (
    #     (2 * np.pi) ** (2.0 / 3)
    #     / np.pi
    #     * R_sun.value
    #     / (G.to("m3 / (kg d2)").value * M_sun.value) ** (1.0 / 3)
    #     / (27+365*2 * 5)
    # )
    # # freq = _get_frequency_grid(1/(0.5*27), 1/0.5, A)[:len(power)]
    # freq = _get_frequency_grid(1/(0.5*27), 1/0.5, A)
    # power = power[:len(freq)]

    power_trend = estimate_trend(freq, power, 5)

    top_peaks = find_top_peaks(freq, power, npeaks=5)
    print(top_peaks)
    print(1/top_peaks[0])
    top_freq = top_peaks[0][0]

    SDE2 = compute_SDE(power-power_trend, 5)

    # print(what)

    SDE, SDE_max, _, uncertainty_trend, SDE_naive, (X, Y) = spectra(freq, power, 5)
    # print(SDE_max)
    # print(power, SDE_raw, SDE)
    fig, ax = plt.subplots(3,1, figsize=(5,11))
    # freq = freq[::-1]
    # periods = 1/freq
    # periods = freq
    # ax[0].plot(freq, power)
    # ax[0].plot(freq, power_trend)
    # ax[1].plot(freq, uncertainty_trend)
    # ax[2].plot(freq, SDE)

    ax[0].plot(freq/top_freq, power)
    ax[0].plot(freq/top_freq, power_trend)
    # ax[0].plot(freq[X], Y, 'o')
    # ax[0].plot(freq[idx_sorted_by_height], power[idx_sorted_by_height], c="C3", marker='.', ls="none")
    ax[0].plot(top_peaks[0]/top_freq, top_peaks[1], c="k", marker='o', ls="none")
    ax[0].axvline(1/(0.7965984123373484*0.98)/top_freq)
    ax[0].axvline(1/(0.7965984123373484*1.02)/top_freq)
    ax[0].axvline(2/0.7965984123373484/top_freq)
    ax[0].axvline(1/(2*0.7965984123373484)/top_freq)
    # ax[0].axvline(1/(2.791128698397543*0.98))
    # ax[0].axvline(1/(2.791128698397543*1.02))
    # ax[1].plot(freq[M], (power-power_trend)[M])
    # ax[2].plot(freq, power_clipped)
    # ax[2].plot(power_median_filtered[0], power_median_filtered[1])
    ax[1].plot(freq, power-power_trend)
    # ax[2].axhline(med_sd2)
    # ax[2].plot(freq, power_median_filtered)

    # ax[2].plot(power_gaussian_filtered[0], power_gaussian_filtered[1])
    # ax[2].plot(freq, SDE)
    ax[2].plot(freq, SDE2)

    # ax[4].plot(freq, power_uncertainty)
    # ax[4].plot(1/freq, SDE)

    plt.show()



# def spectra4(SR, oversampling_factor):
#     # using SDE definition with median instead of mean, and correcting the trend of SDE after calculating it for the whole sample
#     N = 1 + 3.322 * np.log10(len(SR))
#     N = int(np.round(N))

#     indices = np.arange(len(SR))

#     index_chunks = np.array_split(indices, N)
 


#     SDE_raw = (SR - np.median(SR[::oversampling_factor])) / (1.4826 * median_abs_deviation(SR[::oversampling_factor]))

#     X = [np.mean(indices[x]) for x in index_chunks]
#     # X = [np.mean(indices[x]) for x in index_chunks]
#     pre_chunks = np.concatenate([SDE_raw[x] for x in index_chunks[slice(2)]])
#     post_chunks = np.concatenate([SDE_raw[x] for x in index_chunks[slice(-2,N)]])
#     fill_pre = np.median(pre_chunks)
#     fill_post = np.median(post_chunks)
#     Y = [np.median(SDE_raw[x]) for x in index_chunks]
#     # SR_chunks = np.array_split(SR, N)
#     # trend_chunks = [np.median(x) for x in SR_chunks]

#     f = interp1d(X, Y, kind="cubic", assume_sorted=True, fill_value=(fill_pre, fill_post), bounds_error=False)

#     xnew = indices
#     SR_trend = f(xnew)


#     SDE = SDE_raw - SR_trend #(SR - SR_trend) / SR_scatter


#     return SDE_raw, SDE, SR_trend, X, Y#, SR_scatter