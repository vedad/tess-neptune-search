#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation


__all__ = ["spectra", "period_uncertainty"]

def spectra(freq, power, oversampling_factor):

    # calculating SDE with the median and MAD for different bins with interpolation. Has the advantage of accounting for higher scatter at lower frequencies

    # ``power`` is the signal residue (SR) output from cuvarbase.bls

    os = oversampling_factor

    N = 1 + 3.322 * np.log10(len(power))
    N = int(np.round(N))

    # power = power[::-1]

    index_ary = np.arange(len(power))
    index_subary = np.array_split(index_ary, N)

    def _get_fill_values(x0, func, **kwargs):
        first_two = slice(1)
        last_two = slice(-1,N)
        _y1 = np.concatenate([x0[x] for x in index_subary[first_two]])
        _y2 = np.concatenate([x0[x] for x in index_subary[last_two]])

        y1 = func(_y1[::os], **kwargs)
        y2 = func(_y2[::os], **kwargs)

        return (y1, y2)

    # periods = 1/freq[::-1]
    # get power spectrum trend
    # frequency bins and power bins for interpolation
    X = [np.mean(freq[x][::os]) for x in index_subary]
    # X = [np.mean(periods[x][::os]) for x in index_subary]
    Y = [np.median(power[x][::os]) for x in index_subary]

    # interpolate oversampled grid to get the median trend of the power
    f = interp1d(X, Y,
                 kind="cubic",
                 assume_sorted=True, 
                 fill_value=_get_fill_values(power, np.median),
                 bounds_error=False
                 )
    power_trend = f(freq)

    # get power spectrum uncertainty trend
    Y_err = [median_abs_deviation(
                    power[x][::os], scale="normal"
                    )
             for x in index_subary]

    f = interp1d(X, Y_err,
                 kind="cubic",
                 assume_sorted=True,
                 fill_value=_get_fill_values(
                                power, median_abs_deviation, scale="normal"
                                ),
                bounds_error=False)
    power_uncertainty_trend = f(freq)

    # SDE as defined in Kovacs et al. (2002)
    SDE = (power - power_trend) / power_uncertainty_trend

    return SDE, SDE.argmax(), power_trend, power_uncertainty_trend


def period_uncertainty(periods, power):
    # Determine estimate for uncertainty in period
    # Method: Full width at half maximum
    try:
        # Upper limit
        index_highest_power = np.argmax(power)
        idx = index_highest_power
        while True:
            idx += 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_upper = idx
                break
        # Lower limit
        idx = index_highest_power
        while True:
            idx -= 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_lower = idx
                break
        period_fwhm = periods[idx_upper] - periods[idx_lower]
        # TLS version, doesn't quite make sense to me (should be + if taking arithmetic average?)
        # period_uncertainty = 0.5 * (periods[idx_upper] - periods[idx_lower])
        # assume normal distribution of peak to get std
        period_uncertainty = period_fwhm / (2 * np.sqrt(2 * np.log(2)))
    except:
        period_uncertainty = float("inf")
    return period_uncertainty


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import pickle
    from astropy.constants import R_sun, M_sun, G
    from grid import _get_frequency_grid

    filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-1-5473_0000000341553254.01-s07-s08-s10-s34-s35-s36-s37.pickle"

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-1-11941_0000000219324481.01-s27.pickle"

    # filename = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/results/tests/20230726/bls_power_PLA-3-9136_0000000097514673.01-s27.pickle"

    with open(filename, "rb") as handle:
        power = pickle.load(handle)

    A = (
        (2 * np.pi) ** (2.0 / 3)
        / np.pi
        * R_sun.value
        / (G.to("m3 / (kg d2)").value * M_sun.value) ** (1.0 / 3)
        / (27+365*2 * 5)
    )
    # freq = _get_frequency_grid(1/(0.5*27), 1/0.5, A)[:len(power)]
    freq = _get_frequency_grid(1/(0.5*27), 1/0.5, A)
    power = power[:len(freq)]

    SDE, SDE_max, power_trend, uncertainty_trend = spectra(freq, power, 5)
    print(SDE_max)
    # print(power, SDE_raw, SDE)
    fig, ax = plt.subplots(3,1, figsize=(5,10))
    # freq = freq[::-1]
    # periods = 1/freq
    # periods = freq
    # ax[0].plot(freq, power)
    # ax[0].plot(freq, power_trend)
    # ax[1].plot(freq, uncertainty_trend)
    # ax[2].plot(freq, SDE)

    ax[0].plot(freq, power)
    ax[0].plot(freq, power_trend)
    ax[1].plot(freq, uncertainty_trend)
    ax[2].plot(1/freq, SDE)

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