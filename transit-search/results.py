#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u

data_file = "/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data_results/tls_0000000000021371.01-s0011_sde=5.pickle"

with open(data_file, 'rb') as handle:
    res = pickle.load(handle)

print(res["duration"])
def plot_periodogram(results, ax):
    ax.axvline(results.period, alpha=0.2, lw=3)
    ax.set_xlim(np.min(results.periods), np.max(results.periods))
    ax.set_ylim(0, results.power.max()*1.1)
    for n in range(2, 4):
        ax.axvline(n*results.period, alpha=0.2, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.2, lw=1, linestyle="dashed")
    ax.set_ylabel("SDE")
    ax.set_xlabel("period (days)")
    ax.plot(results.periods, results.power, color='black', lw=0.5)
    ax.set_xlim(0, max(results.periods))

    return ax

def plot_folded(results, ax):
    ax.plot(
        results.model_folded_phase,
        results.model_folded_model,
        color='C0')

    ax.errorbar(
        results.folded_phase,
        results.folded_y,
        color='k',
        fmt='.',
        alpha=0.5,
        zorder=2)
    ax.set_xlim(0.49, 0.51)
    ax.set_xlabel('phase')
    ax.set_ylabel('relative flux')

    return ax

def plot_model(results, ax):
    ax = plot_data(results, ax)

    ax.plot(results.model_lightcurve_time, results.model_lightcurve_model,
            "C0", lw=2)
    
    return ax

def plot_data(results, ax):

    X, Y = np.concatenate(results.data_x), np.concatenate(results.data_y)

    ax.errorbar(
        X,
        Y,
        color='k',
        fmt='.',
        mew=0,
        alpha=0.5,
        zorder=2)
    
    ax.set_xlim(X.min()-1, X.max()+1)
    ax.set_xlabel('time (TJD)')
    ax.set_ylabel('raw flux')

    return ax

def plot_residuals(results, ax):

    X, Y = np.concatenate(results.data_x), np.concatenate(results.data_y)
    residuals = Y - results.model_lightcurve_model

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

    Rp = results.rp_rs * results.radius.to(u.R_earth)

    info = ("\n"
        "{:<5} = {:.1f}\n".format("SDE", results.SDE)+
        "{:<5} = {:.1f}\n".format("S/N", results.snr)+
        "{:<5} = {:.3f}\n\n".format("FAP", results.FAP)+
#
        "{:<5} = {:.5f}\n".format("P", results.period)+
        "{:<5} = {:.3f}\n".format("r/R", results.rp_rs)+
        "{:<5} = {:.3f}\n".format("dur", results.duration*24)+
        "{:<5} = {:.4f}\n".format("depth", (1-results.depth)*1e3)+
        "{:<5} = {:.1f}\n\n".format("Rp", Rp)+
        #
        "{:<5} = {:.1f}\n".format("Tmag", results.Tmag) +
        "{:<5} = {:.0f}\n".format("Teff", results.teff) +
        "{:<5} = {:.2f}\n".format("Rs", results.radius)
    )

    ax.text(0, 1, f"TIC {results.cid}", multialignment="left", va="top", ha="left", 
            transform=ax.transAxes, fontweight="bold", fontfamily="monospace")
    ax.text(0, 1, info, multialignment="left", va="top", ha="left", 
            transform=ax.transAxes, fontfamily="monospace")

    ax.axis("off")
    return ax
    

def plot_sheet(results, savedir=None):
    X, Y = np.concatenate(results.data_x), np.concatenate(results.data_y)

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
    # ax["a"] = plot_residuals(results, ax["a"])
    ax["t"] = plot_information(results, ax["t"])
    ax["B"] = plot_folded(results, ax["B"])
    ax["C"] = plot_model(results, ax["C"])
    ax["D"] = plot_periodogram(results, ax["D"])

    if savedir is not None:
        savedir += f"/{results.cid}_SDE={int(np.round(results.SDE)):.0f}.pdf"
        fig.savefig(savedir, bbox_inches="tight")


fig4 = plot_sheet(res, savedir="/Users/u2271802/Astronomy/projects/neptunes/tess-neptune-search/transit-search/example_data_sheets")
plt.show()