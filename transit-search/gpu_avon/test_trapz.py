#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt



def phase(t, freq, phi0=0.):
    phi = (t * freq - phi0)
    phi -= np.floor(phi)

    return phi


def transit_model(phi, y0=0.0, delta=1., q=0.01, qin=0.5, phi0=0.5):

    q_ingress = q * qin
    q_total = q - 2*q_ingress
    t1 = -0.5*q
    t2 = -0.5*q + q_ingress
    t3 = 0.5*q - q_ingress
    t4 = 0.5*q

    y = y0 * np.ones_like(phi)
    
    transit = (phi > -0.5*q_total) & (phi < 0.5*q_total)
    transit_ingress = (phi > t1) & (phi < t2)
    transit_egress = (phi > t3) & (phi < t4)
    slope_ingress = -delta / q_ingress
    slope_egress = delta / q_ingress
    intercept = slope_ingress * t1
    # y = a * x + b
    # a = slope
    # x = phi
    # b = a *x - 0
    # transit = phi < q_total
    y[transit] -= delta
    y[transit_ingress] = slope_ingress*phi[transit_ingress] - intercept
    y[transit_egress] = slope_egress*phi[transit_egress] - intercept
    return y

def plot_bls_model(ax, phase, model, **kwargs):

    ax.plot(phase, model, '.', **kwargs)

t = np.linspace(0, 3, 500)
freq = 1/1.5

phi = phase(t, freq, phi0=0.)
phi[phi > 0.5] -= 1
model = transit_model(phi, y0=0, delta=0.5, q=0.5, qin=0.25, phi0=0.5)

fig, ax = plt.subplots()

ax = plot_bls_model(ax, phi, model)


plt.show()
