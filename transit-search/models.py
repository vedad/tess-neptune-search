#!/usr/bin/env python

from numpy import arccos, sqrt, pi, clip, select, finfo, cos
from numba import vectorize

# import jax
# import jax.numpy as jnp

# from tinygp import kernels, GaussianProcess

# jax.config.update("jax_enable_x64", True)


# def build_gp(params):
#     kernel = kernels.quasisep.Matern32(
#                       scale=jnp.exp(params["log_scale"],
#                       sigma=jnp.exp(params["log_sigma"]))
#                       )
#     return GaussianProcess(
#         kernel,
#         params["x"],
#         diag=params["yerr"]**2 + jnp.exp(params["log_jitter"]),
#         # mean=params["mean"],
#     )


# @jax.jit
# def loss(params):
#     gp = build_gp(params)
#     return -gp.log_probability(params["y"])

class qpower2:

    def __call__(self, *args):
        return qpower2._qpower2(*args)

    @staticmethod
    @vectorize(nopython=True)
    def esolve(M, ecc):
        """
        Taken from https://github.com/pmaxted/pycheops/blob/master/pycheops/funcs.py

        Solve Kepler's equation M = E - ecc.sin(E) 

        :param M: mean anomaly (scalar or array)
        :param ecc: eccentricity (scalar or array)

        :returns: eccentric anomaly, E

        Algorithm is from Markley 1995, CeMDA, 63, 101 via pyAstronomy class
        keplerOrbit.py

        :Example:

        Test precision using random values::
        
        >>> from pycheops.funcs import esolve
        >>> from numpy import pi, sin, abs, max
        >>> from numpy.random import uniform
        >>> ecc = uniform(0,1,1000)
        >>> M = uniform(-2*pi,4*pi,1000)
        >>> E = esolve(M, ecc)
        >>> maxerr = max(abs(E - ecc*sin(E) - (M % (2*pi)) ))
        >>> print("Maximum error = {:0.2e}".format(maxerr))
        Maximum error = 8.88e-16

        """
        M = M % (2*pi)
        if ecc == 0:
            return M
        if M > pi:
            M = 2*pi - M
            flip = True
        else:
            flip = False
        alpha = (3*pi + 1.6*(pi-abs(M))/(1+ecc) )/(pi - 6/pi)
        d = 3*(1 - ecc) + alpha*ecc
        r = 3*alpha*d * (d-1+ecc)*M + M**3
        q = 2*alpha*d*(1-ecc) - M**2
        w = (abs(r) + sqrt(q**3 + r**2))**(2/3)
        E = (2*r*w/(w**2 + w*q + q**2) + M) / d
        f_0 = E - ecc*sin(E) - M
        f_1 = 1 - ecc*cos(E)
        f_2 = ecc*sin(E)
        f_3 = 1-f_1
        d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
        d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3**2)*f_3/6)
        E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4**2*f_3/6 - d_4**3*f_2/24)
        if flip:
            E =  2*pi - E
        return E

    @staticmethod
    def t2z(t, tzero, P, sini, rstar, ecc=0, omdeg=90, returnMask=False):
        """
        Taken from https://github.com/pmaxted/pycheops/blob/master/pycheops/funcs.py

        Calculate star-planet separation relative to scaled stellar radius, z

        Optionally, return a flag/mask to indicate cases where the planet is
        further from the observer than the star, i.e., whether phases with z<1 are
        transits (mask==True) or eclipses (mask==False)

        :param t: time of observation (scalar or array)
        :param tzero: time of inferior conjunction, i.e., mid-transit
        :param P: orbital period
        :param sini: sine of orbital inclination
        :param rstar: scaled stellar radius, R_star/a
        :param ecc: eccentricity (optional, default=0)
        :param omdeg: longitude of periastron in degrees (optional, default=90)
        :param returnFlag: return a flag to distinguish transits from eclipses.

        N.B. omdeg is the longitude of periastron for the star's orbit

        :returns: z [, mask]

        :Example:
        
        >>> from pycheops.funcs import t2z
        >>> from numpy import linspace
        >>> import matplotlib.pyplot as plt
        >>> t = linspace(0,1,1000)
        >>> sini = 0.999
        >>> rstar = 0.1
        >>> plt.plot(t, t2z(t,0,1,sini,rstar))
        >>> plt.xlim(0,1)
        >>> plt.ylim(0,12)
        >>> ecc = 0.1
        >>> for omdeg in (0, 90, 180, 270):
        >>>     plt.plot(t, t2z(t,0,1,sini,rstar,ecc,omdeg))
        >>> plt.show()
            
        """
        from numpy import pi, cos, arctan
        if ecc == 0:
            nu = 2*pi*(t-tzero)/P
            omrad = 0.5*pi
            z = sqrt(1 - cos(nu)**2*sini**2)/rstar
        else:
            tp = tzero2tperi(tzero,P,sini,ecc,omdeg,return_nan_on_error=True)
            if tp is nan:
                if returnMask:
                    return full_like(t,nan),full_like(t,True,dtype=bool)
                else:
                    return full_like(t,nan)
            M = 2*pi*(t-tp)/P
            E = esolve(M,ecc)
            nu = 2*arctan(sqrt((1+ecc)/(1-ecc))*tan(E/2))
            omrad = pi*omdeg/180
            # Equation (5.63) from Hilditch
            z = (((1-ecc**2)/
                (1+ecc*cos(nu))*sqrt(1-sin(omrad+nu)**2*sini**2))/rstar)
        if returnMask:
            return z, sin(nu + omrad)*sini < 0
        else:
            return z
        
    @staticmethod
    def tzero2tperi(tzero,P,sini,ecc,omdeg,
            return_nan_on_error=False):
        """
        Calculate time of periastron from time of mid-transit

        Uses the method by Lacy, 1992AJ....104.2213L

        :param tzero: times of mid-transit
        :param P: orbital period
        :param sini: sine of orbital inclination 
        :param ecc: eccentricity 
        :param omdeg: longitude of periastron in degrees

        :returns: time of periastron prior to tzero

        :Example:
        >>> from pycheops.funcs import tzero2tperi
        >>> tzero = 54321.6789
        >>> P = 1.23456
        >>> sini = 0.987
        >>> ecc = 0.654
        >>> omdeg = 89.01
        >>> print("{:0.4f}".format(tzero2tperi(tzero,P,sini,ecc,omdeg)))
        54321.6784

        """
        def _delta(th, sin2i, omrad, ecc):
            # Equation (4.9) from Hilditch
            return (1-ecc**2)*(
                    sqrt(1-sin2i*sin(th+omrad)**2)/(1+ecc*cos(th)))

        omrad = omdeg*pi/180
        sin2i = sini**2
        theta = 0.5*pi-omrad
        if (1-sin2i) > finfo(0.).eps :
            ta = theta-0.125*pi
            tb = theta
            tc = theta+0.125*pi
            fa = _delta(ta, sin2i, omrad, ecc)
            fb = _delta(tb, sin2i, omrad, ecc)
            fc = _delta(tc, sin2i, omrad, ecc)
            if ((fb>fa)|(fb>fc)):
                t_ = linspace(0,2*pi,1024)
                d_ = _delta(t_, sin2i, omrad, ecc)
                try:
                    i_= argrelextrema(d_, less)[0]
                    t_ = t_[i_]
                    if len(t_)>1:
                        i_ = (abs(t_ - tb)).argmin()
                        t_ = t_[i_]
                    ta,tb,tc = (t_-0.01, t_, t_+0.01)
                except:
                    if return_nan_on_error: return nan
                    print(sin2i, omrad, ecc)
                    print(ta, tb, tc)
                    print(fa, fb, fc)
                    raise ValueError('tzero2tperi grid search fail')
            try:
                theta = brent(_delta, args=(sin2i, omrad, ecc), brack=(ta, tb, tc))
            except ValueError:
                if return_nan_on_error: return nan
                print(sin2i, omrad, ecc)
                print(ta, tb, tc)
                print(fa, fb, fc)
                raise ValueError('Not a bracketing interval.')

        if theta == pi:
            E = pi 
        else:
            E = 2*arctan(sqrt((1-ecc)/(1+ecc))*tan(theta/2))
        return tzero - (E - ecc*sin(E))*P/(2*pi)

    @staticmethod
    def _qpower2(z,p,c,alpha):
        r"""
        Fast and accurate transit light curves for the power-2 limb-darkening law

        The power-2 limb-darkening law is

        .. math::
            I(\mu) = 1 - c (1 - \mu^\alpha)

        Light curves are calculated using the qpower2 approximation [2]_. The
        approximation is accurate to better than 100ppm for radius ratio k < 0.1.

        **N.B.** qpower2 is untested/inaccurate for values of k > 0.2

        .. [2] Maxted, P.F.L. & Gill, S., 2019A&A...622A..33M 

        :param z: star-planet separation on the sky cf. star radius (array)
        :param k: planet-star radius ratio (scalar, k<1) 
        :param c: power-2 limb darkening coefficient
        :param a: power-2 limb darkening exponent

        :returns: light curve (observed flux)  

        :Example:

        >>> from pycheops.models import qpower2
        >>> from pycheops.funcs import t2z
        >>> from numpy import linspace
        >>> import matplotlib.pyplot as plt
        >>> t = linspace(-0.025,0.025,1000)
        >>> sini = 0.999
        >>> rstar = 0.05
        >>> ecc = 0.2
        >>> om = 120
        >>> tzero = 0.0
        >>> P = 0.1
        >>> z=t2z(t,tzero,P,sini,rstar,ecc,om)
        >>> c = 0.5
        >>> a = 0.7
        >>> k = 0.1
        >>> f = qpower2(z,k,c,a)
        >>> plt.plot(t,f)
        >>> plt.show()

        """
        # From Maxted & Gill 2019, A&A, 622, A33
        # from numpy import arccos, sqrt, pi, clip, select, finfo
        I_0 = (alpha+2)/(pi*(alpha-c*alpha+2))
        g = 0.5*alpha
        def q1(z,p,c,alpha):
            zt = clip(abs(z), 0,1-p)
            s = 1-zt**2
            c0 = (1-c+c*s**g)
            c2 = 0.5*alpha*c*s**(g-2)*((alpha-1)*zt**2-1)
            return 1-I_0*pi*p**2*(c0 + 0.25*p**2*c2 - 0.125*alpha*c*p**2*s**(g-1))
        def q2(z,p,c,alpha):
            zt = clip(abs(z), 1-p,1+p)
            d = clip((zt**2 - p**2 + 1)/(2*zt),0,1)
            ra = 0.5*(zt-p+d)
            rb = 0.5*(1+d)
            sa = clip(1-ra**2,finfo(0.0).eps,1)
            sb = clip(1-rb**2,finfo(0.0).eps,1)
            q = clip((zt-d)/p,-1,1)
            w2 = p**2-(d-zt)**2
            w = sqrt(clip(w2,finfo(0.0).eps,1))
            b0 = 1 - c + c*sa**g
            b1 = -alpha*c*ra*sa**(g-1)
            b2 = 0.5*alpha*c*sa**(g-2)*((alpha-1)*ra**2-1)
            a0 = b0 + b1*(zt-ra) + b2*(zt-ra)**2
            a1 = b1+2*b2*(zt-ra)
            aq = arccos(q)
            J1 = ( (a0*(d-zt)-(2/3)*a1*w2 + 0.25*b2*(d-zt)*(2*(d-zt)**2-p**2))*w
                    + (a0*p**2 + 0.25*b2*p**4)*aq )
            J2 = alpha*c*sa**(g-1)*p**4*(0.125*aq +
                    (1/12)*q*(q**2-2.5)*sqrt(clip(1-q**2,0,1)) )
            d0 = 1 - c + c*sb**g
            d1 = -alpha*c*rb*sb**(g-1)
            K1 = ((d0-rb*d1)*arccos(d) +
                    ((rb*d+(2/3)*(1-d**2))*d1 - d*d0)*sqrt(clip(1-d**2,0,1)) )
            K2 = (1/3)*c*alpha*sb**(g+0.5)*(1-d)
            return 1 - I_0*(J1 - J2 + K1 - K2)
        return select( [z <= (1-p), abs(z-1) < p],
            [q1(z, p, c, alpha), q2(z, p, c, alpha)], default=1)



if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0.4, 0.6, 500)
    z = qpower2.t2z(t, 0.5, 2, 1, 0.1)
    model = qpower2()(z, 0.05, 0.5, 0.1)
    # model = qpower2(z, 0.05, 0.5, 0.1)

    plt.plot(t, model)
    plt.show()