import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import solve_ivp
import seaborn as sns

import dustpy

year = dustpy.constants.year
au = dustpy.constants.au


def get_transparent_cmap(cmap=None):
    cmap = cmap or 'Reds'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap = cmap.copy()
    N = cmap.N
    cmap = cmap(np.arange(N))
    cmap[:, -1] = np.linspace(0.0, 1.0, N)
    return ListedColormap(cmap)


def _scale(t, tmin=1e2, tmax=1e6, dxmin=0.5, dxmax=0.1):
    """define a scaling function that scales time scales to arrow length"""
    sign = np.sign(t)
    t = min(max(tmin, np.abs(t)), tmax)
    return sign * (dxmin + (dxmax - dxmin) * (np.log10(t) - np.log10(tmin)) / (np.log10(tmax) - np.log10(tmin)))


scale = np.vectorize(_scale)


def plot_quiver(dustpy_file, trajectories=[10, 30, 100], nr=14, na=10, vmin=1e-5, vmax=1e1, cmap=None, rasterized=True):
    """Plots my review quiver plot based on a dustpy snapshot"""

    # ## Read a snapshot

    writer = dustpy.hdf5writer()
    s = writer.read.output(dustpy_file)

    # Get the grid and dust surface density $\Sigma_d$

    a = s.dust.a[0, :]
    r = s.grid.r

    A = np.mean(s.grid.m[1:] / s.grid.m[:-1])
    B = 2 * (A - 1) / (A + 1)
    sig_d = s.dust.Sigma / B

    # Compute the time scales

    # drift time scale interpolation function
    t_drift = RectBivariateSpline(np.log10(r), np.log10(a), r[:, None] / (s.dust.v.rad - s.gas.v.rad[:, None]))

    # def t_drift(x, y):
    #     return 10.**_t_drift(x, y).T

    # growth time scale interpolation function
    _tg = (1. / (s.dust.eps * s.grid.OmegaK))[None, :] * np.ones_like(s.dust.a.T)
    t_grow = interp2d(np.log10(r), np.log10(a), _tg)

    # compute the size limits $St _ {frag}, a _ {drift}$

    gamma = np.abs(2 * s.gas.eta / (s.gas.Hp / s.grid.r)**2)
    a_drift = 2 * s.dust.Sigma.sum(-1) * (s.grid.OmegaK * s.grid.r / s.gas.cs)**2 / (np.pi * gamma * s.dust.rhos.mean())
    St_frag = (s.dust.v.frag / s.gas.cs)**2 / (3.0 * s.gas.alpha)
    _a_frag_ep = 2.0 * St_frag * s.gas.Sigma / (np.pi * s.dust.rhos.mean())
    a_frag_ep = interp1d(s.grid.r, _a_frag_ep, fill_value=_a_frag_ep[0], bounds_error=False)

    if trajectories is not None:
        #
        # Trajectories
        # Computes trajectories in the given time scale / velocity space
        #

        def dydt(t, y, amin=a[0], amax=10 * a[-1], rmin=r[0], rmax=r[-1]):
            "time derivative of the vector (particle size, radius)"
            a = y[0]
            r = y[1]

            if (a < amin) or (a > amax) or (a > a_frag_ep(r)):
                dadt = a / (1e8 * year)
            else:
                dadt = a / t_grow(np.log10(r), np.log10(a))[0]

            if (r < rmin) or (r > rmax):
                drdt = -r / (1e8 * year)
            else:
                drdt = r / t_drift(np.log10(r), np.log10(a))[0, 0]

            return np.array([dadt, drdt])

        # define a fine grid due to the very different time scales
        t_grid = np.linspace(0, 1.1e6, 4000) * year

        SOLS = []

        for r0 in trajectories:
            # print('integrating at r0 = {} AU'.format(r0))

            y0 = np.array([1e-4, r0 * au])
            res = solve_ivp(dydt, [0, t_grid[-1]], y0, t_eval=t_grid, vectorized=False, method='LSODA')  # 'LSODA')  # 'BDF')
            if not res['success']:
                raise ValueError(res['message'])
            SOLS += [res]

    # define the grid on which the arrows are drawn
    # and compute the growth and drift time scales on that grid

    arrow_r = np.geomspace(r[0], r[-1], nr)
    arrow_a = np.geomspace(a[0], a[-1], na)

    arrow_d = t_drift(np.log10(arrow_r), np.log10(arrow_a)).T / year
    arrow_g = t_grow(np.log10(arrow_r), np.log10(arrow_a)) / year

    #
    # Plot settings
    #
    # quiver plot properties:
    quiver_kwargs = {'units': 'inches',
                     'scale': 2,  # this scales the length of the arrows
                     'width': 0.015,  # this their width & head size
                     'scale_units': 'inches',
                     'alpha': 1}

    # settings for the quiver key:
    kprops = {'labelpos': 'E', 'alpha': 1, 'labelsep': 0.05}
    fprops = {'size': 'small'}

    # position of the quiver key
    xo = 0.0
    yo = 0.0

    # Begin Figure

    cols = sns.color_palette('Set1')
    cmap = get_transparent_cmap(cmap)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(arrow_r[[0, -1]] / au)
    ax.set_ylim(a[[0, -1]])

    # plot dust surface density

    ax.pcolormesh(r / au, a, sig_d.T, norm=LogNorm(vmin, vmax), cmap=cmap, label=r'$\Sigma_{dust}$', rasterized=rasterized)

    # plot the drift and fragmentation limits

    ax.plot(r / au, a_drift, label='drift limit', c=cols[1], lw=3)
    cc = ax.contour(r / au, a, s.dust.St.T - St_frag, [0], colors=[cols[1]], linewidths=3, linestyles='--')
    ax.plot([], [], c=cc.colors[0], ls=cc.linestyles, lw=cc.linewidths, label='fragmentation limit')

    # stokes contours

    st_radius = 150.
    St_levels = 10.**np.arange(-5, 2)
    cc = ax.contour(r / au, a, s.dust.St.T, St_levels, colors='0.5', linestyles='-', linewidths=.5)
    plt.clabel(cc, St_levels,
               fmt=lambda v: f'$\\mathsf{{St}} = 10^{{{np.log10(v):.0f}}}$', fontsize='small',
               manual=[(st_radius, .4e-3),
                       (st_radius, .4e-2),
                       (st_radius, .4e-1),
                       (st_radius, .4e0),
                       (st_radius, .4e1),
                       (st_radius, .4e2),
                       ])

    # plot the arrows

    ax.quiver(arrow_r / au, arrow_a, scale(arrow_d), np.zeros_like(arrow_d), color=cols[8], **quiver_kwargs)
    ax.quiver(arrow_r / au, arrow_a, np.zeros_like(arrow_g), scale(arrow_g), color=cols[8], **quiver_kwargs)
    Qxy = ax.quiver(arrow_r / au, arrow_a, scale(arrow_d), scale(arrow_g), color='k', **quiver_kwargs)

    # plot the trajectories
    if trajectories is not None:
        for SOL in SOLS:
            mask = SOL.y[0] <= a[-1]
            line, = ax.loglog(SOL.y[1][mask] / au, SOL.y[0][mask], '--', c=cols[0])
            for _t in [3, 4, 5, 6]:
                _r = np.interp(10.**_t * year, SOL.t, SOL.y[1])
                _a = np.interp(10.**_t * year, SOL.t, SOL.y[0])
                if np.isclose(_a, SOL.y[0][-1]):
                    continue
                ax.scatter(_r / au, _a, c='k', s=45, zorder=100, fc=line.get_color())
                ax.text(_r / au, _a, f'{_t}', horizontalalignment='center', verticalalignment='center', fontsize='xx-small', color='w', zorder=101)
        line.set_label('trajectory')

    # plot the quiver key

    rect = plt.Rectangle((xo - 0.01, yo), 0.3, 0.45, facecolor="w", alpha=0.75, transform=ax.transAxes, zorder=3, lw=.1, ec='k')
    ax.minorticks_off()
    ax.add_patch(rect)

    qk1 = ax.quiverkey(Qxy, xo + 0.08, yo + 0.05 * 5, scale(1e6), r'$10^6$ years', color='k', **kprops, fontproperties=fprops)
    qk2 = ax.quiverkey(Qxy, xo + 0.08, yo + 0.05 * 4, scale(1e5), r'$10^5$ years', color='k', **kprops, fontproperties=fprops)
    qk3 = ax.quiverkey(Qxy, xo + 0.08, yo + 0.05 * 3, scale(1e4), r'$10^4$ years', color='k', **kprops, fontproperties=fprops)
    qk4 = ax.quiverkey(Qxy, xo + 0.08, yo + 0.05 * 2, scale(1e3), r'$10^3$ years', color='k', **kprops, fontproperties=fprops)
    qk5 = ax.quiverkey(Qxy, xo + 0.08, yo + 0.05 * 1, scale(1e2), r'$10^2$ years', color='k', **kprops, fontproperties=fprops)
    for qk in [qk1, qk2, qk3, qk4, qk5]:
        qk.set_zorder(4)
    ax.legend(handlelength=3, loc=(0.01, 0.27), fontsize=fprops['size']).get_frame().set_alpha(0)

    ax.set_xlabel('radius [au]')
    ax.set_ylabel('particle size [cm]')

    ax.text(0.8, 0.99, f't = {num2tex(s.t / year)} yr', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    return fig, ax


def num2tex(n, x=2, y=2):
    """This function turns a real number into a tex-string numbers >10^x and <10^-x
    are returned as e.g., $1.23 \times 10^{5}$, otherwise as e.g., $1234$.
    Unnecessary digit in the tex-string are removed

    Arguments:
    ----------
    n = number to be converted to tex string

    Keywords:
    ---------
    x : int
    :    threshold exponent

    y : int
    :    number of digits

    Example:
    --------
    >>> num2tex([3e3,3e4,3e5],5,1)
    '$3000.0$, $30000.0$, $3\\times 10^{5}$'

    """
    from numpy import array, log10
    s = None
    for i in array(n, ndmin=1):
        if i == 0:
            t = r'$0$'
        else:
            if log10(i) > x or log10(i) < -x:
                t = ('%2.' + str(y) + 'e') % i
                t = t[0:t.find('e')]
                t = r'$' + t + r' \times 10^{%i}$' % round(log10(i / float(t)))
            else:
                t = ('%' + str(y) + '.' + str(y) + 'f') % i
                t = r'$' + t + '$'
        #
        # some corrections
        #
        if y == 0:
            nz = ''
        else:
            nz = '.' + str(0).zfill(y)
        t = t.replace('1' + nz + ' \times ', '')
        t = t.replace(nz + ' ', '')
        #
        # we don't need 1\times 10^{x}, just 10^{x}
        #
        t = t.replace(r'$1\times', '$')
        #
        # if there are more input numbers, attache them with a comma
        #
        if s is not None:
            s = s + ', ' + t
        else:
            s = t
    return s


def Kanagawa2017_gap_profile(r, a, q, h, alpha0):
    """Function calculates the gap profile according Kanagawa et al. (2017).

    Parameters
    ----------
    r : array
        Radial grid
    a : float
        Semi-majo axis of planet
    q : float
        Planet-star mass ratio
    h : float
        Aspect ratio at planet location
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array
        Pertubation of surface density due to planet"""

    # Unperturbed return value
    ret = np.ones_like(r)

    # Distance to planet (normalized)
    dist = np.abs(r - a) / a

    K = q**2 / (h**5 * alpha0)  # K
    Kp = q**2 / (h**3 * alpha0)  # K prime
    Kp4 = Kp**(0.25)  # Fourth root of K prime
    SigMin = 1. / (1 + 0.04 * K)  # Sigma minimum
    SigGap = 4 / Kp4 * dist - 0.32  # Sigma gap
    dr1 = (0.25 * SigMin + 0.08) * Kp**0.25  # Delta r1
    dr2 = 0.33 * Kp**0.25  # Delta r2

    # Gap edges
    mask1 = np.logical_and(dr1 < dist, dist < dr2)
    ret = np.where(mask1, SigGap, ret)
    # Gap center
    mask2 = dist < dr1
    ret = np.where(mask2, SigMin, ret)

    return ret
