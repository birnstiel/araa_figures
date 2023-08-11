import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import seaborn as sns

import dustpy

year = dustpy.constants.year
au = dustpy.constants.au


def read_paletton_text(file='paletton.txt'):
    "reads a text file as produced by palleton.com (textfile output)"

    txt = Path(file).read_text()

    colors = np.array([
        line.split('=')[2].strip()[4:-1].split(',')
        for line in txt.split('\n')
        if not line == '' and not line.startswith(('#', '*'))
    ]).astype(int).reshape(-1, 5, 3)
    return colors


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


def plot_quiver(dustpy_file, trajectories=[10, 30, 100], nr=14, na=10, vmin=1e-5, vmax=1e1, cmap=None, rasterized=True, cols=None, figsize=(8, 5), ax=None):
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

            y0 = np.array([a[0], r0 * au])
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'invalid value encountered in log10')
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

    # Begin Figure

    if cols is None:
        cols = sns.color_palette('Set1')
    cmap = get_transparent_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
    else:
        fig = ax.figure
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(arrow_r[[0, -1]] / au)
    ax.set_ylim(a[[0, -1]])

    # plot dust surface density

    # either as pcolormesh
    # cc = ax.pcolormesh(r / au, a, sig_d.T, norm=LogNorm(vmin, vmax), cmap=cmap, label=r'$\Sigma_{dust}$', rasterized=rasterized)

    # or as contours
    levels = 10.**np.arange(int(np.log10(vmin)), np.ceil(np.log10(vmax)))
    cc = ax.contourf(r / au, a, sig_d.T, levels=levels, norm=LogNorm(vmin, vmax), cmap=cmap)

    # color bar

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.95])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.width, pos.height / 20])
    cb = plt.colorbar(cc, cax=cax, orientation='horizontal')
    cb.set_label(r'$\sigma(r, a)$ [g / cm$^2$]')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')

    # plot the drift and fragmentation limits

    ax.plot(r / au, a_drift, label='drift limit', c=cols[0], lw=2)
    cc = ax.contour(r / au, a, s.dust.St.T - St_frag, [0], colors=[cols[0]], linewidths=2, linestyles='--')
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
                       ])

    # plot the arrows

    ax.quiver(arrow_r / au, arrow_a, scale(arrow_d), np.zeros_like(arrow_d), color=cols[1], **quiver_kwargs)
    ax.quiver(arrow_r / au, arrow_a, np.zeros_like(arrow_g), scale(arrow_g), color=cols[1], **quiver_kwargs)
    Qxy = ax.quiver(arrow_r / au, arrow_a, scale(arrow_d), scale(arrow_g), color='k', **quiver_kwargs)

    # plot the trajectories
    if trajectories is not None:
        for SOL in SOLS:
            # remove too large, or unphysical parts of the solution
            mask = SOL.y[0] <= a[-1]
            mask = mask & (~np.isnan(SOL.y).any(0)) & (~np.isinf(SOL.y).any(0))

            line, = ax.loglog(SOL.y[1][mask] / au, SOL.y[0][mask], '--', c=cols[2])
            for it, _t in enumerate([3, 4, 5, 6]):
                _r = np.interp(10.**_t * year, SOL.t[mask], SOL.y[1][mask])
                _a = np.interp(10.**_t * year, SOL.t[mask], SOL.y[0][mask])
                if np.isclose(_a, SOL.y[0][mask][-1]):
                    continue
                sca = ax.scatter(_r / au, _a, c='k', s=45, zorder=100 + it, fc=line.get_color())
                ax.text(_r / au, _a, f'{_t}', horizontalalignment='center', verticalalignment='center', fontsize='xx-small', color='w', zorder=sca.zorder + 1)
        line.set_label('trajectory')

    # plot the quiver key

    # position of the quiver key
    xo = 0.646
    yo = 0.675

    rect = plt.Rectangle((xo - 0.01, yo), 0.36, 0.32, facecolor="w", alpha=0.85, transform=ax.transAxes, zorder=30, lw=.1, ec='k')
    ax.add_patch(rect)

    qkeys = []
    qkeys += [ax.quiverkey(Qxy, xo + 0.08, yo - 0.025 + 0.05 * 3, scale(1e6), r'$10^6$ years', color='k', **kprops, fontproperties=fprops)]
    qkeys += [ax.quiverkey(Qxy, xo + 0.08, yo - 0.025 + 0.05 * 2, scale(1e4), r'$10^4$ years', color='k', **kprops, fontproperties=fprops)]
    qkeys += [ax.quiverkey(Qxy, xo + 0.08, yo - 0.025 + 0.05 * 1, scale(1e2), r'$10^2$ years', color='k', **kprops, fontproperties=fprops)]
    for qk in qkeys:
        qk.set_zorder(40)
    leg = ax.legend(handlelength=3, loc=(xo-0.01, yo+0.14), fontsize=fprops['size'])
    leg.get_frame().set_alpha(0)
    leg.zorder = 40

    ax.set_xlabel('radius [au]')
    ax.set_ylabel('particle size [cm]')

    ax.text(0.03, 0.99, f't = {num2tex(s.t / year)} yr', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # minor grid only on x-axis
    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', left=False)

    return fig, ax, SOLS


def plot_size_distri(dustpy_files, radii_au=[3, -30, 100], cols=None, figsize=(8, 5), times=None, legend=False, ax=None):
    """plots the size distibutions of a dustpy simulation"""

    if not isinstance(dustpy_files, list):
        dustpy_files = [dustpy_files]

    writer = dustpy.hdf5writer()
    writer.datadir = str(Path(dustpy_files[0]).parent)

    # get the simulation snapshots
    simtimes = writer.read.sequence('t') / year

    # if not specified, we use all times
    # otherwise, we select the snapshots closes to the selected times
    if times is None:
        i_snap = np.arange(len(dustpy_files))
    else:
        i_snap = [np.abs(simtimes - _t).argmin() for _t in times]

    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    else:
        f = ax.figure

    N_snap = len(i_snap)
    lines = []
    labels = []
    
    # the part that is to be done for every file
    for i_plot, i_file in enumerate(i_snap):

        # Read a snapshot
        s = writer.read.output(dustpy_files[i_file])

        # Get the grid and dust surface density $\Sigma_d$

        a = s.dust.a[0, :]
        r = s.grid.r

        A = np.mean(s.grid.m[1:] / s.grid.m[:-1])
        B = 2 * (A - 1) / (A + 1)
        # normalized
        sig_d = s.dust.Sigma / s.dust.Sigma.sum(1)[:, None] / B

        # if a radius is negative, replace it with nearby pressure bump
        # first: find local pressure maxima
        rmax = np.array([s.grid.r[i] / au for i in argrelextrema(s.gas.P, np.greater)])[0]
        for ir, _r in enumerate(radii_au):
            if (_r < 0) & (len(rmax) > 0):
                radii_au[ir] = rmax[np.abs(rmax - np.abs(_r)).argmin()]
        radii_au = [_r for _r in radii_au if _r > 0]

        for i, _r in enumerate(radii_au):
            ir = np.abs(r - _r * au).argmin()

            Sig_i = sig_d[ir, :]
            # n_of_a = 3 * Sig_i / (B * m * a)

            if N_snap < 10:
                lw = 0.5 * (i_plot + 1)
            else:
                lw = 1 + 1.5 * (i_plot == N_snap - 1)

            line, = ax.loglog(a, Sig_i / B, c=f'C{i}',
                               alpha=0.2 + 0.5 * i_plot / (N_snap - 1) + 0.3 * (i_plot == N_snap - 1),
                               lw=lw, zorder=100+i)
            label = f'{_r:.1f} au'

            if i == 0 and legend:
                lines += ax.plot([], [], c='k', alpha=line.get_alpha(), lw=line.get_linewidth())
                labels+= [f'$t = {times[i_plot]/1e6:.2f}$ Myr']

            if i_plot == N_snap - 1:
                lines += [line]
                labels += [label]

    idx = np.argsort([line.zorder for line in lines])
    lines = [lines[i] for i in idx]
    labels = [labels[i] for i in idx]
    
    leg=ax.legend(lines, labels)
    leg.zorder=200

    ax.set_xlim(left=a[0])
    ax.set_ylim(1e-6, 1e+1)
    ax.set_xlabel('particle size [cm]')
    ax.set_ylabel(r'normalized $\sigma(r, a)$ [g / cm$^2$]')

    # add reference slopes
    pos = ax.get_position()
    for slope, y0, angle in zip([0.5, 1.5], [1e-2, 1e-6], [15, 39]):
        angle = np.arctan(slope / ax.get_data_ratio() * pos.height / pos.width * f.get_figheight() / f.get_figwidth()) * 180 / np.pi
        ax.loglog(a, y0 * (a / a[0])**slope, c='0.0', ls='--', zorder=150)
        
        ax.text(10 * a[0], y0 * 10**slope * 0.5, f'$n(a)\propto a^{{{slope-4:.1f}}}$', rotation=angle, color='0.0', ha='center', va='center', zorder=150)

    # minor grid only on x-axis
    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', left=False)

    return ax


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
