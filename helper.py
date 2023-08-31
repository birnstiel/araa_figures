import sys
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline

output_dir = (Path(__file__).parent.resolve() / 'output').absolute()
output_dir.mkdir(exist_ok=True)

data_dir = (Path(__file__).parent.resolve() / 'data').absolute()

sqdeg_per_sr = 4 * np.pi**2 / 360.0**2

araa_style = {
    'image.cmap': 'rocket',
    'text.usetex': False,
    'figure.dpi': 72.0,
    'font.cursive': ['Myriad Pro'],
    'font.family': ['sans-serif'],
    'font.sans-serif': ['Myriad Pro'],
    'font.size': 10.0,
    'font.stretch': 'normal',
    'font.style': 'normal',
    'font.variant': 'normal',
    'font.weight': 'normal',
    'axes.labelweight': 'bold'
}

label_font_dict = {'fontstyle': 'italic', 'fontweight': 'bold'}


class image():
    "simple image class that reads a FITS file and provides the data and x/y grids for plotting"

    def __init__(self, fname, clip=None):
        h = fits.getheader(fname)

        d = np.squeeze(fits.getdata(fname)).T
        d[np.isnan(d)] = 0.0
        d = d / d.max()

        naxis = d.ndim

        dx = h['CDELT1'] * 3600
        dy = h['CDELT2'] * 3600

        x = dx * (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1)
        y = dy * (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1)

        if naxis == 3:
            nz = h['NAXIS3']
            dz = h['CDELT3']
            z = h['CRVAL3'] + np.arange(nz) * dz

        if clip is not None:
            xmask = np.abs(x) < clip
            ymask = np.abs(y) < clip
            xymask = xmask[:, None] & ymask[None, :]
            x = x[xmask]
            y = y[ymask]
            if naxis == 2:
                d = d[xymask].reshape(len(x), len(y))
            elif naxis == 3:
                d = d[xymask].reshape(len(x), len(y), len(z))
            else:
                raise ValueError('data needs to be 2D or 3D')

        self.xi = np.hstack((x - 0.5 * dx, x[-1] + 0.5 * dx))
        self.yi = np.hstack((y - 0.5 * dy, y[-1] + 0.5 * dy))

        self.A_pix_sr = -h['CDELT1'] * h['CDELT2'] / sqdeg_per_sr

        self.x = x
        self.y = y
        self.h = h
        self.data = d

        if naxis == 3:
            self.z = z
            self.zi = np.hstack((z - 0.5 * dz, z[-1] + 0.5 * dz))


def plot_ring(ax, r, z, inc, PA, nphi=50, **kwargs):
    """Plots a circular ring with the given orientation on a 2d axes object.

    Parameters
    ----------
    ax : matplotlib axes
        axes into which to draw
    r : float
        radius of the ring
    z : float
        elevation of the ring above mid-plane
    inc : float
        inclination of the ring ot of the mid-plane
    PA : float
        position axis
    nphi : int, optional
        how many points to use for drawing the ring, by default 50
    """
    phi = np.linspace(0, 2 * np.pi, nphi)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = z * np.ones_like(x)

    # unit vectors

    ex = [1, 0, 0]
    ey = [0, 1, 0]
    ez = [0, 0, 1]

    points = np.array([x, y, z])

    # compute the rotation

    rotINC = R.from_rotvec([inc, 0, 0])

    ex2 = rotINC.apply(ex)
    ey2 = rotINC.apply(ey)
    ez2 = rotINC.apply(ez)

    rotPA = R.from_rotvec(PA * ez2)

    ex3 = rotPA.apply(ex2)
    ey3 = rotPA.apply(ey2)
    # ez3 = rotPA.apply(ez2)

    xo = points.T.dot(ex3)
    yo = points.T.dot(ey3)
    # zo = points.T.dot(ez3)

    ax.plot(xo, yo, **kwargs)


def apply_araa_style(ax):
    """Applies some of the styling for the ARA&A figures.

    Parameters
    ----------
    ax : matplotlib axes
        which axes to update
    ndec : int, optional
        how many digits to show on the ticks, by default 0
    """
    from matplotlib.ticker import StrMethodFormatter
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    for axis in [ax.xaxis, ax.yaxis]:
        axis.label.set_fontweight('bold')
        axis.label.set_fontstyle('italic')


class Capturing(list):
    """Context manager capturing standard output of whatever is called in it.

    Keywords
    --------
    stderr : bool
        if True will capture the standard error instead of standard output.
        defaulats to False

    Examples
    --------
    >>> with Capturing() as output:
    >>>     do_something(my_object)

    `output` is now a list containing the lines printed by the function call.

    This can also be concatenated

    >>> with Capturing() as output:
    >>>    print('hello world')
    >>> print('displays on screen')
    displays on screen

    >>> with output:
    >>>     print('hello world2')
    >>> print('output:', output)
    output: ['hello world', 'hello world2']

    >>> import warnings
    >>> with output, Capturing(stderr=True) as err:
    >>>     print('hello world2')
    >>>     warnings.warn('testwarning')
    >>> print(output)
    output: ['hello world', 'hello world2']

    >>> print('error:', err[0].split(':')[-1])
    error:  testwarning

    Mostly copied from [this stackoverflow answer](http://stackoverflow.com/a/16571630/2108771)

    """

    def __init__(self, /, *args, stderr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._error = stderr

    def __enter__(self):
        """Start capturing output when entering the context"""
        if self._error:
            self._std = sys.stderr
            sys.stderr = self._stringio = StringIO()
        else:
            self._std = sys.stdout
            sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """Get & return the collected output when exiting context"""
        self.extend(self._stringio.getvalue().splitlines())
        if self._error:
            sys.stderr = self._std
        else:
            sys.stdout = self._std


def plot_time_path(x, y, time, snaps=[], label=None, spline=True, xlog=False, ylog=False, tlog=False, eps=0.05, ax=None, scatter_kw=None, k=3, **kwargs):
    """plot an evolutionary track along 2D and marks points in time

    Parameters
    ----------
    x : array
        x-track
    y : array
        y-track
    time : time
        time axis along x and y

    snaps : list, optional
        the snapshots at which to mark points along th path, by default []
    label : str, optional
        label for that track, by default None
    spline : bool, optional
        if true, then spline-interpolate the given points to smooth the track, by default True
    tlog : bool, optional
        if time interpolation is done on a log or linear resolution, by default False
    eps : float, optional
        by how much to extrapolate the arrow forward and back in time
        - change sign to flip the arror
        - set to zero for no arrow
        - give a 2-element array for different fractional parts backward/forward
    ax : matplotlib.Axes
        axes into which to plot, by default None which will create axes
    xlog : bool, optional
        if True, interpolation is done on log x values, by default False
    ylog : bool, optional
        if True, interpolation is done on log y values, by default False
    k : int, optional
        order of the spline, by default 3
    scatter_kw : dict
        keywords for the scatter plot making the circles

    Other Keywords are passed to the `ax.plot` command.

    Example:
    >>> x = np.geomspace(0.1, 2, 15)
    >>> y = np.sin(x)
    >>> t = x**2
    >>> f, ax = plt.subplots()
    >>> helper.plot_time_path(x, y, t, snaps=[0.1, 1, 2, 3], spline=False, ax=ax)
    >>> helper.plot_time_path(x, y, t, snaps=[0.1, 1, 2, 3], tlog=True, spline=True, ax=ax)
    """
    if ax is None:
        _, ax = plt.subplots()

    # create the splines
    k = k * int(spline)

    # define the log or linear conversions used in the interpolation
    def ident(a):
        return a

    x_conv = ident
    x_inv = ident
    y_conv = ident
    y_inv = ident
    t_conv = ident

    def log_forward(a):
        return np.log10(a)

    def log_backward(a):
        return 10.**a

    if tlog:
        t_conv = log_forward

    if xlog:
        x_conv = log_forward
        x_inv = log_backward

    if ylog:
        y_conv = log_forward
        y_inv = log_backward

    # define the spline interpolation    

    _spl_x = make_interp_spline(t_conv(time), x_conv(x), k=k)
    _spl_y = make_interp_spline(t_conv(time), y_conv(y), k=k)

    def spl_x(t):
        return x_inv(_spl_x(t_conv(t), extrapolate=True))

    def spl_y(t):
        return y_inv(_spl_y(t_conv(t), extrapolate=True))

    # get a smoother interpolation between the points
    if spline:
        if tlog:
            time_smooth = np.geomspace(time[0], time[-1], 50)
        else:
            time_smooth = np.linspace(time[0], time[-1], 50)

        x_smooth = spl_x(time_smooth)
        y_smooth = spl_y(time_smooth)
    else:
        time_smooth = time
        x_smooth = x
        y_smooth = y

    line, = ax.plot(x_smooth, y_smooth, label=label, **kwargs)

    eps = eps * np.ones(2)
    if not np.all(eps == 0.0):
        # add an arrow head slightly extrapolated

        _x0 = spl_x(time[-1] * (1 - eps[0]))
        _y0 = spl_y(time[-1] * (1 - eps[0]))
        _x1 = spl_x(time[-1] * (1 + eps[1]))
        _y1 = spl_y(time[-1] * (1 + eps[1]))

        # opt = dict(color=line.get_color(), arrowstyle=f'simple,head_width=.75,head_length=.75,lw={line.get_linewidth()}', connectionstyle='arc3,rad=0')
        lw = line.get_linewidth()
        opt = dict(color=line.get_color(),
                   linestyle=line.get_linestyle(),
                   alpha=line.get_alpha(),
                   linewidth=lw,
                   arrowstyle=f'-|>,head_width={0.3*lw},head_length={0.6*lw}')
        ax.annotate('', xy=(_x1, _y1), xycoords='data', xytext=(_x0, _y0), textcoords='data', arrowprops=opt, size=5, annotation_clip=False)

    # add the time circles
    if len(snaps) > 0:

        for it, _t in enumerate(snaps):
            if _t > time_smooth[-1]:
                continue
            _x = spl_x(_t)
            _y = spl_y(_t)

            # set properties of the circles
            if scatter_kw is None:
                scatter_kw = {}
            scatter_kw['s'] = scatter_kw.get('s', 100)
            scatter_kw['fc'] = scatter_kw.get('fc', line.get_color())
            scatter_kw['zorder'] = scatter_kw.get('zorder', 110)

            sca = ax.scatter(_x, _y, **scatter_kw)
            ax.text(_x, _y, f'{_t:.1g}', horizontalalignment='center', verticalalignment='center', fontsize='xx-small', color='w', zorder=sca.zorder + 1)
