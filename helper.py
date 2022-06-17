import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.spatial.transform import Rotation as R

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

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
        d = d / d.max()

        dx = h['CDELT1'] * 3600
        dy = h['CDELT2'] * 3600

        x = dx * (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1)
        y = dy * (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1)

        if clip is not None:
            xmask = np.abs(x) < clip
            ymask = np.abs(y) < clip
            xymask = xmask[:, None] & ymask[None, :]
            x = x[xmask]
            y = y[ymask]
            d = d[xymask].reshape(len(x), len(y))

        self.xi = np.hstack((x - 0.5 * dx, x[-1] + 0.5 * dx))
        self.yi = np.hstack((y - 0.5 * dy, y[-1] + 0.5 * dy))

        self.A_pix_sr = -h['CDELT1'] * h['CDELT2'] / sqdeg_per_sr

        self.x = x
        self.y = y
        self.h = h
        self.data = d


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


def apply_araa_style(ax, ndec=0):
    """Applies some of the styling for the ARA&A figures.

    Parameters
    ----------
    ax : matplotlib axes
        which axes to update
    ndec : int, optional
        how many digits to show on the ticks, by default 0
    """
    from matplotlib.ticker import StrMethodFormatter
    ax.xaxis.set_major_formatter(StrMethodFormatter(f'{{x:,.{ndec:d}f}}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter(f'{{x:,.{ndec:d}f}}'))

    for axis in [ax.xaxis, ax.yaxis]:
        axis.label.set_fontweight('bold')
        axis.label.set_fontstyle('italic')
