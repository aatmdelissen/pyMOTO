import os
import platform
import numbers
import sys
from pathlib import Path
import numpy as np
if platform.system() == 'Darwin':  # Avoid "Python is not installed as a framework (Mac OS X)" error
    # Change backend
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pymoto import Module
from .assembly import DomainDefinition


class FigModule(Module):
    """ Abstract base class for any module which produces a figure

    Keyword Args:
        saveto (str): Save images of each iteration to the specified location. (default = ``None``)
        overwrite (bool): Overwrite saved image every time the figure is updated, else prefix ``_0000`` is added to the
          filename (default = ``False``)
        show (bool): Show the figure on the screen
    """
    def __init__(self, *args, saveto=None, overwrite=False, show=True, **kwargs):
        self.fig = plt.figure()
        if saveto is not None:
            self.saveloc, self.saveext = os.path.splitext(saveto)
            dir = os.path.dirname(saveto)
            if not os.path.exists(dir):  # Create dir
                os.makedirs(dir)
        else:
            self.saveloc, self.saveext = None, None
        self.overwrite = overwrite
        self.show = show
        self.iter = 0
        super().__init__(*args, **kwargs)

    def _update_fig(self):
        if self.iter == 0 and self.show:
            plt.show(block=False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.saveloc is not None:
            filen = "{0:s}{1:s}".format(self.saveloc, self.saveext) if self.overwrite else \
                "{0:s}_{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext)
            self.fig.savefig(filen)

        self.iter += 1

    def __del__(self):
        plt.close(self.fig)


class PlotDomain(FigModule):
    """ Plots the densities of a domain (2D or 3D)

    Input Signal:
      - ``x`` (`np.ndarray`): The field to be shown of size ``(domain.nel)``

    Args:
        domain: The domain layout

    Keyword Args:
        saveto (str): Save images of each iteration to the specified location. (default = ``None``)
        overwrite (bool): Overwrite saved image every time the figure is updated, else prefix ``_0000`` is added to the
          filename (default = ``False``)
        show (bool): Show the figure on the screen
        clim: Color limits. In 2D ``[cmin, cmax]``: the values of minimum and maximum color. In 3D ``clipval``: the
          value below which elements are clipped.
        cmap (str): Colormap (only for 2D)
    """
    def _prepare(self, domain: DomainDefinition, clim=None, cmap='gray_r'):
        self.clim = clim
        self.cmap = cmap
        self.domain = domain

    def _response(self, x):
        if self.domain.dim == 2:
            self._plot_2d(x)
        elif self.domain.dim == 3:
            self._plot_3d(x)
        else:
            raise NotImplementedError("Only 2D and 3D plots are implemented")
        assert len(self.fig.axes) > 0, "Figure must contain axes"
        self.fig.axes[0].set_title(f"{self.sig_in[0].tag}, Iteration {self.iter}")

        self._update_fig()

    def _plot_2d(self, x):
        data = x.reshape((self.domain.nelx, self.domain.nely), order='F').T  # TODO Put this reshape inside domain?
        if hasattr(self, 'im'):
            self.im.set_data(data)
        else:
            ax = self.fig.add_subplot(111)
            Lx = self.domain.nelx * self.domain.unitx
            Ly = self.domain.nely * self.domain.unity
            self.im = ax.imshow(data, cmap=self.cmap, origin='lower', extent=(0.0, Lx, 0.0, Ly))
            self.cbar = self.fig.colorbar(self.im, orientation='horizontal')
            ax.set(xlabel='x', ylabel='y')
        vmin, vmax = np.min(data), np.max(data)
        if vmin < 0:
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs
        clim = [vmin, vmax] if self.clim is None else self.clim
        self.im.set_clim(vmin=clim[0], vmax=clim[1])

    def _plot_3d(self, x):
        # prepare some coordinates, and attach rgb values to each
        ei, ej, ek = np.indices((self.domain.nelx, self.domain.nely, self.domain.nelz))
        els = self.domain.get_elemnumber(ei, ej, ek)
        densities = x[els]

        clip_lim = self.clim if self.clim is not None and isinstance(self.clim, numbers.Number) else 0.4

        sel = densities > clip_lim

        # combine the color components
        colors = np.zeros(sel.shape + (3,))
        colors[..., 0] = np.clip(1-densities, 0, 1)
        colors[..., 1] = np.clip(1-densities, 0, 1)
        colors[..., 2] = np.clip(1-densities, 0, 1)

        # and plot everything
        if len(self.fig.axes) == 0:
            # flake8: noqa: F401
            from mpl_toolkits.mplot3d import Axes3D  # This import is needed in order to support 3d plotting
            ax = self.fig.add_subplot(projection='3d')
            max_ext = max(self.domain.nelx, self.domain.nely, self.domain.nelz)
            ax.set(xlabel='x', ylabel='y', zlabel='z',
                   xlim=[(self.domain.nelx-max_ext)/2, (self.domain.nelx+max_ext)/2],
                   ylim=[(self.domain.nely-max_ext)/2, (self.domain.nely+max_ext)/2],
                   zlim=[(self.domain.nelz-max_ext)/2, (self.domain.nelz+max_ext)/2])
        else:
            ax = self.fig.axes[0]

        if hasattr(self, 'fac'):
            for i, f in self.fac.items():
                f.remove()

        self.fac = ax.voxels(sel, facecolors=colors, linewidth=0.5, edgecolors='k')


class PlotGraph(FigModule):
    """ Plot an X-Y graph

    Input Signals:
      - ``x`` (`numpy.ndarray`): X-values
      - ``*args`` (`numpy.ndarray`): Y-values, which must match the dimension of ``x``

    Keyword Args:
        saveto (str): Save images of each iteration to the specified location. (default = ``None``)
        overwrite (bool): Overwrite saved image every time the figure is updated, else prefix ``_0000`` is added to the
          filename (default = ``False``)
        show (bool): Show the figure on the screen
        style (str): Line/marker style (*e.g.* ``"."``)
    """

    def _prepare(self, style: str = None):
        self.style = style

    def _response(self, x, *ys):
        if not hasattr(self, 'ax'):
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel(self.sig_in[0].tag)
            self.ax.set_ylabel(self.sig_in[1].tag)

        if not hasattr(self, 'line'):
            self.line = []
            for i, s in enumerate(self.sig_in[1:]):
                if self.style is None:
                    self.line.append(self.ax.plot([], [], label=s.tag)[0])
                else:
                    self.line.append(self.ax.plot([], [], self.style, label=s.tag)[0])
            self.ax.legend()

        ymin, ymax = np.inf, -np.inf
        for i, y in enumerate(ys):
            self.line[i].set_xdata(x)
            self.line[i].set_ydata(y)
            ymin, ymax = min(ymin, np.min(y)), max(ymax, np.max(y))
        self.ax.set_xlim([np.min(x), np.max(x)])
        dy = ymax - ymin
        self.ax.set_ylim([ymin-0.05*dy, ymax+0.05*dy])

        self._update_fig()


class PlotIter(FigModule):
    """ Plot iteration history of one or more variables

    Input Signals:
      - ``*args`` (`Numeric` or `numpy.ndarray`): Y-values to show for each iteration

    Keyword Args:
        saveto (str): Save images of each iteration to the specified location. (default = ``None``)
        overwrite (bool): Overwrite saved image every time the figure is updated, else prefix ``_0000`` is added to the
          filename (default = ``False``)
        show (bool): Show the figure on the screen
        ylim: Provide y-axis limits for the plot
    """
    def _prepare(self, ylim=None, log_scale=False):
        self.minlim = 1e+200
        self.maxlim = -1e+200
        self.ylim = ylim
        self.log_scale = log_scale

    def _response(self, *args):
        if not hasattr(self, 'ax'):
            self.ax = self.fig.add_subplot(111)
            self.ax.set_yscale('linear' if not self.log_scale else 'log')
            self.ax.set_xlabel("Iteration")

        if not hasattr(self, 'line'):
            self.line = []
            for i, s in enumerate(self.sig_in):
                self.line.append(None)
                self.line[i], = self.ax.plot([], [], '.', label=s.tag)
                self.ax.legend()

        for i, xx in enumerate(args):
            try:
                xadd = xx.reshape(xx.size)
                self.line[i].set_ydata(np.concatenate([self.line[i].get_ydata(), xadd]))
                self.line[i].set_xdata(np.concatenate([self.line[i].get_xdata(), self.iter*np.ones_like(xadd)]))
            except AttributeError:  # In case xx is not numpy, it doesn't have "reshape" nor "size" attributes
                xadd = xx
                self.line[i].set_ydata(np.append(self.line[i].get_ydata(), xadd))
                self.line[i].set_xdata(np.append(self.line[i].get_xdata(), self.iter))

            self.minlim = min(self.minlim, np.min(xadd))
            self.maxlim = max(self.maxlim, np.max(xadd))

        self.ax.set_xlim([-0.5, self.iter+0.5])
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        elif np.isfinite(self.minlim) and np.isfinite(self.maxlim):
            if self.log_scale:
                dy = (np.log10(self.maxlim) - np.log10(self.minlim))*0.05
                ll = 10**(np.log10(self.minlim) - dy)
                ul = 10**(np.log10(self.maxlim) + dy)
            else:
                dy = (self.maxlim - self.minlim)*0.05
                ll = self.minlim - dy
                ul = self.maxlim + dy

            if ll == ul:
                dy = abs(np.nextafter(abs(ll), 1) - abs(ll))
                ll = ll - 1e5*dy
                ul = ul + 1e5*dy
            self.ax.set_ylim([ll, ul])

        self._update_fig()


class WriteToVTI(Module):
    """ Writes vectors to a Paraview VTI file

    See also: :attr:`DomainDefinition.write_to_vti()`

    The size of the vectors should be a multiple of ``nel`` or ``nnodes``. Based on their size they are marked as
    cell-data or point-data in the VTI file. For 2D data (size is equal to ``2*nnodes``), the z-dimension is padded
    with zeros to have 3-dimensional data. Also block-vectors of multiple dimensions (*e.g.* ``(2, 3*nnodes)``) are
    accepted, which get the suffixed as ``_00``.

    Input Signals:
      - ``*args`` (`numpy.ndarray`): Vectors to write to VTI. The signal tags are used as name.

    Args:
        domain: The domain layout
        saveto (str): Location to save the VTI file
        overwrite (bool): Write a new file for each iteration
        scale (float): Scaling factor for the domain
    """
    def _prepare(self, domain: DomainDefinition, saveto: str, overwrite: bool = False, scale=1.):
        self.domain = domain
        self.saveto = saveto
        Path(saveto).parent.mkdir(parents=True, exist_ok=True)
        self.iter = 0
        self.scale = scale
        self.overwrite = overwrite

    def _response(self, *args):
        data = {}
        for s in self.sig_in:
            data[s.tag] = s.state
        pth = os.path.splitext(self.saveto)
        if self.overwrite:
            filen = pth[0] + pth[1]
        else:
            filen = pth[0] + '.{0:04d}'.format(self.iter) + pth[1]
        self.domain.write_to_vti(data, filename=filen, scale=self.scale)
        self.iter += 1


class ScalarToFile(Module):
    """ Writes iteration data to a log file

    This function can also handle small vectors of scalars, i.e. eigenfrequencies or multiple constraints.

    Input Signals:
      - ``*args`` (`Numeric` or `np.ndarray`): Values to write to file. The signal tags are used as name.

    Args:
        saveto: Location to save the log file, supports .txt or .csv
        fmt (optional): Value format (e.g. 'e', 'f', '.3e', '.5g', '.3f')
        separator (optional): Value separator, .csv files will automatically use a comma
    """
    def _prepare(self, saveto: str, fmt: str = '.10e', separator: str = '\t'):
        self.saveto = saveto
        Path(saveto).parent.mkdir(parents=True, exist_ok=True)
        self.iter = 0

        # Test the format
        3.14.__format__(fmt)
        self.format = fmt

        self.separator = "," if ".csv" in self.saveto else separator

    def _response(self, *args):
        tags = [] if self.iter == 0 else None

        # Add iteration as first column
        dat = [self.iter.__format__('d')]
        if tags is not None:
            tags.append('Iteration')

        # Add all signals
        for s in self.sig_in:
            if np.size(np.asarray(s.state)) > 1:
                it = np.nditer(s.state, flags=['multi_index'])
                while not it.finished:
                    dat.append(it.value.__format__(self.format))
                    if tags is not None:
                        tags.append(f"{s.tag}{list(it.multi_index)}")
                    it.iternext()
            else:
                dat.append(s.state.__format__(self.format))
                if tags is not None:
                    tags.append(s.tag)

        # Write to file
        if tags is not None:
            assert len(tags) == len(dat)
            with open(self.saveto, "w+") as f:
                # Write header line
                f.write(self.separator.join(tags))
                f.write("\n")

        with open(self.saveto, "a+") as f:
            # Write data
            f.write(self.separator.join(dat))
            f.write("\n")

        self.iter += 1


class TransientToVTI(Module):
    """ Writes transient response vectors to a Paraview VTI file

    This module utilizes a series file to properly associate the correct time with each timestep:
    https://gitlab.kitware.com/paraview/paraview/blob/v5.5.0/Documentation/release/ParaView-5.5.0.md#json-based-new-meta-file-format-for-series-added

    See also: :attr:`DomainDefinition.write_to_vti()`

    The size of the vectors should be a multiple of ``nel`` or ``nnodes``. Based on their size they are marked as
    cell-data or point-data in the VTI file. For 2D data (size is equal to ``2*nnodes``), the z-dimension is padded
    with zeros to have 3-dimensional data. Also block-vectors of multiple dimensions (*e.g.* ``(2, 3*nnodes)``) are
    accepted, which get the suffixed as ``_00``.

    Input Signals:
      - ``*args`` (`numpy.ndarray`): Vectors to write to VTI. The signal tags are used as name.

    Args:
        domain: The domain layout
        transient_tags(list): Tag name of Signal(s) that vary over time
        saveto (str): Location to save folders with transient responses for specific iterations
        delta_t (float): Length of timestep
        interval (int): Iteration interval for saving data
        scale (float): Scaling factor for the domain
    """

    def _prepare(self, domain: DomainDefinition, transient_tags: list, saveto: str, delta_t: float, interval: int=10, scale=1.):
        self.domain = domain
        self.path = os.path.split(saveto)
        Path(saveto).parent.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.transient_tags = transient_tags
        self.it = 0
        self.interval = interval
        self.dt = delta_t

    def _response(self, *args):
        if self.it % self.interval == 0:
            #prepare folder of transient responses for specific optimization iteration
            saveto = self.path[0] + '/transient_{0:03d}/'.format(self.it) + self.path[1]

            #prepare time series file
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            f = open(saveto + '.series', 'a')
            f.write("{\n")
            f.write("\t\"file-series-version\" : \"1.0\",\n")
            f.write("\t\"files\" : [\n")

            #prepare data from signals
            data = {}
            for s in self.sig_in:
                data[s.tag] = s.state
            pth = os.path.splitext(saveto)
            datavtk = data.copy()

            #loop over time steps, then generate new vtk file and add reference to series file for each time step
            for t in range(data[self.transient_tags[0]].shape[1]):
                for r in self.transient_tags:
                    datavtk[r] = data[r][:, t]
                filename = pth[0] + pth[1] + '.{0:04d}'.format(t)
                self.domain.write_to_vti(datavtk, filename=filename, scale=self.scale)
                f.write('\t\t{{ "name" : "{0}", "time": {1} }},\n'.format(self.path[1]+'.{0:04d}'.format(t) + '.vti', t*self.dt))
            f.write("\t]\n}")

        self.it += 1