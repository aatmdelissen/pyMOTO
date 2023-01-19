import os
import platform
from pathlib import Path
import numpy as np
if platform.system() == 'Darwin':  # Avoid "Python is not installed as a framework (Mac OS X)" error
    # Change backend
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pymoto import Module
from .assembly import DomainDefinition


class PlotDomain(Module):
    """ Plots the densities of a domain (2D or 3D) """
    def _prepare(self, domain: DomainDefinition, saveto: str = None, clim=None, cmap='gray_r'):
        self.clim = clim
        self.cmap = cmap
        self.domain = domain
        self.fig = plt.figure()
        if saveto is not None:
            self.saveloc, self.saveext = os.path.splitext(saveto)
            dir = os.path.dirname(saveto)
            if not os.path.exists(dir):
                os.makedirs(dir)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = 0

    def _response(self, x):
        if self.domain.dim == 2:
            self.plot_2d(x)
        elif self.domain.dim == 3:
            self.plot_3d(x)
        else:
            raise NotImplementedError("Only 2D and 3D plots are implemented")
        assert len(self.fig.axes) > 0, "Figure must contain axes"
        self.fig.axes[0].set_title(f"{self.sig_in[0].tag}, Iteration {self.iter}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.saveloc is not None:
            self.fig.savefig("{0:s}_{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext))

        self.iter += 1

    def plot_2d(self, x):
        data = x.reshape((self.domain.nelx, self.domain.nely), order='F').T  # TODO Put this reshape inside domain?
        if hasattr(self, 'im'):
            self.im.set_data(data)
        else:
            ax = self.fig.add_subplot(111)
            self.im = ax.imshow(data, origin='lower', cmap=self.cmap)
            self.cbar = self.fig.colorbar(self.im, orientation='horizontal')
            ax.set(xlabel='x', ylabel='y')
            plt.show(block=False)
        clim = [np.min(data), np.max(data)] if self.clim is None else self.clim
        self.im.set_clim(vmin=clim[0], vmax=clim[1])

    def plot_3d(self, x):
        # prepare some coordinates, and attach rgb values to each
        ei, ej, ek = np.indices((self.domain.nelx, self.domain.nely, self.domain.nelz))
        els = self.domain.get_elemnumber(ei, ej, ek)
        densities = x[els]

        sel = densities > 0.4

        # combine the color components
        colors = np.zeros(sel.shape + (3,))
        colors[..., 0] = np.clip(1-densities, 0, 1)
        colors[..., 1] = np.clip(1-densities, 0, 1)
        colors[..., 2] = np.clip(1-densities, 0, 1)

        # and plot everything
        if len(self.fig.axes) == 0:
            from mpl_toolkits.mplot3d import Axes3D  # TODO can this be removed?
            ax = self.fig.add_subplot(projection='3d')
            max_ext = max(self.domain.nelx, self.domain.nely, self.domain.nelz)
            ax.set(xlabel='x', ylabel='y', zlabel='z',
                   xlim=[(self.domain.nelx-max_ext)/2, (self.domain.nelx+max_ext)/2],
                   ylim=[(self.domain.nely-max_ext)/2, (self.domain.nely+max_ext)/2],
                   zlim=[(self.domain.nelz-max_ext)/2, (self.domain.nelz+max_ext)/2])
            plt.show(block=False)
        else:
            ax = self.fig.axes[0]

        if hasattr(self, 'fac'):
            for i, f in self.fac.items():
                f.remove()

        self.fac = ax.voxels(sel,
                             facecolors=colors,
                             edgecolors='k',  # np.clip(2*colors - 0.5, 0, 1),  # brighter
                             linewidth=0.5)


class PlotGraph(Module):
    """ Plot a X-Y graph """
    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def _response(self, x, *ys):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax.set_xlabel(self.sig_in[0].tag)
            self.ax.set_ylabel(self.sig_in[1].tag)

        if not hasattr(self, 'line'):
            self.line = []
            for i, s in enumerate(self.sig_in[1:]):
                self.line.append(plt.plot([], [], label=s.tag)[0])
            self.ax.legend()
            plt.show(block=False)

        ymin, ymax = np.inf, -np.inf
        for i, y in enumerate(ys):
            self.line[i].set_xdata(x)
            self.line[i].set_ydata(y)
            ymin, ymax = min(ymin, min(y)), max(ymax, max(y))
        self.ax.set_xlim([min(x), max(x)])
        self.ax.set_ylim([ymin, ymax])


class PlotIter(Module):
    """ Plot iteration history of one or more variables """
    def _prepare(self):
        self.iter = 0
        self.minlim = 1e+200
        self.maxlim = -1e+200

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    def _response(self, *args):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, 1)

        if not hasattr(self, 'line'):
            self.line = []
            for i, s in enumerate(self.sig_in):
                self.line.append(None)
                self.line[i], = plt.plot([], [], '.', label=s.tag)

                self.ax.set_yscale('linear')
                self.ax.set_xlabel("Iteration")
                self.ax.legend()
                plt.show(block=False)

        for i, xx in enumerate(args):
            try:
                xadd = xx.reshape(xx.size)
                self.line[i].set_ydata(np.concatenate([self.line[i].get_ydata(), xadd]))
                self.line[i].set_xdata(np.concatenate([self.line[i].get_xdata(), self.iter*np.ones_like(xadd)]))
            except:  # TODO what is the exception?
                xadd = xx
                self.line[i].set_ydata(np.append(self.line[i].get_ydata(), xadd))
                self.line[i].set_xdata(np.append(self.line[i].get_xdata(), self.iter))

            self.minlim = min(self.minlim, np.min(xadd))
            self.maxlim = max(self.maxlim, np.max(xadd))

        # dy = max((self.maxlim - self.minlim)/10, 1e-5 * self.maxlim)

        self.ax.set_xlim([-0.5, self.iter+0.5])
        if np.isfinite(self.minlim) and np.isfinite(self.maxlim):
            self.ax.set_ylim([self.minlim*0.95, self.maxlim*1.05])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # self.fig.savefig(self.filen)

        self.iter += 1

        return []


class WriteToParaview(Module):
    """ Writes vectors to a Paraview VTI file"""
    def _prepare(self, domain: DomainDefinition, saveto: str, scale=1.0):
        self.domain = domain
        self.saveto = saveto
        Path(saveto).parent.mkdir(parents=True, exist_ok=True)
        self.iter = 0
        self.scale = scale

    def _response(self, *args):
        data = {}
        for s in self.sig_in:
            data[s.tag] = s.state
        pth = os.path.splitext(self.saveto)
        filen = pth[0] + '.{0:04d}'.format(self.iter) + pth[1]
        self.domain.write_to_vti(data, filename=filen, scale=self.scale)
        self.iter += 1
