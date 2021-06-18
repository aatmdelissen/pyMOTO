from pymodular import Module
import matplotlib.pyplot as plt
import numpy as np
from .assembly import DomainDefinition
import os


class PlotDomain2D(Module):
    def _prepare(self, domain: DomainDefinition, saveto: str = None):
        self.shape = (domain.nx, domain.ny)
        self.fig, _ = plt.subplots(1, 1)
        if saveto is not None:
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = 0

    def _response(self, x):
        ax = self.fig.axes[0]
        data = x.reshape(self.shape, order='F').T
        if hasattr(self, 'im'):
            self.im.set_data(data)
            self.im.set_clim(vmin=np.min(data), vmax=np.max(data))
        else:
            ax.set_title(self.sig_in[0].tag)
            self.im = ax.imshow(data, origin='lower', cmap='gray_r')
            self.cbar = self.fig.colorbar(self.im, orientation='horizontal')
            plt.show(block=False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.saveloc is not None:
            self.fig.savefig("{0:s}_{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext))

        self.iter += 1


class PlotIter(Module):
    def _prepare(self):
        self.iter = 0
        self.minlim = 1e+200
        self.maxlim = -1e+200

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
            except:
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