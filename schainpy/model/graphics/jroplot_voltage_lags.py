
import os
import time
import math
import datetime
import numpy
from schainpy.model.proc.jroproc_base import ProcessingUnit, Operation, MPDecorator  #YONG

from .jroplot_spectra import RTIPlot, NoisePlot

from schainpy.utils import log
from .plotting_codes import *

from schainpy.model.graphics.jroplot_base import Plot, plt

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter


class RTIDPPlot(RTIPlot):
    '''
    Written by R. Flores
    '''
    '''Plot for RTI Double Pulse Experiment Using Cross Products Analysis
    '''

    CODE = 'RTIDP'
    colormap = 'jro'
    plot_name = 'RTI'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = 3
        self.nplots = self.nrows
        #self.height=10
        if self.showSNR:
            self.nrows += 1
            self.nplots += 1

        self.ylabel = 'Height [km]'
        self.xlabel = 'Time (LT)'

        self.cb_label = 'Intensity (dB)'

        self.titles = ['{} Channel {}'.format(
            self.plot_name.upper(), '0x1'),'{} Channel {}'.format(
                self.plot_name.upper(), '0'),'{} Channel {}'.format(
                    self.plot_name.upper(), '1')]

    def update(self, dataOut):

        data = {}
        meta = {}
        data[self.CODE] = dataOut.data_for_RTI_DP
        data['NRANGE'] = dataOut.NDP

        return data, meta

    def plot(self):

        self.x = self.data.times
        self.y = self.data.yrange[0: self.data['NRANGE']]
        self.z = self.data[self.CODE]
        self.z = numpy.ma.masked_invalid(self.z)

        if self.decimation is None:
            x, y, z = self.fill_gaps(self.x, self.y, self.z)
        else:
            x, y, z = self.fill_gaps(*self.decimate())

        for n, ax in enumerate(self.axes):

            self.zmax = self.zmax if self.zmax is not None else numpy.max(
                self.z[1][0,12:40])
            self.zmin = self.zmin if self.zmin is not None else numpy.min(
                self.z[1][0,12:40])

            if ax.firsttime:

                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]

                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )
            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )


class RTILPPlot(RTIPlot):

    '''
    Written by R. Flores
    '''
    '''
       Plot for RTI Long Pulse Using Cross Products Analysis
    '''

    CODE = 'RTILP'
    colormap = 'jro'
    plot_name = 'RTI LP'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = 2
        self.nplots = self.nrows
        if self.showSNR:
            self.nrows += 1
            self.nplots += 1

        self.ylabel = 'Height [km]'
        self.xlabel = 'Time (LT)'

        self.cb_label = 'Intensity (dB)'

        self.titles = ['{} Channel {}'.format(
            self.plot_name.upper(), '0'),'{} Channel {}'.format(
                self.plot_name.upper(), '1'),'{} Channel {}'.format(
                    self.plot_name.upper(), '2'),'{} Channel {}'.format(
                        self.plot_name.upper(), '3')]


    def update(self, dataOut):

        data = {}
        meta = {}
        data['rti'] = dataOut.data_for_RTI_LP
        data['NRANGE'] = dataOut.NRANGE

        return data, meta
    def plot(self):

        NRANGE = self.data['NRANGE'][-1]
        self.x = self.data.times
        self.y = self.data.yrange[0:NRANGE]

        self.z = self.data['rti']

        self.z = numpy.ma.masked_invalid(self.z)

        if self.decimation is None:
            x, y, z = self.fill_gaps(self.x, self.y, self.z)
        else:
            x, y, z = self.fill_gaps(*self.decimate())

        for n, ax in enumerate(self.axes):

            self.zmax = self.zmax if self.zmax is not None else numpy.max(
                self.z[1][0,12:40])
            self.zmin = self.zmin if self.zmin is not None else numpy.min(
                self.z[1][0,12:40])

            if ax.firsttime:

                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]


                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )
                #plt.tight_layout()
            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )
                #plt.tight_layout()


class DenRTIPlot(RTIPlot):
    '''
    Written by R. Flores
    '''
    '''
       RTI Plot for Electron Densities
    '''

    CODE = 'denrti'
    colormap = 'jet'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = self.data.shape(self.CODE)[0]
        self.nplots = self.nrows

        self.ylabel = 'Range [km]'
        self.xlabel = 'Time (LT)'

        self.plots_adjust.update({'wspace': 0.8, 'hspace':0.2, 'left': 0.2, 'right': 0.9, 'bottom': 0.18})

        if self.CODE == 'denrti':
            self.cb_label = r'$\mathrm{N_e}$ Electron Density ($\mathrm{1/cm^3}$)'

        self.titles = ['Electron Density RTI']

    def update(self, dataOut):

        data = {}
        meta = {}

        data['denrti'] = dataOut.DensityFinal*1.e-6 #To Plot in cm^-3

        return data, meta

    def plot(self):

        self.x = self.data.times
        self.y = self.data.yrange

        self.z = self.data[self.CODE]

        self.z = numpy.ma.masked_invalid(self.z)

        if self.decimation is None:
            x, y, z = self.fill_gaps(self.x, self.y, self.z)
        else:
            x, y, z = self.fill_gaps(*self.decimate())

        for n, ax in enumerate(self.axes):

            self.zmax = self.zmax if self.zmax is not None else numpy.max(
                self.z[n])
            self.zmin = self.zmin if self.zmin is not None else numpy.min(
                self.z[n])

            if ax.firsttime:

                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                if numpy.log10(self.zmin)<0:
                    self.zmin=1
                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       #vmin=self.zmin,
                                       #vmax=self.zmax,
                                       cmap=self.cmaps[n],
                                       norm=colors.LogNorm(vmin=self.zmin,vmax=self.zmax)
                                       )

            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       #vmin=self.zmin,
                                       #vmax=self.zmax,
                                       cmap=self.cmaps[n],
                                       norm=colors.LogNorm(vmin=self.zmin,vmax=self.zmax)
                                       )


class ETempRTIPlot(RTIPlot):
    '''
    Written by R. Flores
    '''
    '''
       Plot for Electron Temperature
    '''

    CODE = 'ETemp'
    colormap = 'jet'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = self.data.shape(self.CODE)[0]
        self.nplots = self.nrows

        self.ylabel = 'Range [km]'
        self.xlabel = 'Time (LT)'
        self.plots_adjust.update({'wspace': 0.8, 'hspace':0.2, 'left': 0.2, 'right': 0.9, 'bottom': 0.18})
        if self.CODE == 'ETemp':
            self.cb_label = 'Electron Temperature (K)'
            self.titles = ['Electron Temperature RTI']
        if self.CODE == 'ITemp':
            self.cb_label = 'Ion Temperature (K)'
            self.titles = ['Ion Temperature RTI']
        if self.CODE == 'HeFracLP':
            self.cb_label ='He+ Fraction'
            self.titles = ['He+ Fraction RTI']
            self.zmax=0.16
        if self.CODE == 'HFracLP':
            self.cb_label ='H+ Fraction'
            self.titles = ['H+ Fraction RTI']

    def update(self, dataOut):

        data = {}
        meta = {}

        data['ETemp'] = dataOut.ElecTempFinal

        return data, meta

    def plot(self):

        self.x = self.data.times
        self.y = self.data.yrange
        self.z = self.data[self.CODE]

        self.z = numpy.ma.masked_invalid(self.z)

        if self.decimation is None:
            x, y, z = self.fill_gaps(self.x, self.y, self.z)
        else:
            x, y, z = self.fill_gaps(*self.decimate())

        for n, ax in enumerate(self.axes):

            self.zmax = self.zmax if self.zmax is not None else numpy.max(
                self.z[n])
            self.zmin = self.zmin if self.zmin is not None else numpy.min(
                self.z[n])

            if ax.firsttime:

                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]

                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )
                #plt.tight_layout()

            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T * self.factors[n],
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=self.cmaps[n]
                                       )


class ITempRTIPlot(ETempRTIPlot):
    '''
    Written by R. Flores
    '''
    '''
       Plot for Ion Temperature
    '''

    CODE = 'ITemp'
    colormap = 'jet'
    plot_name = 'Ion Temperature'

    def update(self, dataOut):

        data = {}
        meta = {}

        data['ITemp'] = dataOut.IonTempFinal

        return data, meta


class HFracRTIPlot(ETempRTIPlot):
    '''
    Written by R. Flores
    '''
    '''
       Plot for H+ LP
    '''

    CODE = 'HFracLP'
    colormap = 'jet'
    plot_name = 'H+ Frac'

    def update(self, dataOut):

        data = {}
        meta = {}
        data['HFracLP'] = dataOut.PhyFinal

        return data, meta


class HeFracRTIPlot(ETempRTIPlot):
    '''
    Written by R. Flores
    '''
    '''
       Plot for He+ LP
    '''

    CODE = 'HeFracLP'
    colormap = 'jet'
    plot_name = 'He+ Frac'

    def update(self, dataOut):

        data = {}
        meta = {}
        data['HeFracLP'] = dataOut.PheFinal

        return data, meta


class TempsDPPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for Electron - Ion Temperatures
    '''

    CODE = 'tempsDP'
    #plot_name = 'Temperatures'
    plot_type = 'scatterbuffer'

    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Temperature (K)'
        self.titles = ['Electron/Ion Temperatures']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['Te'] = dataOut.te2
        data['Ti'] = dataOut.ti2
        data['Te_error'] = dataOut.ete2
        data['Ti_error'] = dataOut.eti2

        meta['yrange'] = dataOut.heightList[0:dataOut.NSHTS]

        return data, meta

    def plot(self):

        y = self.data.yrange

        self.xmin = -100
        self.xmax = 5000

        ax = self.axes[0]

        data = self.data[-1]

        Te = data['Te']
        Ti = data['Ti']
        errTe = data['Te_error']
        errTi = data['Ti_error']

        if ax.firsttime:
            ax.errorbar(Te, y, xerr=errTe, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='Te')
            ax.errorbar(Ti, y, fmt='k^', xerr=errTi,elinewidth=1.0,color='k',linewidth=2.0, label='Ti')
            plt.legend(loc='lower right')
            self.ystep_given = 50
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

        else:
            self.clear_figures()
            ax.errorbar(Te, y, xerr=errTe, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='Te')
            ax.errorbar(Ti, y, fmt='k^', xerr=errTi,elinewidth=1.0,color='k',linewidth=2.0, label='Ti')
            plt.legend(loc='lower right')
            ax.yaxis.set_minor_locator(MultipleLocator(15))


class TempsHPPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for Temperatures Hybrid Experiment
    '''

    CODE = 'temps_LP'
    #plot_name = 'Temperatures'
    plot_type = 'scatterbuffer'


    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Temperature (K)'
        self.titles = ['Electron/Ion Temperatures']
        self.width = 3.5
        self.height = 6.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}


        data['Te'] = numpy.concatenate((dataOut.te2[:dataOut.cut],dataOut.te[dataOut.cut:]))
        data['Ti'] = numpy.concatenate((dataOut.ti2[:dataOut.cut],dataOut.ti[dataOut.cut:]))
        data['Te_error'] = numpy.concatenate((dataOut.ete2[:dataOut.cut],dataOut.ete[dataOut.cut:]))
        data['Ti_error'] = numpy.concatenate((dataOut.eti2[:dataOut.cut],dataOut.eti[dataOut.cut:]))

        meta['yrange'] = dataOut.heightList[0:dataOut.NACF]

        return data, meta

    def plot(self):


        self.y = self.data.yrange
        self.xmin = -100
        self.xmax = 4500
        ax = self.axes[0]

        data = self.data[-1]

        Te = data['Te']
        Ti = data['Ti']
        errTe = data['Te_error']
        errTi = data['Ti_error']

        if ax.firsttime:

            ax.errorbar(Te, self.y, xerr=errTe, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='Te')
            ax.errorbar(Ti, self.y, fmt='k^', xerr=errTi,elinewidth=1.0,color='k',linewidth=2.0, label='Ti')
            plt.legend(loc='lower right')
            self.ystep_given = 200
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

        else:
            self.clear_figures()
            ax.errorbar(Te, self.y, xerr=errTe, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='Te')
            ax.errorbar(Ti, self.y, fmt='k^', xerr=errTi,elinewidth=1.0,color='k',linewidth=2.0, label='Ti')
            plt.legend(loc='lower right')
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')


class FracsHPPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for Composition LP
    '''

    CODE = 'fracs_LP'
    plot_type = 'scatterbuffer'


    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Frac'
        self.titles = ['Composition']
        self.width = 3.5
        self.height = 6.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        #aux_nan=numpy.zeros(dataOut.cut,'float32')
        #aux_nan[:]=numpy.nan
        #data['ph'] = numpy.concatenate((aux_nan,dataOut.ph[dataOut.cut:]))
        #data['eph'] = numpy.concatenate((aux_nan,dataOut.eph[dataOut.cut:]))

        data['ph'] = dataOut.ph[dataOut.cut:]
        data['eph'] = dataOut.eph[dataOut.cut:]
        data['phe'] = dataOut.phe[dataOut.cut:]
        data['ephe'] = dataOut.ephe[dataOut.cut:]

        data['cut'] = dataOut.cut

        meta['yrange'] = dataOut.heightList[0:dataOut.NACF]


        return data, meta

    def plot(self):

        data = self.data[-1]

        ph = data['ph']
        eph = data['eph']
        phe = data['phe']
        ephe = data['ephe']
        cut = data['cut']
        self.y = self.data.yrange

        self.xmin = 0
        self.xmax = 1
        ax = self.axes[0]

        if ax.firsttime:

            ax.errorbar(ph, self.y[cut:], xerr=eph, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='H+')
            ax.errorbar(phe, self.y[cut:], fmt='k^', xerr=ephe,elinewidth=1.0,color='k',linewidth=2.0, label='He+')
            plt.legend(loc='lower right')
            self.xstep_given = 0.2
            self.ystep_given = 200
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

        else:
            self.clear_figures()
            ax.errorbar(ph, self.y[cut:], xerr=eph, fmt='r^',elinewidth=1.0,color='r',linewidth=2.0, label='H+')
            ax.errorbar(phe, self.y[cut:], fmt='k^', xerr=ephe,elinewidth=1.0,color='k',linewidth=2.0, label='He+')
            plt.legend(loc='lower right')
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

class EDensityPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for electron density
    '''

    CODE = 'den'
    #plot_name = 'Electron Density'
    plot_type = 'scatterbuffer'

    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = r'$\mathrm{N_e}$ Electron Density ($\mathrm{1/cm^3}$)'
        self.titles = ['Electron Density']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['den_power'] = dataOut.ph2[:dataOut.NSHTS]
        data['den_Faraday'] = dataOut.dphi[:dataOut.NSHTS]
        data['den_error'] = dataOut.sdp2[:dataOut.NSHTS]
        #data['err_Faraday'] = dataOut.sdn1[:dataOut.NSHTS]
        #print(numpy.shape(data['den_power']))
        #print(numpy.shape(data['den_Faraday']))
        #print(numpy.shape(data['den_error']))

        data['NSHTS'] = dataOut.NSHTS

        meta['yrange'] = dataOut.heightList[0:dataOut.NSHTS]

        return data, meta

    def plot(self):

        y = self.data.yrange

        #self.xmin = 1e3
        #self.xmax = 1e7

        ax = self.axes[0]

        data = self.data[-1]

        DenPow = data['den_power']
        DenFar = data['den_Faraday']
        errDenPow = data['den_error']
        #errFaraday = data['err_Faraday']

        NSHTS = data['NSHTS']

        if self.CODE == 'denLP':
            DenPowLP = data['den_LP']
            errDenPowLP = data['den_LP_error']
            cut = data['cut']

        if ax.firsttime:
            self.autoxticks=False
            #ax.errorbar(DenFar, y[:NSHTS], xerr=1, fmt='h-',elinewidth=1.0,color='g',linewidth=1.0, label='Faraday Profile',markersize=2)
            ax.errorbar(DenFar, y[:NSHTS], xerr=1, fmt='h-',elinewidth=1.0,color='g',linewidth=1.0, label='Faraday',markersize=2,linestyle='-')
            #ax.errorbar(DenPow, y[:NSHTS], fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='b',linewidth=1.0, label='Power Profile',markersize=2)
            ax.errorbar(DenPow, y[:NSHTS], fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='k',linewidth=1.0, label='Power',markersize=2,linestyle='-')

            if self.CODE=='denLP':
                ax.errorbar(DenPowLP[cut:], y[cut:], xerr=errDenPowLP[cut:], fmt='r^-',elinewidth=1.0,color='r',linewidth=1.0, label='LP Profile',markersize=2)

            plt.legend(loc='upper left',fontsize=8.5)
            #plt.legend(loc='lower left',fontsize=8.5)
            ax.set_xscale("log")#, nonposx='clip')
            grid_y_ticks=numpy.arange(numpy.nanmin(y),numpy.nanmax(y),50)
            self.ystep_given=100
            if self.CODE=='denLP':
                self.ystep_given=200
            ax.set_yticks(grid_y_ticks,minor=True)
            locmaj = LogLocator(base=10,numticks=12)
            ax.xaxis.set_major_locator(locmaj)
            locmin = LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(which='minor')

        else:
            dataBefore = self.data[-2]
            DenPowBefore = dataBefore['den_power']
            self.clear_figures()
            #ax.errorbar(DenFar, y[:NSHTS], xerr=1, fmt='h-',elinewidth=1.0,color='g',linewidth=1.0, label='Faraday Profile',markersize=2)
            ax.errorbar(DenFar, y[:NSHTS], xerr=1, fmt='h-',elinewidth=1.0,color='g',linewidth=1.0, label='Faraday',markersize=2,linestyle='-')
            #ax.errorbar(DenPow, y[:NSHTS], fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='b',linewidth=1.0, label='Power Profile',markersize=2)
            ax.errorbar(DenPow, y[:NSHTS], fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='k',linewidth=1.0, label='Power',markersize=2,linestyle='-')
            ax.errorbar(DenPowBefore, y[:NSHTS], elinewidth=1.0,color='r',linewidth=0.5,linestyle="dashed")

            if self.CODE=='denLP':
                ax.errorbar(DenPowLP[cut:], y[cut:], fmt='r^-', xerr=errDenPowLP[cut:],elinewidth=1.0,color='r',linewidth=1.0, label='LP Profile',markersize=2)

            ax.set_xscale("log")#, nonposx='clip')
            grid_y_ticks=numpy.arange(numpy.nanmin(y),numpy.nanmax(y),50)
            ax.set_yticks(grid_y_ticks,minor=True)
            locmaj = LogLocator(base=10,numticks=12)
            ax.xaxis.set_major_locator(locmaj)
            locmin = LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(which='minor')
            plt.legend(loc='upper left',fontsize=8.5)
            #plt.legend(loc='lower left',fontsize=8.5)

class RelativeDenPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for electron density
    '''

    CODE = 'den'
    #plot_name = 'Electron Density'
    plot_type = 'scatterbuffer'

    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = r'$\mathrm{N_e}$ Relative Electron Density ($\mathrm{1/cm^3}$)'
        self.titles = ['Electron Density']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['den_power'] = dataOut.ph2
        data['den_error'] = dataOut.sdp2

        meta['yrange'] = dataOut.heightList

        return data, meta

    def plot(self):

        y = self.data.yrange

        ax = self.axes[0]

        data = self.data[-1]

        DenPow = data['den_power']
        errDenPow = data['den_error']

        if ax.firsttime:
            self.autoxticks=False
            ax.errorbar(DenPow, y, fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='k',linewidth=1.0, label='Power',markersize=2,linestyle='-')

            plt.legend(loc='upper left',fontsize=8.5)
            #plt.legend(loc='lower left',fontsize=8.5)
            ax.set_xscale("log")#, nonposx='clip')
            grid_y_ticks=numpy.arange(numpy.nanmin(y),numpy.nanmax(y),50)
            self.ystep_given=100
            ax.set_yticks(grid_y_ticks,minor=True)
            locmaj = LogLocator(base=10,numticks=12)
            ax.xaxis.set_major_locator(locmaj)
            locmin = LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(which='minor')

        else:
            dataBefore = self.data[-2]
            DenPowBefore = dataBefore['den_power']
            self.clear_figures()
            ax.errorbar(DenPow, y, fmt='k^-', xerr=errDenPow,elinewidth=1.0,color='k',linewidth=1.0, label='Power',markersize=2,linestyle='-')
            ax.errorbar(DenPowBefore, y, elinewidth=1.0,color='r',linewidth=0.5,linestyle="dashed")

            ax.set_xscale("log")#, nonposx='clip')
            grid_y_ticks=numpy.arange(numpy.nanmin(y),numpy.nanmax(y),50)
            ax.set_yticks(grid_y_ticks,minor=True)
            locmaj = LogLocator(base=10,numticks=12)
            ax.xaxis.set_major_locator(locmaj)
            locmin = LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(which='minor')
            plt.legend(loc='upper left',fontsize=8.5)
            #plt.legend(loc='lower left',fontsize=8.5)

class FaradayAnglePlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for electron density
    '''

    CODE = 'angle'
    plot_name = 'Faraday Angle'
    plot_type = 'scatterbuffer'

    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Faraday Angle (º)'
        self.titles = ['Electron Density']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['angle'] = numpy.degrees(dataOut.phi)
        #'''
        #print(dataOut.phi_uwrp)
        #print(data['angle'])
        #exit(1)
        #'''
        data['dphi'] = dataOut.dphi_uc*10
        #print(dataOut.dphi)

        #data['NSHTS'] = dataOut.NSHTS

        #meta['yrange'] = dataOut.heightList[0:dataOut.NSHTS]

        return data, meta

    def plot(self):

        data = self.data[-1]
        self.x = data[self.CODE]
        dphi = data['dphi']
        self.y = self.data.yrange
        self.xmin = -360#-180
        self.xmax = 360#180
        ax = self.axes[0]

        if ax.firsttime:
            self.autoxticks=False
            #if self.CODE=='den':
            ax.plot(self.x, self.y,marker='o',color='g',linewidth=1.0,markersize=2)
            ax.plot(dphi, self.y,marker='o',color='blue',linewidth=1.0,markersize=2)

            grid_y_ticks=numpy.arange(numpy.nanmin(self.y),numpy.nanmax(self.y),50)
            self.ystep_given=100
            if self.CODE=='denLP':
                self.ystep_given=200
            ax.set_yticks(grid_y_ticks,minor=True)
            ax.grid(which='minor')
            #plt.tight_layout()
        else:

            self.clear_figures()
            #if self.CODE=='den':
            #print(numpy.shape(self.x))
            ax.plot(self.x, self.y, marker='o',color='g',linewidth=1.0, markersize=2)
            ax.plot(dphi, self.y,marker='o',color='blue',linewidth=1.0,markersize=2)

            grid_y_ticks=numpy.arange(numpy.nanmin(self.y),numpy.nanmax(self.y),50)
            ax.set_yticks(grid_y_ticks,minor=True)
            ax.grid(which='minor')

class EDensityHPPlot(EDensityPlot):
    '''
    Written by R. Flores
    '''
    '''
       Plot for Electron Density Hybrid Experiment
    '''

    CODE = 'denLP'
    plot_name = 'Electron Density'
    plot_type = 'scatterbuffer'

    def update(self, dataOut):
        data = {}
        meta = {}

        data['den_power'] = dataOut.ph2[:dataOut.NSHTS]
        data['den_Faraday']=dataOut.dphi[:dataOut.NSHTS]
        data['den_error']=dataOut.sdp2[:dataOut.NSHTS]
        data['den_LP']=dataOut.ne[:dataOut.NACF]
        data['den_LP_error']=dataOut.ene[:dataOut.NACF]*dataOut.ne[:dataOut.NACF]*0.434
        #self.ene=10**dataOut.ene[:dataOut.NACF]
        data['NSHTS']=dataOut.NSHTS
        data['cut']=dataOut.cut

        return data, meta


class ACFsPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for ACFs Double Pulse Experiment
    '''

    CODE = 'acfs'
    #plot_name = 'ACF'
    plot_type = 'scatterbuffer'


    def setup(self):
        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Lag (ms)'
        self.titles = ['ACFs']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['ACFs'] = dataOut.acfs_to_plot
        data['ACFs_error'] = dataOut.acfs_error_to_plot
        data['lags'] = dataOut.lags_to_plot
        data['Lag_contaminated_1'] = dataOut.x_igcej_to_plot
        data['Lag_contaminated_2'] = dataOut.x_ibad_to_plot
        data['Height_contaminated_1'] = dataOut.y_igcej_to_plot
        data['Height_contaminated_2'] = dataOut.y_ibad_to_plot

        meta['yrange'] = numpy.array([])
        #meta['NSHTS'] = dataOut.NSHTS
        #meta['DPL'] = dataOut.DPL
        data['NSHTS'] = dataOut.NSHTS #This is metadata
        data['DPL'] = dataOut.DPL #This is metadata

        return data, meta

    def plot(self):

        data = self.data[-1]
        #NSHTS = self.meta['NSHTS']
        #DPL = self.meta['DPL']
        NSHTS = data['NSHTS'] #This is metadata
        DPL = data['DPL'] #This is metadata

        lags = data['lags']
        ACFs = data['ACFs']
        errACFs = data['ACFs_error']
        BadLag1 = data['Lag_contaminated_1']
        BadLag2 = data['Lag_contaminated_2']
        BadHei1 = data['Height_contaminated_1']
        BadHei2 = data['Height_contaminated_2']

        self.xmin = 0.0
        self.xmax = 2.0
        self.y = ACFs

        ax = self.axes[0]

        if ax.firsttime:

            for i in range(NSHTS):
                x_aux = numpy.isfinite(lags[i,:])
                y_aux = numpy.isfinite(ACFs[i,:])
                yerr_aux = numpy.isfinite(errACFs[i,:])
                x_igcej_aux = numpy.isfinite(BadLag1[i,:])
                y_igcej_aux = numpy.isfinite(BadHei1[i,:])
                x_ibad_aux = numpy.isfinite(BadLag2[i,:])
                y_ibad_aux = numpy.isfinite(BadHei2[i,:])
                if lags[i,:][~numpy.isnan(lags[i,:])].shape[0]>2:
                    ax.errorbar(lags[i,x_aux], ACFs[i,y_aux], yerr=errACFs[i,x_aux],color='b',marker='o',linewidth=1.0,markersize=2)
                ax.plot(BadLag1[i,x_igcej_aux],BadHei1[i,y_igcej_aux],'x',color='red',markersize=2)
                ax.plot(BadLag2[i,x_ibad_aux],BadHei2[i,y_ibad_aux],'X',color='red',markersize=2)

            self.xstep_given = (self.xmax-self.xmin)/(DPL-1)
            self.ystep_given = 50
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

        else:
            self.clear_figures()
            for i in range(NSHTS):
                x_aux = numpy.isfinite(lags[i,:])
                y_aux = numpy.isfinite(ACFs[i,:])
                yerr_aux = numpy.isfinite(errACFs[i,:])
                x_igcej_aux = numpy.isfinite(BadLag1[i,:])
                y_igcej_aux = numpy.isfinite(BadHei1[i,:])
                x_ibad_aux = numpy.isfinite(BadLag2[i,:])
                y_ibad_aux = numpy.isfinite(BadHei2[i,:])
                if lags[i,:][~numpy.isnan(lags[i,:])].shape[0]>2:
                    ax.errorbar(lags[i,x_aux], ACFs[i,y_aux], yerr=errACFs[i,x_aux],linewidth=1.0,markersize=2,color='b',marker='o')
                ax.plot(BadLag1[i,x_igcej_aux],BadHei1[i,y_igcej_aux],'x',color='red',markersize=2)
                ax.plot(BadLag2[i,x_ibad_aux],BadHei2[i,y_ibad_aux],'X',color='red',markersize=2)
            ax.yaxis.set_minor_locator(MultipleLocator(15))

class ACFsLPPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for ACFs Double Pulse Experiment
    '''

    CODE = 'acfs_LP'
    #plot_name = 'ACF'
    plot_type = 'scatterbuffer'


    def setup(self):
        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Range [km]'
        self.xlabel = 'Lag (ms)'
        self.titles = ['ACFs']
        self.width = 3.5
        self.height = 5.5
        self.colorbar = False
        self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        aux=numpy.zeros((dataOut.NACF,dataOut.IBITS),'float32')
        errors=numpy.zeros((dataOut.NACF,dataOut.IBITS),'float32')
        lags_LP_to_plot=numpy.zeros((dataOut.NACF,dataOut.IBITS),'float32')

        for i in range(dataOut.NACF):
            for j in range(dataOut.IBITS):
                if numpy.abs(dataOut.errors[j,i]/dataOut.output_LP_integrated.real[0,i,0])<1.0:
                    aux[i,j]=dataOut.output_LP_integrated.real[j,i,0]/dataOut.output_LP_integrated.real[0,i,0]
                    aux[i,j]=max(min(aux[i,j],1.0),-1.0)*dataOut.DH+dataOut.heightList[i]
                    lags_LP_to_plot[i,j]=dataOut.lags_LP[j]
                    errors[i,j]=dataOut.errors[j,i]/dataOut.output_LP_integrated.real[0,i,0]*dataOut.DH
                else:
                    aux[i,j]=numpy.nan
                    lags_LP_to_plot[i,j]=numpy.nan
                    errors[i,j]=numpy.nan

        data['ACFs'] = aux
        data['ACFs_error'] = errors
        data['lags'] = lags_LP_to_plot

        meta['yrange'] = numpy.array([])
        #meta['NACF'] = dataOut.NACF
        #meta['NLAG'] = dataOut.NLAG
        data['NACF'] = dataOut.NACF #This is metadata
        data['NLAG'] = dataOut.NLAG #This is metadata

        return data, meta

    def plot(self):

        data = self.data[-1]
        #NACF = self.meta['NACF']
        #NLAG = self.meta['NLAG']
        NACF = data['NACF'] #This is metadata
        NLAG = data['NLAG'] #This is metadata

        lags = data['lags']
        ACFs = data['ACFs']
        errACFs = data['ACFs_error']

        self.xmin = 0.0
        self.xmax = 1.5

        self.y = ACFs

        ax = self.axes[0]

        if ax.firsttime:

            for i in range(NACF):
                x_aux = numpy.isfinite(lags[i,:])
                y_aux = numpy.isfinite(ACFs[i,:])
                yerr_aux = numpy.isfinite(errACFs[i,:])

                if lags[i,:][~numpy.isnan(lags[i,:])].shape[0]>2:
                    ax.errorbar(lags[i,x_aux], ACFs[i,y_aux], yerr=errACFs[i,x_aux],color='b',linewidth=1.0,markersize=2,ecolor='r')

            #self.xstep_given = (self.xmax-self.xmin)/(self.data.NLAG-1)
            self.xstep_given=0.3
            self.ystep_given = 200
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.grid(which='minor')

        else:
            self.clear_figures()

            for i in range(NACF):
                x_aux = numpy.isfinite(lags[i,:])
                y_aux = numpy.isfinite(ACFs[i,:])
                yerr_aux = numpy.isfinite(errACFs[i,:])

                if lags[i,:][~numpy.isnan(lags[i,:])].shape[0]>2:
                    ax.errorbar(lags[i,x_aux], ACFs[i,y_aux], yerr=errACFs[i,x_aux],color='b',linewidth=1.0,markersize=2,ecolor='r')

            ax.yaxis.set_minor_locator(MultipleLocator(15))


class CrossProductsPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for cross products
    '''

    CODE = 'crossprod'
    plot_name = 'Cross Products'
    plot_type = 'scatterbuffer'

    def setup(self):

        self.ncols = 3
        self.nrows = 1
        self.nplots = 3
        self.ylabel = 'Range [km]'
        self.width = 3.5*self.nplots
        self.height = 5.5
        self.colorbar = False
        self.titles = []

    def update(self, dataOut):

        data = {}
        meta = {}

        data['crossprod'] = dataOut.crossprods
        data['NDP'] = dataOut.NDP

        return data, meta

    def plot(self):

        self.x = self.data['crossprod'][:,-1,:,:,:,:]
        self.y = self.data.heights[0:self.data['NDP']]

        for n, ax in enumerate(self.axes):

            self.xmin=numpy.min(numpy.concatenate((self.x[n][0,20:30,0,0],self.x[n][1,20:30,0,0],self.x[n][2,20:30,0,0],self.x[n][3,20:30,0,0])))
            self.xmax=numpy.max(numpy.concatenate((self.x[n][0,20:30,0,0],self.x[n][1,20:30,0,0],self.x[n][2,20:30,0,0],self.x[n][3,20:30,0,0])))

            if ax.firsttime:

                self.autoxticks=False
                if n==0:
                    label1='kax'
                    label2='kay'
                    label3='kbx'
                    label4='kby'
                    self.xlimits=[(self.xmin,self.xmax)]
                elif n==1:
                    label1='kax2'
                    label2='kay2'
                    label3='kbx2'
                    label4='kby2'
                    self.xlimits.append((self.xmin,self.xmax))
                elif n==2:
                    label1='kaxay'
                    label2='kbxby'
                    label3='kaxbx'
                    label4='kaxby'
                    self.xlimits.append((self.xmin,self.xmax))

                ax.plotline1 = ax.plot(self.x[n][0,:,0,0], self.y, color='r',linewidth=2.0, label=label1)
                ax.plotline2 = ax.plot(self.x[n][1,:,0,0], self.y, color='k',linewidth=2.0, label=label2)
                ax.plotline3 = ax.plot(self.x[n][2,:,0,0], self.y, color='b',linewidth=2.0, label=label3)
                ax.plotline4 = ax.plot(self.x[n][3,:,0,0], self.y, color='m',linewidth=2.0, label=label4)
                ax.legend(loc='upper right')
                ax.set_xlim(self.xmin, self.xmax)
                self.titles.append('{}'.format(self.plot_name.upper()))

            else:

                if n==0:
                    self.xlimits=[(self.xmin,self.xmax)]
                else:
                    self.xlimits.append((self.xmin,self.xmax))

                ax.set_xlim(self.xmin, self.xmax)

                ax.plotline1[0].set_data(self.x[n][0,:,0,0],self.y)
                ax.plotline2[0].set_data(self.x[n][1,:,0,0],self.y)
                ax.plotline3[0].set_data(self.x[n][2,:,0,0],self.y)
                ax.plotline4[0].set_data(self.x[n][3,:,0,0],self.y)
                self.titles.append('{}'.format(self.plot_name.upper()))


class CrossProductsLPPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for cross products LP
    '''

    CODE = 'crossprodslp'
    plot_name = 'Cross Products LP'
    plot_type = 'scatterbuffer'


    def setup(self):

        self.ncols = 2
        self.nrows = 1
        self.nplots = 2
        self.ylabel = 'Range [km]'
        self.xlabel = 'dB'
        self.width = 3.5*self.nplots
        self.height = 5.5
        self.colorbar = False
        self.titles = []
        self.plots_adjust.update({'wspace': .8 ,'left': 0.17, 'right': 0.88, 'bottom': 0.1})

    def update(self, dataOut):
        data = {}
        meta = {}

        data['crossprodslp'] = 10*numpy.log10(numpy.abs(dataOut.output_LP))

        data['NRANGE'] = dataOut.NRANGE #This is metadata
        data['NLAG'] = dataOut.NLAG #This is metadata

        return data, meta

    def plot(self):

        NRANGE = self.data['NRANGE'][-1]
        NLAG = self.data['NLAG'][-1]

        x = self.data[self.CODE][:,-1,:,:]
        self.y = self.data.yrange[0:NRANGE]

        label_array=numpy.array(['lag '+ str(x) for x in range(NLAG)])
        color_array=['r','k','g','b','c','m','y','orange','steelblue','purple','peru','darksalmon','grey','limegreen','olive','midnightblue']


        for n, ax in enumerate(self.axes):

            self.xmin=28#30
            self.xmax=70#70
            #self.xmin=numpy.min(numpy.concatenate((self.x[0,:,n],self.x[1,:,n])))
            #self.xmax=numpy.max(numpy.concatenate((self.x[0,:,n],self.x[1,:,n])))

            if ax.firsttime:

                self.autoxticks=False
                if n == 0:
                    self.plotline_array=numpy.zeros((2,NLAG),dtype=object)

                for i in range(NLAG):
                    self.plotline_array[n,i], = ax.plot(x[i,:,n], self.y, color=color_array[i],linewidth=1.0, label=label_array[i])

                ax.legend(loc='upper right')
                ax.set_xlim(self.xmin, self.xmax)
                if n==0:
                    self.titles.append('{} CH0'.format(self.plot_name.upper()))
                if n==1:
                    self.titles.append('{} CH1'.format(self.plot_name.upper()))
            else:
                for i in range(NLAG):
                    self.plotline_array[n,i].set_data(x[i,:,n],self.y)

                if n==0:
                    self.titles.append('{} CH0'.format(self.plot_name.upper()))
                if n==1:
                    self.titles.append('{} CH1'.format(self.plot_name.upper()))


class NoiseDPPlot(NoisePlot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for noise Double Pulse
    '''

    CODE = 'noise'
    #plot_name = 'Noise'
    #plot_type = 'scatterbuffer'

    def update(self, dataOut):

        data = {}
        meta = {}
        data['noise'] = 10*numpy.log10(dataOut.noise_final)

        return data, meta


class XmitWaveformPlot(Plot):
    '''
    Written by R. Flores
    '''
    '''
    Plot for xmit waveform
    '''

    CODE = 'xmit'
    plot_name = 'Xmit Waveform'
    plot_type = 'scatterbuffer'


    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = ''
        self.xlabel = 'Number of Lag'
        self.width = 5.5
        self.height = 3.5
        self.colorbar = False
        self.plots_adjust.update({'right': 0.85 })
        self.titles = [self.plot_name]
        #self.plots_adjust.update({'left': 0.17, 'right': 0.88, 'bottom': 0.1})

        #if not self.titles:
            #self.titles = self.data.parameters \
                #if self.data.parameters else ['{}'.format(self.plot_name.upper())]

    def update(self, dataOut):

        data = {}
        meta = {}

        y_1=numpy.arctan2(dataOut.output_LP[:,0,2].imag,dataOut.output_LP[:,0,2].real)* 180 / (numpy.pi*10)
        y_2=numpy.abs(dataOut.output_LP[:,0,2])
        norm=numpy.max(y_2)
        norm=max(norm,0.1)
        y_2=y_2/norm

        meta['yrange'] = numpy.array([])

        data['xmit'] = numpy.vstack((y_1,y_2))
        data['NLAG'] = dataOut.NLAG

        return data, meta

    def plot(self):

        data = self.data[-1]
        NLAG = data['NLAG']
        x = numpy.arange(0,NLAG,1,'float32')
        y = data['xmit']

        self.xmin = 0
        self.xmax = NLAG-1
        self.ymin = -1.0
        self.ymax = 1.0
        ax = self.axes[0]

        if ax.firsttime:
            ax.plotline0=ax.plot(x,y[0,:],color='blue')
            ax.plotline1=ax.plot(x,y[1,:],color='red')
            secax=ax.secondary_xaxis(location=0.5)
            secax.xaxis.tick_bottom()
            secax.tick_params( labelleft=False, labeltop=False,
                      labelright=False, labelbottom=False)

            self.xstep_given = 3
            self.ystep_given = .25
            secax.set_xticks(numpy.linspace(self.xmin, self.xmax, 6)) #only works on matplotlib.version>3.2

        else:
            ax.plotline0[0].set_data(x,y[0,:])
            ax.plotline1[0].set_data(x,y[1,:])
