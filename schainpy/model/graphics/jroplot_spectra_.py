# Copyright (c) 2012-2021 Jicamarca Radio Observatory
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
"""Classes to plot Spectra data

"""

import os
import numpy

from schainpy.model.graphics.jroplot_base import Plot, plt, log

from matplotlib import __version__ as plt_version

if plt_version >='3.3.4':
    EXTRA_POINTS = 0
else:
    EXTRA_POINTS = 1


class SpectraPlot(Plot):
    '''
    Plot for Spectra data
    '''

    CODE = 'spc'
    colormap = 'jet'
    plot_type = 'pcolor'
    buffering = False

    def setup(self):

        self.nplots = len(self.data.channels)
        self.ncols = int(numpy.sqrt(self.nplots) + 0.9)
        self.nrows = int((1.0 * self.nplots / self.ncols) + 0.9)
        self.height = 2.6 * self.nrows
        self.cb_label = 'dB'
        if self.showprofile:
            self.width = 4 * self.ncols
        else:
            self.width = 3.5 * self.ncols
        self.plots_adjust.update({'wspace': 0.8, 'hspace':0.2, 'left': 0.2, 'right': 0.9, 'bottom': 0.18})
        self.ylabel = 'Range [km]'

    def update(self, dataOut):

        data = {}
        meta = {}
        spc = 10 * numpy.log10(dataOut.data_spc / dataOut.normFactor)
        data['spc'] = spc
        data['rti'] = dataOut.getPower()
        if hasattr(dataOut, 'LagPlot'): #Double Pulse
            max_hei_id = dataOut.nHeights - 2*dataOut.LagPlot
            data['noise'] = 10*numpy.log10(dataOut.getNoise(ymin_index=53,ymax_index=max_hei_id)/dataOut.normFactor)
            data['noise'][0] = 10*numpy.log10(dataOut.getNoise(ymin_index=53)[0]/dataOut.normFactor)
        else:
            data['noise'] = 10 * numpy.log10(dataOut.getNoise() / dataOut.normFactor)
        extrapoints = spc.shape[1] % dataOut.nFFTPoints
        extrapoints=1
        meta['xrange'] = (dataOut.getFreqRange(EXTRA_POINTS) / 1000., dataOut.getAcfRange(EXTRA_POINTS), dataOut.getVelRange(EXTRA_POINTS))
        if self.CODE == 'spc_moments':
            data['moments'] = dataOut.moments
        if self.CODE == 'gaussian_fit':
            data['gaussfit'] = dataOut.DGauFitParams

        return data, meta

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        if (self.CODE == 'spc_moments') | (self.CODE == 'gaussian_fit'):
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []

        y = self.data.yrange
        self.y = y

        data = self.data[-1]
        z = data['spc']

        for n, ax in enumerate(self.axes):
            noise = data['noise'][n]

            if self.CODE == 'spc_moments':
                mean = data['moments'][n, 1]
            if self.CODE == 'gaussian_fit':
                gau0 = data['gaussfit'][n][2,:,0]
                gau1 = data['gaussfit'][n][2,:,1]
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else numpy.nanmin(x)#-self.xmax
                #self.zmin = self.zmin if self.zmin else numpy.nanmin(z)
                #self.zmax = self.zmax if self.zmax else numpy.nanmax(z)
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap),
                                       shading = 'auto' # Cambio de actualizacion
                                       )

                if self.showprofile:
                    ax.plt_profile = self.pf_axes[n].plot(
                        data['rti'][n], y)[0]
                    ax.plt_noise = self.pf_axes[n].plot(numpy.repeat(noise, len(y)), y,
                                                        color="k", linestyle="dashed", lw=1)[0]
                if self.CODE == 'spc_moments':
                    ax.plt_mean = ax.plot(mean, y, color='k', lw=1)[0]
                if self.CODE == 'gaussian_fit':
                    ax.plt_gau0 = ax.plot(gau0, y, color='r', lw=1)[0]
                    ax.plt_gau1 = ax.plot(gau1, y, color='y', lw=1)[0]
            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.set_array(z[n].T.ravel())
                if self.showprofile:
                    ax.plt_profile.set_data(data['rti'][n], y)
                    ax.plt_noise.set_data(numpy.repeat(noise, len(y)), y)
                if self.CODE == 'spc_moments':
                    ax.plt_mean.set_data(mean, y)
                if self.CODE == 'gaussian_fit':
                    ax.plt_gau0.set_data(gau0, y)
                    ax.plt_gau1.set_data(gau1, y)
            self.titles.append('CH {}: {:3.2f}dB'.format(n, noise))

class SpectraObliquePlot(Plot):
    '''
    Plot for Spectra data

    Written by R. Flores
    '''

    CODE = 'spc_oblique'
    colormap = 'jet'
    plot_type = 'pcolor'

    def setup(self):
        self.xaxis = "oblique"
        self.nplots = len(self.data.channels)
        self.ncols = int(numpy.sqrt(self.nplots) + 0.9)
        self.nrows = int((1.0 * self.nplots / self.ncols) + 0.9)
        self.height = 2.6 * self.nrows
        self.cb_label = 'dB'
        if self.showprofile:
            self.width = 4 * self.ncols
        else:
            self.width = 3.5 * self.ncols
        self.plots_adjust.update({'wspace': 0.8, 'hspace':0.2, 'left': 0.2, 'right': 0.9, 'bottom': 0.18})
        self.ylabel = 'Range [km]'

    def update(self, dataOut):

        data = {}
        meta = {}
        spc = 10*numpy.log10(dataOut.data_spc/dataOut.normFactor)
        data['spc'] = spc
        data['rti'] = dataOut.getPower()
        data['noise'] = 10*numpy.log10(dataOut.getNoise()/dataOut.normFactor)
        meta['xrange'] = (dataOut.getFreqRange(EXTRA_POINTS)/1000., dataOut.getAcfRange(EXTRA_POINTS), dataOut.getVelRange(EXTRA_POINTS))

        data['shift1'] = dataOut.Dop_EEJ_T1[0]
        data['shift2'] = dataOut.Dop_EEJ_T2[0]
        data['max_val_2'] = dataOut.Oblique_params[0,-1,:]
        data['shift1_error'] = dataOut.Err_Dop_EEJ_T1[0]
        data['shift2_error'] = dataOut.Err_Dop_EEJ_T2[0]

        return data, meta

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []

        y = self.data.yrange
        self.y = y
        data = self.data[-1]
        z = data['spc']

        for n, ax in enumerate(self.axes):
            noise = self.data['noise'][n][-1]
            shift1 = data['shift1']
            shift2 = data['shift2']
            max_val_2 = data['max_val_2']
            err1 = data['shift1_error']
            err2 = data['shift2_error']
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(z)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(z)
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )

                if self.showprofile:
                    ax.plt_profile = self.pf_axes[n].plot(
                        self.data['rti'][n][-1], y)[0]
                    ax.plt_noise = self.pf_axes[n].plot(numpy.repeat(noise, len(y)), y,
                                                        color="k", linestyle="dashed", lw=1)[0]

                self.ploterr1 = ax.errorbar(shift1, y, xerr=err1, fmt='k^', elinewidth=2.2, marker='o', linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)
                self.ploterr2 = ax.errorbar(shift2, y, xerr=err2, fmt='m^',elinewidth=2.2,marker='o',linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)
                self.ploterr3 = ax.errorbar(max_val_2, y, xerr=0, fmt='g^',elinewidth=2.2,marker='o',linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)
            else:
                self.ploterr1.remove()
                self.ploterr2.remove()
                self.ploterr3.remove()
                ax.plt.set_array(z[n].T.ravel())
                if self.showprofile:
                    ax.plt_profile.set_data(self.data['rti'][n][-1], y)
                    ax.plt_noise.set_data(numpy.repeat(noise, len(y)), y)
                self.ploterr1 = ax.errorbar(shift1, y, xerr=err1, fmt='k^', elinewidth=2.2, marker='o', linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)
                self.ploterr2 = ax.errorbar(shift2, y, xerr=err2, fmt='m^',elinewidth=2.2,marker='o',linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)
                self.ploterr3 = ax.errorbar(max_val_2, y, xerr=0, fmt='g^',elinewidth=2.2,marker='o',linestyle='None',markersize=2.5,capsize=0.3,markeredgewidth=0.2)

            self.titles.append('CH {}: {:3.2f}dB'.format(n, noise))


class CrossSpectraPlot(Plot):

    CODE = 'cspc'
    colormap = 'jet'
    plot_type = 'pcolor'
    zmin_coh = None
    zmax_coh = None
    zmin_phase = None
    zmax_phase = None
    
    def setup(self):

        self.ncols = 2
        self.nplots = len(self.data.pairs) * 2
        self.nrows = int((1.0 * self.nplots / self.ncols) + 0.9)
        self.width = 3.1 * self.ncols
        self.height = 5 * self.nrows
        self.ylabel = 'Range [km]'
        self.showprofile = False
        self.plots_adjust.update({'left': 0.08, 'right': 0.92, 'wspace': 0.5, 'hspace':0.4, 'top':0.95, 'bottom': 0.08})

    def update(self, dataOut):

        data = {}
        meta = {}

        spc = dataOut.data_spc
        cspc = dataOut.data_cspc
        extrapoints = spc.shape[1] % dataOut.nFFTPoints
        meta['xrange'] = (dataOut.getFreqRange(EXTRA_POINTS) / 1000., dataOut.getAcfRange(EXTRA_POINTS), dataOut.getVelRange(EXTRA_POINTS))
        meta['pairs'] = dataOut.pairsList

        tmp = []

        for n, pair in enumerate(meta['pairs']):
            out = cspc[n] / numpy.sqrt(spc[pair[0]] * spc[pair[1]])
            coh = numpy.abs(out)
            phase = numpy.arctan2(out.imag, out.real) * 180 / numpy.pi
            tmp.append(coh)
            tmp.append(phase)

        data['cspc'] = numpy.array(tmp)

        return data, meta

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []

        y = self.data.yrange
        self.y = y

        data = self.data[-1]
        cspc = data['cspc']

        for n in range(len(self.data.pairs)):
            pair = self.data.pairs[n]
            coh = cspc[n * 2]
            phase = cspc[n * 2 + 1]
            ax = self.axes[2 * n]
            if ax.firsttime:
                ax.plt = ax.pcolormesh(x, y, coh.T,
                                       vmin=0,
                                       vmax=1,
                                       cmap=plt.get_cmap(self.colormap_coh)
                                       )
            else:
                ax.plt.set_array(coh.T.ravel())
            self.titles.append(
                'Coherence Ch{} * Ch{}'.format(pair[0], pair[1]))

            ax = self.axes[2 * n + 1]
            if ax.firsttime:
                ax.plt = ax.pcolormesh(x, y, phase.T,
                                       vmin=-180,
                                       vmax=180,
                                       cmap=plt.get_cmap(self.colormap_phase)
                                       )
            else:
                ax.plt.set_array(phase.T.ravel())
            self.titles.append('Phase CH{} * CH{}'.format(pair[0], pair[1]))


class CrossSpectra4Plot(Plot):

    CODE = 'cspc'
    colormap = 'jet'
    plot_type = 'pcolor'
    zmin_coh = None
    zmax_coh = None
    zmin_phase = None
    zmax_phase = None
    
    def setup(self):

        self.ncols = 4
        self.nrows = len(self.data.pairs)
        self.nplots = self.nrows * 4
        self.width = 3.1 * self.ncols
        self.height = 5 * self.nrows
        self.ylabel = 'Range [km]'
        self.showprofile = False
        self.plots_adjust.update({'left': 0.08, 'right': 0.92, 'wspace': 0.5, 'hspace':0.4, 'top':0.95, 'bottom': 0.08})

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []


        y = self.data.heights
        self.y = y
        nspc = self.data['spc']
        #print(numpy.shape(self.data['spc']))
        spc = self.data['cspc'][0]
        #print(numpy.shape(nspc))
        #exit()
        #nspc[1,:,:] = numpy.flip(nspc[1,:,:],axis=0)
        #print(numpy.shape(spc))
        #exit()
        cspc = self.data['cspc'][1]

        #xflip=numpy.flip(x)
        #print(numpy.shape(cspc))
        #exit()

        for n in range(self.nrows):
            noise = self.data['noise'][:,-1]
            pair = self.data.pairs[n]
            #print(pair)
            #exit()
            ax = self.axes[4 * n]
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(nspc)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(nspc)
                ax.plt = ax.pcolormesh(x , y , nspc[pair[0]].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                #print(numpy.shape(nspc[pair[0]].T))
                #exit()
                ax.plt.set_array(nspc[pair[0]].T.ravel())
            self.titles.append('CH {}: {:3.2f}dB'.format(pair[0], noise[pair[0]]))

            ax = self.axes[4 * n + 1]

            if ax.firsttime:
                ax.plt = ax.pcolormesh(x , y, numpy.flip(nspc[pair[1]],axis=0).T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:

                ax.plt.set_array(numpy.flip(nspc[pair[1]],axis=0).T.ravel())
            self.titles.append('CH {}: {:3.2f}dB'.format(pair[1], noise[pair[1]]))

            out = cspc[n] / numpy.sqrt(spc[pair[0]] * spc[pair[1]])
            coh = numpy.abs(out)
            phase = numpy.arctan2(out.imag, out.real) * 180 / numpy.pi

            ax = self.axes[4 * n + 2]
            if ax.firsttime:
                ax.plt = ax.pcolormesh(x, y, numpy.flip(coh,axis=0).T,
                                       vmin=0,
                                       vmax=1,
                                       cmap=plt.get_cmap(self.colormap_coh)
                                       )
            else:
                ax.plt.set_array(numpy.flip(coh,axis=0).T.ravel())
            self.titles.append(
                'Coherence Ch{} * Ch{}'.format(pair[0], pair[1]))

            ax = self.axes[4 * n + 3]
            if ax.firsttime:
                ax.plt = ax.pcolormesh(x, y, numpy.flip(phase,axis=0).T,
                                       vmin=-180,
                                       vmax=180,
                                       cmap=plt.get_cmap(self.colormap_phase)
                                       )
            else:
                ax.plt.set_array(numpy.flip(phase,axis=0).T.ravel())
            self.titles.append('Phase CH{} * CH{}'.format(pair[0], pair[1]))


class CrossSpectra2Plot(Plot):

    CODE = 'cspc'
    colormap = 'jet'
    plot_type = 'pcolor'
    zmin_coh = None
    zmax_coh = None
    zmin_phase = None
    zmax_phase = None

    def setup(self):

        self.ncols = 1
        self.nrows = len(self.data.pairs)
        self.nplots = self.nrows * 1
        self.width = 3.1 * self.ncols
        self.height = 5 * self.nrows
        self.ylabel = 'Range [km]'
        self.showprofile = False
        self.plots_adjust.update({'left': 0.22, 'right': .90, 'wspace': 0.5, 'hspace':0.4, 'top':0.95, 'bottom': 0.08})

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []


        y = self.data.heights
        self.y = y
        #nspc = self.data['spc']
        #print(numpy.shape(self.data['spc']))
        #spc = self.data['cspc'][0]
        #print(numpy.shape(spc))
        #exit()
        cspc = self.data['cspc'][1]
        #print(numpy.shape(cspc))
        #exit()

        for n in range(self.nrows):
            noise = self.data['noise'][:,-1]
            pair = self.data.pairs[n]
            #print(pair)            #exit()



            out = cspc[n]# / numpy.sqrt(spc[pair[0]] * spc[pair[1]])

            #print(out[:,53])
            #exit()
            cross = numpy.abs(out)
            z = cross/self.data.nFactor
            #print("here")
            #print(dataOut.data_spc[0,0,0])
            #exit()

            cross = 10*numpy.log10(z)
            #print(numpy.shape(cross))
            #print(cross[0,:])
            #print(self.data.nFactor)
            #exit()
            #phase = numpy.arctan2(out.imag, out.real) * 180 / numpy.pi

            ax = self.axes[1 * n]
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(cross)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(cross)
                ax.plt = ax.pcolormesh(x, y, cross.T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                ax.plt.set_array(cross.T.ravel())
            self.titles.append(
                'Cross Spectra Power Ch{} * Ch{}'.format(pair[0], pair[1]))


class CrossSpectra3Plot(Plot):

    CODE = 'cspc'
    colormap = 'jet'
    plot_type = 'pcolor'
    zmin_coh = None
    zmax_coh = None
    zmin_phase = None
    zmax_phase = None

    def setup(self):

        self.ncols = 3
        self.nrows = len(self.data.pairs)
        self.nplots = self.nrows * 3
        self.width = 3.1 * self.ncols
        self.height = 5 * self.nrows
        self.ylabel = 'Range [km]'
        self.showprofile = False
        self.plots_adjust.update({'left': 0.22, 'right': .90, 'wspace': 0.5, 'hspace':0.4, 'top':0.95, 'bottom': 0.08})

    def plot(self):

        if self.xaxis == "frequency":
            x = self.data.xrange[0]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2]
            self.xlabel = "Velocity (m/s)"

        self.titles = []


        y = self.data.heights
        self.y = y
        #nspc = self.data['spc']
        #print(numpy.shape(self.data['spc']))
        #spc = self.data['cspc'][0]
        #print(numpy.shape(spc))
        #exit()
        cspc = self.data['cspc'][1]
        #print(numpy.shape(cspc))
        #exit()

        for n in range(self.nrows):
            noise = self.data['noise'][:,-1]
            pair = self.data.pairs[n]
            #print(pair)            #exit()



            out = cspc[n]# / numpy.sqrt(spc[pair[0]] * spc[pair[1]])

            #print(out[:,53])
            #exit()
            cross = numpy.abs(out)
            z = cross/self.data.nFactor
            cross = 10*numpy.log10(z)

            out_r= out.real/self.data.nFactor
            #out_r = 10*numpy.log10(out_r)

            out_i= out.imag/self.data.nFactor
            #out_i = 10*numpy.log10(out_i)
            #print(numpy.shape(cross))
            #print(cross[0,:])
            #print(self.data.nFactor)
            #exit()
            #phase = numpy.arctan2(out.imag, out.real) * 180 / numpy.pi

            ax = self.axes[3 * n]
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(cross)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(cross)
                ax.plt = ax.pcolormesh(x, y, cross.T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                ax.plt.set_array(cross.T.ravel())
            self.titles.append(
                'Cross Spectra Power Ch{} * Ch{}'.format(pair[0], pair[1]))

            ax = self.axes[3 * n + 1]
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(cross)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(cross)
                ax.plt = ax.pcolormesh(x, y, out_r.T,
                                       vmin=-1.e6,
                                       vmax=0,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                ax.plt.set_array(out_r.T.ravel())
            self.titles.append(
                'Cross Spectra Real Ch{} * Ch{}'.format(pair[0], pair[1]))

            ax = self.axes[3 * n + 2]


            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.zmin = self.zmin if self.zmin else numpy.nanmin(cross)
                self.zmax = self.zmax if self.zmax else numpy.nanmax(cross)
                ax.plt = ax.pcolormesh(x, y, out_i.T,
                                       vmin=-1.e6,
                                       vmax=1.e6,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                ax.plt.set_array(out_i.T.ravel())
            self.titles.append(
                'Cross Spectra Imag Ch{} * Ch{}'.format(pair[0], pair[1]))

class RTIPlot(Plot):
    '''
    Plot for RTI data
    '''

    CODE = 'rti'
    colormap = 'jet'
    plot_type = 'pcolorbuffer'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = len(self.data.channels)
        self.nplots = len(self.data.channels)
        self.ylabel = 'Range [km]'
        self.xlabel = 'Local Time'
        self.cb_label = 'dB'
        self.plots_adjust.update({'hspace':0.8, 'left': 0.1, 'bottom': 0.1, 'right':0.95})
        self.titles = ['{} Channel {}'.format(
            self.CODE.upper(), x) for x in range(self.nrows)]

    def update(self, dataOut):

        data = {}
        meta = {}
        data['rti'] = dataOut.getPower()
        data['noise'] = 10 * numpy.log10(dataOut.getNoise() / dataOut.normFactor)

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
            self.zmin = self.zmin if self.zmin else numpy.min(self.z)
            self.zmax = self.zmax if self.zmax else numpy.max(self.z)
            if ax.firsttime:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
                if self.showprofile:
                    ax.plot_profile = self.pf_axes[n].plot(
                        self.data['rti'][n][-1], self.y)[0]
                    ax.plot_noise = self.pf_axes[n].plot(numpy.repeat(self.data['noise'][n][-1], len(self.y)), self.y,
                                                         color="k", linestyle="dashed", lw=1)[0]
            else:
                if self.zlimits is not None:
                    self.zmin, self.zmax = self.zlimits[n]
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
                if self.showprofile:
                    ax.plot_profile.set_data(self.data['rti'][n][-1], self.y)
                    ax.plot_noise.set_data(numpy.repeat(
                        self.data['noise'][n][-1], len(self.y)), self.y)


class SpectrogramPlot(Plot):
    '''
    Plot for Spectrogram data
    '''

    CODE = 'Spectrogram_Profile'
    colormap = 'binary'
    plot_type = 'pcolorbuffer'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = len(self.data.channels)
        self.nplots = len(self.data.channels)
        self.xlabel = 'Time'
        #self.cb_label = 'dB'
        self.plots_adjust.update({'hspace':1.2, 'left': 0.1, 'bottom': 0.12, 'right':0.95})
        self.titles = []

        #self.titles = ['{} Channel {} \n H = {} km ({} - {})'.format(
            #self.CODE.upper(), x, self.data.heightList[self.data.hei], self.data.heightList[self.data.hei],self.data.heightList[self.data.hei]+(self.data.DH*self.data.nProfiles)) for x in range(self.nrows)]

        self.titles = ['{} Channel {}'.format(
            self.CODE.upper(), x) for x in range(self.nrows)]


    def update(self, dataOut):
        data = {}
        meta = {}

        maxHei = 1620#+12000
        maxHei = 1180
        indb = numpy.where(dataOut.heightList <= maxHei)
        hei = indb[0][-1]
        #print(dataOut.heightList)

        factor = dataOut.nIncohInt
        z = dataOut.data_spc[:,:,hei] / factor
        z = numpy.where(numpy.isfinite(z), z, numpy.NAN)
        #buffer = 10 * numpy.log10(z)

        meta['xrange'] = (dataOut.getFreqRange(EXTRA_POINTS)/1000., dataOut.getAcfRange(EXTRA_POINTS), dataOut.getVelRange(EXTRA_POINTS))


        #self.hei = hei
        #self.heightList = dataOut.heightList
        #self.DH = (dataOut.heightList[1] - dataOut.heightList[0])/dataOut.step
        #self.nProfiles = dataOut.nProfiles

        data['Spectrogram_Profile'] = 10 * numpy.log10(z)

        data['hei'] = hei
        data['DH'] = (dataOut.heightList[1] - dataOut.heightList[0])/dataOut.step
        data['nProfiles'] = dataOut.nProfiles
        #meta['yrange'] = dataOut.heightList[0:dataOut.NSHTS]
        '''
        import matplotlib.pyplot as plt
        plt.plot(10 * numpy.log10(z[0,:]))
        plt.show()

        from time import sleep
        sleep(10)
        '''
        return data, meta

    def plot(self):

        self.x = self.data.times
        self.z = self.data[self.CODE]
        self.y = self.data.xrange[0]

        hei = self.data['hei'][-1]
        DH = self.data['DH'][-1]
        nProfiles = self.data['nProfiles'][-1]

        self.ylabel = "Frequency (kHz)"

        self.z = numpy.ma.masked_invalid(self.z)

        if self.decimation is None:
            x, y, z = self.fill_gaps(self.x, self.y, self.z)
        else:
            x, y, z = self.fill_gaps(*self.decimate())

        for n, ax in enumerate(self.axes):
            self.zmin = self.zmin if self.zmin else numpy.min(self.z)
            self.zmax = self.zmax if self.zmax else numpy.max(self.z)
            data = self.data[-1]
            if ax.firsttime:
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )
            else:
                ax.plt.remove()
                ax.plt = ax.pcolormesh(x, y, z[n].T,
                                       vmin=self.zmin,
                                       vmax=self.zmax,
                                       cmap=plt.get_cmap(self.colormap)
                                       )

        #self.titles.append('Spectrogram')

        #self.titles.append('{} Channel {} \n H = {} km ({} - {})'.format(
            #self.CODE.upper(), x, y[hei], y[hei],y[hei]+(DH*nProfiles)))

class CoherencePlot(RTIPlot):
    '''
    Plot for Coherence data
    '''

    CODE = 'coh'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = len(self.data.pairs)
        self.nplots = len(self.data.pairs)
        self.ylabel = 'Range [km]'
        self.xlabel = 'Time'
        self.plots_adjust.update({'hspace':0.6, 'left': 0.1, 'bottom': 0.1, 'right':0.95})
        if self.CODE == 'coh':
            self.cb_label = ''
            self.titles = [
                'Coherence Map Ch{} * Ch{}'.format(x[0], x[1]) for x in self.data.pairs]
        else:
            self.cb_label = 'Degrees'
            self.titles = [
                'Phase Map Ch{} * Ch{}'.format(x[0], x[1]) for x in self.data.pairs]

    def update(self, dataOut):

        data = {}
        meta = {}
        data['coh'] = dataOut.getCoherence()
        meta['pairs'] = dataOut.pairsList

        return data, meta

class PhasePlot(CoherencePlot):
    '''
    Plot for Phase map data
    '''

    CODE = 'phase'
    colormap = 'seismic'

    def update(self, dataOut):

        data = {}
        meta = {}
        data['phase'] = dataOut.getCoherence(phase=True)
        meta['pairs'] = dataOut.pairsList

        return data, meta

class NoisePlot(Plot):
    '''
    Plot for noise
    '''

    CODE = 'noise'
    plot_type = 'scatterbuffer'

    def setup(self):
        self.xaxis = 'time'
        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.ylabel = 'Intensity [dB]'
        self.xlabel = 'Time'
        self.titles = ['Noise']
        self.colorbar = False
        self.plots_adjust.update({'right': 0.85 })

    def update(self, dataOut):

        data = {}
        meta = {}
        data['noise'] = 10 * numpy.log10(dataOut.getNoise() / dataOut.normFactor).reshape(dataOut.nChannels, 1)
        meta['yrange'] = numpy.array([])

        return data, meta

    def plot(self):

        x = self.data.times
        xmin = self.data.min_time
        xmax = xmin + self.xrange * 60 * 60
        Y = self.data['noise']

        if self.axes[0].firsttime:
            self.ymin = numpy.nanmin(Y) - 5
            self.ymax = numpy.nanmax(Y) + 5
            for ch in self.data.channels:
                y = Y[ch]
                self.axes[0].plot(x, y, lw=1, label='Ch{}'.format(ch))
            plt.legend(bbox_to_anchor=(1.18, 1.0))
        else:
            for ch in self.data.channels:
                y = Y[ch]
                self.axes[0].lines[ch].set_data(x, y)

        self.ymin = numpy.nanmin(Y) - 5
        self.ymax = numpy.nanmax(Y) + 10


class PowerProfilePlot(Plot):

    CODE = 'pow_profile'
    plot_type = 'scatter'

    def setup(self):

        self.ncols = 1
        self.nrows = 1
        self.nplots = 1
        self.height = 4
        self.width = 3
        self.ylabel = 'Range [km]'
        self.xlabel = 'Intensity [dB]'
        self.titles = ['Power Profile']
        self.colorbar = False

    def update(self, dataOut):

        data = {}
        meta = {}
        data[self.CODE] = dataOut.getPower()

        return data, meta

    def plot(self):

        y = self.data.yrange
        self.y = y

        x = self.data[-1][self.CODE]

        if self.xmin is None: self.xmin = numpy.nanmin(x) * 0.9
        if self.xmax is None: self.xmax = numpy.nanmax(x) * 1.1

        if self.axes[0].firsttime:
            for ch in self.data.channels:
                self.axes[0].plot(x[ch], y, lw=1, label='Ch{}'.format(ch))
            plt.legend()
        else:
            for ch in self.data.channels:
                self.axes[0].lines[ch].set_data(x[ch], y)


class SpectraCutPlot(Plot):

    CODE = 'spc_cut'
    plot_type = 'scatter'
    buffering = False

    def setup(self):

        self.nplots = len(self.data.channels)
        self.ncols = int(numpy.sqrt(self.nplots) + 0.9)
        self.nrows = int((1.0 * self.nplots / self.ncols) + 0.9)
        self.width = 3.4 * self.ncols + 1.5
        self.height = 3 * self.nrows
        self.ylabel = 'Power [dB]'
        self.colorbar = False
        self.plots_adjust.update({'left':0.1, 'hspace':0.3, 'right': 0.75, 'bottom':0.08})

    def update(self, dataOut):

        data = {}
        meta = {}
        spc = 10 * numpy.log10(dataOut.data_pre[0] / dataOut.normFactor)
        data['spc'] = spc
        meta['xrange'] = (dataOut.getFreqRange(EXTRA_POINTS) / 1000., dataOut.getAcfRange(EXTRA_POINTS), dataOut.getVelRange(EXTRA_POINTS))
        if self.CODE == 'cut_gaussian_fit':
            data['gauss_fit0'] = 10 * numpy.log10(dataOut.GaussFit0 / dataOut.normFactor)
            data['gauss_fit1'] = 10 * numpy.log10(dataOut.GaussFit1 / dataOut.normFactor)
        return data, meta

    def plot(self):
        if self.xaxis == "frequency":
            x = self.data.xrange[0][1:]
            self.xlabel = "Frequency (kHz)"
        elif self.xaxis == "time":
            x = self.data.xrange[1]
            self.xlabel = "Time (ms)"
        else:
            x = self.data.xrange[2][:-1]
            self.xlabel = "Velocity (m/s)"

        if self.CODE == 'cut_gaussian_fit':
            x = self.data.xrange[2][:-1]
            self.xlabel = "Velocity (m/s)"

        self.titles = []

        y = self.data.yrange
        data = self.data[-1]
        z = data['spc']

        if self.height_index:
            index = numpy.array(self.height_index)
        else:
            index = numpy.arange(0, len(y), int((len(y)) / 9))

        for n, ax in enumerate(self.axes):
            if self.CODE == 'cut_gaussian_fit':
                gau0 = data['gauss_fit0']
                gau1 = data['gauss_fit1']
            if ax.firsttime:
                self.xmax = self.xmax if self.xmax else numpy.nanmax(x)
                self.xmin = self.xmin if self.xmin else -self.xmax
                self.ymin = self.ymin if self.ymin else numpy.nanmin(z[:,:,index])
                self.ymax = self.ymax if self.ymax else numpy.nanmax(z[:,:,index])

                ax.plt = ax.plot(x, z[n, :, index].T, lw=0.25)
                if self.CODE == 'cut_gaussian_fit':
                    ax.plt_gau0 = ax.plot(x, gau0[n, :, index].T, lw=1, linestyle='-.')
                    for i, line in enumerate(ax.plt_gau0):
                        line.set_color(ax.plt[i].get_color())
                    ax.plt_gau1 = ax.plot(x, gau1[n, :, index].T, lw=1, linestyle='--')
                    for i, line in enumerate(ax.plt_gau1):
                        line.set_color(ax.plt[i].get_color())
                labels = ['Range = {:2.1f}km'.format(y[i]) for i in index]
                self.figures[0].legend(ax.plt, labels, loc='center right')
            else:
                for i, line in enumerate(ax.plt):
                    line.set_data(x, z[n, :, index[i]].T)
                for i, line in enumerate(ax.plt_gau0):
                    line.set_data(x, gau0[n, :, index[i]].T)
                    line.set_color(ax.plt[i].get_color())
                for i, line in enumerate(ax.plt_gau1):
                    line.set_data(x, gau1[n, :, index[i]].T)
                    line.set_color(ax.plt[i].get_color())
            self.titles.append('CH {}'.format(n))


class BeaconPhase(Plot):

    __isConfig = None
    __nsubplots = None

    PREFIX = 'beacon_phase'

    def __init__(self):
        Plot.__init__(self)
        self.timerange = 24 * 60 * 60
        self.isConfig = False
        self.__nsubplots = 1
        self.counter_imagwr = 0
        self.WIDTH = 800
        self.HEIGHT = 400
        self.WIDTHPROF = 120
        self.HEIGHTPROF = 0
        self.xdata = None
        self.ydata = None

        self.PLOT_CODE = BEACON_CODE

        self.FTP_WEI = None
        self.EXP_CODE = None
        self.SUB_EXP_CODE = None
        self.PLOT_POS = None

        self.filename_phase = None

        self.figfile = None

        self.xmin = None
        self.xmax = None

    def getSubplots(self):

        ncol = 1
        nrow = 1

        return nrow, ncol

    def setup(self, id, nplots, wintitle, showprofile=True, show=True):

        self.__showprofile = showprofile
        self.nplots = nplots

        ncolspan = 7
        colspan = 6
        self.__nsubplots = 2

        self.createFigure(id=id,
                          wintitle=wintitle,
                          widthplot=self.WIDTH + self.WIDTHPROF,
                          heightplot=self.HEIGHT + self.HEIGHTPROF,
                          show=show)

        nrow, ncol = self.getSubplots()

        self.addAxes(nrow, ncol * ncolspan, 0, 0, colspan, 1)

    def save_phase(self, filename_phase):
        f = open(filename_phase, 'w+')
        f.write('\n\n')
        f.write('JICAMARCA RADIO OBSERVATORY - Beacon Phase \n')
        f.write('DD MM YYYY  HH MM SS   pair(2,0) pair(2,1) pair(2,3) pair(2,4)\n\n')
        f.close()

    def save_data(self, filename_phase, data, data_datetime):
        f = open(filename_phase, 'a')
        timetuple_data = data_datetime.timetuple()
        day = str(timetuple_data.tm_mday)
        month = str(timetuple_data.tm_mon)
        year = str(timetuple_data.tm_year)
        hour = str(timetuple_data.tm_hour)
        minute = str(timetuple_data.tm_min)
        second = str(timetuple_data.tm_sec)
        f.write(day + ' ' + month + ' ' + year + '  ' + hour + ' ' + minute + ' ' + second + '   ' + str(data[0]) + '   ' + str(data[1]) + '   ' + str(data[2]) + '   ' + str(data[3]) + '\n')
        f.close()

    def plot(self):
        log.warning('TODO: Not yet implemented...')

    def run(self, dataOut, id, wintitle="", pairsList=None, showprofile='True',
            xmin=None, xmax=None, ymin=None, ymax=None, hmin=None, hmax=None,
            timerange=None,
            save=False, figpath='./', figfile=None, show=True, ftp=False, wr_period=1,
            server=None, folder=None, username=None, password=None,
            ftp_wei=0, exp_code=0, sub_exp_code=0, plot_pos=0):

        if dataOut.flagNoData:
            return dataOut

        if not isTimeInHourRange(dataOut.datatime, xmin, xmax):
            return

        if pairsList == None:
            pairsIndexList = dataOut.pairsIndexList[:10]
        else:
            pairsIndexList = []
            for pair in pairsList:
                if pair not in dataOut.pairsList:
                    raise ValueError("Pair %s is not in dataOut.pairsList" % (pair))
                pairsIndexList.append(dataOut.pairsList.index(pair))

        if pairsIndexList == []:
            return

 #         if len(pairsIndexList) > 4:
 #             pairsIndexList = pairsIndexList[0:4]

        hmin_index = None
        hmax_index = None

        if hmin != None and hmax != None:
            indexes = numpy.arange(dataOut.nHeights)
            hmin_list = indexes[dataOut.heightList >= hmin]
            hmax_list = indexes[dataOut.heightList <= hmax]

            if hmin_list.any():
                hmin_index = hmin_list[0]

            if hmax_list.any():
                hmax_index = hmax_list[-1] + 1

        x = dataOut.getTimeRange()

        thisDatetime = dataOut.datatime

        title = wintitle + " Signal Phase"  # : %s" %(thisDatetime.strftime("%d-%b-%Y"))
        xlabel = "Local Time"
        ylabel = "Phase (degrees)"

        update_figfile = False

        nplots = len(pairsIndexList)
        # phase = numpy.zeros((len(pairsIndexList),len(dataOut.beacon_heiIndexList)))
        phase_beacon = numpy.zeros(len(pairsIndexList))
        for i in range(nplots):
            pair = dataOut.pairsList[pairsIndexList[i]]
            ccf = numpy.average(dataOut.data_cspc[pairsIndexList[i], :, hmin_index:hmax_index], axis=0)
            powa = numpy.average(dataOut.data_spc[pair[0], :, hmin_index:hmax_index], axis=0)
            powb = numpy.average(dataOut.data_spc[pair[1], :, hmin_index:hmax_index], axis=0)
            avgcoherenceComplex = ccf / numpy.sqrt(powa * powb)
            phase = numpy.arctan2(avgcoherenceComplex.imag, avgcoherenceComplex.real) * 180 / numpy.pi

            if dataOut.beacon_heiIndexList:
                phase_beacon[i] = numpy.average(phase[dataOut.beacon_heiIndexList])
            else:
                phase_beacon[i] = numpy.average(phase)

        if not self.isConfig:

            nplots = len(pairsIndexList)

            self.setup(id=id,
                       nplots=nplots,
                       wintitle=wintitle,
                       showprofile=showprofile,
                       show=show)

            if timerange != None:
                self.timerange = timerange

            self.xmin, self.xmax = self.getTimeLim(x, xmin, xmax, timerange)

            if ymin == None: ymin = 0
            if ymax == None: ymax = 360

            self.FTP_WEI = ftp_wei
            self.EXP_CODE = exp_code
            self.SUB_EXP_CODE = sub_exp_code
            self.PLOT_POS = plot_pos

            self.name = thisDatetime.strftime("%Y%m%d_%H%M%S")
            self.isConfig = True
            self.figfile = figfile
            self.xdata = numpy.array([])
            self.ydata = numpy.array([])

            update_figfile = True

            # open file beacon phase
            path = '%s%03d' % (self.PREFIX, self.id)
            beacon_file = os.path.join(path, '%s.txt' % self.name)
            self.filename_phase = os.path.join(figpath, beacon_file)
            # self.save_phase(self.filename_phase)


        # store data beacon phase
        # self.save_data(self.filename_phase, phase_beacon, thisDatetime)

        self.setWinTitle(title)


        title = "Phase Plot %s" % (thisDatetime.strftime("%Y/%m/%d %H:%M:%S"))

        legendlabels = ["Pair (%d,%d)" % (pair[0], pair[1]) for pair in dataOut.pairsList]

        axes = self.axesList[0]

        self.xdata = numpy.hstack((self.xdata, x[0:1]))

        if len(self.ydata) == 0:
            self.ydata = phase_beacon.reshape(-1, 1)
        else:
            self.ydata = numpy.hstack((self.ydata, phase_beacon.reshape(-1, 1)))


        axes.pmultilineyaxis(x=self.xdata, y=self.ydata,
                    xmin=self.xmin, xmax=self.xmax, ymin=ymin, ymax=ymax,
                    xlabel=xlabel, ylabel=ylabel, title=title, legendlabels=legendlabels, marker='x', markersize=8, linestyle="solid",
                    XAxisAsTime=True, grid='both'
                    )

        self.draw()

        if dataOut.ltctime >= self.xmax:
            self.counter_imagwr = wr_period
            self.isConfig = False
            update_figfile = True

        self.save(figpath=figpath,
                  figfile=figfile,
                  save=save,
                  ftp=ftp,
                  wr_period=wr_period,
                  thisDatetime=thisDatetime,
                  update_figfile=update_figfile)

        return dataOut
