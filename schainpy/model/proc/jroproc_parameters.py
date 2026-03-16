# v3.0-devel
import numpy
import math
from scipy import optimize, interpolate, signal, stats, ndimage
from scipy.fftpack import fft
import scipy
from scipy.optimize import least_squares
import re
import datetime
import copy
import sys
import importlib
import itertools
from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool
import time

import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b #optimize with bounds on state papameters
from .jroproc_base import ProcessingUnit, Operation, MPDecorator
from schainpy.model.data.jrodata import Parameters, hildebrand_sekhon
# from scipy import asarray as ar,exp
ar = numpy.asarray
exp = numpy.exp
from scipy.optimize import fmin, curve_fit
from schainpy.utils import log
import warnings
# from numpy import NaN
from numpy import nan

NaN = nan

from scipy.optimize.optimize import OptimizeWarning
warnings.filterwarnings('ignore')


SPEED_OF_LIGHT = 299792458

'''solving pickling issue'''

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

# @MPDecorator
class ParametersProc(ProcessingUnit):

    METHODS = {}
    nSeconds = None

    def __init__(self):
        ProcessingUnit.__init__(self)

        self.buffer = None
        self.firstdatatime = None
        self.profIndex = 0
        self.dataOut = Parameters()
        self.setupReq = False #Agregar a todas las unidades de proc

    def __updateObjFromInput(self):

        self.dataOut.inputUnit = self.dataIn.type

        self.dataOut.timeZone = self.dataIn.timeZone
        self.dataOut.dstFlag = self.dataIn.dstFlag
        self.dataOut.errorCount = self.dataIn.errorCount
        self.dataOut.useLocalTime = self.dataIn.useLocalTime

        self.dataOut.radarControllerHeaderObj = self.dataIn.radarControllerHeaderObj.copy()
        self.dataOut.systemHeaderObj = self.dataIn.systemHeaderObj.copy()
        self.dataOut.channelList = self.dataIn.channelList
        self.dataOut.heightList = self.dataIn.heightList
        self.dataOut.dtype = numpy.dtype([('real','<f4'),('imag','<f4')])
        # self.dataOut.nHeights = self.dataIn.nHeights
        # self.dataOut.nChannels = self.dataIn.nChannels
        # self.dataOut.nBaud = self.dataIn.nBaud
        # self.dataOut.nCode = self.dataIn.nCode
        # self.dataOut.code = self.dataIn.code
        # self.dataOut.nProfiles = self.dataOut.nFFTPoints
        self.dataOut.flagDiscontinuousBlock = self.dataIn.flagDiscontinuousBlock
        # self.dataOut.utctime = self.firstdatatime
        self.dataOut.utctime = self.dataIn.utctime
        self.dataOut.flagDecodeData = self.dataIn.flagDecodeData #asumo q la data esta decodificada
        self.dataOut.flagDeflipData = self.dataIn.flagDeflipData #asumo q la data esta sin flip
        self.dataOut.nCohInt = self.dataIn.nCohInt
        # self.dataOut.nIncohInt = 1
        # self.dataOut.ippSeconds = self.dataIn.ippSeconds
        # self.dataOut.windowOfFilter = self.dataIn.windowOfFilter
        self.dataOut.timeInterval1 = self.dataIn.timeInterval
        self.dataOut.heightList = self.dataIn.heightList
        self.dataOut.frequency = self.dataIn.frequency
        #self.dataOut.noise = self.dataIn.noise

    def run(self):

        #----------------------    Voltage Data    ---------------------------

        if self.dataIn.type == "Voltage":

            self.__updateObjFromInput()
            self.dataOut.data_pre = self.dataIn.data.copy()
            self.dataOut.flagNoData = False
            self.dataOut.utctimeInit = self.dataIn.utctime
            self.dataOut.paramInterval = self.dataIn.nProfiles*self.dataIn.nCohInt*self.dataIn.ippSeconds
            if hasattr(self.dataIn, 'dataPP_POW'):
                self.dataOut.dataPP_POW = self.dataIn.dataPP_POW

            if hasattr(self.dataIn, 'dataPP_POWER'):
                self.dataOut.dataPP_POWER = self.dataIn.dataPP_POWER

            if hasattr(self.dataIn, 'dataPP_DOP'):
                self.dataOut.dataPP_DOP = self.dataIn.dataPP_DOP

            if hasattr(self.dataIn, 'dataPP_SNR'):
                self.dataOut.dataPP_SNR = self.dataIn.dataPP_SNR

            if hasattr(self.dataIn, 'dataPP_WIDTH'):
                self.dataOut.dataPP_WIDTH = self.dataIn.dataPP_WIDTH
            return

        #----------------------    Spectra Data    ---------------------------

        if self.dataIn.type == "Spectra":

            self.dataOut.data_pre = [self.dataIn.data_spc, self.dataIn.data_cspc]
            self.dataOut.data_spc = self.dataIn.data_spc
            self.dataOut.data_cspc = self.dataIn.data_cspc
            # for JULIA processing
            self.dataOut.data_diffcspc = self.dataIn.data_diffcspc     
            self.dataOut.nDiffIncohInt = self.dataIn.nDiffIncohInt
            # for JULIA processing
            self.dataOut.nProfiles = self.dataIn.nProfiles
            self.dataOut.nIncohInt = self.dataIn.nIncohInt
            self.dataOut.nFFTPoints = self.dataIn.nFFTPoints
            self.dataOut.ippFactor = self.dataIn.ippFactor
            self.dataOut.abscissaList = self.dataIn.getVelRange(1)
            self.dataOut.spc_noise = self.dataIn.getNoise()
            self.dataOut.spc_range = (self.dataIn.getFreqRange(1) , self.dataIn.getAcfRange(1) , self.dataIn.getVelRange(1))
            # self.dataOut.normFactor = self.dataIn.normFactor
            self.dataOut.pairsList = self.dataIn.pairsList
            self.dataOut.groupList = self.dataIn.pairsList
            self.dataOut.flagNoData = False

            if hasattr(self.dataIn, 'ChanDist'): #Distances of receiver channels
                self.dataOut.ChanDist = self.dataIn.ChanDist
            else: self.dataOut.ChanDist = None

            #if hasattr(self.dataIn, 'VelRange'): #Velocities range
            #    self.dataOut.VelRange = self.dataIn.VelRange
            #else: self.dataOut.VelRange = None

            if hasattr(self.dataIn, 'RadarConst'): #Radar Constant
                self.dataOut.RadarConst = self.dataIn.RadarConst

            if hasattr(self.dataIn, 'NPW'): #NPW
                self.dataOut.NPW = self.dataIn.NPW

            if hasattr(self.dataIn, 'COFA'): #COFA
                self.dataOut.COFA = self.dataIn.COFA



        #----------------------    Correlation Data    ---------------------------

        if self.dataIn.type == "Correlation":
            acf_ind, ccf_ind, acf_pairs, ccf_pairs, data_acf, data_ccf = self.dataIn.splitFunctions()

            self.dataOut.data_pre = (self.dataIn.data_cf[acf_ind,:], self.dataIn.data_cf[ccf_ind,:,:])
            self.dataOut.normFactor = (self.dataIn.normFactor[acf_ind,:], self.dataIn.normFactor[ccf_ind,:])
            self.dataOut.groupList = (acf_pairs, ccf_pairs)

            self.dataOut.abscissaList = self.dataIn.lagRange
            self.dataOut.noise = self.dataIn.noise
            self.dataOut.data_snr = self.dataIn.SNR
            self.dataOut.flagNoData = False
            self.dataOut.nAvg = self.dataIn.nAvg

        #----------------------    Parameters Data    ---------------------------

        if self.dataIn.type == "Parameters":
            self.dataOut.copy(self.dataIn)
            self.dataOut.flagNoData = False

            return True

        self.__updateObjFromInput()
        self.dataOut.utctimeInit = self.dataIn.utctime
        self.dataOut.paramInterval = self.dataIn.timeInterval

        return


def target(tups):

    obj, args = tups

    return obj.FitGau(args)

class RemoveWideGC(Operation):
    ''' This class remove the wide clutter and replace it with a simple interpolation points
        This mainly applies to CLAIRE radar

        ClutterWidth :    Width to look for the clutter peak

        Input:

        self.dataOut.data_pre :        SPC and CSPC
        self.dataOut.spc_range :       To select wind and rainfall velocities

        Affected:

        self.dataOut.data_pre :        It is used for the new SPC and CSPC ranges of wind

        Written by D. Scipión 25.02.2021
    '''
    def __init__(self):
        Operation.__init__(self)
        self.i = 0
        self.ich = 0
        self.ir = 0
    
    def run(self, dataOut, ClutterWidth=2.5):

        self.spc = dataOut.data_pre[0].copy()
        self.spc_out = dataOut.data_pre[0].copy()
        self.Num_Chn = self.spc.shape[0]
        self.Num_Hei = self.spc.shape[2]
        VelRange = dataOut.spc_range[2][:-1]
        dv = VelRange[1]-VelRange[0]

        # Find the velocities that corresponds to zero
        gc_values = numpy.squeeze(numpy.where(numpy.abs(VelRange) <= ClutterWidth))

        # Removing novalid data from the spectra
        for ich in range(self.Num_Chn) :
            for ir in range(self.Num_Hei) :
                # Estimate the noise at each range
                HSn = hildebrand_sekhon(self.spc[ich,:,ir],dataOut.nIncohInt)

                # Removing the noise floor at each range
                novalid = numpy.where(self.spc[ich,:,ir] < HSn)
                self.spc[ich,novalid,ir] = HSn

                junk = numpy.append(numpy.insert(numpy.squeeze(self.spc[ich,gc_values,ir]),0,HSn),HSn)
                j1index = numpy.squeeze(numpy.where(numpy.diff(junk)>0))
                j2index = numpy.squeeze(numpy.where(numpy.diff(junk)<0))
                if ((numpy.size(j1index)<=1) | (numpy.size(j2index)<=1)) : 
                    continue
                junk3 = numpy.squeeze(numpy.diff(j1index))
                junk4 = numpy.squeeze(numpy.diff(j2index))
                
                valleyindex = j2index[numpy.where(junk4>1)]
                peakindex = j1index[numpy.where(junk3>1)]

                isvalid = numpy.squeeze(numpy.where(numpy.abs(VelRange[gc_values[peakindex]]) <= 2.5*dv))
                if numpy.size(isvalid) == 0 :
                    continue
                if numpy.size(isvalid) >1 :
                    vindex = numpy.argmax(self.spc[ich,gc_values[peakindex[isvalid]],ir])
                    isvalid = isvalid[vindex]
                
                # clutter peak
                gcpeak = peakindex[isvalid]
                vl = numpy.where(valleyindex < gcpeak)
                if numpy.size(vl) == 0:
                    continue
                gcvl = valleyindex[vl[0][-1]]
                vr = numpy.where(valleyindex > gcpeak)
                if numpy.size(vr) == 0:
                    continue
                gcvr = valleyindex[vr[0][0]]

                # Removing the clutter
                interpindex = numpy.array([gc_values[gcvl], gc_values[gcvr]])
                gcindex = gc_values[gcvl+1:gcvr-1]
                self.spc_out[ich,gcindex,ir] = numpy.interp(VelRange[gcindex],VelRange[interpindex],self.spc[ich,interpindex,ir])

        dataOut.data_pre[0] = self.spc_out

        return dataOut

class SpectralFilters(Operation):
    ''' This class allows to replace the novalid values with noise for each channel 
        This applies to CLAIRE RADAR

        PositiveLimit :    RightLimit of novalid data
        NegativeLimit :    LeftLimit of novalid data

        Input:

        self.dataOut.data_pre :        SPC and CSPC
        self.dataOut.spc_range :       To select wind and rainfall velocities

        Affected:

        self.dataOut.data_pre :        It is used for the new SPC and CSPC ranges of wind

        Written by D. Scipión 29.01.2021
    '''
    def __init__(self):
        Operation.__init__(self)
        self.i = 0
    
    def run(self, dataOut, ):

        self.spc = dataOut.data_pre[0].copy()
        self.Num_Chn = self.spc.shape[0]
        VelRange = dataOut.spc_range[2]

        # novalid corresponds to data within the Negative and PositiveLimit
        

        # Removing novalid data from the spectra
        for i in range(self.Num_Chn):
            self.spc[i,novalid,:] = dataOut.noise[i]
        dataOut.data_pre[0] = self.spc
        return dataOut



class GaussianFit(Operation):

    '''
        Function that fit of one and two generalized gaussians (gg) based
        on the PSD shape across an "power band" identified from a cumsum of
        the measured spectrum - noise.

        Input:
            self.dataOut.data_pre    :    SelfSpectra

        Output:
            self.dataOut.SPCparam :    SPC_ch1, SPC_ch2

    '''
    def __init__(self):
        Operation.__init__(self)
        self.i=0


    # def run(self, dataOut, num_intg=7, pnoise=1., SNRlimit=-9): #num_intg: Incoherent integrations, pnoise: Noise, vel_arr: range of velocities, similar to the ftt points
    def run(self, dataOut, SNRdBlimit=-9, method='generalized'):
        """This routine will find a couple of generalized Gaussians to a power spectrum
        methods: generalized, squared
        input: spc
        output:
            noise, amplitude0,shift0,width0,p0,Amplitude1,shift1,width1,p1
        """
        print ('Entering ',method,' double Gaussian fit')
        self.spc = dataOut.data_pre[0].copy()
        self.Num_Hei = self.spc.shape[2]
        self.Num_Bin = self.spc.shape[1]
        self.Num_Chn = self.spc.shape[0]

        start_time = time.time()

        pool = Pool(processes=self.Num_Chn)
        args = [(dataOut.spc_range[2], ich, dataOut.spc_noise[ich], dataOut.nIncohInt, SNRdBlimit) for ich in range(self.Num_Chn)]
        objs = [self for __ in range(self.Num_Chn)]
        attrs = list(zip(objs, args))
        DGauFitParam = pool.map(target, attrs)
        # Parameters:
        # 0. Noise, 1. Amplitude, 2. Shift, 3. Width 4. Power
        dataOut.DGauFitParams = numpy.asarray(DGauFitParam)

        # Double Gaussian Curves
        gau0 = numpy.zeros([self.Num_Chn,self.Num_Bin,self.Num_Hei])
        gau0[:] = numpy.NaN
        gau1 = numpy.zeros([self.Num_Chn,self.Num_Bin,self.Num_Hei])
        gau1[:] = numpy.NaN
        x_mtr = numpy.transpose(numpy.tile(dataOut.getVelRange(1)[:-1], (self.Num_Hei,1)))
        for iCh in range(self.Num_Chn):
            N0 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][0,:,0]] * self.Num_Bin))
            N1 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][0,:,1]] * self.Num_Bin))
            A0 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][1,:,0]] * self.Num_Bin))
            A1 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][1,:,1]] * self.Num_Bin))
            v0 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][2,:,0]] * self.Num_Bin))
            v1 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][2,:,1]] * self.Num_Bin))
            s0 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][3,:,0]] * self.Num_Bin))
            s1 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][3,:,1]] * self.Num_Bin))
            if method == 'generalized':
                p0 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][4,:,0]] * self.Num_Bin))
                p1 = numpy.transpose(numpy.transpose([dataOut.DGauFitParams[iCh][4,:,1]] * self.Num_Bin))
            elif method == 'squared':
                p0 = 2.
                p1 = 2.
            gau0[iCh] = A0*numpy.exp(-0.5*numpy.abs((x_mtr-v0)/s0)**p0)+N0
            gau1[iCh] = A1*numpy.exp(-0.5*numpy.abs((x_mtr-v1)/s1)**p1)+N1
        dataOut.GaussFit0 = gau0
        dataOut.GaussFit1 = gau1

        print('Leaving ',method ,' double Gaussian fit')
        return dataOut

    def FitGau(self, X):
        # print('Entering FitGau')
        # Assigning the variables
        Vrange, ch, wnoise, num_intg, SNRlimit = X
        # Noise Limits
        noisebl = wnoise * 0.9
        noisebh = wnoise * 1.1
        # Radar Velocity
        Va = max(Vrange)
        deltav = Vrange[1] - Vrange[0]
        x = numpy.arange(self.Num_Bin)

        # print ('stop 0')

        # 5 parameters, 2 Gaussians
        DGauFitParam = numpy.zeros([5, self.Num_Hei,2])
        DGauFitParam[:] = numpy.NaN

        # SPCparam = []
        # SPC_ch1 = numpy.zeros([self.Num_Bin,self.Num_Hei])
        # SPC_ch2 = numpy.zeros([self.Num_Bin,self.Num_Hei])
        # SPC_ch1[:] = 0 #numpy.NaN
        # SPC_ch2[:] = 0 #numpy.NaN
        # print ('stop 1')
        for ht in range(self.Num_Hei):
            # print (ht)
            # print ('stop 2')
            # Spectra at each range
            spc =  numpy.asarray(self.spc)[ch,:,ht]
            snr = ( spc.mean() - wnoise ) / wnoise
            snrdB = 10.*numpy.log10(snr)

            #print ('stop 3')
            if snrdB < SNRlimit :
                # snr = numpy.NaN
                # SPC_ch1[:,ht] = 0#numpy.NaN
                # SPC_ch1[:,ht] = 0#numpy.NaN
                # SPCparam = (SPC_ch1,SPC_ch2)
                # print ('SNR less than SNRth')
                continue
            # wnoise = hildebrand_sekhon(spc,num_intg)
            # print ('stop 2.01')
            #############################################
            # normalizing spc and noise
            # This part differs from gg1
            # spc_norm_max = max(spc) #commented by D. Scipión 19.03.2021
            #spc = spc / spc_norm_max
            # pnoise = pnoise #/ spc_norm_max #commented by D. Scipión 19.03.2021
            #############################################

            # print ('stop 2.1')
            fatspectra=1.0
            # noise per channel.... we might want to use the noise at each range
            
            # wnoise = noise_ #/ spc_norm_max #commented by D. Scipión 19.03.2021
                #wnoise,stdv,i_max,index =enoise(spc,num_intg) #noise estimate using Hildebrand Sekhon, only wnoise is used
                #if wnoise>1.1*pnoise: # to be tested later
                #    wnoise=pnoise
            # noisebl = wnoise*0.9
            # noisebh = wnoise*1.1
            spc = spc - wnoise # signal

            # print ('stop 2.2')
            minx = numpy.argmin(spc)
            #spcs=spc.copy()
            spcs = numpy.roll(spc,-minx)
            cum = numpy.cumsum(spcs)
            # tot_noise = wnoise * self.Num_Bin  #64;

            # print ('stop 2.3')
            # snr = sum(spcs) / tot_noise
            # snrdB = 10.*numpy.log10(snr)
            #print ('stop 3')
            # if snrdB < SNRlimit :
                # snr = numpy.NaN
                # SPC_ch1[:,ht] = 0#numpy.NaN
                # SPC_ch1[:,ht] = 0#numpy.NaN
                # SPCparam = (SPC_ch1,SPC_ch2)
                # print ('SNR less than SNRth')
                # continue


            #if snrdB<-18 or numpy.isnan(snrdB) or num_intg<4:
            #    return [None,]*4,[None,]*4,None,snrdB,None,None,[None,]*5,[None,]*9,None
            # print ('stop 4')
            cummax = max(cum)
            epsi = 0.08 * fatspectra # cumsum to narrow down the energy region
            cumlo = cummax * epsi
            cumhi = cummax * (1-epsi)
            powerindex = numpy.array(numpy.where(numpy.logical_and(cum>cumlo, cum<cumhi))[0])

            # print ('stop 5')
            if len(powerindex) < 1:# case for powerindex 0
                # print ('powerindex < 1')
                continue
            powerlo = powerindex[0]
            powerhi = powerindex[-1]
            powerwidth = powerhi-powerlo
            if powerwidth <= 1:
                # print('powerwidth <= 1')
                continue

            # print ('stop 6')
            firstpeak = powerlo + powerwidth/10.# first gaussian energy location
            secondpeak = powerhi - powerwidth/10. #second gaussian energy location
            midpeak = (firstpeak + secondpeak)/2.
            firstamp = spcs[int(firstpeak)]
            secondamp = spcs[int(secondpeak)]
            midamp = spcs[int(midpeak)]

            y_data = spc + wnoise

            '''    single Gaussian    '''
            shift0 = numpy.mod(midpeak+minx, self.Num_Bin )
            width0 = powerwidth/4.#Initialization entire power of spectrum divided by 4
            power0 = 2.
            amplitude0 = midamp
            state0 = [shift0,width0,amplitude0,power0,wnoise]
            bnds = ((0,self.Num_Bin-1),(1,powerwidth),(0,None),(0.5,3.),(noisebl,noisebh))
            lsq1 = fmin_l_bfgs_b(self.misfit1, state0, args=(y_data,x,num_intg), bounds=bnds, approx_grad=True)
            # print ('stop 7.1')
            # print (bnds)

            chiSq1=lsq1[1]

            # print ('stop 8')
            if fatspectra<1.0 and powerwidth<4:
                    choice=0
                    Amplitude0=lsq1[0][2]
                    shift0=lsq1[0][0]
                    width0=lsq1[0][1]
                    p0=lsq1[0][3]
                    Amplitude1=0.
                    shift1=0.
                    width1=0.
                    p1=0.
                    noise=lsq1[0][4]
                    #return (numpy.array([shift0,width0,Amplitude0,p0]),
                    #        numpy.array([shift1,width1,Amplitude1,p1]),noise,snrdB,chiSq1,6.,sigmas1,[None,]*9,choice)
            # print ('stop 9')
            '''    two Gaussians    '''
            #shift0=numpy.mod(firstpeak+minx,64); shift1=numpy.mod(secondpeak+minx,64)
            shift0 = numpy.mod(firstpeak+minx, self.Num_Bin )
            shift1 = numpy.mod(secondpeak+minx, self.Num_Bin )
            width0 = powerwidth/6.
            width1 = width0
            power0 = 2.
            power1 = power0
            amplitude0 = firstamp
            amplitude1 = secondamp
            state0 = [shift0,width0,amplitude0,power0,shift1,width1,amplitude1,power1,wnoise]
            #bnds=((0,63),(1,powerwidth/2.),(0,None),(0.5,3.),(0,63),(1,powerwidth/2.),(0,None),(0.5,3.),(noisebl,noisebh))
            bnds=((0,self.Num_Bin-1),(1,powerwidth/2.),(0,None),(0.5,3.),(0,self.Num_Bin-1),(1,powerwidth/2.),(0,None),(0.5,3.),(noisebl,noisebh))
            #bnds=(( 0,(self.Num_Bin-1) ),(1,powerwidth/2.),(0,None),(0.5,3.),( 0,(self.Num_Bin-1)),(1,powerwidth/2.),(0,None),(0.5,3.),(0.1,0.5))

            # print ('stop 10')
            lsq2 = fmin_l_bfgs_b( self.misfit2 , state0 , args=(y_data,x,num_intg) , bounds=bnds , approx_grad=True )

            # print ('stop 11')
            chiSq2 = lsq2[1]

            # print ('stop 12')

            oneG = (chiSq1<5 and chiSq1/chiSq2<2.0) and (abs(lsq2[0][0]-lsq2[0][4])<(lsq2[0][1]+lsq2[0][5])/3. or abs(lsq2[0][0]-lsq2[0][4])<10)

            # print ('stop 13')
            if snrdB>-12: # when SNR is strong pick the peak with least shift (LOS velocity) error
                if oneG:
                    choice = 0
                else:
                    w1 = lsq2[0][1]; w2 = lsq2[0][5]
                    a1 = lsq2[0][2]; a2 = lsq2[0][6]
                    p1 = lsq2[0][3]; p2 = lsq2[0][7]
                    s1 = (2**(1+1./p1))*scipy.special.gamma(1./p1)/p1
                    s2 = (2**(1+1./p2))*scipy.special.gamma(1./p2)/p2
                    gp1 = a1*w1*s1; gp2 = a2*w2*s2 # power content of each ggaussian with proper p scaling

                    if gp1>gp2:
                        if a1>0.7*a2:
                            choice = 1
                        else:
                            choice = 2
                    elif gp2>gp1:
                        if a2>0.7*a1:
                            choice = 2
                        else:
                            choice = 1
                    else:
                        choice = numpy.argmax([a1,a2])+1
                        #else:
                        #choice=argmin([std2a,std2b])+1

            else: # with low SNR go to the most energetic peak
                choice = numpy.argmax([lsq1[0][2]*lsq1[0][1],lsq2[0][2]*lsq2[0][1],lsq2[0][6]*lsq2[0][5]])

            # print ('stop 14')
            shift0 = lsq2[0][0]
            vel0 = Vrange[0] + shift0 * deltav
            shift1 = lsq2[0][4]
            # vel1=Vrange[0] + shift1 * deltav

            # max_vel = 1.0
            # Va = max(Vrange)
            # deltav = Vrange[1]-Vrange[0]
            # print ('stop 15')
            #first peak will be 0, second peak will be 1
            # if vel0 > -1.0 and vel0 < max_vel : #first peak is in the correct range # Commented by D.Scipión 19.03.2021
            if vel0 > -Va and vel0 < Va : #first peak is in the correct range
                shift0 = lsq2[0][0]
                width0 = lsq2[0][1]
                Amplitude0 = lsq2[0][2]
                p0 = lsq2[0][3]

                shift1 = lsq2[0][4]
                width1 = lsq2[0][5]
                Amplitude1 = lsq2[0][6]
                p1 = lsq2[0][7]
                noise = lsq2[0][8]
            else:
                shift1 = lsq2[0][0]
                width1 = lsq2[0][1]
                Amplitude1 = lsq2[0][2]
                p1 = lsq2[0][3]

                shift0 = lsq2[0][4]
                width0 = lsq2[0][5]
                Amplitude0 = lsq2[0][6]
                p0 = lsq2[0][7]
                noise = lsq2[0][8]

            if Amplitude0<0.05: # in case the peak is noise
                shift0,width0,Amplitude0,p0 = 4*[numpy.NaN]
            if Amplitude1<0.05:
                shift1,width1,Amplitude1,p1 = 4*[numpy.NaN]

            # print ('stop 16 ') 
            # SPC_ch1[:,ht] = noise + Amplitude0*numpy.exp(-0.5*(abs(x-shift0)/width0)**p0)
            # SPC_ch2[:,ht] = noise + Amplitude1*numpy.exp(-0.5*(abs(x-shift1)/width1)**p1)
            # SPCparam = (SPC_ch1,SPC_ch2)

            DGauFitParam[0,ht,0] = noise
            DGauFitParam[0,ht,1] = noise
            DGauFitParam[1,ht,0] = Amplitude0
            DGauFitParam[1,ht,1] = Amplitude1
            DGauFitParam[2,ht,0] = Vrange[0] + shift0 * deltav
            DGauFitParam[2,ht,1] = Vrange[0] + shift1 * deltav
            DGauFitParam[3,ht,0] = width0 * deltav
            DGauFitParam[3,ht,1] = width1 * deltav
            DGauFitParam[4,ht,0] = p0
            DGauFitParam[4,ht,1] = p1

        return DGauFitParam

    def y_model1(self,x,state):
        shift0, width0, amplitude0, power0, noise = state
        model0 = amplitude0*numpy.exp(-0.5*abs((x - shift0)/width0)**power0)
        model0u = amplitude0*numpy.exp(-0.5*abs((x - shift0 - self.Num_Bin)/width0)**power0)
        model0d = amplitude0*numpy.exp(-0.5*abs((x - shift0 + self.Num_Bin)/width0)**power0)
        return model0 + model0u + model0d + noise

    def y_model2(self,x,state): #Equation for two generalized Gaussians with Nyquist
        shift0, width0, amplitude0, power0, shift1, width1, amplitude1, power1, noise = state
        model0 = amplitude0*numpy.exp(-0.5*abs((x-shift0)/width0)**power0)
        model0u = amplitude0*numpy.exp(-0.5*abs((x - shift0 - self.Num_Bin)/width0)**power0)
        model0d = amplitude0*numpy.exp(-0.5*abs((x - shift0 + self.Num_Bin)/width0)**power0)

        model1 = amplitude1*numpy.exp(-0.5*abs((x - shift1)/width1)**power1)
        model1u = amplitude1*numpy.exp(-0.5*abs((x - shift1 - self.Num_Bin)/width1)**power1)
        model1d = amplitude1*numpy.exp(-0.5*abs((x - shift1 + self.Num_Bin)/width1)**power1)
        return model0 + model0u + model0d + model1 + model1u + model1d + noise

    def misfit1(self,state,y_data,x,num_intg): # This function compares how close real data is with the model data, the close it is, the better it is.

        return num_intg*sum((numpy.log(y_data)-numpy.log(self.y_model1(x,state)))**2)#/(64-5.) # /(64-5.) can be commented

    def misfit2(self,state,y_data,x,num_intg):
        return num_intg*sum((numpy.log(y_data)-numpy.log(self.y_model2(x,state)))**2)#/(64-9.)

class Oblique_Gauss_Fit(Operation):
    '''
    Written by R. Flores
    '''
    def __init__(self):
        Operation.__init__(self)

    def Gauss_fit(self,spc,x,nGauss):


        def gaussian(x, a, b, c, d):
            val = a * numpy.exp(-(x - b)**2 / (2*c**2)) + d
            return val

        if nGauss == 'first':
            spc_1_aux = numpy.copy(spc[:numpy.argmax(spc)+1])
            spc_2_aux = numpy.flip(spc_1_aux)
            spc_3_aux = numpy.concatenate((spc_1_aux,spc_2_aux[1:]))

            len_dif = len(x)-len(spc_3_aux)

            spc_zeros = numpy.ones(len_dif)*spc_1_aux[0]

            spc_new = numpy.concatenate((spc_3_aux,spc_zeros))

            y = spc_new

        elif nGauss == 'second':
            y = spc


        # estimate starting values from the data
        a = y.max()
        b = x[numpy.argmax(y)]
        if nGauss == 'first':
            c = 1.#b#b#numpy.std(spc)
        elif nGauss == 'second':
            c = b
        else:
            print("ERROR")

        d = numpy.mean(y[-100:])

        # define a least squares function to optimize
        def minfunc(params):
            return sum((y-gaussian(x,params[0],params[1],params[2],params[3]))**2)

        # fit
        popt = fmin(minfunc,[a,b,c,d],disp=False)
        #popt,fopt,niter,funcalls = fmin(minfunc,[a,b,c,d])


        return gaussian(x, popt[0], popt[1], popt[2], popt[3]), popt[0], popt[1], popt[2], popt[3]

    def Gauss_fit_2(self,spc,x,nGauss):


        def gaussian(x, a, b, c, d):
            val = a * numpy.exp(-(x - b)**2 / (2*c**2)) + d
            return val

        if nGauss == 'first':
            spc_1_aux = numpy.copy(spc[:numpy.argmax(spc)+1])
            spc_2_aux = numpy.flip(spc_1_aux)
            spc_3_aux = numpy.concatenate((spc_1_aux,spc_2_aux[1:]))

            len_dif = len(x)-len(spc_3_aux)

            spc_zeros = numpy.ones(len_dif)*spc_1_aux[0]

            spc_new = numpy.concatenate((spc_3_aux,spc_zeros))

            y = spc_new

        elif nGauss == 'second':
            y = spc


        # estimate starting values from the data
        a = y.max()
        b = x[numpy.argmax(y)]
        if nGauss == 'first':
            c = 1.#b#b#numpy.std(spc)
        elif nGauss == 'second':
            c = b
        else:
            print("ERROR")

        d = numpy.mean(y[-100:])

        # define a least squares function to optimize
        popt,pcov = curve_fit(gaussian,x,y,p0=[a,b,c,d])
        #popt,fopt,niter,funcalls = fmin(minfunc,[a,b,c,d])


        #return gaussian(x, popt[0], popt[1], popt[2], popt[3]), popt[0], popt[1], popt[2], popt[3]
        return gaussian(x, popt[0], popt[1], popt[2], popt[3]),popt[0], popt[1], popt[2], popt[3]

    def Double_Gauss_fit(self,spc,x,A1,B1,C1,A2,B2,C2,D):

        def double_gaussian(x, a1, b1, c1, a2, b2, c2, d):
            val = a1 * numpy.exp(-(x - b1)**2 / (2*c1**2)) + a2 * numpy.exp(-(x - b2)**2 / (2*c2**2)) + d
            return val


        y = spc

        # estimate starting values from the data
        a1 = A1
        b1 = B1
        c1 = C1#numpy.std(spc)

        a2 = A2#y.max()
        b2 = B2#x[numpy.argmax(y)]
        c2 = C2#numpy.std(spc)
        d = D

        # define a least squares function to optimize
        def minfunc(params):
            return sum((y-double_gaussian(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))**2)

        # fit
        popt = fmin(minfunc,[a1,b1,c1,a2,b2,c2,d],disp=False)

        return double_gaussian(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]

    def Double_Gauss_fit_2(self,spc,x,A1,B1,C1,A2,B2,C2,D):

        def double_gaussian(x, a1, b1, c1, a2, b2, c2, d):
            val = a1 * numpy.exp(-(x - b1)**2 / (2*c1**2)) + a2 * numpy.exp(-(x - b2)**2 / (2*c2**2)) + d
            return val


        y = spc

        # estimate starting values from the data
        a1 = A1
        b1 = B1
        c1 = C1#numpy.std(spc)

        a2 = A2#y.max()
        b2 = B2#x[numpy.argmax(y)]
        c2 = C2#numpy.std(spc)
        d = D

        # fit

        popt,pcov = curve_fit(double_gaussian,x,y,p0=[a1,b1,c1,a2,b2,c2,d])

        error = numpy.sqrt(numpy.diag(pcov))

        return popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], error[0], error[1], error[2], error[3], error[4], error[5], error[6]

    def windowing_double(self,spc,x,A1,B1,C1,A2,B2,C2,D):
        from scipy.optimize import curve_fit,fmin

        def R_gaussian(x, a, b, c):
                N = int(numpy.shape(x)[0])
                val = a * numpy.exp(-((x)*c*2*2*numpy.pi)**2 / (2))* numpy.exp(1.j*b*x*4*numpy.pi)
                return val

        def T(x,N):
            T = 1-abs(x)/N
            return T

        def R_T_spc_fun(x, a1, b1, c1, a2, b2, c2, d):

            N = int(numpy.shape(x)[0])

            x_max = x[-1]

            x_pos = x[1600:]
            x_neg = x[:1600]

            R_T_neg_1 = R_gaussian(x, a1, b1, c1)[:1600]*T(x_neg,-x[0])
            R_T_pos_1 = R_gaussian(x, a1, b1, c1)[1600:]*T(x_pos,x[-1])
            R_T_sum_1 = R_T_pos_1 + R_T_neg_1
            R_T_spc_1 = numpy.fft.fft(R_T_sum_1).real
            R_T_spc_1 = numpy.fft.fftshift(R_T_spc_1)
            max_val_1 = numpy.max(R_T_spc_1)
            R_T_spc_1 = R_T_spc_1*a1/max_val_1

            R_T_neg_2 = R_gaussian(x, a2, b2, c2)[:1600]*T(x_neg,-x[0])
            R_T_pos_2 = R_gaussian(x, a2, b2, c2)[1600:]*T(x_pos,x[-1])
            R_T_sum_2 = R_T_pos_2 + R_T_neg_2
            R_T_spc_2 = numpy.fft.fft(R_T_sum_2).real
            R_T_spc_2 = numpy.fft.fftshift(R_T_spc_2)
            max_val_2 = numpy.max(R_T_spc_2)
            R_T_spc_2 = R_T_spc_2*a2/max_val_2

            R_T_d = d*numpy.fft.fftshift(signal.unit_impulse(N))
            R_T_d_neg = R_T_d[:1600]*T(x_neg,-x[0])
            R_T_d_pos = R_T_d[1600:]*T(x_pos,x[-1])
            R_T_d_sum = R_T_d_pos + R_T_d_neg
            R_T_spc_3 = numpy.fft.fft(R_T_d_sum).real
            R_T_spc_3 = numpy.fft.fftshift(R_T_spc_3)

            R_T_final = R_T_spc_1 + R_T_spc_2 + R_T_spc_3

            return R_T_final

        y = spc#gaussian(x, a, meanY, sigmaY) + a*0.1*numpy.random.normal(0, 1, size=len(x))

        from scipy.stats import norm
        mean,std=norm.fit(spc)

        # estimate starting values from the data
        a1 = A1
        b1 = B1
        c1 = C1#numpy.std(spc)

        a2 = A2#y.max()
        b2 = B2#x[numpy.argmax(y)]
        c2 = C2#numpy.std(spc)
        d = D

        ippSeconds = 250*20*1.e-6/3

        x_t = ippSeconds * (numpy.arange(1600) -1600 / 2.)

        x_t = numpy.linspace(x_t[0],x_t[-1],3200)

        x_freq = numpy.fft.fftfreq(1600,d=ippSeconds)
        x_freq = numpy.fft.fftshift(x_freq)

        # define a least squares function to optimize
        def minfunc(params):
            #print(params[2])
            #print(numpy.shape(params[2]))
            return sum((y-R_T_spc_fun(x_t,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))**2/1)#y**2)

        # fit

        popt_full = fmin(minfunc,[a1,b1,c1,a2,b2,c2,d],full_output=True)
        #print("nIter", popt_full[2])
        popt = popt_full[0]

        #return R_T_spc_fun(x_t,popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]
        return popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]

    def Double_Gauss_fit_weight(self,spc,x,A1,B1,C1,A2,B2,C2,D):
        from scipy.optimize import curve_fit,fmin

        def double_gaussian(x, a1, b1, c1, a2, b2, c2, d):
            val = a1 * numpy.exp(-(x - b1)**2 / (2*c1**2)) + a2 * numpy.exp(-(x - b2)**2 / (2*c2**2)) + d
            return val

        y = spc

        from scipy.stats import norm
        mean,std=norm.fit(spc)

        # estimate starting values from the data
        a1 = A1
        b1 = B1
        c1 = C1#numpy.std(spc)

        a2 = A2#y.max()
        b2 = B2#x[numpy.argmax(y)]
        c2 = C2#numpy.std(spc)
        d = D

        y_clean = signal.medfilt(y)
        # define a least squares function to optimize
        def minfunc(params):
            return sum((y-double_gaussian(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))**2/(y_clean**2/1))

        # fit
        popt_full = fmin(minfunc,[a1,b1,c1,a2,b2,c2,d], disp =False, full_output=True)
        #print("nIter", popt_full[2])
        popt = popt_full[0]
        #popt,pcov = curve_fit(double_gaussian,x,y,p0=[a1,b1,c1,a2,b2,c2,d])

        #return double_gaussian(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]
        return popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]

    def DH_mode(self,spectra,VelRange):

        from scipy.optimize import curve_fit

        def double_gauss(x, a1,b1,c1, a2,b2,c2, d):
            val = a1 * numpy.exp(-(x - b1)**2 / (2*c1**2)) + a2 * numpy.exp(-(x - b2)**2 / (2*c2**2)) + d
            return val

        spec = (spectra.copy()).flatten()
        amp=spec.max()
        params=numpy.array([amp,-400,30,amp/4,-200,150,1.0e7])
        #try:
        popt,pcov=curve_fit(double_gauss, VelRange, spec, p0=params,bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf]))

        error = numpy.sqrt(numpy.diag(pcov))
            #doppler_2=popt[4]
            #err_2 = numpy.sqrt(pcov[4][4])

        #except:
            #pass
            #doppler_2=numpy.NAN
            #err_2 = numpy.NAN

        #return doppler_2, err_2

        return popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], error[0], error[1], error[2], error[3], error[4], error[5], error[6]

    def Tri_Marco(self,spc,freq,a1,b1,c1,a2,b2,c2,d):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        def tri_gaussian(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, d):
            z1 = (x-b1)/c1
            z2 = (x-b2)/c2
            z3 = (x-b3)/c3
            val = a1 * numpy.exp(-z1**2/2) + a2 * numpy.exp(-z2**2/2) + a3 * numpy.exp(-z3**2/2) + d
            return val

        from scipy.signal import medfilt
        Nincoh = 20
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)
        c1 = abs(c1)
        c2 = abs(c2)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-tri_gaussian(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]))/spcm

        # fit
        #bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
        bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0,0,0,0],[numpy.inf,-100,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,600,numpy.inf,numpy.inf])
        #bounds=([0,-180,0,0,-100,30,0,110,0,0],[numpy.inf,-110,20,numpy.inf,33,80,numpy.inf,150,16,numpy.inf])
        #bounds=([0,-540,0,0,-300,100,0,330,0,0],[numpy.inf,-330,60,numpy.inf,100,240,numpy.inf,450,80,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,spc_max]
        #print(a1,b1,c1,a2,b2,c2,d)
        popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,a2/4,-b1,c1,d],x_scale=params_scale,bounds=bounds)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        A2f = popt.x[3]; B2f = popt.x[4]; C2f = popt.x[5]
        A3f = popt.x[6]; B3f = popt.x[7]; C3f = popt.x[8]
        Df = popt.x[9]

        return A1f, B1f, C1f, A2f, B2f, C2f, Df

    def Tri_Marco(self,spc,freq,a1,b1,c1,a2,b2,c2,d):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        def duo_gaussian(x, a1, b1, c1, a2, b2, c2, d):
            z1 = (x-b1)/c1
            z2 = (x-b2)/c2
            #z3 = (x-b3)/c3
            val = a1 * numpy.exp(-z1**2/2) + a2 * numpy.exp(-z2**2/2) + d
            return val

        from scipy.signal import medfilt
        Nincoh = 20
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)
        c1 = abs(c1)
        c2 = abs(c2)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-tri_gaussian(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))/spcm

        # fit
        #bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
        bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0],[numpy.inf,-100,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf])
        #bounds=([0,-180,0,0,-100,30,0,110,0,0],[numpy.inf,-110,20,numpy.inf,33,80,numpy.inf,150,16,numpy.inf])
        #bounds=([0,-540,0,0,-300,100,0,330,0,0],[numpy.inf,-330,60,numpy.inf,100,240,numpy.inf,450,80,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,spc_max]
        #print(a1,b1,c1,a2,b2,c2,d)
        popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,d],x_scale=params_scale,bounds=bounds)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        A2f = popt.x[3]; B2f = popt.x[4]; C2f = popt.x[5]
        #A3f = popt.x[6]; B3f = popt.x[7]; C3f = popt.x[8]
        Df = popt.x[9]

        return A1f, B1f, C1f, A2f, B2f, C2f, Df

    def double_gaussian_skew(self,x, a1, b1, c1, a2, b2, c2, k2, d):
        #from scipy import special
        z1 = (x-b1)/c1
        z2 = (x-b2)/c2
        h2 = 1-k2*z2
        h2[h2<0] = 0
        y2 = -1/k2*numpy.log(h2)
        val = a1 * numpy.exp(-z1**2/2) + a2 * numpy.exp(-y2**2/2)/(1-k2*z2) + d
        return val

    def gaussian(self, x, a, b, c, d):
        z = (x-b)/c
        val = a * numpy.exp(-z**2/2) + d
        return val

    def double_gaussian(self, x, a1, b1, c1, a2, b2, c2, d):
        z1 = (x-b1)/c1
        z2 = (x-b2)/c2
        val = a1 * numpy.exp(-z1**2/2) + a2 * numpy.exp(-z2**2/2) + d
        return val

    def double_gaussian_double_skew(self,x, a1, b1, c1, k1, a2, b2, c2, k2, d):

        z1 = (x-b1)/c1
        h1 = 1-k1*z1
        h1[h1<0] = 0
        y1 = -1/k1*numpy.log(h1)

        z2 = (x-b2)/c2
        h2 = 1-k2*z2
        h2[h2<0] = 0
        y2 = -1/k2*numpy.log(h2)

        val = a1 * numpy.exp(-y1**2/2)/(1-k1*z1) + a2 * numpy.exp(-y2**2/2)/(1-k2*z2) + d
        return val

    def gaussian_skew(self,x, a2, b2, c2, k2, d):
        #from scipy import special
        z2 = (x-b2)/c2
        h2 = 1-k2*z2
        h2[h2<0] = 0
        y2 = -1/k2*numpy.log(h2)
        val = a2 * numpy.exp(-y2**2/2)/(1-k2*z2) + d
        return val

    def triple_gaussian_skew(self,x, a1, b1, c1, a2, b2, c2, k2, a3, b3, c3, k3, d):
        #from scipy import special
        z1 = (x-b1)/c1
        z2 = (x-b2)/c2
        z3 = (x-b3)/c3
        h2 = 1-k2*z2
        h2[h2<0] = 0
        y2 = -1/k2*numpy.log(h2)
        h3 = 1-k3*z3
        h3[h3<0] = 0
        y3 = -1/k3*numpy.log(h3)
        val = a1 * numpy.exp(-z1**2/2) + a2 * numpy.exp(-y2**2/2)/(1-k2*z2) + a3 * numpy.exp(-y3**2/2)/(1-k3*z3) + d
        return val

    def Double_Gauss_Skew_fit_weight_bound_no_inputs(self,spc,freq):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = 20
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.double_gaussian_skew(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]))/spcm

        # fit
    #    bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
    #    bounds=([0,-numpy.inf,0,0,-numpy.inf,0,-numpy.inf,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,0,numpy.inf])
        #print(a1,b1,c1,a2,b2,c2,k2,d)
        bounds=([0,-numpy.inf,0,0,-400,0,0,0],[numpy.inf,-340,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        #print(bounds)
        #bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        params_scale = [spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,1,spc_max]
        x0_value = numpy.array([spc_max,-400,30,spc_max/4,-200,150,1,1.0e7])
        #popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=1)
        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)
    #    popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x_scale=params_scale,verbose=1)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        A2f = popt.x[3]; B2f = popt.x[4]; C2f = popt.x[5]; K2f = popt.x[6]
        Df = popt.x[7]

        aux = self.gaussian_skew(freq, A2f, B2f, C2f, K2f, Df)
        doppler = freq[numpy.argmax(aux)]

        #return A1f, B1f, C1f, A2f, B2f, C2f, K2f, Df, doppler
        return A1f, B1f, C1f, A2f, B2f, C2f, K2f, Df, doppler

    def Double_Gauss_Double_Skew_fit_weight_bound_no_inputs(self,spc,freq,Nincoh,hei):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        #from scipy.signal import medfilt
        #Nincoh = 20
        #Nincoh = 80
        Nincoh = Nincoh
        #spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)
        spcm = spc/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.double_gaussian_double_skew(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]))/spcm

        # fit
    #    bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
    #    bounds=([0,-numpy.inf,0,0,-numpy.inf,0,-numpy.inf,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,0,numpy.inf])
        #print(a1,b1,c1,a2,b2,c2,k2,d)
        #bounds=([0,-numpy.inf,0,-numpy.inf,0,-400,0,0,0],[numpy.inf,-340,numpy.inf,0,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        #bounds=([0,-numpy.inf,0,-numpy.inf,0,-400,0,0,0],[numpy.inf,-140,numpy.inf,0,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        bounds=([0,-numpy.inf,0,-5,0,-400,0,0,0],[numpy.inf,-200,numpy.inf,5,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])

        #print(bounds)
        #bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        params_scale = [spc_max,freq_max,freq_max,1,spc_max,freq_max,freq_max,1,spc_max]
        ####################x0_value = numpy.array([spc_max,-400,30,-.1,spc_max/4,-200,150,1,1.0e7])

        dop1_x0 = freq[numpy.argmax(spc)]
        ####dop1_x0 = freq[numpy.argmax(spcm)]
        if dop1_x0 < 0:
          dop2_x0 = dop1_x0 + 100
        if dop1_x0 > 0:
          dop2_x0 = dop1_x0 - 100

        ###########x0_value = numpy.array([spc_max,-200.5,30,-.1,spc_max/4,-100.5,150,1,1.0e7])
        x0_value = numpy.array([spc_max,dop1_x0,30,-.1,spc_max/4, dop2_x0,150,1,1.0e7])
        #x0_value = numpy.array([spc_max,-400.5,30,-.1,spc_max/4,-200.5,150,1,1.0e7])
        #popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=1)
        '''
        print("INSIDE 1")
        print("x0_value: ", x0_value)
        print("boundaries: ", bounds)
        import matplotlib.pyplot as plt
        plt.plot(freq,spc)
        plt.plot(freq,self.double_gaussian_double_skew(freq,x0_value[0],x0_value[1],x0_value[2],x0_value[3],x0_value[4],x0_value[5],x0_value[6],x0_value[7],x0_value[8]))
        plt.title(hei)
        plt.show()
        '''
        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)
    #    popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x_scale=params_scale,verbose=1)
        #print(popt)
        #########print("INSIDE 2")
        J = popt.jac

        try:
            cov = numpy.linalg.inv(J.T.dot(J))
            error = numpy.sqrt(numpy.diagonal(cov))
        except:
            error = numpy.ones((9))*numpy.NAN
        #print("error_inside",error)
        #exit(1)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]; K1f = popt.x[3]
        A2f = popt.x[4]; B2f = popt.x[5]; C2f = popt.x[6]; K2f = popt.x[7]
        Df = popt.x[8]
        '''
        A1f_err = error.x[0]; B1f_err= error.x[1]; C1f_err = error.x[2]; K1f_err = error.x[3]
        A2f_err = error.x[4]; B2f_err = error.x[5]; C2f_err = error.x[6]; K2f_err = error.x[7]
        Df_err = error.x[8]
        '''
        aux1 = self.gaussian_skew(freq, A1f, B1f, C1f, K1f, Df)
        doppler1 = freq[numpy.argmax(aux1)]

        aux2 = self.gaussian_skew(freq, A2f, B2f, C2f, K2f, Df)
        doppler2 = freq[numpy.argmax(aux2)]
        #print("error",error)
        #exit(1)


        return A1f, B1f, C1f, K1f, A2f, B2f, C2f, K2f, Df, doppler1, doppler2, error

    def Double_Gauss_fit_weight_bound_no_inputs(self,spc,freq,Nincoh):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = 20
        Nincoh = 80
        Nincoh = Nincoh
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.double_gaussian(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))/spcm

        # fit
    #    bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
    #    bounds=([0,-numpy.inf,0,0,-numpy.inf,0,-numpy.inf,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,0,numpy.inf])
        #print(a1,b1,c1,a2,b2,c2,k2,d)

        dop1_x0 = freq[numpy.argmax(spcm)]

        #####bounds=([0,-numpy.inf,0,0,-400,0,0],[numpy.inf,-340,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf])
        #####bounds=([0,-numpy.inf,0,0,dop1_x0-50,0,0],[numpy.inf,-340,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf])
        bounds=([0,-numpy.inf,0,0,dop1_x0-50,0,0],[numpy.inf,-300,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf])
        #####bounds=([0,-numpy.inf,0,0,-500,0,0],[numpy.inf,-340,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf])
        #bounds=([0,-numpy.inf,0,-numpy.inf,0,-500,0,0,0],[numpy.inf,-240,numpy.inf,0,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        #print(bounds)
        #bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        params_scale = [spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,spc_max]
        #x0_value = numpy.array([spc_max,-400.5,30,spc_max/4,-200.5,150,1.0e7])
        x0_value = numpy.array([spc_max,-400.5,30,spc_max/4,dop1_x0,150,1.0e7])
        #x0_value = numpy.array([spc_max,-420.5,30,-.1,spc_max/4,-50,150,.1,numpy.mean(spc[-50:])])
        #print("before popt")
        #print(x0_value)
        #print("freq: ",freq)
        #popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=1)
        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)
    #    popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x_scale=params_scale,verbose=1)
        #print("after popt")
        J = popt.jac

        try:
            cov = numpy.linalg.inv(J.T.dot(J))
            error = numpy.sqrt(numpy.diagonal(cov))
        except:
            error = numpy.ones((7))*numpy.NAN

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        A2f = popt.x[3]; B2f = popt.x[4]; C2f = popt.x[5]
        Df = popt.x[6]
        #print("before return")
        return A1f, B1f, C1f, A2f, B2f, C2f, Df, error

    def Double_Gauss_Double_Skew_fit_weight_bound_with_inputs(self, spc, freq, a1, b1, c1, a2, b2, c2, k2, d):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = dataOut.nIncohInt
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.double_gaussian_double_skew(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]))/spcm


        bounds=([0,-numpy.inf,0,-numpy.inf,0,-400,0,0,0],[numpy.inf,-340,numpy.inf,0,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,1,spc_max,freq_max,freq_max,1,spc_max]

        x0_value = numpy.array([a1,b1,c1,-.1,a2,b2,c2,k2,d])

        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]; K1f = popt.x[3]
        A2f = popt.x[4]; B2f = popt.x[5]; C2f = popt.x[6]; K2f = popt.x[7]
        Df = popt.x[8]

        aux = self.gaussian_skew(freq, A2f, B2f, C2f, K2f, Df)
        doppler = x[numpy.argmax(aux)]

        return A1f, B1f, C1f, K1f, A2f, B2f, C2f, K2f, Df, doppler

    def Triple_Gauss_Skew_fit_weight_bound_no_inputs(self,spc,freq):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = 20
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.triple_gaussian_skew(freq,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10],params[11]))/spcm

        # fit
    #    bounds=([0,-460,0,0,-400,120,0],[numpy.inf,-340,50,numpy.inf,0,250,numpy.inf])
    #    bounds=([0,-numpy.inf,0,0,-numpy.inf,0,-numpy.inf,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,0,numpy.inf])
        #print(a1,b1,c1,a2,b2,c2,k2,d)
        bounds=([0,-numpy.inf,0,0,-400,0,0,0,0,0,0,0],[numpy.inf,-340,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf])
        #print(bounds)
        #bounds=([0,-numpy.inf,0,0,-numpy.inf,0,0,0],[numpy.inf,-200,numpy.inf,numpy.inf,0,numpy.inf,numpy.inf,numpy.inf])
        params_scale = [spc_max,freq_max,freq_max,spc_max,freq_max,freq_max,1,spc_max,freq_max,freq_max,1,spc_max]
        x0_value = numpy.array([spc_max,-400,30,spc_max/4,-200,150,1,spc_max/4,400,150,1,1.0e7])
        #popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=1)
        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)
    #    popt = least_squares(lsq_func,[a1,b1,c1,a2,b2,c2,k2,d],x_scale=params_scale,verbose=1)

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        A2f = popt.x[3]; B2f = popt.x[4]; C2f = popt.x[5]; K2f = popt.x[6]
        A3f = popt.x[7]; B3f = popt.x[8]; C3f = popt.x[9]; K3f = popt.x[10]
        Df = popt.x[11]

        aux = self.gaussian_skew(freq, A2f, B2f, C2f, K2f, Df)
        doppler = freq[numpy.argmax(aux)]

        return A1f, B1f, C1f, A2f, B2f, C2f, K2f, A3f, B3f, C3f, K3f, Df, doppler

    def CEEJ_Skew_fit_weight_bound_no_inputs(self,spc,freq,Nincoh):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = 20
        Nincoh = 80
        Nincoh = Nincoh
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.gaussian_skew(freq,params[0],params[1],params[2],params[3],params[4]))#/spcm


        bounds=([0,0,0,-numpy.inf,0],[numpy.inf,numpy.inf,numpy.inf,0,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,1,spc_max]

        x0_value = numpy.array([spc_max,freq[numpy.argmax(spc)],30,-.1,numpy.mean(spc[:50])])

        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)

        J = popt.jac

        try:
            error = numpy.ones((9))*numpy.NAN
            cov = numpy.linalg.inv(J.T.dot(J))
            error[:4] = numpy.sqrt(numpy.diagonal(cov))[:4]
            error[-1] = numpy.sqrt(numpy.diagonal(cov))[-1]
        except:
            error = numpy.ones((9))*numpy.NAN

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]; K1f = popt.x[3]
        Df = popt.x[4]

        aux1 = self.gaussian_skew(freq, A1f, B1f, C1f, K1f, Df)
        doppler1 = freq[numpy.argmax(aux1)]
        #print("CEEJ ERROR:",error)

        return A1f, B1f, C1f, K1f, numpy.NAN, numpy.NAN, numpy.NAN, numpy.NAN, Df, doppler1, numpy.NAN, error

    def CEEJ_fit_weight_bound_no_inputs(self,spc,freq,Nincoh):

        from scipy.optimize import least_squares

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        from scipy.signal import medfilt
        Nincoh = 20
        Nincoh = 80
        Nincoh = Nincoh
        spcm = medfilt(spc,11)/numpy.sqrt(Nincoh)

        # define a least squares function to optimize
        def lsq_func(params):
            return (spc-self.gaussian(freq,params[0],params[1],params[2],params[3]))#/spcm


        bounds=([0,0,0,0],[numpy.inf,numpy.inf,numpy.inf,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,spc_max]

        x0_value = numpy.array([spc_max,freq[numpy.argmax(spcm)],30,numpy.mean(spc[:50])])

        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)

        J = popt.jac

        try:
            error = numpy.ones((4))*numpy.NAN
            cov = numpy.linalg.inv(J.T.dot(J))
            error = numpy.sqrt(numpy.diagonal(cov))
        except:
            error = numpy.ones((4))*numpy.NAN

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        Df = popt.x[3]

        return A1f, B1f, C1f, Df, error

    def Simple_fit_bound(self,spc,freq,Nincoh):

        freq_max = numpy.max(numpy.abs(freq))
        spc_max = numpy.max(spc)

        Nincoh = Nincoh

        def lsq_func(params):
            return (spc-self.gaussian(freq,params[0],params[1],params[2],params[3]))

        bounds=([0,-50,0,0],[numpy.inf,+50,numpy.inf,numpy.inf])

        params_scale = [spc_max,freq_max,freq_max,spc_max]

        x0_value = numpy.array([spc_max,-20.5,5,1.0e7])

        popt = least_squares(lsq_func,x0=x0_value,x_scale=params_scale,bounds=bounds,verbose=0)

        J = popt.jac

        try:
            cov = numpy.linalg.inv(J.T.dot(J))
            error = numpy.sqrt(numpy.diagonal(cov))
        except:
            error = numpy.ones((4))*numpy.NAN

        A1f = popt.x[0]; B1f = popt.x[1]; C1f = popt.x[2]
        Df = popt.x[3]

        return A1f, B1f, C1f, Df, error

    def clean_outliers(self,param):

        threshold = 700

        param = numpy.where(param < -threshold, numpy.nan, param)
        param = numpy.where(param > +threshold, numpy.nan, param)

        return param

    def windowing_single(self,spc,x,A,B,C,D,nFFTPoints):
        from scipy.optimize import curve_fit,fmin

        def R_gaussian(x, a, b, c):
                N = int(numpy.shape(x)[0])
                val = a * numpy.exp(-((x)*c*2*2*numpy.pi)**2 / (2))* numpy.exp(1.j*b*x*4*numpy.pi)
                return val

        def T(x,N):
            T = 1-abs(x)/N
            return T

        def R_T_spc_fun(x, a, b, c, d, nFFTPoints):

            N = int(numpy.shape(x)[0])

            x_max = x[-1]

            x_pos = x[int(nFFTPoints/2):]
            x_neg = x[:int(nFFTPoints/2)]

            R_T_neg_1 = R_gaussian(x, a, b, c)[:int(nFFTPoints/2)]*T(x_neg,-x[0])
            R_T_pos_1 = R_gaussian(x, a, b, c)[int(nFFTPoints/2):]*T(x_pos,x[-1])
            R_T_sum_1 = R_T_pos_1 + R_T_neg_1
            R_T_spc_1 = numpy.fft.fft(R_T_sum_1).real
            R_T_spc_1 = numpy.fft.fftshift(R_T_spc_1)
            max_val_1 = numpy.max(R_T_spc_1)
            R_T_spc_1 = R_T_spc_1*a/max_val_1

            R_T_d = d*numpy.fft.fftshift(signal.unit_impulse(N))
            R_T_d_neg = R_T_d[:int(nFFTPoints/2)]*T(x_neg,-x[0])
            R_T_d_pos = R_T_d[int(nFFTPoints/2):]*T(x_pos,x[-1])
            R_T_d_sum = R_T_d_pos + R_T_d_neg
            R_T_spc_3 = numpy.fft.fft(R_T_d_sum).real
            R_T_spc_3 = numpy.fft.fftshift(R_T_spc_3)

            R_T_final = R_T_spc_1 + R_T_spc_3

            return R_T_final

        y = spc#gaussian(x, a, meanY, sigmaY) + a*0.1*numpy.random.normal(0, 1, size=len(x))

        from scipy.stats import norm
        mean,std=norm.fit(spc)

        # estimate starting values from the data
        a = A
        b = B
        c = C#numpy.std(spc)
        d = D
        '''
        ippSeconds = 250*20*1.e-6/3

        x_t = ippSeconds * (numpy.arange(1600) -1600 / 2.)

        x_t = numpy.linspace(x_t[0],x_t[-1],3200)

        x_freq = numpy.fft.fftfreq(1600,d=ippSeconds)
        x_freq = numpy.fft.fftshift(x_freq)
        '''
        # define a least squares function to optimize
        def minfunc(params):
            return sum((y-R_T_spc_fun(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))**2/1)#y**2)

        # fit

        popt_full = fmin(minfunc,[a,b,c,d],full_output=True)
        #print("nIter", popt_full[2])
        popt = popt_full[0]

        #return R_T_spc_fun(x_t,popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]
        return popt[0], popt[1], popt[2], popt[3]

    def run(self, dataOut, mode = 0, Hmin1 = None, Hmax1 = None, Hmin2 = None, Hmax2 = None, Dop = 'Shift'):

        pwcode = 1

        if dataOut.flagDecodeData:
            pwcode = numpy.sum(dataOut.code[0]**2)
        #normFactor = min(self.nFFTPoints,self.nProfiles)*self.nIncohInt*self.nCohInt*pwcode*self.windowOfFilter
        normFactor = dataOut.nProfiles * dataOut.nIncohInt * dataOut.nCohInt * pwcode * dataOut.windowOfFilter
        factor = normFactor
        z = dataOut.data_spc / factor
        z = numpy.where(numpy.isfinite(z), z, numpy.NAN)
        dataOut.power = numpy.average(z, axis=1)
        dataOut.powerdB = 10 * numpy.log10(dataOut.power)

        x = dataOut.getVelRange(0)

        dataOut.Oblique_params = numpy.ones((1,7,dataOut.nHeights))*numpy.NAN
        dataOut.Oblique_param_errors = numpy.ones((1,7,dataOut.nHeights))*numpy.NAN
        dataOut.dplr_2_u = numpy.ones((1,1,dataOut.nHeights))*numpy.NAN

        if mode == 6:
            dataOut.Oblique_params = numpy.ones((1,9,dataOut.nHeights))*numpy.NAN
        elif mode == 7:
            dataOut.Oblique_params = numpy.ones((1,13,dataOut.nHeights))*numpy.NAN
        elif mode == 8:
            dataOut.Oblique_params = numpy.ones((1,10,dataOut.nHeights))*numpy.NAN
        elif mode == 9:
            dataOut.Oblique_params = numpy.ones((1,11,dataOut.nHeights))*numpy.NAN
            dataOut.Oblique_param_errors = numpy.ones((1,9,dataOut.nHeights))*numpy.NAN
        elif mode == 11:
            dataOut.Oblique_params = numpy.ones((1,7,dataOut.nHeights))*numpy.NAN
            dataOut.Oblique_param_errors = numpy.ones((1,7,dataOut.nHeights))*numpy.NAN
        elif mode == 10: #150 km
            dataOut.Oblique_params = numpy.ones((1,4,dataOut.nHeights))*numpy.NAN
            dataOut.Oblique_param_errors = numpy.ones((1,4,dataOut.nHeights))*numpy.NAN
            dataOut.snr_log10 = numpy.ones((1,dataOut.nHeights))*numpy.NAN

        dataOut.VelRange = x



        #l1=range(22,36) #+62
        #l1=range(32,36)
        #l2=range(58,99) #+62

        #if Hmin1 == None or Hmax1 == None or Hmin2 == None or Hmax2 == None:

        minHei1 = 105.
        maxHei1 = 122.5
        maxHei1 = 130.5

        if mode == 10: #150 km
            minHei1 = 100
            maxHei1 = 100

        inda1 = numpy.where(dataOut.heightList >= minHei1)
        indb1 = numpy.where(dataOut.heightList <= maxHei1)

        minIndex1 = inda1[0][0]
        maxIndex1 = indb1[0][-1]

        minHei2 = 150.
        maxHei2 = 201.25
        maxHei2 = 225.3

        if mode == 10: #150 km
            minHei2 = 110
            maxHei2 = 165

        inda2 = numpy.where(dataOut.heightList >= minHei2)
        indb2 = numpy.where(dataOut.heightList <= maxHei2)

        minIndex2 = inda2[0][0]
        maxIndex2 = indb2[0][-1]

        l1=range(minIndex1,maxIndex1)
        l2=range(minIndex2,maxIndex2)

        if mode == 4:
            '''
            for ind in range(dataOut.nHeights):
                if(dataOut.heightList[ind]>=168 and dataOut.heightList[ind]<188):
                    try:
                        dataOut.Oblique_params[0,0,ind],dataOut.Oblique_params[0,1,ind],dataOut.Oblique_params[0,2,ind],dataOut.Oblique_params[0,3,ind],dataOut.Oblique_params[0,4,ind],dataOut.Oblique_params[0,5,ind],dataOut.Oblique_params[0,6,ind],dataOut.Oblique_param_errors[0,0,ind],dataOut.Oblique_param_errors[0,1,ind],dataOut.Oblique_param_errors[0,2,ind],dataOut.Oblique_param_errors[0,3,ind],dataOut.Oblique_param_errors[0,4,ind],dataOut.Oblique_param_errors[0,5,ind],dataOut.Oblique_param_errors[0,6,ind] = self.DH_mode(dataOut.data_spc[0,:,ind],dataOut.VelRange)
                    except:
                        pass
                        '''
            for ind in itertools.chain(l1, l2):

                try:
                    dataOut.Oblique_params[0,0,ind],dataOut.Oblique_params[0,1,ind],dataOut.Oblique_params[0,2,ind],dataOut.Oblique_params[0,3,ind],dataOut.Oblique_params[0,4,ind],dataOut.Oblique_params[0,5,ind],dataOut.Oblique_params[0,6,ind],dataOut.Oblique_param_errors[0,0,ind],dataOut.Oblique_param_errors[0,1,ind],dataOut.Oblique_param_errors[0,2,ind],dataOut.Oblique_param_errors[0,3,ind],dataOut.Oblique_param_errors[0,4,ind],dataOut.Oblique_param_errors[0,5,ind],dataOut.Oblique_param_errors[0,6,ind] = self.DH_mode(dataOut.data_spc[0,:,ind],dataOut.VelRange)
                    dataOut.dplr_2_u[0,0,ind] = dataOut.Oblique_params[0,4,ind]/numpy.sin(numpy.arccos(102/dataOut.heightList[ind]))
                except:
                    pass

        else:
            #print("After: ", dataOut.data_snr[0])
            #######import matplotlib.pyplot as plt
            #######plt.plot(dataOut.data_snr[0],dataOut.heightList,marker='*',linestyle='--')
            #######plt.show()
            #print("l1: ", dataOut.heightList[l1])
            #print("l2: ", dataOut.heightList[l2])
            for hei in itertools.chain(l1, l2):
            #for hei in range(79,81):
                #if numpy.isnan(dataOut.data_snr[0,hei]) or numpy.isnan(numpy.log10(dataOut.data_snr[0,hei])):
                if numpy.isnan(dataOut.snl[0,hei]):# or dataOut.snl[0,hei]<.0:

                    continue #Avoids the analysis when there is only noise

                try:
                    spc = dataOut.data_spc[0,:,hei]

                    if mode == 6: #Skew Weighted Bounded
                        dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_params[0,7,hei],dataOut.Oblique_params[0,8,hei] = self.Double_Gauss_Skew_fit_weight_bound_no_inputs(spc,x)
                        dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,8,hei]/numpy.sin(numpy.arccos(100./dataOut.heightList[hei]))

                    elif mode == 7: #Triple Skew Weighted Bounded
                        dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_params[0,7,hei],dataOut.Oblique_params[0,8,hei],dataOut.Oblique_params[0,9,hei],dataOut.Oblique_params[0,10,hei],dataOut.Oblique_params[0,11,hei],dataOut.Oblique_params[0,12,hei] = self.Triple_Gauss_Skew_fit_weight_bound_no_inputs(spc,x)
                        dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,12,hei]/numpy.sin(numpy.arccos(100./dataOut.heightList[hei]))

                    elif mode == 8: #Double Skewed Weighted Bounded with inputs
                        a1, b1, c1, a2, b2, c2, k2, d, dopp = self.Double_Gauss_Skew_fit_weight_bound_no_inputs(spc,x)
                        dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_params[0,7,hei],dataOut.Oblique_params[0,8,hei],dataOut.Oblique_params[0,9,hei] = self.Double_Gauss_Skew_fit_weight_bound_no_inputs(spc,x, a1, b1, c1, a2, b2, c2, k2, d)
                        dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,9,hei]/numpy.sin(numpy.arccos(100./dataOut.heightList[hei]))

                    elif mode == 9: #Double Skewed Weighted Bounded no inputs
                        #if numpy.max(spc) <= 0:
                        from scipy.signal import medfilt
                        spcm = medfilt(spc,11)
                        if x[numpy.argmax(spcm)] <= 0:
                            #print("EEJ", dataOut.heightList[hei], hei)
                            #if hei != 70:
                                #continue
                            #else:
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_params[0,7,hei],dataOut.Oblique_params[0,8,hei],dataOut.Oblique_params[0,9,hei],dataOut.Oblique_params[0,10,hei],dataOut.Oblique_param_errors[0,:,hei] = self.Double_Gauss_Double_Skew_fit_weight_bound_no_inputs(spcm,x,dataOut.nIncohInt,dataOut.heightList[hei])
                            #if dataOut.Oblique_params[0,-2,hei] < -500 or dataOut.Oblique_params[0,-2,hei] > 500 or dataOut.Oblique_params[0,-1,hei] < -500 or dataOut.Oblique_params[0,-1,hei] > 500:
                            #    dataOut.Oblique_params[0,:,hei] *= numpy.NAN
                            dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,10,hei]/numpy.sin(numpy.arccos(100./dataOut.heightList[hei]))

                        else:
                            #print("CEEJ")
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_params[0,7,hei],dataOut.Oblique_params[0,8,hei],dataOut.Oblique_params[0,9,hei],dataOut.Oblique_params[0,10,hei],dataOut.Oblique_param_errors[0,:,hei] = self.CEEJ_Skew_fit_weight_bound_no_inputs(spcm,x,dataOut.nIncohInt)
                            #if dataOut.Oblique_params[0,-2,hei] < -500 or dataOut.Oblique_params[0,-2,hei] > 500 or dataOut.Oblique_params[0,-1,hei] < -500 or dataOut.Oblique_params[0,-1,hei] > 500:
                            #    dataOut.Oblique_params[0,:,hei] *= numpy.NAN
                            dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,10,hei]/numpy.sin(numpy.arccos(100./dataOut.heightList[hei]))
                    elif mode == 11: #Double Weighted Bounded no inputs
                        #if numpy.max(spc) <= 0:
                        from scipy.signal import medfilt
                        spcm = medfilt(spc,11)

                        if x[numpy.argmax(spcm)] <= 0:
                            #print("EEJ")
                            #print("EEJ",dataOut.heightList[hei])
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_param_errors[0,:,hei] = self.Double_Gauss_fit_weight_bound_no_inputs(spc,x,dataOut.nIncohInt)
                            #if dataOut.Oblique_params[0,-2,hei] < -500 or dataOut.Oblique_params[0,-2,hei] > 500 or dataOut.Oblique_params[0,-1,hei] < -500 or dataOut.Oblique_params[0,-1,hei] > 500:
                            #    dataOut.Oblique_params[0,:,hei] *= numpy.NAN
                        else:
                            #print("CEEJ",dataOut.heightList[hei])
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_param_errors[0,:,hei] = self.CEEJ_fit_weight_bound_no_inputs(spc,x,dataOut.nIncohInt)

                    elif mode == 10: #150km
                        dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_param_errors[0,:,hei] = self.Simple_fit_bound(spc,x,dataOut.nIncohInt)
                        snr = (dataOut.power[0,hei]*factor - dataOut.Oblique_params[0,3,hei])/dataOut.Oblique_params[0,3,hei]
                        dataOut.snr_log10[0,hei] = numpy.log10(snr)

                    else:
                        spc_fit, A1, B1, C1, D1 = self.Gauss_fit_2(spc,x,'first')

                        spc_diff = spc - spc_fit
                        spc_diff[spc_diff < 0] = 0

                        spc_fit_diff, A2, B2, C2, D2 = self.Gauss_fit_2(spc_diff,x,'second')

                        D = (D1+D2)

                        if mode == 0: #Double Fit
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei],dataOut.Oblique_param_errors[0,0,hei],dataOut.Oblique_param_errors[0,1,hei],dataOut.Oblique_param_errors[0,2,hei],dataOut.Oblique_param_errors[0,3,hei],dataOut.Oblique_param_errors[0,4,hei],dataOut.Oblique_param_errors[0,5,hei],dataOut.Oblique_param_errors[0,6,hei] = self.Double_Gauss_fit_2(spc,x,A1,B1,C1,A2,B2,C2,D)
                        #spc_double_fit,dataOut.Oblique_params = self.Double_Gauss_fit(spc,x,A1,B1,C1,A2,B2,C2,D)

                        elif mode == 1: #Double Fit Windowed
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei] = self.windowing_double(spc,dataOut.getFreqRange(0),A1,B1,C1,A2,B2,C2,D)

                        elif mode == 2: #Double Fit Weight
                            dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei] = self.Double_Gauss_fit_weight(spc,x,A1,B1,C1,A2,B2,C2,D)

                        elif mode == 3: #Simple Fit
                            dataOut.Oblique_params[0,0,hei] = A1
                            dataOut.Oblique_params[0,1,hei] = B1
                            dataOut.Oblique_params[0,2,hei] = C1
                            dataOut.Oblique_params[0,3,hei] = A2
                            dataOut.Oblique_params[0,4,hei] = B2
                            dataOut.Oblique_params[0,5,hei] = C2
                            dataOut.Oblique_params[0,6,hei] = D

                        elif mode == 5: #Triple Fit Weight
                            if hei in l1:
                                dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei] = self.duo_Marco(spc,x,A1,B1,C1,A2,B2,C2,D)
                                dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,4,hei]/numpy.sin(numpy.arccos(102/dataOut.heightList[hei]))
                                #print(dataOut.Oblique_params[0,0,hei])
                                #print(dataOut.dplr_2_u[0,0,hei])
                            else:
                                dataOut.Oblique_params[0,0,hei],dataOut.Oblique_params[0,1,hei],dataOut.Oblique_params[0,2,hei],dataOut.Oblique_params[0,3,hei],dataOut.Oblique_params[0,4,hei],dataOut.Oblique_params[0,5,hei],dataOut.Oblique_params[0,6,hei] = self.Double_Gauss_fit_weight(spc,x,A1,B1,C1,A2,B2,C2,D)
                                dataOut.dplr_2_u[0,0,hei] = dataOut.Oblique_params[0,4,hei]/numpy.sin(numpy.arccos(102/dataOut.heightList[hei]))


                except:
                    ###dataOut.Oblique_params[0,:,hei] = dataOut.Oblique_params[0,:,hei]*numpy.NAN
                    pass

        #exit(1)
        dataOut.paramInterval = dataOut.nProfiles*dataOut.nCohInt*dataOut.ippSeconds
        dataOut.lat=-11.95
        dataOut.lon=-76.87
        '''
        dataOut.Oblique_params = numpy.where(dataOut.Oblique_params<-700, numpy.nan, dop_t1)
        dataOut.Oblique_params = numpy.where(dataOut.Oblique_params<+700, numpy.nan, dop_t1)
        Aquí debo exceptuar las amplitudes
        '''
        if mode == 9: #Double Skew Gaussian
            #dataOut.Dop_EEJ_T1 = dataOut.Oblique_params[:,-2,:] #Pos[Max_value]
            #dataOut.Dop_EEJ_T1 = dataOut.Oblique_params[:,1,:] #Shift
            dataOut.Spec_W_T1 = dataOut.Oblique_params[:,2,:]
            #dataOut.Dop_EEJ_T2 = dataOut.Oblique_params[:,-1,:] #Pos[Max_value]
            #dataOut.Dop_EEJ_T2 = dataOut.Oblique_params[:,5,:] #Shift
            dataOut.Spec_W_T2 = dataOut.Oblique_params[:,6,:]
            if Dop == 'Shift':
                dataOut.Dop_EEJ_T1 = dataOut.Oblique_params[:,1,:] #Shift
                dataOut.Dop_EEJ_T2 = dataOut.Oblique_params[:,5,:] #Shift
            elif Dop == 'Max':
                dataOut.Dop_EEJ_T1 = dataOut.Oblique_params[:,-2,:] #Pos[Max_value]
                dataOut.Dop_EEJ_T2 = dataOut.Oblique_params[:,-1,:] #Pos[Max_value]

            dataOut.Err_Dop_EEJ_T1 = dataOut.Oblique_param_errors[:,1,:] #En realidad este es el error?
            dataOut.Err_Spec_W_T1 = dataOut.Oblique_param_errors[:,2,:]
            dataOut.Err_Dop_EEJ_T2 = dataOut.Oblique_param_errors[:,5,:] #En realidad este es el error?
            dataOut.Err_Spec_W_T2 = dataOut.Oblique_param_errors[:,6,:]

        elif mode == 11: #Double Gaussian
            dataOut.Dop_EEJ_T1 = dataOut.Oblique_params[:,1,:]
            dataOut.Spec_W_T1 = dataOut.Oblique_params[:,2,:]
            dataOut.Dop_EEJ_T2 = dataOut.Oblique_params[:,4,:]
            dataOut.Spec_W_T2 = dataOut.Oblique_params[:,5,:]

            dataOut.Err_Dop_EEJ_T1 = dataOut.Oblique_param_errors[:,1,:]
            dataOut.Err_Spec_W_T1 = dataOut.Oblique_param_errors[:,2,:]
            dataOut.Err_Dop_EEJ_T2 = dataOut.Oblique_param_errors[:,4,:]
            dataOut.Err_Spec_W_T2 = dataOut.Oblique_param_errors[:,5,:]

        #print("Before: ", dataOut.Dop_EEJ_T2)
        dataOut.Spec_W_T1 = self.clean_outliers(dataOut.Spec_W_T1)
        dataOut.Spec_W_T2 = self.clean_outliers(dataOut.Spec_W_T2)
        dataOut.Dop_EEJ_T1 = self.clean_outliers(dataOut.Dop_EEJ_T1)
        dataOut.Dop_EEJ_T2 = self.clean_outliers(dataOut.Dop_EEJ_T2)
        #print("After: ", dataOut.Dop_EEJ_T2)
        dataOut.Err_Spec_W_T1 = self.clean_outliers(dataOut.Err_Spec_W_T1)
        dataOut.Err_Spec_W_T2 = self.clean_outliers(dataOut.Err_Spec_W_T2)
        dataOut.Err_Dop_EEJ_T1 = self.clean_outliers(dataOut.Err_Dop_EEJ_T1)
        dataOut.Err_Dop_EEJ_T2 = self.clean_outliers(dataOut.Err_Dop_EEJ_T2)
        #print("Before data_snr: ", dataOut.data_snr)
        #dataOut.data_snr = numpy.where(numpy.isnan(dataOut.Dop_EEJ_T1), numpy.nan, dataOut.data_snr)
        dataOut.snl = numpy.where(numpy.isnan(dataOut.Dop_EEJ_T1), numpy.nan, dataOut.snl)

        #print("After data_snr: ", dataOut.data_snr)
        dataOut.mode = mode
        dataOut.flagNoData = numpy.all(numpy.isnan(dataOut.Dop_EEJ_T1)) #Si todos los valores son NaN no se prosigue
        ###dataOut.flagNoData = False #Descomentar solo para ploteo sino mantener comentado (para guardado)

        return dataOut

class Gaussian_Windowed(Operation):
    '''
    Written by R. Flores
    '''
    def __init__(self):
        Operation.__init__(self)

    def windowing_single(self,spc,x,A,B,C,D,nFFTPoints):
        from scipy.optimize import curve_fit,fmin

        def gaussian(x, a, b, c, d):
            val = a * numpy.exp(-(x - b)**2 / (2*c**2)) + d
            return val

        def R_gaussian(x, a, b, c):
                N = int(numpy.shape(x)[0])
                val = a * numpy.exp(-((x)*c*2*2*numpy.pi)**2 / (2))* numpy.exp(1.j*b*x*4*numpy.pi)
                return val

        def T(x,N):
            T = 1-abs(x)/N
            return T

        def R_T_spc_fun(x, a, b, c, d, nFFTPoints):

            N = int(numpy.shape(x)[0])

            x_max = x[-1]

            x_pos = x[nFFTPoints:]
            x_neg = x[:nFFTPoints]
            #print([int(nFFTPoints/2))
            #print("x: ", x)
            #print("x_neg: ", x_neg)
            #print("x_pos: ", x_pos)


            R_T_neg_1 = R_gaussian(x, a, b, c)[:nFFTPoints]*T(x_neg,-x[0])
            R_T_pos_1 = R_gaussian(x, a, b, c)[nFFTPoints:]*T(x_pos,x[-1])
            #print(T(x_pos,x[-1]),x_pos,x[-1])
            #print(R_T_neg_1.shape,R_T_pos_1.shape)
            R_T_sum_1 = R_T_pos_1 + R_T_neg_1
            R_T_spc_1 = numpy.fft.fft(R_T_sum_1).real
            R_T_spc_1 = numpy.fft.fftshift(R_T_spc_1)
            max_val_1 = numpy.max(R_T_spc_1)
            R_T_spc_1 = R_T_spc_1*a/max_val_1

            R_T_d = d*numpy.fft.fftshift(signal.unit_impulse(N))
            R_T_d_neg = R_T_d[:nFFTPoints]*T(x_neg,-x[0])
            R_T_d_pos = R_T_d[nFFTPoints:]*T(x_pos,x[-1])
            R_T_d_sum = R_T_d_pos + R_T_d_neg
            R_T_spc_3 = numpy.fft.fft(R_T_d_sum).real
            R_T_spc_3 = numpy.fft.fftshift(R_T_spc_3)

            R_T_final = R_T_spc_1 + R_T_spc_3

            return R_T_final

        y = spc#gaussian(x, a, meanY, sigmaY) + a*0.1*numpy.random.normal(0, 1, size=len(x))

        from scipy.stats import norm
        mean,std=norm.fit(spc)

        # estimate starting values from the data
        a = A
        b = B
        c = C#numpy.std(spc)
        d = D
        #'''
        #ippSeconds = 250*20*1.e-6/3

        #x_t = ippSeconds * (numpy.arange(nFFTPoints) - nFFTPoints / 2.)

        #x_t = numpy.linspace(x_t[0],x_t[-1],3200)
        #print("x_t: ", x_t)
        #print("nFFTPoints: ", nFFTPoints)
        x_vel = numpy.linspace(x[0],x[-1],int(2*nFFTPoints))
        #print("x_vel: ", x_vel)
        #x_freq = numpy.fft.fftfreq(1600,d=ippSeconds)
        #x_freq = numpy.fft.fftshift(x_freq)
        #'''
        # define a least squares function to optimize
        def minfunc(params):
            #print("y.shape: ", numpy.shape(y))
            return sum((y-R_T_spc_fun(x_vel,params[0],params[1],params[2],params[3],nFFTPoints))**2/1)#y**2)

        # fit

        popt_full = fmin(minfunc,[a,b,c,d], disp=False)
        #print("nIter", popt_full[2])
        popt = popt_full#[0]

        fun = gaussian(x, popt[0], popt[1], popt[2], popt[3])

        #return R_T_spc_fun(x_t,popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]
        return fun, popt[0], popt[1], popt[2], popt[3]

    def run(self, dataOut):

        from scipy.signal import medfilt
        import matplotlib.pyplot as plt
        dataOut.moments = numpy.ones((dataOut.nChannels,4,dataOut.nHeights))*numpy.NAN
        dataOut.VelRange = dataOut.getVelRange(0)
        for nChannel in range(dataOut.nChannels):
            for hei in range(dataOut.heightList.shape[0]):
                #print("ipp: ", dataOut.ippSeconds)
                spc = numpy.copy(dataOut.data_spc[nChannel,:,hei])

                #print(VelRange)
                #print(dataOut.getFreqRange(64))
                spcm = medfilt(spc,11)
                spc_max = numpy.max(spcm)
                dop1_x0 = dataOut.VelRange[numpy.argmax(spcm)]
                D = numpy.min(spcm)

                fun, A, B, C, D = self.windowing_single(spc,dataOut.VelRange,spc_max,dop1_x0,abs(dop1_x0),D,dataOut.nFFTPoints)
                dataOut.moments[nChannel,0,hei] = A
                dataOut.moments[nChannel,1,hei] = B
                dataOut.moments[nChannel,2,hei] = C
                dataOut.moments[nChannel,3,hei] = D
                '''
                plt.figure()
                plt.plot(VelRange,spc,marker='*',linestyle='')
                plt.plot(VelRange,fun)
                plt.title(dataOut.heightList[hei])
                plt.show()
                '''

        return dataOut

class PrecipitationProc(Operation):

    '''
         Operator that estimates Reflectivity factor (Z), and estimates rainfall Rate (R)

         Input:
            self.dataOut.data_pre    :    SelfSpectra

         Output:

            self.dataOut.data_output :    Reflectivity factor, rainfall Rate


         Parameters affected:
    '''

    def __init__(self):
        Operation.__init__(self)
        self.i=0

    def run(self, dataOut, radar=None, Pt=5000, Gt=295.1209, Gr=70.7945, Lambda=0.6741, aL=2.5118,
            tauW=4e-06, ThetaT=0.1656317, ThetaR=0.36774087, Km2 = 0.93, Altitude=3350, SNRdBlimit=-30,
            channel=None):

        # print ('Entering PrecepitationProc ... ')

        if radar == "MIRA35C" :

            self.spc = dataOut.data_pre[0].copy()
            self.Num_Hei = self.spc.shape[2]
            self.Num_Bin = self.spc.shape[1]
            self.Num_Chn = self.spc.shape[0]
            Ze = self.dBZeMODE2(dataOut)

        else:

            self.spc = dataOut.data_pre[0].copy()

            #NOTA SE DEBE REMOVER EL RANGO DEL PULSO TX
            self.spc[:,:,0:7]= numpy.NaN

            self.Num_Hei = self.spc.shape[2]
            self.Num_Bin = self.spc.shape[1]
            self.Num_Chn = self.spc.shape[0]

            VelRange = dataOut.spc_range[2]

            ''' Se obtiene la constante del RADAR '''

            self.Pt = Pt
            self.Gt = Gt
            self.Gr = Gr
            self.Lambda = Lambda
            self.aL = aL
            self.tauW = tauW
            self.ThetaT = ThetaT 
            self.ThetaR = ThetaR
            self.GSys = 10**(36.63/10) # Ganancia de los LNA 36.63 dB
            self.lt = 10**(1.67/10) # Perdida en cables Tx 1.67 dB
            self.lr = 10**(5.73/10) # Perdida en cables Rx 5.73 dB

            Numerator = ( (4*numpy.pi)**3 * aL**2 * 16 * numpy.log(2) )
            Denominator = ( Pt * Gt * Gr * Lambda**2 * SPEED_OF_LIGHT * tauW * numpy.pi * ThetaT * ThetaR)
            RadarConstant = 10e-26 * Numerator / Denominator #
            ExpConstant = 10**(40/10) #Constante Experimental

            SignalPower = numpy.zeros([self.Num_Chn,self.Num_Bin,self.Num_Hei])
            for i in range(self.Num_Chn):
                SignalPower[i,:,:] = self.spc[i,:,:] - dataOut.noise[i]
                SignalPower[numpy.where(SignalPower < 0)] = 1e-20

            if channel is None:
                SPCmean = numpy.mean(SignalPower, 0)
            else:
                SPCmean = SignalPower[channel]
            Pr = SPCmean[:,:]/dataOut.normFactor

            # Declaring auxiliary variables
            Range = dataOut.heightList*1000. #Range in m
            # replicate the heightlist to obtain a matrix [Num_Bin,Num_Hei]
            rMtrx = numpy.transpose(numpy.transpose([dataOut.heightList*1000.] * self.Num_Bin))
            zMtrx = rMtrx+Altitude
            # replicate the VelRange to obtain a matrix [Num_Bin,Num_Hei]
            VelMtrx = numpy.transpose(numpy.tile(VelRange[:-1], (self.Num_Hei,1)))

            # height dependence to air density Foote and Du Toit (1969)
            delv_z = 1 + 3.68e-5 * zMtrx + 1.71e-9 * zMtrx**2
            VMtrx = VelMtrx / delv_z #Normalized velocity
            VMtrx[numpy.where(VMtrx> 9.6)] = numpy.NaN
            # Diameter is related to the fall speed of falling drops
            D_Vz = -1.667 * numpy.log( 0.9369 - 0.097087 * VMtrx ) # D in [mm]
            # Only valid for D>= 0.16 mm
            D_Vz[numpy.where(D_Vz < 0.16)] = numpy.NaN

            #Calculate Radar Reflectivity ETAn
            ETAn = (RadarConstant *ExpConstant) * Pr * rMtrx**2  #Reflectivity (ETA)
            ETAd = ETAn * 6.18 * exp( -0.6 * D_Vz ) * delv_z
            # Radar Cross Section
            sigmaD = Km2 * (D_Vz * 1e-3 )**6 * numpy.pi**5 / Lambda**4 
            # Drop Size Distribution
            DSD = ETAn / sigmaD
            # Equivalente Reflectivy
            Ze_eqn = numpy.nansum( DSD * D_Vz**6 ,axis=0)
            Ze_org = numpy.nansum(ETAn * Lambda**4, axis=0) / (1e-18*numpy.pi**5 * Km2) # [mm^6 /m^3]
            # RainFall Rate
            RR = 0.0006*numpy.pi * numpy.nansum( D_Vz**3 * DSD * VelMtrx ,0) #mm/hr

        # Censoring the data
        # Removing data with SNRth < 0dB se debe considerar el SNR por canal
        SNRth = 10**(SNRdBlimit/10) #-30dB
        novalid = numpy.where((dataOut.data_snr[0,:] <SNRth) | (dataOut.data_snr[1,:] <SNRth) | (dataOut.data_snr[2,:] <SNRth)) # AND condition. Maybe OR condition better
        W = numpy.nanmean(dataOut.data_dop,0)
        W[novalid] = numpy.NaN
        Ze_org[novalid] = numpy.NaN
        RR[novalid] = numpy.NaN

        dataOut.data_output = RR[8]
        dataOut.data_param = numpy.ones([3,self.Num_Hei])
        dataOut.channelList = [0,1,2]
        
        dataOut.data_param[0]=10*numpy.log10(Ze_org)
        dataOut.data_param[1]=-W
        dataOut.data_param[2]=RR

        # print ('Leaving PrecepitationProc ... ')
        return dataOut

    def dBZeMODE2(self, dataOut): #    Processing for MIRA35C

        NPW = dataOut.NPW
        COFA = dataOut.COFA

        SNR = numpy.array([self.spc[0,:,:] / NPW[0]]) #, self.spc[1,:,:] / NPW[1]])
        RadarConst = dataOut.RadarConst
        #frequency = 34.85*10**9

        ETA = numpy.zeros(([self.Num_Chn ,self.Num_Hei]))
        data_output = numpy.ones([self.Num_Chn , self.Num_Hei])*numpy.NaN

        ETA = numpy.sum(SNR,1)

        ETA = numpy.where(ETA != 0. , ETA, numpy.NaN)

        Ze = numpy.ones([self.Num_Chn, self.Num_Hei] )

        for r in range(self.Num_Hei):

            Ze[0,r] =  ( ETA[0,r] ) * COFA[0,r][0] * RadarConst * ((r/5000.)**2)
            #Ze[1,r] =  ( ETA[1,r] ) * COFA[1,r][0] * RadarConst * ((r/5000.)**2)

        return Ze

#     def GetRadarConstant(self):
#
#         """
#         Constants:
#
#         Pt:     Transmission Power               dB        5kW                5000
#         Gt:     Transmission Gain                dB        24.7 dB            295.1209
#         Gr:     Reception Gain                   dB        18.5 dB            70.7945
#         Lambda: Wavelenght                       m         0.6741 m           0.6741
#         aL:     Attenuation loses                dB        4dB                2.5118
#         tauW:   Width of transmission pulse      s         4us                4e-6
#         ThetaT: Transmission antenna bean angle  rad       0.1656317 rad      0.1656317
#         ThetaR: Reception antenna beam angle     rad       0.36774087 rad     0.36774087
#
#         """
#
#         Numerator = ( (4*numpy.pi)**3 * aL**2 * 16 * numpy.log(2) )
#         Denominator = ( Pt * Gt * Gr * Lambda**2 * SPEED_OF_LIGHT * TauW * numpy.pi * ThetaT * TheraR)
#         RadarConstant =  Numerator / Denominator
#
#         return RadarConstant


class FullSpectralAnalysis(Operation):

    """
        Function that implements Full Spectral Analysis technique.

        Input:
            self.dataOut.data_pre    :    SelfSpectra and CrossSpectra data
            self.dataOut.groupList   :    Pairlist of channels
            self.dataOut.ChanDist    :    Physical distance between receivers


        Output:

            self.dataOut.data_output :    Zonal wind, Meridional wind, and Vertical wind


        Parameters affected:    Winds, height range, SNR

    """
    def run(self, dataOut, Xi01=None, Xi02=None, Xi12=None, Eta01=None, Eta02=None, Eta12=None, SNRdBlimit=-30, 
        minheight=None, maxheight=None, NegativeLimit=None, PositiveLimit=None):

        spc = dataOut.data_pre[0].copy()
        cspc = dataOut.data_pre[1]
        nHeights = spc.shape[2]

        # first_height = 0.75 #km (ref: data header 20170822)
        # resolution_height = 0.075 #km
        '''
            finding height range. check this when radar parameters are changed!
        '''
        if maxheight is not None:
            # range_max = math.ceil((maxheight - first_height) / resolution_height) # theoretical
            range_max = math.ceil(13.26 * maxheight - 3) # empirical, works better
        else:
            range_max = nHeights
        if minheight is not None:
            # range_min = int((minheight - first_height) / resolution_height) # theoretical
            range_min = int(13.26 * minheight - 5) # empirical, works better
            if range_min < 0:
                range_min = 0
        else:
            range_min = 0

        pairsList = dataOut.groupList
        if dataOut.ChanDist is not None :
            ChanDist = dataOut.ChanDist
        else:
            ChanDist = numpy.array([[Xi01, Eta01],[Xi02,Eta02],[Xi12,Eta12]])

        # 4 variables: zonal, meridional, vertical, and average SNR
        data_param = numpy.zeros([4,nHeights]) * numpy.NaN
        velocityX = numpy.zeros([nHeights]) * numpy.NaN
        velocityY = numpy.zeros([nHeights]) * numpy.NaN
        velocityZ = numpy.zeros([nHeights]) * numpy.NaN

        dbSNR = 10*numpy.log10(numpy.average(dataOut.data_snr,0))

        '''***********************************************WIND ESTIMATION**************************************'''
        for Height in range(nHeights):

            if Height >= range_min and Height < range_max:
                # error_code will be useful in future analysis
                [Vzon,Vmer,Vver, error_code] = self.WindEstimation(spc[:,:,Height], cspc[:,:,Height], pairsList, 
                    ChanDist, Height, dataOut.noise, dataOut.spc_range, dbSNR[Height], SNRdBlimit, NegativeLimit, PositiveLimit,dataOut.frequency)

            if abs(Vzon) < 100. and abs(Vmer) < 100.:
                velocityX[Height] = Vzon
                velocityY[Height] = -Vmer
                velocityZ[Height] = Vver
        
        # Censoring data with SNR threshold
        dbSNR [dbSNR < SNRdBlimit] = numpy.NaN

        data_param[0] = velocityX
        data_param[1] = velocityY
        data_param[2] = velocityZ
        data_param[3] = dbSNR
        dataOut.data_param = data_param
        return dataOut

    def moving_average(self,x, N=2):
        """ convolution for smoothenig data. note that last N-1 values are convolution with zeroes """
        return numpy.convolve(x, numpy.ones((N,))/N)[(N-1):]

    def gaus(self,xSamples,Amp,Mu,Sigma):
        return Amp * numpy.exp(-0.5*((xSamples - Mu)/Sigma)**2)

    def Moments(self, ySamples, xSamples):
        Power = numpy.nanmean(ySamples)                                 # Power, 0th Moment
        yNorm = ySamples / numpy.nansum(ySamples)
        RadVel = numpy.nansum(xSamples * yNorm)                         # Radial Velocity, 1st Moment
        Sigma2 = numpy.nansum(yNorm * (xSamples - RadVel)**2)      # Spectral Width, 2nd Moment
        StdDev = numpy.sqrt(numpy.abs(Sigma2))                                            # Desv. Estandar, Ancho espectral
        return numpy.array([Power,RadVel,StdDev])

    def StopWindEstimation(self, error_code):
        Vzon = numpy.NaN
        Vmer = numpy.NaN
        Vver = numpy.NaN
        return Vzon, Vmer, Vver, error_code

    def AntiAliasing(self, interval, maxstep):
        """
            function to prevent errors from aliased values when computing phaseslope
        """
        antialiased = numpy.zeros(len(interval))
        copyinterval = interval.copy()

        antialiased[0] = copyinterval[0]

        for i in range(1,len(antialiased)):
            step = interval[i] - interval[i-1]
            if step > maxstep:
                copyinterval -= 2*numpy.pi
                antialiased[i] = copyinterval[i]
            elif step < maxstep*(-1):
                copyinterval += 2*numpy.pi
                antialiased[i] = copyinterval[i]
            else:
                antialiased[i] = copyinterval[i].copy()

        return antialiased

    def WindEstimation(self, spc, cspc, pairsList, ChanDist, Height, noise, AbbsisaRange, dbSNR, SNRlimit, NegativeLimit, PositiveLimit, radfreq):
        """
            Function that Calculates Zonal, Meridional and Vertical wind velocities.
            Initial Version by E. Bocanegra updated by J. Zibell until Nov. 2019.

            Input:
                spc, cspc       : self spectra and cross spectra data. In Briggs notation something like S_i*(S_i)_conj, (S_j)_conj respectively.
                pairsList       : Pairlist of channels
                ChanDist        : array of xi_ij and eta_ij
                Height          : height at which data is processed
                noise           : noise in [channels] format for specific height
                Abbsisarange    : range of the frequencies or velocities
                dbSNR, SNRlimit : signal to noise ratio in db, lower limit

            Output:
                Vzon, Vmer, Vver         : wind velocities
                error_code               : int that states where code is terminated

                    0 : no error detected
                    1 : Gaussian of mean spc exceeds widthlimit
                    2 : no Gaussian of mean spc found
                    3 : SNR to low or velocity to high -> prec. e.g.
                    4 : at least one Gaussian of cspc exceeds widthlimit
                    5 : zero out of three cspc Gaussian fits converged
                    6 : phase slope fit could not be found
                    7 : arrays used to fit phase have different length
                    8 : frequency range is either too short (len <= 5) or very long (> 30% of cspc)

        """

        error_code = 0

        nChan = spc.shape[0]
        nProf = spc.shape[1]
        nPair = cspc.shape[0]

        SPC_Samples = numpy.zeros([nChan, nProf])           # for normalized spc values for one height
        CSPC_Samples = numpy.zeros([nPair, nProf], dtype=numpy.complex_)      # for normalized cspc values
        phase = numpy.zeros([nPair, nProf])                 # phase between channels
        PhaseSlope = numpy.zeros(nPair)                     # slope of the phases, channelwise
        PhaseInter = numpy.zeros(nPair)                     # intercept to the slope of the phases, channelwise
        xFrec = AbbsisaRange[0][:-1]                        # frequency range
        xVel = AbbsisaRange[2][:-1]                         # velocity range
        xSamples = xFrec                                    # the frequency range is taken
        delta_x = xSamples[1] - xSamples[0]                 # delta_f or delta_x

        # only consider velocities with in NegativeLimit and PositiveLimit 
        if (NegativeLimit is None):
            NegativeLimit = numpy.min(xVel)
        if (PositiveLimit is None):
            PositiveLimit = numpy.max(xVel)
        xvalid = numpy.where((xVel > NegativeLimit) & (xVel < PositiveLimit))
        xSamples_zoom = xSamples[xvalid]

        '''Getting Eij and Nij'''
        Xi01, Xi02, Xi12 = ChanDist[:,0]
        Eta01, Eta02, Eta12 = ChanDist[:,1]

        # spwd limit - updated by D. Scipión 30.03.2021
        widthlimit = 10
        '''************************* SPC is normalized ********************************'''
        spc_norm = spc.copy() 
        # For each channel
        for i in range(nChan):
            spc_sub = spc_norm[i,:] - noise[i]  # only the signal power
            SPC_Samples[i] = spc_sub / (numpy.nansum(spc_sub) * delta_x)

        '''********************** FITTING MEAN SPC GAUSSIAN **********************'''

        """ the gaussian of the mean: first subtract noise, then normalize. this is legal because
            you only fit the curve and don't need the absolute value of height for calculation,
            only for estimation of width. for normalization of cross spectra, you need initial,
            unnormalized self-spectra With noise.

            Technically, you don't even need to normalize the self-spectra, as you only need the
            width of the peak. However, it was left this way. Note that the normalization has a flaw:
            due to subtraction of the noise, some values are below zero. Raw "spc" values should be
            >= 0, as it is the modulus squared of the signals (complex * it's conjugate)
        """
        # initial conditions
        popt = [1e-10,0,1e-10] 
        # Spectra average
        SPCMean = numpy.average(SPC_Samples,0) 
        # Moments in frequency
        SPCMoments = self.Moments(SPCMean[xvalid], xSamples_zoom)

        # Gauss Fit SPC in frequency domain
        if dbSNR > SNRlimit: # only if SNR > SNRth
            try:
                popt,pcov = curve_fit(self.gaus,xSamples_zoom,SPCMean[xvalid],p0=SPCMoments)
                if popt[2] <= 0 or popt[2] > widthlimit: # CONDITION
                    return self.StopWindEstimation(error_code = 1)
                FitGauss = self.gaus(xSamples_zoom,*popt)
            except :#RuntimeError:
                return self.StopWindEstimation(error_code = 2)
        else:
            return self.StopWindEstimation(error_code = 3)

        '''***************************** CSPC Normalization *************************
                The Spc spectra are used to normalize the crossspectra. Peaks from precipitation
                influence the norm which is not desired. First, a range is identified where the
                wind peak is estimated -> sum_wind is sum of those frequencies. Next, the area
                around it gets cut off and values replaced by mean determined by the boundary
                data -> sum_noise (spc is not normalized here, thats why the noise is important)

                The sums are then added and multiplied by range/datapoints, because you need
                an integral and not a sum for normalization.

                A norm is found according to Briggs 92.
        '''
        # for each pair
        for i in range(nPair):
            cspc_norm = cspc[i,:].copy()
            chan_index0 = pairsList[i][0]
            chan_index1 = pairsList[i][1]
            CSPC_Samples[i] = cspc_norm / (numpy.sqrt(numpy.nansum(spc_norm[chan_index0])*numpy.nansum(spc_norm[chan_index1])) * delta_x)
            phase[i] = numpy.arctan2(CSPC_Samples[i].imag, CSPC_Samples[i].real)

        CSPCmoments = numpy.vstack([self.Moments(numpy.abs(CSPC_Samples[0,xvalid]), xSamples_zoom),
                                    self.Moments(numpy.abs(CSPC_Samples[1,xvalid]), xSamples_zoom),
                                    self.Moments(numpy.abs(CSPC_Samples[2,xvalid]), xSamples_zoom)])

        popt01, popt02, popt12 = [1e-10,0,1e-10], [1e-10,0,1e-10] ,[1e-10,0,1e-10]
        FitGauss01, FitGauss02, FitGauss12 = numpy.zeros(len(xSamples)), numpy.zeros(len(xSamples)), numpy.zeros(len(xSamples))

        '''*******************************FIT GAUSS CSPC************************************'''
        try:
            popt01,pcov = curve_fit(self.gaus,xSamples_zoom,numpy.abs(CSPC_Samples[0][xvalid]),p0=CSPCmoments[0])
            if popt01[2] > widthlimit: # CONDITION
                return self.StopWindEstimation(error_code = 4)
            popt02,pcov = curve_fit(self.gaus,xSamples_zoom,numpy.abs(CSPC_Samples[1][xvalid]),p0=CSPCmoments[1])
            if popt02[2] > widthlimit: # CONDITION
                return self.StopWindEstimation(error_code = 4)
            popt12,pcov = curve_fit(self.gaus,xSamples_zoom,numpy.abs(CSPC_Samples[2][xvalid]),p0=CSPCmoments[2])
            if popt12[2] > widthlimit: # CONDITION
                return self.StopWindEstimation(error_code = 4)

            FitGauss01 = self.gaus(xSamples_zoom, *popt01)
            FitGauss02 = self.gaus(xSamples_zoom, *popt02)
            FitGauss12 = self.gaus(xSamples_zoom, *popt12)
        except:
            return self.StopWindEstimation(error_code = 5)


        '''************* Getting Fij ***************'''
        # x-axis point of the gaussian where the center is located from GaussFit of spectra
        GaussCenter = popt[1]
        ClosestCenter = xSamples_zoom[numpy.abs(xSamples_zoom-GaussCenter).argmin()]
        PointGauCenter = numpy.where(xSamples_zoom==ClosestCenter)[0][0]

        # Point where e^-1 is located in the gaussian
        PeMinus1 = numpy.max(FitGauss) * numpy.exp(-1)
        FijClosest = FitGauss[numpy.abs(FitGauss-PeMinus1).argmin()] # The closest point to"Peminus1" in "FitGauss"
        PointFij = numpy.where(FitGauss==FijClosest)[0][0]
        Fij = numpy.abs(xSamples_zoom[PointFij] - xSamples_zoom[PointGauCenter])

        '''********** Taking frequency ranges from mean SPCs **********'''
        GauWidth = popt[2] * 3/2        # Bandwidth of Gau01
        Range = numpy.empty(2)
        Range[0] = GaussCenter - GauWidth
        Range[1] = GaussCenter + GauWidth
        # Point in x-axis where the bandwidth is located (min:max)
        ClosRangeMin = xSamples_zoom[numpy.abs(xSamples_zoom-Range[0]).argmin()]
        ClosRangeMax = xSamples_zoom[numpy.abs(xSamples_zoom-Range[1]).argmin()]
        PointRangeMin = numpy.where(xSamples_zoom==ClosRangeMin)[0][0]
        PointRangeMax = numpy.where(xSamples_zoom==ClosRangeMax)[0][0]
        Range = numpy.array([ PointRangeMin, PointRangeMax ])
        FrecRange = xSamples_zoom[ Range[0] : Range[1] ]

        '''************************** Getting Phase Slope ***************************'''
        for i in range(nPair):
            if len(FrecRange) > 5:
                PhaseRange = phase[i, xvalid[0][Range[0]:Range[1]]].copy()
                mask = ~numpy.isnan(FrecRange) & ~numpy.isnan(PhaseRange)
                if len(FrecRange) == len(PhaseRange):
                    try:
                        slope, intercept, _, _, _ = stats.linregress(FrecRange[mask], self.AntiAliasing(PhaseRange[mask], 4.5))
                        PhaseSlope[i] = slope
                        PhaseInter[i] = intercept
                    except:
                        return self.StopWindEstimation(error_code = 6)
                else:
                    return self.StopWindEstimation(error_code = 7)
            else:
                return self.StopWindEstimation(error_code = 8)

        '''*** Constants A-H correspond to the convention as in Briggs and Vincent 1992 ***'''

        '''Getting constant C'''
        cC=(Fij*numpy.pi)**2

        '''****** Getting constants F and G ******'''
        MijEijNij = numpy.array([[Xi02,Eta02], [Xi12,Eta12]])
        # MijEijNij = numpy.array([[Xi01,Eta01], [Xi02,Eta02], [Xi12,Eta12]])
        # MijResult0 = (-PhaseSlope[0] * cC) / (2*numpy.pi)
        MijResult1 = (-PhaseSlope[1] * cC) / (2*numpy.pi)
        MijResult2 = (-PhaseSlope[2] * cC) / (2*numpy.pi)
        # MijResults = numpy.array([MijResult0, MijResult1, MijResult2])
        MijResults = numpy.array([MijResult1, MijResult2])
        (cF,cG) = numpy.linalg.solve(MijEijNij, MijResults)

        '''****** Getting constants A, B and H ******'''
        W01 = numpy.nanmax( FitGauss01 )
        W02 = numpy.nanmax( FitGauss02 )
        W12 = numpy.nanmax( FitGauss12 )

        WijResult01 = ((cF * Xi01 + cG * Eta01)**2)/cC - numpy.log(W01 / numpy.sqrt(numpy.pi / cC))
        WijResult02 = ((cF * Xi02 + cG * Eta02)**2)/cC - numpy.log(W02 / numpy.sqrt(numpy.pi / cC))
        WijResult12 = ((cF * Xi12 + cG * Eta12)**2)/cC - numpy.log(W12 / numpy.sqrt(numpy.pi / cC))
        WijResults = numpy.array([WijResult01, WijResult02, WijResult12])

        WijEijNij = numpy.array([ [Xi01**2, Eta01**2, 2*Xi01*Eta01] , [Xi02**2, Eta02**2, 2*Xi02*Eta02] , [Xi12**2, Eta12**2, 2*Xi12*Eta12] ])
        (cA,cB,cH) = numpy.linalg.solve(WijEijNij, WijResults)

        VxVy = numpy.array([[cA,cH],[cH,cB]])
        VxVyResults = numpy.array([-cF,-cG])
        (Vmer,Vzon) = numpy.linalg.solve(VxVy, VxVyResults)
        Vver =  -SPCMoments[1]*SPEED_OF_LIGHT/(2*radfreq)
        error_code = 0

        return Vzon, Vmer, Vver, error_code

class SpectralMoments(Operation):

    '''
        Function SpectralMoments()

        Calculates moments (power, mean, standard deviation) and SNR of the signal

        Type of dataIn:    Spectra

        Configuration Parameters:

            proc_type :     (0) First spectral moments routine (Default), 
                            (1) Spectral moment routine similar to JULIA.
            mode_fit  :     (0) No gaussian fit
                            (1) One gaussian fit for 150Km processing.

            exp       :     '150EEJ' To select 128 points window
                            'ESF_EW' To select full window.

        Input:
            channelList    :    simple channel list to select e.g. [2,3,7]
            self.dataOut.data_pre        :    Spectral data
            self.dataOut.abscissaList    :    List of frequencies
            self.dataOut.noise           :    Noise level per channel

        Affected:
            self.dataOut.moments        :    Parameters per channel
            self.dataOut.data_snr       :    SNR per channel

    '''
                  
    def __calculateMoments(self, oldspec, oldfreq, n0,
                           nicoh = None, graph = None, smooth = None, type1 = None, fwindow = None, \
                           snrth = None, dc = None, aliasing = None, oldfd = None, wwauto = None,   \
                           vers= None, Hei= None, debug=False, dbg_hei=None, ymax=0.1, curr_ch=0, sel_ch=[0,1]):

        def __GAUSSWINFIT1(A, flagPDER=0):
            nonlocal truex, xvalid
            nparams = 4
            M=truex.size
            mm=numpy.arange(M,dtype='f4')
            delta = numpy.zeros(M,dtype='f4')
            delta[0] = 1.0 
            Ts = numpy.array([1.0/(2*truex[0])],dtype='f4')[0]
            jj = -1j
            #if self.winauto is None: self.winauto = (1.0 - mm/M)
            winauto = (1.0 - mm/M)
            winauto = winauto/winauto.max()     # Normalized to 1
            #ON_ERROR,2     # IDL sentence: Return to caller if an error occurs
            A[0] = numpy.abs(A[0])
            A[2] = numpy.abs(A[2])
            A[3] = numpy.abs(A[3])
            pi=numpy.array([numpy.pi],dtype='f4')[0]
            if A[2] != 0:
                Z = numpy.exp(-2*numpy.power((pi*A[2]*mm*Ts),2,dtype='f4')+jj*2*pi*A[1]*mm*Ts, dtype='c8')      # Get Z
            else:
                Z = mm*0.0
                A[0] = 0.0
            junkF = numpy.roll(2*fft(winauto*(A[0]*Z+A[3]*delta)).real - \
                                  winauto[0]*(A[0]+A[3]), M//2)                         # *M scale for fft not needed in python
            F = junkF[xvalid]
            if flagPDER == 0:       #NEED PARTIAL?
                return F
            PDER = numpy.zeros((M,nparams))   #YES, MAKE ARRAY.   
            PDER[:,0] = numpy.shift(2*(fft(winauto*Z)*M) - winauto[0], M/2)
            PDER[:,1] = numpy.shift(2*(fft(winauto*jj*2*numpy.pi*mm*Ts*A[0]*Z)*M), M/2)
            PDER[:,2] = numpy.shift(2*(fft(winauto*(-4*numpy.power(numpy.pi*mm*Ts,2)*A[2]*A[0]*Z))*M), M/2)
            PDER[:,3] = numpy.shift(2*(fft(winauto*delta)*M) - winauto[0], M/2)
            PDER = PDER[xvalid,:]
            return F, PDER

        def __curvefit_koki(y, a, Weights, FlagNoDerivative=1,
                            itmax=20, tol=None):
            #ON_ERROR,2 IDL SENTENCE: RETURN TO THE CALLER IF ERROR
            if tol == None:
                tol = numpy.array([1.e-3],dtype='f4')[0] 
            typ=a.dtype
            double = 1 if typ == numpy.float64 else 0
            if typ != numpy.float32:
                a=a.astype(numpy.float32)       #Make params floating
            # if we will be estimating partial derivates then compute machine precision
            if FlagNoDerivative == 1:
                res=numpy.MachAr(float_conv=numpy.float32)
                eps=numpy.sqrt(res.eps)

            nterms = a.size             # Number of parameters
            nfree=numpy.array([numpy.size(y) - nterms],dtype='f4')[0]   # Degrees of freedom
            if nfree <= 0: print('Curvefit - not enough data points.')
            flambda= numpy.array([0.001],dtype='f4')[0]                         # Initial lambda
            #diag=numpy.arange(nterms)*(nterms+1)       # Subscripta of diagonal elements
            # Use diag method in python
            converge=1

            #Define the partial derivative array
            PDER = numpy.zeros((nterms,numpy.size(y)),dtype='f8') if double == 1 else numpy.zeros((nterms,numpy.size(y)),dtype='f4')

            for Niter in range(itmax):      #Iteration loop     

                if FlagNoDerivative == 1:
                    #Evaluate function and estimate partial derivatives
                    yfit = __GAUSSWINFIT1(a)
                    for term in range(nterms):
                        p=a.copy()      # Copy current parameters
                        #Increment size for forward difference derivative
                        inc = eps * abs(p[term])
                        if inc == 0: inc = eps
                        p[term] = p[term] + inc
                        yfit1 = __GAUSSWINFIT1(p)
                        PDER[term,:] = (yfit1-yfit)/inc
                else:
                    #The user's procedure will return partial derivatives
                    yfit,PDER=__GAUSSWINFIT1(a, flagPDER=1)

                beta = numpy.dot(PDER,(y-yfit)*Weights)
                alpha = numpy.dot(PDER * numpy.tile(Weights,(nterms,1)), numpy.transpose(PDER))
                # save current values of return parameters
                sigma1 = numpy.sqrt( 1.0 / numpy.diag(alpha) )  # Current sigma.
                sigma  = sigma1

                chisq1 = numpy.sum(Weights*numpy.power(y-yfit,2,dtype='f4'),dtype='f4')/nfree     # Current chi squared.
                chisq = chisq1
                yfit1 = yfit
                elev7=numpy.array([1.0e7],dtype='f4')[0]
                compara =numpy.sum(abs(y))/elev7/nfree
                done_early = chisq1 < compara

                if done_early:
                    chi2 = chisq         # Return chi-squared (chi2 obsolete-still works)  
                    if done_early: Niter -= 1
                    #save_tp(chisq,Niter,yfit)
                    return yfit, a, converge, sigma, chisq          # return result
                #c = numpy.dot(c, c)    # this operator implemented at the next lines
                c_tmp = numpy.sqrt(numpy.diag(alpha))
                siz=len(c_tmp)
                c=numpy.dot(c_tmp.reshape(siz,1),c_tmp.reshape(1,siz))
                lambdaCount = 0
                while True:
                    lambdaCount += 1
                    # Normalize alpha to have unit diagonal.
                    array = alpha / c
                    # Augment the diagonal.
                    one=numpy.array([1.],dtype='f4')[0]
                    numpy.fill_diagonal(array,numpy.diag(array)*(one+flambda))
                    # Invert modified curvature matrix to find new parameters.
  
                    try:
                        array =  (1.0/array) if array.size == 1  else numpy.linalg.inv(array)
                    except Exception as e:
                        print(e)
                        array[:]=numpy.NaN

                    b = a + numpy.dot(numpy.transpose(beta),array/c) # New params           
                    yfit = __GAUSSWINFIT1(b) # Evaluate function
                    chisq = numpy.sum(Weights*numpy.power(y-yfit,2,dtype='f4'),dtype='f4')/nfree # New chisq
                    sigma = numpy.sqrt(numpy.diag(array)/numpy.diag(alpha)) # New sigma
                    if (numpy.isfinite(chisq) == 0) or \
                        (lambdaCount > 30 and chisq >= chisq1):
                        # Reject changes made this iteration, use old values.
                        yfit  = yfit1 
                        sigma = sigma1
                        chisq = chisq1
                        converge = 0
                        #print('Failed to converge.')
                        chi2 = chisq         # Return chi-squared (chi2 obsolete-still works)
                        if done_early: Niter -= 1
                        return yfit, a, converge, sigma, chisq, chi2          # return result   
                    ten=numpy.array([10.0],dtype='f4')[0]
                    flambda *= ten      # Assume fit got worse
                    if chisq <= chisq1:
                        break
                hundred=numpy.array([100.0],dtype='f4')[0]
                flambda /= hundred

                a=b                     # Save new parameter estimate.             
                if ((chisq1-chisq)/chisq1) <= tol: # Finished?
                    chi2 = chisq         # Return chi-squared (chi2 obsolete-still works)
                    if done_early: Niter -= 1
                    return yfit, a, converge, sigma, chisq, chi2         # return result
            converge = 0
            chi2 = chisq
            #print('Failed to converge.')
            return yfit, a, converge, sigma, chisq, chi2
        

        def spectral_cut(Hei, ind, dbg_hei, freq, fd, snr, n1, w, ymax, spec, spec2, n0, max_spec, ss1, m, bb0, curr_ch, sel_ch):
            if Hei[ind] > dbg_hei[0] and Hei[ind] < dbg_hei[1] and (curr_ch in sel_ch):
               nsa=len(freq)
               aux='H=%iKm, dop: %4.1f, snr: %4.1f, noise: %4.1f, sw: %4.1f'%(Hei[ind],fd, 10*numpy.log10(snr),10*numpy.log10(n1), w)
               plt.subplots()
               plt.ylim(0,ymax)
               plt.plot(freq,spec,'b-',freq,spec2,'b--', freq,numpy.repeat(n1, nsa),'k--', freq,numpy.repeat(n0, nsa),'k-', freq,numpy.repeat(max_spec, nsa),'y.-', numpy.repeat(fd, nsa),numpy.linspace(0,ymax,nsa),'r--', numpy.repeat(freq[ss1], nsa),numpy.linspace(0,ymax,nsa),'g-.', numpy.repeat(freq[m + bb0], nsa),numpy.linspace(0,ymax,nsa),'g-.')
               plt.title(aux)
               plt.show()
                
                        
        if (nicoh is None): nicoh = 1
        if (smooth is None): smooth = 0
        if (type1 is None): type1 = 0
        if (vers is None): vers = 0        
        if (fwindow is None): fwindow = numpy.zeros(oldfreq.size) + 1
        if (snrth is None): snrth = -20.0
        if (dc is None): dc = 0
        if (aliasing is None): aliasing = 0
        if (oldfd is None): oldfd = 0
        if (wwauto is None): wwauto = 0

        if (n0 < 1.e-20):   n0 = 1.e-20

        xvalid = numpy.where(fwindow == 1)[0]
        freq = oldfreq
        truex = oldfreq
        vec_power = numpy.zeros(oldspec.shape[1])
        vec_fd = numpy.zeros(oldspec.shape[1])
        vec_w = numpy.zeros(oldspec.shape[1])
        vec_snr = numpy.zeros(oldspec.shape[1])
        vec_n1 = numpy.empty(oldspec.shape[1])
        vec_fp = numpy.empty(oldspec.shape[1])
        vec_sigma_fd = numpy.empty(oldspec.shape[1])

        for ind in range(oldspec.shape[1]):
            spec = oldspec[:,ind]
            if (smooth == 0):
                spec2 = spec
            else:
                spec2 = scipy.ndimage.filters.uniform_filter1d(spec,size=smooth)
            
            aux = spec2*fwindow
            max_spec = aux.max()
            m = aux.tolist().index(max_spec)

            if m > 2 and m < oldfreq.size - 3:
                newindex = m + numpy.array([-2,-1,0,1,2])
                newfreq = numpy.arange(20)/20.0*(numpy.max(freq[newindex])-numpy.min(freq[newindex]))+numpy.min(freq[newindex])
                tck = interpolate.splrep(freq[newindex], spec2[newindex])
                peakspec = interpolate.splev(newfreq, tck)
                max_spec = numpy.max(peakspec)
                mnew = numpy.argmax(peakspec)
                fp = newfreq[mnew]
            else:
                fp = freq[m]

            if vers ==0:

                # Moments Estimation
                bb = spec2[numpy.arange(m,spec2.size)]
                bb = (bb<n0).nonzero()
                bb = bb[0]

                ss = spec2[numpy.arange(0,m + 1)]
                ss = (ss<n0).nonzero()
                ss = ss[0]

                if (bb.size == 0):
                    bb0 = spec.size - 1 - m
                else:
                    bb0 = bb[0] - 1
                    if (bb0 < 0):
                        bb0 = 0

                if (ss.size == 0):
                    ss1 = 1
                else:
                    ss1 = max(ss) + 1

                if (ss1 > m):
                    ss1 = m

                valid = numpy.arange(int(m + bb0 - ss1 + 1)) + ss1

                signal_power = ((spec2[valid] - n0) * fwindow[valid]).mean()    # D. Scipión added with correct definition
                total_power = (spec2[valid] * fwindow[valid]).mean()            # D. Scipión added with correct definition
                power = ((spec2[valid] - n0) * fwindow[valid]).sum() 
                fd = ((spec2[valid]- n0)*freq[valid] * fwindow[valid]).sum() / power
                w = numpy.sqrt(((spec2[valid] - n0)*fwindow[valid]*(freq[valid]- fd)**2).sum() / power)
                snr = (spec2.mean()-n0)/n0
                if (snr < 1.e-20): snr = 1.e-20
   
                vec_power[ind] = total_power
                vec_fd[ind] = fd
                vec_w[ind] = w
                vec_snr[ind] = snr
            else:
                # Noise by heights
                n1, stdv = self.__get_noise2(spec, nicoh)            
                # Moments Estimation
                bb = spec2[numpy.arange(m,spec2.size)]
                bb = (bb<n1).nonzero()
                bb = bb[0]

                ss = spec2[numpy.arange(0,m + 1)]
                ss = (ss<n1).nonzero()
                ss = ss[0]

                if (bb.size == 0):
                    bb0 = spec.size - 1 - m
                else:
                    bb0 = bb[0] - 1
                    if (bb0 < 0):
                        bb0 = 0

                if (ss.size == 0):
                    ss1 = 1
                else:
                    ss1 = max(ss) + 1

                if (ss1 > m):
                    ss1 = m

                valid = numpy.arange(int(m + bb0 - ss1 + 1)) + ss1
                               
                power = ((spec[valid] - n1)*fwindow[valid]).sum()
                fd = ((spec[valid]- n1)*freq[valid]*fwindow[valid]).sum()/power
                try:
                    w = numpy.sqrt(((spec[valid] - n1)*fwindow[valid]*(freq[valid]- fd)**2).sum()/power)
                except:
                    w = float("NaN")
                snr = power/(n0*fwindow.sum())

                if debug:
                    spectral_cut(Hei, ind, dbg_hei, freq, fd, snr, n1, w, ymax, spec, spec2, n0, max_spec, ss1, m, bb0, curr_ch, sel_ch)
   
                if snr <  1.e-20: snr = 1.e-20

                # Here start gaussean adjustment

                if type1 == 1 and snr > numpy.power(10,0.1*snrth):
    
                    a = numpy.zeros(4,dtype='f4')
                    a[0] = snr * n0
                    a[1] = fd
                    a[2] = w
                    a[3] = n0

                    np = spec.size
                    aold = a.copy()
                    spec2 = spec.copy()
                    oldxvalid = xvalid.copy()

                    for i in range(2):

                        ww = 1.0/(numpy.power(spec2,2)/nicoh)
                        ww[np//2] = 0.0

                        a = aold.copy()
                        xvalid = oldxvalid.copy()
                        #self.show_var(xvalid)

                        gaussfn = __curvefit_koki(spec[xvalid], a, ww[xvalid]) 
                        a = gaussfn[1]
                        converge = gaussfn[2]

                        xvalid = numpy.arange(np)
                        spec2 = __GAUSSWINFIT1(a)

                    xvalid = oldxvalid.copy()
                    power = a[0] * np
                    fd = a[1]
                    sigma_fd = gaussfn[3][1]
                    snr = max(power/ (max(a[3],n0) * len(oldxvalid)) * converge, 1e-20)
                    w = numpy.abs(a[2])
                    n1 = max(a[3], n0)

                    #gauss_adj=[fd,w,snr,n1,fp,sigma_fd]
                else:
                    sigma_fd=numpy.nan # to avoid UnboundLocalError: local variable 'sigma_fd' referenced before assignment

                vec_fd[ind] = fd
                vec_w[ind] = w
                vec_snr[ind] = snr
                vec_n1[ind] = n1
                vec_fp[ind] = fp
                vec_sigma_fd[ind] = sigma_fd
                vec_power[ind] = power  # to compare with type 0 proccessing

        if vers==1:
            return numpy.vstack((vec_snr,  vec_w, vec_fd, vec_n1, vec_fp, vec_sigma_fd, vec_power))
        else:
            return numpy.vstack((vec_snr, vec_power, vec_fd, vec_w))
    
    def __get_noise2(self,POWER, fft_avg, TALK=0):
        '''
        Rutina para cálculo de ruido por alturas(n1). Similar a IDL
        '''
        SPECT_PTS = len(POWER)
        fft_avg = fft_avg*1.0
        NOMIT = 0
        NN = SPECT_PTS - NOMIT
        N  = NN//2
        ARR = numpy.concatenate((POWER[0:N+1],POWER[N+NOMIT+1:SPECT_PTS]))
        ARR = numpy.sort(ARR)
        NUMS_MIN = (SPECT_PTS+7)//8
        RTEST = (1.0+1.0/fft_avg)
        SUM = 0.0
        SUMSQ = 0.0
        J = 0
        for I in range(NN):
            J = J + 1
            SUM = SUM + ARR[I]
            SUMSQ = SUMSQ + ARR[I]*ARR[I]
            AVE = SUM*1.0/J
            if J > NUMS_MIN:
                if (SUMSQ*J <= RTEST*SUM*SUM): RNOISE = AVE
            else:
                if J == NUMS_MIN: RNOISE = AVE
        if TALK == 1: print('Noise Power (2):%4.4f' %RNOISE)
        stdv = numpy.sqrt(SUMSQ/J - numpy.power(SUM/J,2))
        return RNOISE, stdv

    def __get_noise1(self, power, fft_avg, TALK=0):
        '''
        Rutina para cálculo de ruido por alturas(n0). Similar a IDL
        '''
        num_pts = numpy.size(power)
        fft_avg = fft_avg*1.0
        ind = numpy.argsort(power, axis=None, kind='stable')
        ARR = numpy.reshape(power,-1)[ind]
        NUMS_MIN = num_pts//10
        RTEST = (1.0+1.0/fft_avg)
        SUM = 0.0
        SUMSQ = 0.0
        J = 0
        cont = 1
        while cont == 1 and J < num_pts:

            SUM = SUM + ARR[J]
            SUMSQ = SUMSQ + ARR[J]*ARR[J]
            J = J + 1
 
            if J > NUMS_MIN:
                if (SUMSQ*J <= RTEST*SUM*SUM):
                    LNOISE = SUM*1.0/J
                else:
                    J = J - 1
                    SUM = SUM - ARR[J]
                    SUMSQ = SUMSQ - ARR[J]*ARR[J]
                    cont = 0
            else:
                if J == NUMS_MIN: LNOISE = SUM*1.0/J
        if TALK == 1: print('Noise Power (1):%8.8f' %LNOISE)
        stdv = numpy.sqrt(SUMSQ/J - numpy.power(SUM/J,2))
        return LNOISE, stdv

    def __NoiseByChannel(self, num_prof, num_incoh, spectra,talk=0):

        val_frq = numpy.arange(num_prof-2)+1
        val_frq[(num_prof-2)//2:] = val_frq[(num_prof-2)//2:] + 1       
        junkspc = numpy.sum(spectra[val_frq,:], axis=1)
        junkid = numpy.argsort(junkspc)
        noisezone = val_frq[junkid[0:num_prof//2]]
        specnoise = spectra[noisezone,:]
        noise, stdvnoise = self.__get_noise1(specnoise,num_incoh)
        
        if talk:
            print('noise =', noise)
        return noise, stdvnoise

    def run(self, dataOut, proc_type=0, mode_fit=0, exp='150EEJ', debug=False, dbg_hei=None, ymax=1, sel_ch=[0,1]):

        absc = dataOut.abscissaList[:-1]
        nChannel = dataOut.data_pre[0].shape[0]
        nHei = dataOut.data_pre[0].shape[2]
        Hei=dataOut.heightList
        data_param = numpy.zeros((nChannel, 4 + proc_type*3, nHei))
        nProfiles = dataOut.nProfiles
        nCohInt = dataOut.nCohInt
        nIncohInt = dataOut.nIncohInt
        M = numpy.power(numpy.array(1/(nProfiles * nCohInt) ,dtype='float32'),2)
        N = numpy.array(M / nIncohInt,dtype='float32')         

        if proc_type == 1:
            type1 = mode_fit
            fwindow = numpy.zeros(absc.size) + 1
            if exp == '150EEJ':
                b=64
                fwindow[0:absc.size//2 - b] = 0
                fwindow[absc.size//2 + b:] = 0
            vers = 1 # new           

            data = dataOut.data_pre[0] * N
            
            noise = numpy.zeros(nChannel)
            stdvnoise = numpy.zeros(nChannel)
            for ind in range(nChannel):
                noise[ind], stdvnoise[ind] = self.__NoiseByChannel(nProfiles, nIncohInt, data[ind,:,:])
            smooth=3
        else:
            data = dataOut.data_pre[0]
            noise = dataOut.noise
            fwindow = None
            type1 = None
            vers = 0  # old
            nIncohInt = None
            smooth=None

        for ind in range(nChannel):
            data_param[ind,:,:] = self.__calculateMoments(data[ind,:,:] , absc , noise[ind], nicoh=nIncohInt, smooth=smooth, type1=type1, fwindow=fwindow, vers=vers, Hei=Hei, debug=debug, dbg_hei=dbg_hei, ymax=ymax, curr_ch=ind, sel_ch=sel_ch)
            #data_param[ind,:,:] = self.__calculateMoments(data[ind,:,:] , absc , noise[ind], nicoh=nIncohInt, smooth=smooth, type1=type1, fwindow=fwindow, vers=vers, Hei=Hei, debug=debug)
            if exp == 'ESF_EW':
               data_param[ind,0,:]*=(noise[ind]/stdvnoise[ind])
            data_param[ind,3,:]*=(1.0/M)

        if proc_type == 1:
            dataOut.moments = data_param[:,1:,:]
            dataOut.data_dop = data_param[:,2]
            dataOut.data_width = data_param[:,1]
            dataOut.data_snr = data_param[:,0]
            dataOut.data_pow = data_param[:,6]  # to compare with type0 proccessing
            dataOut.spcpar=numpy.stack((dataOut.data_dop,dataOut.data_width,dataOut.data_snr, data_param[:,3], data_param[:,4],data_param[:,5]),axis=2)
            
            if exp == 'ESF_EW':
                spc=dataOut.data_pre[0]* N
                cspc=dataOut.data_pre[1]* N
                nHei=dataOut.data_pre[1].shape[2]
                cross_pairs=dataOut.pairsList
                nDiffIncohInt = dataOut.nDiffIncohInt
                N2=numpy.array(1 / nDiffIncohInt,dtype='float32')
                diffcspectra=dataOut.data_diffcspc.copy()* N2 * M * M
                num_pairs=len(dataOut.pairsList)

                if num_pairs >= 0:
                    fbinv=numpy.where(absc != 0)[0]
                    ccf=numpy.sum(cspc[:,fbinv,:], axis=1)
                    jvpower=numpy.sum(spc[:,fbinv,:], axis=1)
                    coh=ccf/numpy.sqrt(jvpower[cross_pairs[0][0],:]*jvpower[cross_pairs[0][1],:])
                    dccf=numpy.sum(diffcspectra[:,fbinv,:], axis=1)
                    dataOut.ccfpar = numpy.zeros((num_pairs,nHei,3))
                    dataOut.ccfpar[:,:,0]=numpy.abs(coh)
                    dataOut.ccfpar[:,:,1]=numpy.arctan(numpy.imag(coh)/numpy.real(coh))
                    dataOut.ccfpar[:,:,2]=numpy.arctan(numpy.imag(dccf)/numpy.real(dccf))
        else:
            dataOut.moments = data_param[:,1:,:]
            dataOut.data_snr = data_param[:,0]
            dataOut.data_pow = data_param[:,1]
            dataOut.data_dop = data_param[:,2]
            dataOut.data_width = data_param[:,3]
            dataOut.spcpar=numpy.stack((dataOut.data_dop,dataOut.data_width,dataOut.data_snr, dataOut.data_pow),axis=2)
              
        return dataOut
        

class JULIA_DayVelocities(Operation):
    '''
        Function SpectralMoments()

        From espectral parameters calculates:
        
        1. Signal to noise level (SNL)
        2. Vertical velocity
        3. Zonal velocity
        4. Vertical velocity error
        5. Zonal velocity error.

        Type of dataIn:    SpectralMoments

        Configuration Parameters:

            zenith :     Pairs of angles corresponding to the two beams related to the perpendicular to B from the center of the antenna.                           
            zenithCorrection  :     Adjustment angle for the zenith. Default 0.
            heights       :     Range to process 150kM echoes. By default [125,185].
            nchan :     To process 2 or 1 channel. 2 by default.
            chan  :     If nchan = 1, chan indicates which of the 2 channels to process.
            clean       :  2nd cleaning processing (Graphical). Default False
            driftstdv_th       :     Diferencia máxima entre valores promedio consecutivos de vertical.
            zonalstdv_th       :     Diferencia máxima entre valores promedio consecutivos de zonal.

        Input:

        Affected:

    '''

    def __init__(self):
        Operation.__init__(self)
        self.old_drift=None
        self.old_zonal=None
        self.count_drift=0
        self.count_zonal=0
        self.oldTime_drift=None
        self.oldTime_zonal=None
                    
    def newtotal(self, data):
        return numpy.nansum(data)
    
    def data_filter(self, parm, snrth=-20, swth=20, wErrth=500):

        Sz0 = parm.shape # Sz0: h,p
        drift = parm[:,0]
        sw = 2*parm[:,1]
        snr = 10*numpy.log10(parm[:,2])
        Sz = drift.shape # Sz: h
        mask = numpy.ones((Sz[0]))
        th=0
        valid=numpy.where(numpy.isfinite(snr))
        cvalid = len(valid[0])
        if cvalid >= 1:
            # Cálculo del ruido promedio de snr para el i-ésimo grupo de alturas
            nbins = int(numpy.max(snr)-numpy.min(snr))+1 # bin size = 1, similar to IDL
            h = numpy.histogram(snr,bins=nbins)
            hist = h[0]
            values = numpy.round_(h[1])
            moda = values[numpy.where(hist == numpy.max(hist))]
            indNoise = numpy.where(numpy.abs(snr - numpy.min(moda)) < 3)[0] 

            noise = snr[indNoise]
            noise_mean = numpy.sum(noise)/len(noise)
            # Cálculo de media de snr
            med = numpy.median(snr)
            # Establece el umbral de snr
            if  noise_mean > med + 3:
                th = med
            else:
                th = noise_mean + 3
            # Establece máscara
            novalid = numpy.where(snr <= th)[0]
            mask[novalid] = numpy.nan
        # Elimina datos que no sobrepasen el umbral: PARAMETRO
        novalid = numpy.where(snr <= snrth)
        cnovalid = len(novalid[0])
        if cnovalid > 0:
           mask[novalid] = numpy.nan
        novalid = numpy.where(numpy.isnan(snr))
        cnovalid = len(novalid[0])
        if cnovalid > 0:
            mask[novalid] = numpy.nan

        new_parm = numpy.zeros((Sz0[0],Sz0[1]))
        for i in range(Sz0[1]):
            new_parm[:,i] = parm[:,i] * mask 

        return new_parm, th


    def statistics150km(self, veloc , sigma , threshold , old_veloc=None, count=0, \
                        currTime=None, oldTime=None, amountdata=3, clearAll = None, timeFactor=1800, debug = False):
        
        if oldTime == None:
            oldTime = currTime

        step = (threshold/2)*(numpy.abs(currTime - oldTime)//timeFactor + 1)
        factor = 2  
        avg_threshold = 100
        # Calcula la mediana en todas las alturas por tiempo
        val1=numpy.nanmedian(veloc)

        # Calcula la media ponderada en todas las alturas por tiempo
        val2 = self.newtotal(veloc/numpy.power(sigma,2))/self.newtotal(1/numpy.power(sigma,2))
            
        # Verifica la cercanía de los valores calculados de mediana y media, si son cercanos escoge la media ponderada
        op1=numpy.abs(val2-val1)
        op2=threshold/factor
        cond = op1 < op2

        veloc_prof = val2 if cond else val1
        sigma_prof = numpy.nan
        sets=numpy.array([-1])
              
        if op1 > avg_threshold: #Si son muy lejanos no toma en cuenta estos datos
            veloc_prof = numpy.nan

        # Se calcula nuevamente media ponderada, en base a estimado inicial de la media
        # a fin de eliminar valores que están muy lejanos a dicho valor

        if debug:
            print('veloc_prof:', veloc_prof)
            print('veloc:',veloc)
            print('threshold:',threshold)
            print('factor:',factor)
            print('threshold/factor:',threshold/factor)
            print('numpy.abs(veloc-veloc_prof):', numpy.abs(veloc-veloc_prof))
            print('numpy.where(numpy.abs(veloc-veloc_prof) < threshold/factor)[0]:', numpy.where(numpy.abs(veloc-veloc_prof) < threshold/factor)[0])

        junk = numpy.where(numpy.abs(veloc-veloc_prof) < threshold/factor)[0]        
        if junk.size >= amountdata:
            veloc_prof = self.newtotal(veloc[junk]/numpy.power(sigma[junk],2))/self.newtotal(1/numpy.power(sigma[junk],2))
            sigma_prof1 = numpy.sqrt(1/self.newtotal(1/numpy.power(sigma[junk],2)))
            sigma_prof2 = numpy.sqrt(self.newtotal(numpy.power(veloc[junk]-veloc_prof,2)/numpy.power(sigma[junk],2)))*sigma_prof1
            sigma_prof = numpy.sqrt(numpy.power(sigma_prof1,2)+numpy.power(sigma_prof2,2))
            sets = junk            

        # Compara con valor anterior para evitar presencia de "outliers"
        if debug:
            print('old_veloc:',old_veloc)
            print('step:', step)
            
        if old_veloc == None:
            valid=numpy.isfinite(veloc_prof)
        else:
            valid=numpy.abs(veloc_prof-old_veloc) < step 
            
        if debug:
            print('valid:', valid)
            
        if not valid:
            aver_veloc=numpy.nan
            aver_sigma=numpy.nan
            sets=numpy.array([-1])
        else:
            aver_veloc=veloc_prof
            aver_sigma=sigma_prof
        clearAll=0
        if old_veloc != None and count < 5:
            if numpy.abs(veloc_prof-old_veloc) > step:
                clearAll=1
                count=0
                old_veloc=None
        if numpy.isfinite(aver_veloc):
            
            count+=1
            if old_veloc != None:
                old_veloc =  (old_veloc + aver_veloc) * 0.5
            else:
                old_veloc=aver_veloc
            oldTime=currTime
        if debug:
            print('count:',count)
            print('sets:',sets)
        return sets, old_veloc, count, oldTime, aver_veloc, aver_sigma, clearAll


    def run(self, dataOut, zenith, zenithCorrection=0.0, heights=[125, 185], nchan=2, chan=0, clean=False, driftstdv_th=100, zonalstdv_th=200, amountdata=3):

        dataOut.lat=-11.95
        dataOut.lon=-76.87

        nCh=dataOut.spcpar.shape[0]    
        nHei=dataOut.spcpar.shape[1]
        nParam=dataOut.spcpar.shape[2]

        # Selección de alturas           
        hei=dataOut.heightList
        hvalid=numpy.where([hei >= heights[0]][0] & [hei <= heights[1]][0])[0]
        nhvalid=len(hvalid)
        dataOut.heightList = hei[hvalid]      
        parm=numpy.empty((nCh,nhvalid,nParam)); parm[:]=numpy.nan
        parm[:] = dataOut.spcpar[:,hvalid,:]
        # Primer filtrado: Umbral de SNR
        for i in range(nCh):
            parm[i,:,:] = self.data_filter(parm[i,:,:])[0]

        zenith = numpy.array(zenith)
        zenith -= zenithCorrection
        zenith *= numpy.pi/180
        alpha = zenith[0]
        beta = zenith[1]
        dopplerCH0 = parm[0,:,0]
        dopplerCH1 = parm[1,:,0]
        swCH0 = parm[0,:,1]
        swCH1 = parm[1,:,1]
        snrCH0 = 10*numpy.log10(parm[0,:,2])
        snrCH1 = 10*numpy.log10(parm[1,:,2])
        noiseCH0 = parm[0,:,3]
        noiseCH1 = parm[1,:,3]
        wErrCH0 = parm[0,:,5]
        wErrCH1 = parm[1,:,5]

        # Vertical and zonal calculation: nchan=2 by default
        # Only vertical calculation, for offline processing with only one channel with good signal
        if nchan == 1:
            if chan == 1:
                drift = - dopplerCH1
                snr = snrCH1
                noise = noiseCH1
                sw = swCH1
                w_w_err = wErrCH1
            elif chan == 0:
                drift = - dopplerCH0
                snr = snrCH0
                noise = noiseCH0
                sw = swCH0
                w_w_err = wErrCH0
                            
        elif nchan == 2:
            sinB_A = numpy.sin(beta)*numpy.cos(alpha) - numpy.sin(alpha)* numpy.cos(beta)
            drift = -(dopplerCH0 * numpy.sin(beta) - dopplerCH1 * numpy.sin(alpha))/ sinB_A
            zonal = (dopplerCH0 * numpy.cos(beta) - dopplerCH1 * numpy.cos(alpha))/ sinB_A
            snr = (snrCH0 + snrCH1)/2
            noise = (noiseCH0 + noiseCH1)/2
            sw = (swCH0 + swCH1)/2
            w_w_err= numpy.sqrt(numpy.power(wErrCH0 * numpy.sin(beta)/numpy.abs(sinB_A),2) + numpy.power(wErrCH1 * numpy.sin(alpha)/numpy.abs(sinB_A),2))
            w_e_err= numpy.sqrt(numpy.power(wErrCH0 * numpy.cos(beta)/numpy.abs(-1*sinB_A),2) + numpy.power(wErrCH1 * numpy.cos(alpha)/numpy.abs(-1*sinB_A),2))		

        # 150Km statistics to clean data
        clean_drift = drift.copy()
        clean_drift[:] = numpy.nan
        if nchan == 2:
            clean_zonal = zonal.copy()
            clean_zonal[:] = numpy.nan
        
        # Vertical 
        sets1, self.old_drift, self.count_drift, self.oldTime_drift, aver_veloc, aver_sigma, clearAll = self.statistics150km(drift, w_w_err, driftstdv_th, \
                                                        old_veloc=self.old_drift, count=self.count_drift, currTime=dataOut.utctime, \
                                                        oldTime=self.oldTime_drift, amountdata = amountdata, timeFactor=120, debug = False)
        if clearAll == 1:
            mean_zonal = numpy.nan
            sigma_zonal = numpy.nan
        mean_drift = aver_veloc
        sigma_drift = aver_sigma
            
        if sets1.size != 1:
            clean_drift[sets1] = drift[sets1]

        novalid=numpy.where(numpy.isnan(clean_drift))[0]; cnovalid=novalid.size
        if cnovalid > 0: drift[novalid] = numpy.nan
        if cnovalid > 0: snr[novalid] = numpy.nan

        # Zonal
        if nchan == 2:           
            sets2, self.old_zonal, self.count_zonal, self.oldTime_zonal, aver_veloc, aver_sigma, clearAll = self.statistics150km(zonal, w_e_err, zonalstdv_th, \
                                                        old_veloc=self.old_zonal, count=self.count_zonal, currTime=dataOut.utctime, \
                                                        oldTime=self.oldTime_zonal, amountdata = amountdata, timeFactor=600, debug = False)
            if clearAll == 1:
                mean_zonal = numpy.nan
                sigma_zonal = numpy.nan
            mean_zonal = aver_veloc
            sigma_zonal = aver_sigma
            if sets2.size != 1:
                clean_zonal[sets2] = zonal[sets2]

            novalid=numpy.where(numpy.isnan(clean_zonal))[0]; cnovalid=novalid.size
            if cnovalid > 0: zonal[novalid] = numpy.nan
            if cnovalid > 0: snr[novalid] = numpy.nan

        n_avg_par=4
        avg_par=numpy.empty((n_avg_par,)); avg_par[:] = numpy.nan
        avg_par[0,]=mean_drift
        avg_par[1,]=mean_zonal
        avg_par[2,]=sigma_drift
        avg_par[3,]=sigma_zonal

        set1 = 1.0
        navg = set1
        nci = dataOut.nCohInt
        # ----------------------------------
        ipp = 252.0
        nincoh = dataOut.nIncohInt
        nptsfft = dataOut.nProfiles
        hardcoded=False   # if True, similar to IDL processing
        if hardcoded:   
            ipp=200.1
            nincoh=22
            nptsfft=128
        # ----------------------------------            
        nipp = ipp * nci
        height = dataOut.heightList
        nHei = len(height)
        kd = 213.6
        nint = nptsfft * nincoh
        drift1D = drift.copy()
        if nchan == 2:
            zonal1D=zonal.copy()
        snr1D = snr.copy()
        snr1D = 10*numpy.power(10, 0.1*snr1D)
        noise1D = noise.copy()
        noise0 = numpy.nanmedian(noise1D)
        noise = noise0 + noise0
        sw1D = sw.copy()
        pow0 = snr1D * noise0 + noise0
        acf0 = snr1D * noise0 * numpy.exp((-drift1D*nipp*numpy.pi/(1.5e5*1.5))*1j) * (1-0.5*numpy.power(sw1D*nipp*numpy.pi/(1.5e5*1.5),2))
        acf0 /= pow0
        acf1 = acf0
        dt= nint * nipp /1.5e5

        if nchan == 2:
            dccf = pow0 * pow0 * numpy.exp((zonal1D*kd*dt/(height*1e3))*(1j))
        else:
            dccf = numpy.empty(nHei); dccf[:]=numpy.nan # complex?
        dccf /= pow0 * pow0
        sno=(pow0+pow0-noise)/noise
        
        # First parameter: Signal to noise ratio and its error
        sno = numpy.log10(sno) 
        sno10 = 10 * sno
        dsno = 1.0/numpy.sqrt(nint*navg)*(1+1/sno10)

        # Second parameter: Vertical Drifts
        s=numpy.sqrt(numpy.abs(acf0)*numpy.abs(acf1))
        sp = s*(1.0 + 1.0/sno10)
        vzo = -numpy.arctan2(numpy.imag(acf0+acf1),numpy.real(acf0+acf1))* \
             1.5e5*1.5/(nipp*numpy.pi)
        dvzo = numpy.sqrt(1-sp*sp)*0.338*1.5e5/(numpy.sqrt(nint*navg)*sp*nipp)
                
        # Third parameter: Zonal Drifts
        dt = nint*nipp/1.5e5
        ss = numpy.sqrt(numpy.abs(dccf))        
        vxo = numpy.arctan2(numpy.imag(dccf),numpy.real(dccf))*height*1e3/(kd*dt)
        dvxo = numpy.sqrt(1.0-ss*ss)*height*1e3/(numpy.sqrt(nint*navg)*ss*kd*dt)
        
        npar = 5
        par = numpy.empty((npar, nHei)); par[:] = numpy.nan
        
        par[0,:] = sno
        par[1,:] = vzo
        par[2,:] = vxo        
        par[3,:] = dvzo
        par[4,:] = dvxo
        
        # Segundo filtrado:
        # Remoción por altura: Menos de dos datos finitos no son considerados como eco 150Km.
        clean_par=numpy.empty((npar,nHei)); clean_par[:]=numpy.nan
        if clean:               

            for p in range(npar):
                ih=0
                while ih < nHei-1:
                    j=ih
                    if numpy.isfinite(snr1D[ih]):
                        while numpy.isfinite(snr1D[j]):
                            j+=1
                            if j >= nHei:
                                break
                        if j > ih + 1:
                            for k in range(ih,j):
                                clean_par[p][k] = par[p][k]
                        ih = j - 1
                    ih+=1
        else:
            clean_par[:] = par[:]

        mad_output = numpy.vstack((clean_par[0,:], clean_par[1,:], clean_par[2,:], clean_par[3,:], clean_par[4,:]))
        graph = numpy.vstack((clean_par[0,:], clean_par[1,:], clean_par[2,:]))
        dataOut.data_output = mad_output
        dataOut.data_graph = graph
        dataOut.avg_output = avg_par
        dataOut.utctimeInit = dataOut.utctime
        dataOut.outputInterval = dataOut.timeInterval
        
        dataOut.flagNoData = numpy.all(numpy.isnan(dataOut.data_output[0]))  # NAN vectors are not written
        
        return dataOut


class JULIA_NightVelocities(Operation):
    '''
        Function SpreadFVelocities()

        Calculates SNL and drifts

        Type of dataIn:    Parameters

        Configuration Parameters:

            mymode     :        (0) Interferometry, 
                                (1) Doppler beam swinging.
            myproc     :        (0) JULIA_V, 
                                (1) JULIA_EW.                            
            myantenna  :        (0) 1/4 antenna, 
                                (1) 1/2 antenna.
            jset       :        Number of Incoherent integrations.


        Input:
            channelList    :    simple channel list to select e.g. [2,3,7]
            self.dataOut.data_pre        :    Spectral data
            self.dataOut.abscissaList    :    List of frequencies
            self.dataOut.noise           :    Noise level per channel

        Affected:
            self.dataOut.moments        :    Parameters per channel
            self.dataOut.data_snr       :    SNR per channel

    '''
    def __init__(self):
        Operation.__init__(self)

    def newtotal(self, data):
        return numpy.nansum(data)
    
    def data_filter(self, parm, snrth=-17, swth=20, dopth=500.0, debug=False):

        Sz0 = parm.shape # Sz0: h,p
        drift = parm[:,0]
        sw = 2*parm[:,1]
        snr = 10*numpy.log10(parm[:,2])
        Sz = drift.shape # Sz: h
        mask = numpy.ones((Sz[0]))
        th=0
        valid=numpy.where(numpy.isfinite(snr))
        cvalid = len(valid[0])
        if cvalid >= 1:
            # Cálculo del ruido promedio de snr para el i-ésimo grupo de alturas
            nbins = int(numpy.max(snr)-numpy.min(snr))+1 # bin size = 1, similar to IDL
            h = numpy.histogram(snr,bins=nbins)
            hist = h[0]
            values = numpy.round_(h[1])
            moda = values[numpy.where(hist == numpy.max(hist))]
            indNoise = numpy.where(numpy.abs(snr - numpy.min(moda)) < 3)[0] 

            noise = snr[indNoise]
            noise_mean = numpy.sum(noise)/len(noise)
            # Cálculo de media de snr
            med = numpy.median(snr)
            # Establece el umbral de snr
            if  noise_mean > med + 3:
                th = med
            else:
                th = noise_mean + 3
            # Establece máscara
            novalid = numpy.where(snr <= th)[0]
            mask[novalid] = numpy.nan
        # Elimina datos que no sobrepasen el umbral: PARAMETRO
        novalid = numpy.where(snr <= snrth)
        cnovalid = len(novalid[0])
        if cnovalid > 0:
           mask[novalid] = numpy.nan
        novalid = numpy.where(numpy.isnan(snr))
        cnovalid = len(novalid[0])
        if cnovalid > 0:
            mask[novalid] = numpy.nan
        # umbral de velocidad
        if dopth != None:
            novalid = numpy.where(numpy.logical_or(drift< dopth*(-1), drift > dopth))
            cnovalid = len(novalid[0])
            if cnovalid > 0:
               mask[novalid] = numpy.nan
            if debug:
                print('Descartados:%i de %i:' %(cnovalid, len(drift)))
                print('Porcentaje:%3.1f' %(100.0*cnovalid/len(drift)))
           
        new_parm = numpy.zeros((Sz0[0],Sz0[1]))
        for i in range(Sz0[1]):
            new_parm[:,i] = parm[:,i] * mask

        return new_parm, mask


    def run(self, dataOut, zenith, zenithCorrection, mymode=1, dbs_sel=0, myproc=0, myantenna=0, jset=None, clean=False):


        dataOut.lat=-11.95
        dataOut.lon=-76.87
        mode=mymode
        proc=myproc
        antenna=myantenna
        nci=dataOut.nCohInt
        nptsfft=dataOut.nProfiles
        navg= 3 if jset is None else jset
        nint=dataOut.nIncohInt//navg
        navg1=dataOut.nProfiles * nint * navg
        tau1=dataOut.ippSeconds
        nipp=dataOut.radarControllerHeaderObj.ipp
        jlambda=6
        kd=213.6
        hei=dataOut.heightList.copy()
        
        nCh=dataOut.spcpar.shape[0]    
        nHei=dataOut.spcpar.shape[1]
        nParam=dataOut.spcpar.shape[2]

        parm = numpy.zeros((nCh,nHei,nParam))
        parm[:] = dataOut.spcpar[:]
        mask=numpy.ones(nHei)
        mask0=mask.copy()
        # Primer filtrado: Umbral de SNR
        for i in range(nCh):
            parm[i,:,:], mask = self.data_filter(parm[i,:,:], snrth = 0.1) # umbral 0.1 filtra señal que no corresponde a ESF, para interferometría usar -17dB
            mask0 *= mask

        ccf_results=numpy.transpose(dataOut.ccfpar,(2,1,0))
     
        for i in range(3):
            ccf_results[i,:,0] *= mask0

        zenith = numpy.array(zenith)
        zenith -= zenithCorrection
        zenith *= numpy.pi/180
        alpha = zenith[0]
        beta = zenith[1]
        
        w_w = parm[0,:,0]
        w_e = parm[1,:,0]

        if mode==1:
            # Vertical and zonal calculation            
            sinB_A = numpy.sin(beta)*numpy.cos(alpha) - numpy.sin(alpha)* numpy.cos(beta)
            w = -(w_w * numpy.sin(beta) - w_e * numpy.sin(alpha))/ sinB_A
            u = (w_w * numpy.cos(beta) - w_e * numpy.cos(alpha))/ sinB_A

        #Noise
        n0 = parm[0,:,3]
        n1 = parm[1,:,3]
        jn0_1 = numpy.nanmedian(n0)
        jn0_2 = numpy.nanmean(n0)
        jn1_1 = numpy.nanmedian(n1)
        jn1_2 = numpy.nanmean(n1)       
        noise0 = jn0_2 if numpy.abs(jn0_1-jn0_2)/(jn0_1+jn0_2) <= 0.1 else jn0_1
        noise1 = jn1_2 if numpy.abs(jn1_1-jn1_2)/(jn1_1+jn1_2) <= 0.1 else jn1_1
                     
        noise = noise0 + noise0  if mode == 1 else noise0 + noise1
        
        #Power
        apow1 = (parm[0,:,2]/numpy.sqrt(nint))*noise0 + n0
        apow2 = (parm[1,:,2]/numpy.sqrt(nint))*noise1 + n1
    
        #SNR SNR=Detectability/ SQRT(nint) or (Pow-Noise)/Noise
        s_n0 = (apow1 - noise0)/noise0
        s_n1 = (apow2 - noise1)/noise1
        
        swCH0 = parm[0,:,1]
        swCH1 = parm[1,:,1]
        
        if mode == 1:
            aacf1=(1-numpy.square(tau1)*numpy.square(4*numpy.pi/jlambda*swCH0)/2)* \
                    numpy.exp(-4*numpy.pi/jlambda*w*tau1*1j)* \
                    apow1
            aacf2=(1-numpy.square(tau1)*numpy.square(4*numpy.pi/jlambda*swCH1)/2)* \
                    numpy.exp(-4*numpy.pi/jlambda*w*tau1*1j)* \
                    apow2
            dccf_0=numpy.zeros(nHei, dtype=complex)
            
        else:
            aacf1=(1-numpy.square(tau1)*numpy.square(4*numpy.pi/jlambda*swCH0)/2)* \
                    numpy.exp(4*numpy.pi/jlambda*w_w*tau1*1j)* \
                     apow1
            aacf2=(1-numpy.square(tau1)*numpy.square(4*numpy.pi/jlambda*swCH1)/2)* \
                    numpy.exp(4*numpy.pi/jlambda*w_e*tau1*1j)* \
                    apow2
            dccf_0=numpy.power(ccf_results[0,:,0],2)*apow1*apow2* \
                                                                 numpy.exp( \
                                                                           ( \
                                                                            (1+1*(antenna==1))* \
                                                                            (-1+2*(proc == 1))* \
                                                                            ccf_results[2,:,0] \
                                                                             )*1j)

        nsamp=len(hei)
        pow0 = numpy.empty(nsamp); pow0[:] = numpy.nan
        pow1 = numpy.empty(nsamp); pow1[:] = numpy.nan
        acf0 = numpy.empty(nsamp, dtype=complex); acf0[:] = numpy.nan
        acf1 = numpy.empty(nsamp, dtype=complex); acf1[:] = numpy.nan
        dccf = numpy.empty(nsamp, dtype=complex); dccf[:] = numpy.nan
        dop0 = numpy.empty(nsamp); dop0[:] = numpy.nan
        dop1 = numpy.empty(nsamp); dop1[:] = numpy.nan
        p_w = numpy.empty(nsamp); p_w[:] = numpy.nan
        p_u = numpy.empty(nsamp); p_u[:] = numpy.nan

        if mode == 0 or (mode == 1 and dbs_sel == 0):
            ih=0
            while ih < nsamp-10:
                j=ih
                if numpy.isfinite(s_n0[ih]) and numpy.isfinite(s_n1[ih]):
                    while numpy.isfinite(s_n0[j]) and numpy.isfinite(s_n1[j]):
                        j+=1
                    if j > ih + 2:
                        for k in range(ih,j):
                            pow0[k] = apow1[k]
                            pow1[k] = apow2[k]
                            acf0[k] = aacf1[k]
                            acf1[k] = aacf2[k]
                            dccf[k] = dccf_0[k]
                    ih = j - 1
                ih+=1
        else:
            ih=0
            while ih < nsamp-10:
                j=ih
                if numpy.isfinite(s_n0[ih]):
                    while numpy.isfinite(s_n0[j]) and j < nsamp-10:
                        j+=1
                    #if j > ih + 6:
                    if j > ih + 2:
                    #if j > ih + 3:
                        for k in range(ih,j):
                            pow0[k] = apow1[k]
                            #acf0[k] = aacf1[k]
                            #dccf[k] = dccf_0[k]
                            p_w[k] = w[k]
                            dop0[k] = w_w[k]
                    ih = j - 1
                ih+=1               
            ih=0
            while ih < nsamp-10:
                j=ih
                if numpy.isfinite(s_n1[ih]):
                    while numpy.isfinite(s_n1[j]) and j < nsamp-10:
                        j+=1
                    #if j > ih + 6:
                    if j > ih + 2:
                    #if j > ih + 3:
                        for k in range(ih,j):
                            pow1[k] = apow2[k]
                            #acf1[k] = aacf2[k]
                            p_u[k] = u[k]
                            dop1[k] = w_e[k]
                    ih = j - 1
                ih+=1
            
            acf0 = numpy.zeros(nsamp, dtype=complex)
            acf1 = numpy.zeros(nsamp, dtype=complex)
            dccf = numpy.zeros(nsamp, dtype=complex)
            
        acf0 /= pow0
        acf1 /= pow1
        dccf /= pow0 * pow1
        
        if mode == 0 or (mode == 1 and dbs_sel == 0):
            sno=(pow0+pow1-noise)/noise
            # First parameter: Signal to noise ratio and its error
            sno=numpy.log10(sno)
            dsno=1.0/numpy.sqrt(nint*navg)*(1+1/sno)
            # Second parameter: Vertical Drifts
            s=numpy.sqrt(numpy.abs(acf0)*numpy.abs(acf1))
            ind=numpy.where(numpy.abs(s)>=1.0)
            if numpy.size(ind)>0:
                s[ind]=numpy.sqrt(0.9999)
            sp=s*(1.0 + 1.0/sno)
            vzo=-numpy.arctan2(numpy.imag(acf0+acf1),numpy.real(acf0+acf1))* \
             1.5e5*1.5/(nipp*numpy.pi)
            dvzo=numpy.sqrt(1-sp*sp)*0.338*1.5e5/(numpy.sqrt(nint*navg)*sp*nipp)
            ind=numpy.where(dvzo<=0.1)
            if numpy.size(ind)>0:
                dvzo[ind]=0.1
            # Third parameter: Zonal Drifts
            dt=nint*nipp/1.5e5
            ss=numpy.sqrt(numpy.abs(dccf))
            ind=numpy.where(ss>=1.0)
            if numpy.size(ind)>0:
                ss[ind]=numpy.sqrt(0.99999)
            ind=numpy.where(ss<=0.1)
            if numpy.size(ind)>0:
                ss[ind]=numpy.sqrt(0.1)
            vxo=numpy.arctan2(numpy.imag(dccf),numpy.real(dccf))*hei*1e3/(kd*dt)
            dvxo=numpy.sqrt(1.0-ss*ss)*hei*1e3/(numpy.sqrt(nint*navg)*ss*kd*dt)      
            ind=numpy.where(dvxo<=0.1)
            if numpy.size(ind)>0:
                dvxo[ind]=0.1
        else:
            sno0=(pow0-noise0)/noise0
            sno1=(pow1-noise1)/noise1
        
            # First parameter: Signal to noise ratio and its error
            sno0=numpy.log10(sno0)        
            dsno0=1.0/numpy.sqrt(nint*navg)*(1+1/sno0)
            sno1=numpy.log10(sno1)
            dsno1=1.0/numpy.sqrt(nint*navg)*(1+1/sno1)

        npar=6
        par = numpy.empty((npar, nHei)); par[:] = numpy.nan
        
        if mode == 0:
            par[0,:] = sno
            par[1,:] = vxo
            par[2,:] = dvxo
            par[3,:] = vzo
            par[4,:] = dvzo
            
        elif mode == 1 and dbs_sel == 0:
            par[0,:] = sno
            par[1,:] = vzo     
        else:
            par[0,:] = sno0
            par[1,:] = sno1
            par[2,:] = dop0
            par[3,:] = dop1
            #par[4,:] = p_w
            #par[5,:] = p_u

        if mode == 0:    
            winds = numpy.vstack((par[0,:], par[1,:], par[2,:], par[3,:], par[4,:]))
        elif mode == 1 and dbs_sel == 0:
            winds = numpy.vstack((par[0,:], par[1,:]))
        else:
            winds = numpy.vstack((par[0,:], par[1,:], par[2,:], par[3,:]))
                            
        dataOut.data_output = winds
        dataOut.data_snr = par[0,:]     

        dataOut.utctimeInit = dataOut.utctime
        dataOut.outputInterval = dataOut.timeInterval
        
        aux1= numpy.all(numpy.isnan(dataOut.data_output[0]))  # NAN vectors are not written
        aux2= numpy.all(numpy.isnan(dataOut.data_output[1]))  # NAN vectors are not written
        dataOut.flagNoData = aux1 or aux2
        
        return dataOut
                
class SALags(Operation):
    '''
    Function GetMoments()

    Input:
        self.dataOut.data_pre
        self.dataOut.abscissaList
        self.dataOut.noise
        self.dataOut.normFactor
        self.dataOut.data_snr
        self.dataOut.groupList
        self.dataOut.nChannels

    Affected:
        self.dataOut.data_param

    '''
    def run(self, dataOut):
        data_acf = dataOut.data_pre[0]
        data_ccf = dataOut.data_pre[1]
        normFactor_acf = dataOut.normFactor[0]
        normFactor_ccf = dataOut.normFactor[1]
        pairs_acf = dataOut.groupList[0]
        pairs_ccf = dataOut.groupList[1]

        nHeights = dataOut.nHeights
        absc = dataOut.abscissaList
        noise = dataOut.noise
        SNR = dataOut.data_snr
        nChannels = dataOut.nChannels
#         pairsList = dataOut.groupList
#         pairsAutoCorr, pairsCrossCorr = self.__getPairsAutoCorr(pairsList, nChannels)

        for l in range(len(pairs_acf)):
            data_acf[l,:,:] = data_acf[l,:,:]/normFactor_acf[l,:]

        for l in range(len(pairs_ccf)):
            data_ccf[l,:,:] = data_ccf[l,:,:]/normFactor_ccf[l,:]

        dataOut.data_param = numpy.zeros((len(pairs_ccf)*2 + 1, nHeights))
        dataOut.data_param[:-1,:] = self.__calculateTaus(data_acf, data_ccf, absc)
        dataOut.data_param[-1,:] = self.__calculateLag1Phase(data_acf, absc)
        return

#     def __getPairsAutoCorr(self, pairsList, nChannels):
#
#         pairsAutoCorr = numpy.zeros(nChannels, dtype = 'int')*numpy.nan
#
#         for l in range(len(pairsList)):
#             firstChannel = pairsList[l][0]
#             secondChannel = pairsList[l][1]
#
#             #Obteniendo pares de Autocorrelacion
#             if firstChannel == secondChannel:
#                 pairsAutoCorr[firstChannel] = int(l)
#
#         pairsAutoCorr = pairsAutoCorr.astype(int)
#
#         pairsCrossCorr = range(len(pairsList))
#         pairsCrossCorr = numpy.delete(pairsCrossCorr,pairsAutoCorr)
#
#         return pairsAutoCorr, pairsCrossCorr

    def __calculateTaus(self, data_acf, data_ccf, lagRange):

        lag0 = data_acf.shape[1]/2
        #Funcion de Autocorrelacion
        mean_acf = stats.nanmean(data_acf, axis = 0)

        #Obtencion Indice de TauCross
        ind_ccf = data_ccf.argmax(axis = 1)
        #Obtencion Indice de TauAuto
        ind_acf = numpy.zeros(ind_ccf.shape,dtype = 'int')
        ccf_lag0 = data_ccf[:,lag0,:]

        for i in range(ccf_lag0.shape[0]):
            ind_acf[i,:] = numpy.abs(mean_acf - ccf_lag0[i,:]).argmin(axis = 0)

        #Obtencion de TauCross y TauAuto
        tau_ccf = lagRange[ind_ccf]
        tau_acf  = lagRange[ind_acf]

        Nan1, Nan2 = numpy.where(tau_ccf == lagRange[0])

        tau_ccf[Nan1,Nan2] = numpy.nan
        tau_acf[Nan1,Nan2] = numpy.nan
        tau = numpy.vstack((tau_ccf,tau_acf))

        return tau

    def __calculateLag1Phase(self, data, lagTRange):
        data1 = stats.nanmean(data, axis = 0)
        lag1 = numpy.where(lagTRange == 0)[0][0] + 1

        phase = numpy.angle(data1[lag1,:])

        return phase

def fit_func( x, a0, a1, a2): #, a3, a4, a5):
    z = (x - a1) / a2
    y = a0 * numpy.exp(-z**2 / a2)  #+ a3 + a4 * x + a5 * x**2
    return y


class SpectralFitting(Operation):
    '''
        Function GetMoments()

        Input:
        Output:
        Variables modified:
    '''
    def __calculateMoments(self, oldspec, oldfreq, n0, nicoh = None, graph = None, smooth = None, type1 = None, fwindow = None, snrth = None, dc = None, aliasing = None, oldfd = None, wwauto = None):
        
        if (nicoh is None): nicoh = 1
        if (graph is None): graph = 0    
        if (smooth is None): smooth = 0
        elif (self.smooth < 3): smooth = 0

        if (type1 is None): type1 = 0
        if (fwindow is None): fwindow = numpy.zeros(oldfreq.size) + 1
        if (snrth is None): snrth = -3
        if (dc is None): dc = 0
        if (aliasing is None): aliasing = 0
        if (oldfd is None): oldfd = 0
        if (wwauto is None): wwauto = 0
         
        if (n0 < 1.e-20):   n0 = 1.e-20
        
        freq = oldfreq
        vec_power = numpy.zeros(oldspec.shape[1])
        vec_fd = numpy.zeros(oldspec.shape[1])
        vec_w = numpy.zeros(oldspec.shape[1])
        vec_snr = numpy.zeros(oldspec.shape[1])
        
        oldspec = numpy.ma.masked_invalid(oldspec)

        for ind in range(oldspec.shape[1]):
                        
            spec = oldspec[:,ind]
            aux = spec*fwindow
            max_spec = aux.max()
            m = list(aux).index(max_spec)
                       
            #Smooth    
            if (smooth == 0):   spec2 = spec
            else:   spec2 = scipy.ndimage.filters.uniform_filter1d(spec,size=smooth)
    
            #    Calculo de Momentos
            bb = spec2[list(range(m,spec2.size))]
            bb = (bb<n0).nonzero()
            bb = bb[0]
            
            ss = spec2[list(range(0,m + 1))]
            ss = (ss<n0).nonzero()
            ss = ss[0]
            
            if (bb.size == 0):
                bb0 = spec.size - 1 - m
            else:   
                bb0 = bb[0] - 1
                if (bb0 < 0):
                    bb0 = 0
                    
            if (ss.size == 0):   ss1 = 1
            else: ss1 = max(ss) + 1
            
            if (ss1 > m):   ss1 = m
            
            valid = numpy.asarray(list(range(int(m + bb0 - ss1 + 1)))) + ss1               
            power = ((spec2[valid] - n0)*fwindow[valid]).sum()
            fd = ((spec2[valid]- n0)*freq[valid]*fwindow[valid]).sum()/power
            w = math.sqrt(((spec2[valid] - n0)*fwindow[valid]*(freq[valid]- fd)**2).sum()/power)
            snr = (spec2.mean()-n0)/n0               
            
            if (snr < 1.e-20) :  
                snr = 1.e-20
            
            vec_power[ind] = power
            vec_fd[ind] = fd
            vec_w[ind] = w
            vec_snr[ind] = snr
        
        moments = numpy.vstack((vec_snr, vec_power, vec_fd, vec_w))
        return moments    

      #def __DiffCoherent(self,snrth, spectra, cspectra, nProf, heights,nChan, nHei, nPairs, channels, noise, crosspairs):
    def __DiffCoherent(self, spectra, cspectra, dataOut, noise, snrth, coh_th, hei_th):
        
        import matplotlib.pyplot as plt
        nProf = dataOut.nProfiles
        heights = dataOut.heightList
        nHei = len(heights)
        channels = dataOut.channelList
        nChan = len(channels)
        crosspairs = dataOut.groupList
        nPairs = len(crosspairs)
        #Separar espectros incoherentes de coherentes snr > 20 dB'
        snr_th = 10**(snrth/10.0)
        my_incoh_spectra = numpy.zeros([nChan, nProf,nHei], dtype='float')
        my_incoh_cspectra = numpy.zeros([nPairs,nProf, nHei], dtype='complex')
        my_incoh_aver = numpy.zeros([nChan, nHei])
        my_coh_aver = numpy.zeros([nChan, nHei])

        coh_spectra = numpy.zeros([nChan, nProf, nHei], dtype='float')
        coh_cspectra = numpy.zeros([nPairs, nProf, nHei], dtype='complex')
        coh_aver = numpy.zeros([nChan, nHei])
 
        incoh_spectra = numpy.zeros([nChan, nProf, nHei], dtype='float')
        incoh_cspectra = numpy.zeros([nPairs, nProf, nHei], dtype='complex')
        incoh_aver = numpy.zeros([nChan, nHei])
        power = numpy.sum(spectra, axis=1)
        
        if coh_th == None : coh_th = numpy.array([0.75,0.65,0.15]) # 0.65
        if hei_th == None : hei_th = numpy.array([60,300,650])
        for ic in range(2):
            pair = crosspairs[ic]
            #si el SNR es mayor que el SNR threshold los datos se toman coherentes
            s_n0 = power[pair[0],:]/noise[pair[0]]
            s_n1 = power[pair[1],:]/noise[pair[1]]
            
            valid1 =(s_n0>=snr_th).nonzero()
            valid2 = (s_n1>=snr_th).nonzero()
            #valid = valid2 + valid1 #numpy.concatenate((valid1,valid2), axis=None)
            valid1 =  numpy.array(valid1[0])
            valid2 =  numpy.array(valid2[0])
            valid = valid1
            for iv in range(len(valid2)):
                #for ivv in range(len(valid1)) :
                indv = numpy.array((valid1 == valid2[iv]).nonzero())
                if len(indv[0]) == 0 :
                   valid =  numpy.concatenate((valid,valid2[iv]), axis=None)
            if len(valid)>0:
                my_coh_aver[pair[0],valid]=1	    
                my_coh_aver[pair[1],valid]=1
            # si la coherencia es mayor a la coherencia threshold los datos se toman
            #print my_coh_aver[0,:]
            coh = numpy.squeeze(numpy.nansum(cspectra[ic,:,:], axis=0)/numpy.sqrt(numpy.nansum(spectra[pair[0],:,:], axis=0)*numpy.nansum(spectra[pair[1],:,:], axis=0)))
            #print('coh',numpy.absolute(coh))
            for ih in range(len(hei_th)):
                hvalid = (heights>hei_th[ih]).nonzero()
                hvalid = hvalid[0]
                if len(hvalid)>0:
                    valid = (numpy.absolute(coh[hvalid])>coh_th[ih]).nonzero()
                    valid = valid[0]
                    #print('hvalid:',hvalid)
                    #print('valid', valid)
                    if len(valid)>0:
                        my_coh_aver[pair[0],hvalid[valid]] =1
                        my_coh_aver[pair[1],hvalid[valid]] =1
        
            coh_echoes = (my_coh_aver[pair[0],:] == 1).nonzero()
            incoh_echoes = (my_coh_aver[pair[0],:] != 1).nonzero()
            incoh_echoes = incoh_echoes[0]
            if len(incoh_echoes) > 0:
                my_incoh_spectra[pair[0],:,incoh_echoes] = spectra[pair[0],:,incoh_echoes]
                my_incoh_spectra[pair[1],:,incoh_echoes] = spectra[pair[1],:,incoh_echoes]
                my_incoh_cspectra[ic,:,incoh_echoes] = cspectra[ic,:,incoh_echoes]
                my_incoh_aver[pair[0],incoh_echoes] = 1
                my_incoh_aver[pair[1],incoh_echoes] = 1

        
        for ic in range(2):
            pair = crosspairs[ic]

            valid1 =(my_coh_aver[pair[0],:]==1 ).nonzero()
            valid2 = (my_coh_aver[pair[1],:]==1).nonzero()
            valid1 = numpy.array(valid1[0])
            valid2 = numpy.array(valid2[0])
            valid = valid1
            #print valid1 , valid2
            for iv in range(len(valid2)):
                #for ivv in range(len(valid1)) :
                indv = numpy.array((valid1 == valid2[iv]).nonzero())
                if len(indv[0]) == 0 :
                   valid =  numpy.concatenate((valid,valid2[iv]), axis=None)
            #print valid
            #valid = numpy.concatenate((valid1,valid2), axis=None)
            valid1 =(my_coh_aver[pair[0],:] !=1 ).nonzero()
            valid2 = (my_coh_aver[pair[1],:] !=1).nonzero()
            valid1 = numpy.array(valid1[0])
            valid2 = numpy.array(valid2[0])
            incoh_echoes = valid1
            #print valid1, valid2
            #incoh_echoes= numpy.concatenate((valid1,valid2), axis=None)
            for iv in range(len(valid2)):
                #for ivv in range(len(valid1)) :
                indv = numpy.array((valid1 == valid2[iv]).nonzero())
                if len(indv[0]) == 0 :
                   incoh_echoes = numpy.concatenate(( incoh_echoes,valid2[iv]), axis=None)
            #print incoh_echoes
            if len(valid)>0:
                #print pair
                coh_spectra[pair[0],:,valid] = spectra[pair[0],:,valid]
                coh_spectra[pair[1],:,valid] = spectra[pair[1],:,valid]
                coh_cspectra[ic,:,valid] = cspectra[ic,:,valid]
                coh_aver[pair[0],valid]=1
                coh_aver[pair[1],valid]=1
            if len(incoh_echoes)>0:
                incoh_spectra[pair[0],:,incoh_echoes] = spectra[pair[0],:,incoh_echoes]
                incoh_spectra[pair[1],:,incoh_echoes] = spectra[pair[1],:,incoh_echoes]
                incoh_cspectra[ic,:,incoh_echoes] = cspectra[ic,:,incoh_echoes]
                incoh_aver[pair[0],incoh_echoes]=1
                incoh_aver[pair[1],incoh_echoes]=1
                #plt.imshow(spectra[0,:,:],vmin=20000000)
                #plt.show()
        #my_incoh_aver = my_incoh_aver+1
	    
        #spec = my_incoh_spectra.copy()
        #cspec = my_incoh_cspectra.copy()
        #print('######################', spec)
        #print(self.numpy)
        #return spec, cspec,coh_aver
        return  my_incoh_spectra ,my_incoh_cspectra,my_incoh_aver,my_coh_aver, incoh_spectra, coh_spectra, incoh_cspectra, coh_cspectra, incoh_aver, coh_aver
    
    def __CleanCoherent(self,snrth, spectra, cspectra, coh_aver,dataOut, noise,clean_coh_echoes,index):

        import matplotlib.pyplot as plt
        nProf = dataOut.nProfiles
        heights = dataOut.heightList
        nHei = len(heights)
        channels = dataOut.channelList
        nChan = len(channels)
        crosspairs = dataOut.groupList
        nPairs = len(crosspairs)
        
        #data = dataOut.data_pre[0]
        absc = dataOut.abscissaList[:-1]
        #noise = dataOut.noise
        #nChannel = data.shape[0]
        data_param = numpy.zeros((nChan, 4, spectra.shape[2]))
             
        
        #plt.plot(absc)
        #plt.show()
        clean_coh_spectra = spectra.copy()
        clean_coh_cspectra = cspectra.copy()
        clean_coh_aver = coh_aver.copy()

        spwd_th=[10,6]  #spwd_th[0] --> For satellites ; spwd_th[1] --> For special events like SUN.
        coh_th = 0.75

        rtime0 = [6,18] # periodo sin ESF
        rtime1 = [10.5,13.5] # periodo con alta coherencia y alto ancho espectral (esperado): SOL.

        time = index*5./60
        if clean_coh_echoes == 1 :
           for ind in range(nChan):
              data_param[ind,:,:] = self.__calculateMoments( spectra[ind,:,:] , absc , noise[ind] )
        #print data_param[:,3]
           spwd = data_param[:,3]
            #print spwd.shape
        #  SPECB_JULIA,header=anal_header,jspectra=spectra,vel=velocities,hei=heights, num_aver=1, mode_fit=0,smoothing=smoothing,jvelr=velr,jspwd=spwd,jsnr=snr,jnoise=noise,jstdvnoise=stdvnoise
           #spwd1=[ 1.65607,      1.43416,     0.500373,     0.208361,     0.000000,      26.7767,      22.5936,      26.7530,      20.6962,      29.1098,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,      28.0300,      27.0511,      27.8810,      26.3126,      27.8445,      24.6181,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000,     0.000000]
           #spwd=numpy.array([spwd1,spwd1,spwd1,spwd1])
           #print spwd.shape, heights.shape,coh_aver.shape
      # para obtener spwd
           for ic in range(nPairs):
              pair = crosspairs[ic]
              coh = numpy.squeeze(numpy.sum(cspectra[ic,:,:], axis=1)/numpy.sqrt(numpy.sum(spectra[pair[0],:,:], axis=1)*numpy.sum(spectra[pair[1],:,:], axis=1)))
              for ih in range(nHei) :
        # Considering heights higher than 200km in order to avoid removing phenomena like EEJ.
                 if heights[ih] >= 200 and coh_aver[pair[0],ih] == 1 and coh_aver[pair[1],ih] == 1 :
          # Checking coherence
                    if (numpy.abs(coh[ih]) <= coh_th) or (time >= rtime0[0] and time <= rtime0[1]) :
            # Checking spectral widths
                       if (spwd[pair[0],ih] > spwd_th[0]) or (spwd[pair[1],ih] > spwd_th[0]) :
              # satelite
                          clean_coh_spectra[pair,ih,:] = 0.0
                          clean_coh_cspectra[ic,ih,:] =  0.0
                          clean_coh_aver[pair,ih] = 0
                       else :
                            if ((spwd[pair[0],ih] < spwd_th[1]) or (spwd[pair[1],ih] < spwd_th[1])) :
                # Especial event like sun.
                               clean_coh_spectra[pair,ih,:] = 0.0
                               clean_coh_cspectra[ic,ih,:] =  0.0
                               clean_coh_aver[pair,ih] = 0

        return clean_coh_spectra, clean_coh_cspectra, clean_coh_aver

    isConfig = False
    __dataReady = False
    bloques =  None
    bloque0 = None

    def __init__(self):
        Operation.__init__(self)
        self.i=0
        self.isConfig = False
        

    def setup(self,nChan,nProf,nHei,nBlocks):
        self.__dataReady = False
        self.bloques = numpy.zeros([2, nProf, nHei,nBlocks], dtype= complex)
        self.bloque0 = numpy.zeros([nChan, nProf, nHei, nBlocks])
    
       #def CleanRayleigh(self,dataOut,spectra,cspectra,out_spectra,out_cspectra,sat_spectra,sat_cspectra,crosspairs,heights, channels, nProf,nHei,nChan,nPairs,nIncohInt,nBlocks):
    def CleanRayleigh(self,dataOut,spectra,cspectra,save_drifts):
        #import matplotlib.pyplot as plt
        #for k in range(149):

         #   self.bloque0[:,:,:,k]   = spectra[:,:,0:nHei]
         #   self.bloques[:,:,:,k]   = cspectra[:,:,0:nHei]
        #if self.i==nBlocks:
         #   self.i==0
        rfunc = cspectra.copy() #self.bloques
        n_funct = len(rfunc[0,:,0,0])
        val_spc = spectra*0.0 #self.bloque0*0.0
        val_cspc = cspectra*0.0 #self.bloques*0.0
        in_sat_spectra = spectra.copy()  #self.bloque0
        in_sat_cspectra = cspectra.copy()  #self.bloques            

        #print( rfunc.shape)
        min_hei = 200
        nProf = dataOut.nProfiles
        heights = dataOut.heightList
        nHei = len(heights)
        channels = dataOut.channelList
        nChan = len(channels)
        crosspairs = dataOut.groupList
        nPairs = len(crosspairs)
        hval=(heights >= min_hei).nonzero()
        ih=hval[0]
        #print numpy.absolute(rfunc[:,0,0,14])
        for ih in range(hval[0][0],nHei):
            for ifreq in range(nProf):
                for ii in range(n_funct):
                    
                    func2clean = 10*numpy.log10(numpy.absolute(rfunc[:,ii,ifreq,ih]))
                    #print numpy.amin(func2clean)
                    val = (numpy.isfinite(func2clean)==True).nonzero()
                    if len(val)>0:                   
                       min_val = numpy.around(numpy.amin(func2clean)-2) #> (-40)
                       if min_val <= -40 : min_val = -40
                       max_val = numpy.around(numpy.amax(func2clean)+2) #< 200
                       if max_val >= 200 : max_val = 200
                       #print min_val, max_val
                       step = 1
                            #Getting bins and the histogram
                       x_dist = min_val + numpy.arange(1 + ((max_val-(min_val))/step))*step
                       y_dist,binstep = numpy.histogram(func2clean,bins=range(int(min_val),int(max_val+2),step))                                                
                       mean = numpy.sum(x_dist * y_dist) / numpy.sum(y_dist)
                       sigma = numpy.sqrt(numpy.sum(y_dist * (x_dist - mean)**2) / numpy.sum(y_dist))
                       parg = [numpy.amax(y_dist),mean,sigma]
                       try :
                           gauss_fit, covariance = curve_fit(fit_func, x_dist, y_dist,p0=parg)
                           mode = gauss_fit[1]
                           stdv = gauss_fit[2] 
                       except:
                           mode = mean
                           stdv = sigma 
#                        if ih == 14 and ii == 0 and ifreq ==0 : 
#                            print x_dist.shape, y_dist.shape
#                            print x_dist, y_dist
#                            print min_val, max_val, binstep
#                            print func2clean
#                            print mean,sigma
#                            mean1,std = norm.fit(y_dist)
#                            print mean1, std, gauss_fit
#                            print fit_func(x_dist,gauss_fit[0],gauss_fit[1],gauss_fit[2])
                       #  7.84616      53.9307      3.61863
                       #stdv = 3.61863 # 2.99089
                       #mode = 53.9307 #7.79008

                       #Removing echoes greater than mode + 3*stdv
                       factor_stdv = 2.5
                       noval = (abs(func2clean - mode)>=(factor_stdv*stdv)).nonzero()
                       
                       if len(noval[0]) > 0:
                            novall = ((func2clean - mode) >= (factor_stdv*stdv)).nonzero()
                            cross_pairs = crosspairs[ii]
                                #Getting coherent echoes which are removed.
                            if len(novall[0]) > 0:
                                    #val_spc[(0,1),novall[a],ih] = 1
                                    #val_spc[,(2,3),novall[a],ih] = 1
                                val_spc[novall[0],cross_pairs[0],ifreq,ih] = 1
                                val_spc[novall[0],cross_pairs[1],ifreq,ih] = 1
                                val_cspc[novall[0],ii,ifreq,ih] = 1
                                #print("OUT NOVALL 1")
                                #Removing coherent from ISR data
#                             if ih == 17 and ii == 0 and ifreq ==0 : 
#                                 print spectra[:,cross_pairs[0],ifreq,ih]
                            spectra[noval,cross_pairs[0],ifreq,ih] = numpy.nan
                            spectra[noval,cross_pairs[1],ifreq,ih] = numpy.nan
                            cspectra[noval,ii,ifreq,ih] = numpy.nan
#                             if ih == 17 and ii == 0 and ifreq ==0 : 
#                                 print spectra[:,cross_pairs[0],ifreq,ih]
#                                print noval, len(noval[0])
#                                print novall, len(novall[0])
#                                print factor_stdv*stdv
#                                print func2clean-mode
#                                print val_spc[:,cross_pairs[0],ifreq,ih]
#                                print spectra[:,cross_pairs[0],ifreq,ih]
                    #no sale es para savedrifts >2
            '''                channels = channels 
                            cross_pairs = cross_pairs
                                #print("OUT NOVALL 2")

                            vcross0 = (cross_pairs[0] == channels[ii]).nonzero()
                            vcross1 = (cross_pairs[1] == channels[ii]).nonzero()
                            vcross = numpy.concatenate((vcross0,vcross1),axis=None)
                                #print('vcros =', vcross)
                        
                                #Getting coherent echoes which are removed.
                            if len(novall) > 0:
                                    #val_spc[novall,ii,ifreq,ih] = 1
                                 val_spc[ii,ifreq,ih,novall] = 1
                                 if len(vcross) > 0:
                                    val_cspc[vcross,ifreq,ih,novall] = 1                                    

                                #Removing coherent from ISR data.
                            self.bloque0[ii,ifreq,ih,noval] = numpy.nan
                            if len(vcross) > 0:
                                self.bloques[vcross,ifreq,ih,noval] = numpy.nan    
            '''
            #Getting average of the spectra and cross-spectra from incoherent echoes.
        out_spectra = numpy.zeros([nChan,nProf,nHei], dtype=float) #+numpy.nan
        out_cspectra = numpy.zeros([nPairs,nProf,nHei], dtype=complex) #+numpy.nan
        for ih in range(nHei):
            for ifreq in range(nProf):
                for ich in range(nChan):                    
                    tmp = spectra[:,ich,ifreq,ih] 
                    valid = (numpy.isfinite(tmp[:])==True).nonzero()
#                     if ich == 0 and ifreq == 0 and ih == 17 :
#                         print tmp
#                         print valid
#                         print len(valid[0])
                    #print('TMP',tmp)              
                    if len(valid[0]) >0 :
                       out_spectra[ich,ifreq,ih] = numpy.nansum(tmp)/len(valid[0])
                    #for icr in range(nPairs):
                for icr in range(nPairs):
                    tmp = numpy.squeeze(cspectra[:,icr,ifreq,ih])
                    valid = (numpy.isfinite(tmp)==True).nonzero()
                    if len(valid[0]) > 0:
                        out_cspectra[icr,ifreq,ih] = numpy.nansum(tmp)/len(valid[0])
       # print('##########################################################')
            #Removing fake coherent echoes (at least 4 points around the point)
            
        val_spectra = numpy.sum(val_spc,0)
        val_cspectra = numpy.sum(val_cspc,0)
        
        val_spectra = self.REM_ISOLATED_POINTS(val_spectra,4)
        val_cspectra = self.REM_ISOLATED_POINTS(val_cspectra,4)
        
        for i in range(nChan):
            for j in range(nProf):
                for k in range(nHei):
                    if numpy.isfinite(val_spectra[i,j,k]) and val_spectra[i,j,k] < 1 :
                        val_spc[:,i,j,k] = 0.0
        for i in range(nPairs):
            for j in range(nProf):
                for k in range(nHei):
                    if numpy.isfinite(val_cspectra[i,j,k]) and val_cspectra[i,j,k] < 1 :
                        val_cspc[:,i,j,k] = 0.0
#         val_spc = numpy.reshape(val_spc, (len(spectra[:,0,0,0]),nProf*nHei*nChan))
#         if numpy.isfinite(val_spectra)==str(True):
#            noval = (val_spectra<1).nonzero()
#            if len(noval) > 0:
#                val_spc[:,noval] = 0.0
#                val_spc = numpy.reshape(val_spc, (149,nChan,nProf,nHei))

            #val_cspc = numpy.reshape(val_spc, (149,nChan*nHei*nProf))
            #if numpy.isfinite(val_cspectra)==str(True):
             #   noval = (val_cspectra<1).nonzero()
              #  if len(noval) > 0:
               #     val_cspc[:,noval] = 0.0
                #    val_cspc = numpy.reshape(val_cspc, (149,nChan,nProf,nHei))

        tmp_sat_spectra = spectra.copy()
        tmp_sat_spectra = tmp_sat_spectra*numpy.nan
        tmp_sat_cspectra = cspectra.copy()
        tmp_sat_cspectra = tmp_sat_cspectra*numpy.nan
        
#         fig = plt.figure(figsize=(6,5))
#         left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#         ax = fig.add_axes([left, bottom, width, height]) 
#         cp = ax.contour(10*numpy.log10(numpy.absolute(spectra[0,0,:,:])))
#         ax.clabel(cp, inline=True,fontsize=10)
#         plt.show()
        
        val = (val_spc > 0).nonzero()
        if len(val[0]) > 0:              
                tmp_sat_spectra[val] = in_sat_spectra[val]
            
        val = (val_cspc > 0).nonzero()
        if len(val[0]) > 0:
                tmp_sat_cspectra[val] = in_sat_cspectra[val]

            #Getting average of the spectra and cross-spectra from incoherent echoes.
        sat_spectra = numpy.zeros((nChan,nProf,nHei), dtype=float)
        sat_cspectra = numpy.zeros((nPairs,nProf,nHei), dtype=complex)
        for ih in range(nHei):
            for ifreq in range(nProf):
                for ich in range(nChan):
                    tmp = numpy.squeeze(tmp_sat_spectra[:,ich,ifreq,ih])
                    valid = (numpy.isfinite(tmp)).nonzero()                    
                    if len(valid[0]) > 0:
                        sat_spectra[ich,ifreq,ih] = numpy.nansum(tmp)/len(valid[0])

                for icr in range(nPairs):
                    tmp = numpy.squeeze(tmp_sat_cspectra[:,icr,ifreq,ih])
                    valid = (numpy.isfinite(tmp)).nonzero()
                    if len(valid[0]) > 0:
                        sat_cspectra[icr,ifreq,ih] = numpy.nansum(tmp)/len(valid[0])
            #self.__dataReady= True
            #sat_spectra, sat_cspectra= sat_spectra, sat_cspectra          
        #if not self.__dataReady:
            #return None, None
        return out_spectra, out_cspectra,sat_spectra,sat_cspectra
    def REM_ISOLATED_POINTS(self,array,rth):
#         import matplotlib.pyplot as plt
        if rth == None : rth = 4
 
        num_prof = len(array[0,:,0])
        num_hei = len(array[0,0,:])
        n2d = len(array[:,0,0])
 
        for ii in range(n2d) :
          #print ii,n2d
          tmp = array[ii,:,:]
          #print tmp.shape, array[ii,101,:],array[ii,102,:]
         
#           fig = plt.figure(figsize=(6,5))
#           left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#           ax = fig.add_axes([left, bottom, width, height]) 
#           x = range(num_prof)
#           y = range(num_hei)  
#           cp = ax.contour(y,x,tmp)
#           ax.clabel(cp, inline=True,fontsize=10)
#           plt.show()
          
          #indxs = WHERE(FINITE(tmp) AND tmp GT 0,cindxs)
          tmp = numpy.reshape(tmp,num_prof*num_hei)
          indxs1 = (numpy.isfinite(tmp)==True).nonzero()
          indxs2 = (tmp > 0).nonzero()
          
          indxs1 = (indxs1[0])
          indxs2 = indxs2[0]
          #indxs1 = numpy.array(indxs1[0])
          #indxs2 = numpy.array(indxs2[0])
          indxs = None
          #print indxs1 , indxs2
          for iv in range(len(indxs2)):
                indv = numpy.array((indxs1 == indxs2[iv]).nonzero())
                #print len(indxs2), indv
                if len(indv[0]) > 0  :
                   indxs =  numpy.concatenate((indxs,indxs2[iv]), axis=None)
#           print indxs
          indxs = indxs[1:]
          #print indxs, len(indxs)
          if len(indxs) < 4 :
            array[ii,:,:] = 0.
            return
          
          xpos = numpy.mod(indxs ,num_hei)
          ypos = (indxs / num_hei)
          sx = numpy.argsort(xpos) # Ordering respect to "x" (time)
          #print sx
          xpos = xpos[sx]
          ypos = ypos[sx]
  
   # *********************************** Cleaning isolated points **********************************
          ic = 0
          while True : 
            r = numpy.sqrt(list(numpy.power((xpos[ic]-xpos),2)+ numpy.power((ypos[ic]-ypos),2)))
            #no_coh = WHERE(FINITE(r) AND (r LE rth),cno_coh)
            #plt.plot(r)
            #plt.show()
            no_coh1 = (numpy.isfinite(r)==True).nonzero()
            no_coh2 = (r <= rth).nonzero()
            #print r, no_coh1, no_coh2
            no_coh1 = numpy.array(no_coh1[0])
            no_coh2 = numpy.array(no_coh2[0])
            no_coh = None
          #print valid1 , valid2
            for iv in range(len(no_coh2)):
                indv = numpy.array((no_coh1 == no_coh2[iv]).nonzero())
                if len(indv[0]) > 0  :
                   no_coh =  numpy.concatenate((no_coh,no_coh2[iv]), axis=None)
            no_coh = no_coh[1:]
            #print len(no_coh), no_coh
            if len(no_coh) < 4 :
               #print xpos[ic], ypos[ic], ic
#                plt.plot(r)
#                plt.show()
               xpos[ic] = numpy.nan
               ypos[ic] = numpy.nan
            
            ic = ic + 1      
            if  (ic == len(indxs)) : 
                break
          #print( xpos, ypos)

          indxs = (numpy.isfinite(list(xpos))==True).nonzero()
          #print indxs[0] 
          if len(indxs[0]) < 4 :
             array[ii,:,:] = 0.
             return
    
          xpos = xpos[indxs[0]]
          ypos = ypos[indxs[0]]
          for i in range(0,len(ypos)):
    	      ypos[i]=int(ypos[i])
          junk = tmp
          tmp = junk*0.0
          
          tmp[list(xpos + (ypos*num_hei))] = junk[list(xpos + (ypos*num_hei))] 
          array[ii,:,:] = numpy.reshape(tmp,(num_prof,num_hei))
          
          #print array.shape
          #tmp = numpy.reshape(tmp,(num_prof,num_hei))
          #print tmp.shape
          
#           fig = plt.figure(figsize=(6,5))
#           left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#           ax = fig.add_axes([left, bottom, width, height]) 
#           x = range(num_prof)
#           y = range(num_hei)      
#           cp = ax.contour(y,x,array[ii,:,:])
#           ax.clabel(cp, inline=True,fontsize=10)
#           plt.show()
        return array
    def moments(self,doppler,yarray,npoints):
        ytemp = yarray
        #val = WHERE(ytemp GT 0,cval)    
        #if cval == 0 : val = range(npoints-1)
        val = (ytemp > 0).nonzero()
        val = val[0]
                #print('hvalid:',hvalid)
                #print('valid', valid)
        if len(val) == 0 : val = range(npoints-1) 
        
        ynew = 0.5*(ytemp[val[0]]+ytemp[val[len(val)-1]])
        ytemp[len(ytemp):] = [ynew]

        index = 0
        index = numpy.argmax(ytemp)
        ytemp = numpy.roll(ytemp,int(npoints/2)-1-index)
        ytemp = ytemp[0:npoints-1]

        fmom = numpy.sum(doppler*ytemp)/numpy.sum(ytemp)+(index-(npoints/2-1))*numpy.abs(doppler[1]-doppler[0])
        smom = numpy.sum(doppler*doppler*ytemp)/numpy.sum(ytemp)
        return [fmom,numpy.sqrt(smom)]
    # **********************************************************************************************           
    index = 0
    fint = 0
    buffer = 0
    buffer2 = 0
    buffer3 = 0
    def run(self, dataOut, getSNR = True, path=None, file=None, groupList=None):
        nChannels = dataOut.nChannels
        nHeights= dataOut.heightList.size
        nProf = dataOut.nProfiles
        tini=time.localtime(dataOut.utctime)
        if (tini.tm_min % 5) == 0 and (tini.tm_sec < 5 and self.fint==0): 
#            print tini.tm_min
           self.index = 0
           jspc = self.buffer
           jcspc = self.buffer2
           jnoise = self.buffer3
           self.buffer = dataOut.data_spc
           self.buffer2 = dataOut.data_cspc
           self.buffer3 = dataOut.noise
           self.fint = 1
           if numpy.any(jspc) :
               jspc= numpy.reshape(jspc,(int(len(jspc)/4),nChannels,nProf,nHeights))
               jcspc= numpy.reshape(jcspc,(int(len(jcspc)/2),2,nProf,nHeights))
               jnoise= numpy.reshape(jnoise,(int(len(jnoise)/4),nChannels))
           else:
               dataOut.flagNoData = True
               return dataOut
        else :
           if (tini.tm_min % 5) == 0 : self.fint = 1
           else : self.fint = 0
           self.index += 1
           if numpy.any(self.buffer):       
              self.buffer = numpy.concatenate((self.buffer,dataOut.data_spc), axis=0)
              self.buffer2 = numpy.concatenate((self.buffer2,dataOut.data_cspc), axis=0)
              self.buffer3 = numpy.concatenate((self.buffer3,dataOut.noise), axis=0)
           else:
               self.buffer = dataOut.data_spc
               self.buffer2 = dataOut.data_cspc
               self.buffer3 = dataOut.noise
           dataOut.flagNoData = True
           return dataOut
        if path != None:
            sys.path.append(path)
        self.library = importlib.import_module(file)

        #To be inserted as a parameter
        groupArray = numpy.array(groupList)
        #groupArray = numpy.array([[0,1],[2,3]]) 
        dataOut.groupList = groupArray

        nGroups = groupArray.shape[0]
        nChannels = dataOut.nChannels
        nHeights = dataOut.heightList.size

        #Parameters Array
        dataOut.data_param = None
        dataOut.data_paramC = None

        #Set constants
        constants = self.library.setConstants(dataOut)
        dataOut.constants = constants
        M = dataOut.normFactor
        N = dataOut.nFFTPoints
        ippSeconds = dataOut.ippSeconds
        K = dataOut.nIncohInt
        pairsArray = numpy.array(dataOut.pairsList)

        snrth= 20
        spectra = dataOut.data_spc
        cspectra = dataOut.data_cspc
        nProf = dataOut.nProfiles
        heights = dataOut.heightList
        nHei = len(heights)        
        channels = dataOut.channelList
        nChan = len(channels)
        nIncohInt = dataOut.nIncohInt
        crosspairs = dataOut.groupList
        noise = dataOut.noise
        jnoise = jnoise/N
        noise = numpy.nansum(jnoise,axis=0)#/len(jnoise)
        power = numpy.sum(spectra, axis=1)
        nPairs = len(crosspairs)
        absc = dataOut.abscissaList[:-1]

        if not self.isConfig:
            self.isConfig = True

        index = tini.tm_hour*12+tini.tm_min/5
        jspc = jspc/N/N
        jcspc = jcspc/N/N
        tmp_spectra,tmp_cspectra,sat_spectra,sat_cspectra = self.CleanRayleigh(dataOut,jspc,jcspc,2)
        jspectra = tmp_spectra*len(jspc[:,0,0,0])
        jcspectra = tmp_cspectra*len(jspc[:,0,0,0])
        my_incoh_spectra ,my_incoh_cspectra,my_incoh_aver,my_coh_aver, incoh_spectra, coh_spectra, incoh_cspectra, coh_cspectra, incoh_aver, coh_aver = self.__DiffCoherent(jspectra, jcspectra, dataOut, noise, snrth, None, None)
        clean_coh_spectra, clean_coh_cspectra, clean_coh_aver = self.__CleanCoherent(snrth, coh_spectra, coh_cspectra, coh_aver, dataOut, noise,1,index)                                        
        dataOut.data_spc = incoh_spectra
        dataOut.data_cspc = incoh_cspectra
        
        clean_num_aver = incoh_aver*len(jspc[:,0,0,0])
        coh_num_aver = clean_coh_aver*len(jspc[:,0,0,0])
        #List of possible combinations
        listComb = itertools.combinations(numpy.arange(groupArray.shape[1]),2)
        indCross = numpy.zeros(len(list(listComb)), dtype = 'int')

        if getSNR:
            listChannels = groupArray.reshape((groupArray.size))
            listChannels.sort()
            dataOut.data_SNR = self.__getSNR(dataOut.data_spc[listChannels,:,:], noise[listChannels])
        if dataOut.data_paramC is None:
                    dataOut.data_paramC = numpy.zeros((nGroups*4, nHeights,2))*numpy.nan 
        for i in range(nGroups): 
            coord = groupArray[i,:]
            #Input data array
            data = dataOut.data_spc[coord,:,:]/(M*N)
            data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))

            #Cross Spectra data array for Covariance Matrixes
            ind = 0
            for pairs in listComb:
                pairsSel = numpy.array([coord[x],coord[y]])
                indCross[ind] = int(numpy.where(numpy.all(pairsArray == pairsSel, axis = 1))[0][0])
                ind += 1
            dataCross = dataOut.data_cspc[indCross,:,:]/(M*N)
            dataCross = dataCross**2
            nhei = nHeights
            poweri = numpy.sum(dataOut.data_spc[:,1:nProf-0,:],axis=1)/clean_num_aver[:,:]
            if i == 0 : my_noises = numpy.zeros(4,dtype=float) #FLTARR(4)
            n0i = numpy.nanmin(poweri[0+i*2,0:nhei-0])/(nProf-1)
            n1i = numpy.nanmin(poweri[1+i*2,0:nhei-0])/(nProf-1)
            n0 = n0i
            n1=  n1i
            my_noises[2*i+0] = n0
            my_noises[2*i+1] = n1
            snrth = -16.0
            snrth = 10**(snrth/10.0)

            for h in range(nHeights):
                d = data[:,h]
                smooth = clean_num_aver[i+1,h] #dataOut.data_spc[:,1:nProf-0,:]
                signalpn0 = (dataOut.data_spc[i*2,1:(nProf-0),h])/smooth
                signalpn1 = (dataOut.data_spc[i*2+1,1:(nProf-0),h])/smooth
                signal0 = signalpn0-n0
                signal1 = signalpn1-n1
                snr0 = numpy.sum(signal0/n0)/(nProf-1)
                snr1 = numpy.sum(signal1/n1)/(nProf-1)
                if snr0 > snrth and snr1 > snrth and clean_num_aver[i+1,h] > 0 :
                #Covariance Matrix
                    D = numpy.diag(d**2)
                    ind = 0
                    for pairs in listComb:
                    #Coordinates in Covariance Matrix
                        x = pairs[0]    
                        y = pairs[1]
                    #Channel Index
                        S12 = dataCross[ind,:,h]
                        D12 = numpy.diag(S12)
                    #Completing Covariance Matrix with Cross Spectras
                        D[x*N:(x+1)*N,y*N:(y+1)*N] = D12
                        D[y*N:(y+1)*N,x*N:(x+1)*N] = D12
                        ind += 1
                    diagD = numpy.zeros(256)
                    if h == 17 : 
                        for ii in range(256): diagD[ii] =  D[ii,ii]
                    #Dinv=numpy.linalg.inv(D)
                    #L=numpy.linalg.cholesky(Dinv)
                    try:
                       Dinv=numpy.linalg.inv(D)
                       L=numpy.linalg.cholesky(Dinv)
                    except:
                       Dinv = D*numpy.nan
                       L= D*numpy.nan
                    LT=L.T

                    dp = numpy.dot(LT,d)
                
                #Initial values
                    data_spc = dataOut.data_spc[coord,:,h]

                    if (h>0)and(error1[3]<5):
                        p0 = dataOut.data_param[i,:,h-1]
                    else:
                        p0 = numpy.array(self.library.initialValuesFunction(data_spc, constants))# sin el i(data_spc, constants, i)
                    try:
                    #Least Squares
                    #print (dp,LT,constants)
                    #value =self.__residFunction(p0,dp,LT,constants)
                    #print ("valueREADY",value.shape, type(value))
                    #optimize.leastsq(value)
                        minp,covp,infodict,mesg,ier = optimize.leastsq(self.__residFunction,p0,args=(dp,LT,constants),full_output=True)
                    #minp,covp = optimize.leastsq(self.__residFunction,p0,args=(dp,LT,constants))
                    #Chi square error
                    #print(minp,covp.infodict,mesg,ier)
                    #print("REALIZA OPTIMIZ")
                        error0 = numpy.sum(infodict['fvec']**2)/(2*N)
                    #Error with Jacobian
                        error1 = self.library.errorFunction(minp,constants,LT)
#                         print self.__residFunction(p0,dp,LT, constants)  
#                         print infodict['fvec']
#                         print self.__residFunction(minp,dp,LT,constants)

                    except:
                        minp = p0*numpy.nan
                        error0 = numpy.nan
                        error1 = p0*numpy.nan
                    #print ("EXCEPT 0000000000")
#                     s_sq = (self.__residFunction(minp,dp,LT,constants)).sum()/(len(dp)-len(p0))
#                     covp = covp*s_sq
#                     #print("TRY___________________________________________1")
#                     error = [] 
#                     for ip in range(len(minp)):
#                         try:
#                             error.append(numpy.absolute(covp[ip][ip])**0.5)                    
#                         except:
#                             error.append( 0.00 )
                else :
                    data_spc = dataOut.data_spc[coord,:,h]
                    p0 = numpy.array(self.library.initialValuesFunction(data_spc, constants))
                    minp = p0*numpy.nan
                    error0 = numpy.nan
                    error1 = p0*numpy.nan                                         
                #Save
                if dataOut.data_param is None:
                    dataOut.data_param = numpy.zeros((nGroups, p0.size, nHeights))*numpy.nan
                    dataOut.data_error = numpy.zeros((nGroups, p0.size + 1, nHeights))*numpy.nan
                
                dataOut.data_error[i,:,h] = numpy.hstack((error0,error1))
                dataOut.data_param[i,:,h] = minp

            for ht in range(nHeights-1) :
                smooth = coh_num_aver[i+1,ht] #datc[0,ht,0,beam] 
                dataOut.data_paramC[4*i,ht,1] = smooth
                signalpn0 = (coh_spectra[i*2  ,1:(nProf-0),ht])/smooth #coh_spectra
                signalpn1 = (coh_spectra[i*2+1,1:(nProf-0),ht])/smooth
       
                #val0 = WHERE(signalpn0 > 0,cval0)
                val0 = (signalpn0 > 0).nonzero()
                val0 = val0[0]
                #print('hvalid:',hvalid)
                #print('valid', valid)
                if len(val0) == 0 : val0_npoints = nProf 
                else : val0_npoints = len(val0) 
                
                #val1 = WHERE(signalpn1 > 0,cval1)
                val1 = (signalpn1 > 0).nonzero()
                val1 = val1[0]
                if len(val1) == 0 : val1_npoints = nProf
                else : val1_npoints = len(val1)

                dataOut.data_paramC[0+4*i,ht,0] = numpy.sum((signalpn0/val0_npoints))/n0
                dataOut.data_paramC[1+4*i,ht,0] = numpy.sum((signalpn1/val1_npoints))/n1
        
                signal0 = (signalpn0-n0) # > 0
                vali = (signal0 < 0).nonzero()
                vali = vali[0]
                if len(vali) > 0 : signal0[vali] = 0
                signal1 = (signalpn1-n1) #> 0
                vali = (signal1 < 0).nonzero()
                vali = vali[0]
                if len(vali) > 0 : signal1[vali] = 0
                snr0 = numpy.sum(signal0/n0)/(nProf-1)
                snr1 = numpy.sum(signal1/n1)/(nProf-1)
                doppler = absc[1:]
                if snr0 >= snrth and snr1 >= snrth and smooth :
                    signalpn0_n0 = signalpn0
                    signalpn0_n0[val0] = signalpn0[val0] - n0
                    mom0 = self.moments(doppler,signalpn0-n0,nProf)
#                     sigtmp= numpy.transpose(numpy.tile(signalpn0, [4,1]))
#                     momt= self.__calculateMoments( sigtmp, doppler , n0 )
                    signalpn1_n1 = signalpn1
                    signalpn1_n1[val1] = signalpn1[val1] - n1
                    mom1 = self.moments(doppler,signalpn1_n1,nProf)
                    dataOut.data_paramC[2+4*i,ht,0] = (mom0[0]+mom1[0])/2.
                    dataOut.data_paramC[3+4*i,ht,0] = (mom0[1]+mom1[1])/2.
#                     if graph == 1 :
#                     window, 13
#                     plot,doppler,signalpn0
#                     oplot,doppler,signalpn1,linest=1
#                     oplot,mom0(0)*doppler/doppler,signalpn0
#                     oplot,mom1(0)*doppler/doppler,signalpn1
#                     print,interval/12.,beam,45+ht*15,snr0,snr1,mom0(0),mom1(0),mom0(1),mom1(1)
                    #ENDIF
                #ENDIF
            #ENDFOR End height

        dataOut.data_spc = jspectra
        if getSNR:
            listChannels = groupArray.reshape((groupArray.size))
            listChannels.sort()

            dataOut.data_snr = self.__getSNR(dataOut.data_spc[listChannels,:,:], my_noises[listChannels])
        return dataOut
    
    def __residFunction(self, p, dp, LT, constants):

        fm = self.library.modelFunction(p, constants)
        fmp=numpy.dot(LT,fm)
        return  dp-fmp

    def __getSNR(self, z, noise):

        avg = numpy.average(z, axis=1)
        SNR = (avg.T-noise)/noise
        SNR = SNR.T
        return SNR

    def __chisq(self, p, chindex, hindex):
        #similar to Resid but calculates CHI**2
        [LT,d,fm]=setupLTdfm(p,chindex,hindex)
        dp=numpy.dot(LT,d)
        fmp=numpy.dot(LT,fm)
        chisq=numpy.dot((dp-fmp).T,(dp-fmp))
        return chisq

class WindProfiler(Operation):

    __isConfig = False

    __initime = None
    __lastdatatime = None
    __integrationtime = None

    __buffer = None

    __dataReady = False

    __firstdata = None

    n = None

    def __init__(self):
        Operation.__init__(self)

    def __calculateCosDir(self, elev, azim):
        zen = (90 - elev)*numpy.pi/180
        azim = azim*numpy.pi/180
        cosDirX = numpy.sqrt((1-numpy.cos(zen)**2)/((1+numpy.tan(azim)**2)))
        cosDirY = numpy.sqrt(1-numpy.cos(zen)**2-cosDirX**2)

        signX = numpy.sign(numpy.cos(azim))
        signY = numpy.sign(numpy.sin(azim))

        cosDirX = numpy.copysign(cosDirX, signX)
        cosDirY = numpy.copysign(cosDirY, signY)
        return cosDirX, cosDirY

    def __calculateAngles(self, theta_x, theta_y, azimuth):

        dir_cosw = numpy.sqrt(1-theta_x**2-theta_y**2)
        zenith_arr = numpy.arccos(dir_cosw)
        azimuth_arr = numpy.arctan2(theta_x,theta_y) + azimuth*math.pi/180

        dir_cosu = numpy.sin(azimuth_arr)*numpy.sin(zenith_arr)
        dir_cosv = numpy.cos(azimuth_arr)*numpy.sin(zenith_arr)

        return azimuth_arr, zenith_arr, dir_cosu, dir_cosv, dir_cosw

    def __calculateMatA(self, dir_cosu, dir_cosv, dir_cosw, horOnly):

        if horOnly:
            A = numpy.c_[dir_cosu,dir_cosv]
        else:
            A = numpy.c_[dir_cosu,dir_cosv,dir_cosw]
        A = numpy.asmatrix(A)
        A1 = numpy.linalg.inv(A.transpose()*A)*A.transpose()

        return A1

    def __correctValues(self, heiRang, phi, velRadial, SNR):
        listPhi = phi.tolist()
        maxid = listPhi.index(max(listPhi))
        minid = listPhi.index(min(listPhi))

        rango = list(range(len(phi)))
   #     rango = numpy.delete(rango,maxid)

        heiRang1 = heiRang*math.cos(phi[maxid])
        heiRangAux = heiRang*math.cos(phi[minid])
        indOut = (heiRang1 < heiRangAux[0]).nonzero()
        heiRang1 = numpy.delete(heiRang1,indOut)

        velRadial1 = numpy.zeros([len(phi),len(heiRang1)])
        SNR1 = numpy.zeros([len(phi),len(heiRang1)])

        for i in rango:
            x = heiRang*math.cos(phi[i])
            y1 = velRadial[i,:]
            f1 = interpolate.interp1d(x,y1,kind = 'cubic')

            x1 = heiRang1
            y11 = f1(x1)

            y2 = SNR[i,:]
            f2 = interpolate.interp1d(x,y2,kind = 'cubic')
            y21 = f2(x1)

            velRadial1[i,:] = y11
            SNR1[i,:] = y21

        return heiRang1, velRadial1, SNR1

    def __calculateVelUVW(self, A, velRadial):

        #Operacion Matricial
#         velUVW = numpy.zeros((velRadial.shape[1],3))
#         for ind in range(velRadial.shape[1]):
#             velUVW[ind,:] = numpy.dot(A,velRadial[:,ind])
#         velUVW = velUVW.transpose()
        velUVW = numpy.zeros((A.shape[0],velRadial.shape[1]))
        velUVW[:,:] = numpy.dot(A,velRadial)


        return velUVW

#     def techniqueDBS(self, velRadial0, dirCosx, disrCosy, azimuth, correct, horizontalOnly, heiRang, SNR0):

    def techniqueDBS(self, kwargs):
        """
        Function that implements Doppler Beam Swinging (DBS) technique.

        Input:    Radial velocities, Direction cosines (x and y) of the Beam, Antenna azimuth,
                    Direction correction (if necessary), Ranges and SNR

        Output:    Winds estimation (Zonal, Meridional and Vertical)

        Parameters affected:    Winds, height range, SNR
        """
        velRadial0 = kwargs['velRadial']
        heiRang = kwargs['heightList']
        SNR0 = kwargs['SNR']

        if 'dirCosx' in kwargs and 'dirCosy' in kwargs:
            theta_x = numpy.array(kwargs['dirCosx'])
            theta_y = numpy.array(kwargs['dirCosy'])
        else:
            elev = numpy.array(kwargs['elevation'])
            azim = numpy.array(kwargs['azimuth'])
            theta_x, theta_y = self.__calculateCosDir(elev, azim)
        azimuth = kwargs['correctAzimuth']
        if 'horizontalOnly' in kwargs:
            horizontalOnly = kwargs['horizontalOnly']
        else:   horizontalOnly = False
        if 'correctFactor' in kwargs:
            correctFactor = kwargs['correctFactor']
        else:   correctFactor = 1
        if 'channelList' in kwargs:
            channelList = kwargs['channelList']
            if len(channelList) == 2:
                horizontalOnly = True
            arrayChannel = numpy.array(channelList)
            param = param[arrayChannel,:,:]
            theta_x = theta_x[arrayChannel]
            theta_y = theta_y[arrayChannel]

        azimuth_arr, zenith_arr, dir_cosu, dir_cosv, dir_cosw = self.__calculateAngles(theta_x, theta_y, azimuth)
        heiRang1, velRadial1, SNR1 = self.__correctValues(heiRang, zenith_arr, correctFactor*velRadial0, SNR0)
        A = self.__calculateMatA(dir_cosu, dir_cosv, dir_cosw, horizontalOnly)

        #Calculo de Componentes de la velocidad con DBS
        winds = self.__calculateVelUVW(A,velRadial1)

        return winds, heiRang1, SNR1

    def __calculateDistance(self, posx, posy, pairs_ccf, azimuth = None):

        nPairs = len(pairs_ccf)
        posx = numpy.asarray(posx)
        posy = numpy.asarray(posy)

        #Rotacion Inversa para alinear con el azimuth
        if azimuth!= None:
            azimuth = azimuth*math.pi/180
            posx1 = posx*math.cos(azimuth) + posy*math.sin(azimuth)
            posy1 = -posx*math.sin(azimuth) + posy*math.cos(azimuth)
        else:
            posx1 = posx
            posy1 = posy

        #Calculo de Distancias
        distx = numpy.zeros(nPairs)
        disty = numpy.zeros(nPairs)
        dist = numpy.zeros(nPairs)
        ang = numpy.zeros(nPairs)

        for i in range(nPairs):
            distx[i] = posx1[pairs_ccf[i][1]] - posx1[pairs_ccf[i][0]]
            disty[i] = posy1[pairs_ccf[i][1]] - posy1[pairs_ccf[i][0]]
            dist[i] = numpy.sqrt(distx[i]**2 + disty[i]**2)
            ang[i] = numpy.arctan2(disty[i],distx[i])

        return distx, disty, dist, ang
        #Calculo de Matrices
#         nPairs = len(pairs)
#         ang1 = numpy.zeros((nPairs, 2, 1))
#         dist1 = numpy.zeros((nPairs, 2, 1))
#
#         for j in range(nPairs):
#             dist1[j,0,0] = dist[pairs[j][0]]
#             dist1[j,1,0] = dist[pairs[j][1]]
#             ang1[j,0,0] = ang[pairs[j][0]]
#             ang1[j,1,0] = ang[pairs[j][1]]
#
#         return distx,disty, dist1,ang1


    def __calculateVelVer(self, phase, lagTRange, _lambda):

        Ts = lagTRange[1] - lagTRange[0]
        velW = -_lambda*phase/(4*math.pi*Ts)

        return velW

    def __calculateVelHorDir(self, dist, tau1, tau2, ang):
        nPairs = tau1.shape[0]
        nHeights = tau1.shape[1]
        vel = numpy.zeros((nPairs,3,nHeights))
        dist1 = numpy.reshape(dist, (dist.size,1))

        angCos = numpy.cos(ang)
        angSin = numpy.sin(ang)

        vel0 = dist1*tau1/(2*tau2**2)
        vel[:,0,:] = (vel0*angCos).sum(axis = 1)
        vel[:,1,:] = (vel0*angSin).sum(axis = 1)

        ind = numpy.where(numpy.isinf(vel))
        vel[ind] = numpy.nan

        return vel

#     def __getPairsAutoCorr(self, pairsList, nChannels):
#
#         pairsAutoCorr = numpy.zeros(nChannels, dtype = 'int')*numpy.nan
#
#         for l in range(len(pairsList)):
#             firstChannel = pairsList[l][0]
#             secondChannel = pairsList[l][1]
#
#             #Obteniendo pares de Autocorrelacion
#             if firstChannel == secondChannel:
#                 pairsAutoCorr[firstChannel] = int(l)
#
#         pairsAutoCorr = pairsAutoCorr.astype(int)
#
#         pairsCrossCorr = range(len(pairsList))
#         pairsCrossCorr = numpy.delete(pairsCrossCorr,pairsAutoCorr)
#
#         return pairsAutoCorr, pairsCrossCorr

#     def techniqueSA(self, pairsSelected, pairsList, nChannels, tau, azimuth, _lambda, position_x, position_y, lagTRange, correctFactor):
    def techniqueSA(self, kwargs):

        """
        Function that implements Spaced Antenna (SA) technique.

        Input:    Radial velocities, Direction cosines (x and y) of the Beam, Antenna azimuth,
                    Direction correction (if necessary), Ranges and SNR

        Output:    Winds estimation (Zonal, Meridional and Vertical)

        Parameters affected:    Winds
        """
        position_x = kwargs['positionX']
        position_y = kwargs['positionY']
        azimuth = kwargs['azimuth']

        if 'correctFactor' in kwargs:
            correctFactor = kwargs['correctFactor']
        else:
            correctFactor = 1

        groupList = kwargs['groupList']
        pairs_ccf = groupList[1]
        tau = kwargs['tau']
        _lambda = kwargs['_lambda']

        #Cross Correlation pairs obtained
#         pairsAutoCorr, pairsCrossCorr = self.__getPairsAutoCorr(pairssList, nChannels)
#         pairsArray = numpy.array(pairsList)[pairsCrossCorr]
#         pairsSelArray = numpy.array(pairsSelected)
#         pairs = []
#
#         #Wind estimation pairs obtained
#         for i in range(pairsSelArray.shape[0]/2):
#             ind1 = numpy.where(numpy.all(pairsArray == pairsSelArray[2*i], axis = 1))[0][0]
#             ind2 = numpy.where(numpy.all(pairsArray == pairsSelArray[2*i + 1], axis = 1))[0][0]
#             pairs.append((ind1,ind2))

        indtau = tau.shape[0]/2
        tau1 = tau[:indtau,:]
        tau2 = tau[indtau:-1,:]
#         tau1 = tau1[pairs,:]
#         tau2 = tau2[pairs,:]
        phase1 = tau[-1,:]

        #---------------------------------------------------------------------
        #Metodo Directo
        distx, disty, dist, ang = self.__calculateDistance(position_x, position_y, pairs_ccf,azimuth)
        winds = self.__calculateVelHorDir(dist, tau1, tau2, ang)
        winds = stats.nanmean(winds, axis=0)
        #---------------------------------------------------------------------
        #Metodo General
#         distx, disty, dist = self.calculateDistance(position_x,position_y,pairsCrossCorr, pairsList, azimuth)
#         #Calculo Coeficientes de Funcion de Correlacion
#         F,G,A,B,H = self.calculateCoef(tau1,tau2,distx,disty,n)
#         #Calculo de Velocidades
#         winds = self.calculateVelUV(F,G,A,B,H)

        #---------------------------------------------------------------------
        winds[2,:] = self.__calculateVelVer(phase1, lagTRange, _lambda)
        winds = correctFactor*winds
        return winds

    def __checkTime(self, currentTime, paramInterval, outputInterval):

        dataTime = currentTime + paramInterval
        deltaTime = dataTime - self.__initime

        if deltaTime >= outputInterval or deltaTime < 0:
            self.__dataReady = True
        return

    def techniqueMeteors(self, arrayMeteor, meteorThresh, heightMin, heightMax):
        '''
        Function that implements winds estimation technique with detected meteors.

        Input:    Detected meteors, Minimum meteor quantity to wind estimation

        Output:    Winds estimation (Zonal and Meridional)

        Parameters affected:    Winds
        '''
        #Settings
        nInt = (heightMax - heightMin)/2
        nInt = int(nInt)
        winds = numpy.zeros((2,nInt))*numpy.nan

        #Filter errors
        error = numpy.where(arrayMeteor[:,-1] == 0)[0]
        finalMeteor = arrayMeteor[error,:]

        #Meteor Histogram
        finalHeights = finalMeteor[:,2]
        hist = numpy.histogram(finalHeights, bins = nInt, range = (heightMin,heightMax))
        nMeteorsPerI = hist[0]
        heightPerI = hist[1]

        #Sort of meteors
        indSort = finalHeights.argsort()
        finalMeteor2 = finalMeteor[indSort,:]

        #    Calculating winds
        ind1 = 0
        ind2 = 0

        for i in range(nInt):
            nMet = nMeteorsPerI[i]
            ind1 = ind2
            ind2 = ind1 + nMet

            meteorAux = finalMeteor2[ind1:ind2,:]

            if meteorAux.shape[0] >= meteorThresh:
                vel = meteorAux[:, 6]
                zen = meteorAux[:, 4]*numpy.pi/180
                azim = meteorAux[:, 3]*numpy.pi/180

                n = numpy.cos(zen)
        #         m = (1 - n**2)/(1 - numpy.tan(azim)**2)
        #         l = m*numpy.tan(azim)
                l = numpy.sin(zen)*numpy.sin(azim)
                m = numpy.sin(zen)*numpy.cos(azim)

                A = numpy.vstack((l, m)).transpose()
                A1 = numpy.dot(numpy.linalg.inv( numpy.dot(A.transpose(),A) ),A.transpose())
                windsAux = numpy.dot(A1, vel)

                winds[0,i] = windsAux[0]
                winds[1,i] = windsAux[1]

        return winds, heightPerI[:-1]

    def techniqueNSM_SA(self, **kwargs):
        metArray = kwargs['metArray']
        heightList = kwargs['heightList']
        timeList = kwargs['timeList']

        rx_location = kwargs['rx_location']
        groupList = kwargs['groupList']
        azimuth = kwargs['azimuth']
        dfactor = kwargs['dfactor']
        k = kwargs['k']

        azimuth1, dist = self.__calculateAzimuth1(rx_location, groupList, azimuth)
        d = dist*dfactor
        #Phase calculation
        metArray1 = self.__getPhaseSlope(metArray, heightList, timeList)

        metArray1[:,-2] = metArray1[:,-2]*metArray1[:,2]*1000/(k*d[metArray1[:,1].astype(int)]) #angles into velocities

        velEst = numpy.zeros((heightList.size,2))*numpy.nan
        azimuth1 = azimuth1*numpy.pi/180

        for i in range(heightList.size):
            h = heightList[i]
            indH = numpy.where((metArray1[:,2] == h)&(numpy.abs(metArray1[:,-2]) < 100))[0]
            metHeight = metArray1[indH,:]
            if metHeight.shape[0] >= 2:
                velAux = numpy.asmatrix(metHeight[:,-2]).T    #Radial Velocities
                iazim = metHeight[:,1].astype(int)
                azimAux = numpy.asmatrix(azimuth1[iazim]).T    #Azimuths
                A = numpy.hstack((numpy.cos(azimAux),numpy.sin(azimAux)))
                A = numpy.asmatrix(A)
                A1 = numpy.linalg.pinv(A.transpose()*A)*A.transpose()
                velHor = numpy.dot(A1,velAux)

                velEst[i,:] = numpy.squeeze(velHor)
        return velEst

    def __getPhaseSlope(self, metArray, heightList, timeList):
        meteorList = []
        #utctime sec1 height SNR velRad ph0 ph1 ph2 coh0 coh1 coh2
        #Putting back together the meteor matrix
        utctime = metArray[:,0]
        uniqueTime = numpy.unique(utctime)

        phaseDerThresh = 0.5
        ippSeconds = timeList[1] - timeList[0]
        sec = numpy.where(timeList>1)[0][0]
        nPairs = metArray.shape[1] - 6
        nHeights = len(heightList)

        for t in uniqueTime:
            metArray1 = metArray[utctime==t,:]
#         phaseDerThresh = numpy.pi/4 #reducir Phase thresh
            tmet = metArray1[:,1].astype(int)
            hmet = metArray1[:,2].astype(int)

            metPhase = numpy.zeros((nPairs, heightList.size, timeList.size - 1))
            metPhase[:,:] = numpy.nan
            metPhase[:,hmet,tmet] = metArray1[:,6:].T

            #Delete short trails
            metBool = ~numpy.isnan(metPhase[0,:,:])
            heightVect = numpy.sum(metBool, axis = 1)
            metBool[heightVect<sec,:] = False
            metPhase[:,heightVect<sec,:] = numpy.nan

            #Derivative
            metDer = numpy.abs(metPhase[:,:,1:] - metPhase[:,:,:-1])
            phDerAux = numpy.dstack((numpy.full((nPairs,nHeights,1), False, dtype=bool),metDer > phaseDerThresh))
            metPhase[phDerAux] = numpy.nan

            #--------------------------METEOR DETECTION    -----------------------------------------
            indMet = numpy.where(numpy.any(metBool,axis=1))[0]

            for p in numpy.arange(nPairs):
                phase = metPhase[p,:,:]
                phDer = metDer[p,:,:]

                for h in indMet:
                    height = heightList[h]
                    phase1 = phase[h,:] #82
                    phDer1 = phDer[h,:]

                    phase1[~numpy.isnan(phase1)] = numpy.unwrap(phase1[~numpy.isnan(phase1)])   #Unwrap

                    indValid = numpy.where(~numpy.isnan(phase1))[0]
                    initMet = indValid[0]
                    endMet = 0

                    for i in range(len(indValid)-1):

                        #Time difference
                        inow = indValid[i]
                        inext = indValid[i+1]
                        idiff = inext - inow
                        #Phase difference
                        phDiff = numpy.abs(phase1[inext] - phase1[inow])

                        if idiff>sec or phDiff>numpy.pi/4 or inext==indValid[-1]:   #End of Meteor
                            sizeTrail = inow - initMet + 1
                            if sizeTrail>3*sec:  #Too short meteors
                                x = numpy.arange(initMet,inow+1)*ippSeconds
                                y = phase1[initMet:inow+1]
                                ynnan = ~numpy.isnan(y)
                                x = x[ynnan]
                                y = y[ynnan]
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                                ylin = x*slope + intercept
                                rsq = r_value**2
                                if rsq > 0.5:
                                    vel = slope#*height*1000/(k*d)
                                    estAux = numpy.array([utctime,p,height, vel, rsq])
                                    meteorList.append(estAux)
                            initMet = inext
        metArray2 = numpy.array(meteorList)

        return metArray2

    def __calculateAzimuth1(self, rx_location, pairslist, azimuth0):

        azimuth1 = numpy.zeros(len(pairslist))
        dist = numpy.zeros(len(pairslist))

        for i in range(len(rx_location)):
            ch0 = pairslist[i][0]
            ch1 = pairslist[i][1]

            diffX = rx_location[ch0][0] - rx_location[ch1][0]
            diffY = rx_location[ch0][1] - rx_location[ch1][1]
            azimuth1[i] = numpy.arctan2(diffY,diffX)*180/numpy.pi
            dist[i] = numpy.sqrt(diffX**2 + diffY**2)

        azimuth1 -= azimuth0
        return azimuth1, dist

    def techniqueNSM_DBS(self, **kwargs):
        metArray = kwargs['metArray']
        heightList = kwargs['heightList']
        timeList = kwargs['timeList']
        azimuth = kwargs['azimuth']
        theta_x = numpy.array(kwargs['theta_x'])
        theta_y = numpy.array(kwargs['theta_y'])

        utctime = metArray[:,0]
        cmet = metArray[:,1].astype(int)
        hmet = metArray[:,3].astype(int)
        SNRmet = metArray[:,4]
        vmet = metArray[:,5]
        spcmet = metArray[:,6]

        nChan = numpy.max(cmet) + 1
        nHeights = len(heightList)

        azimuth_arr, zenith_arr, dir_cosu, dir_cosv, dir_cosw = self.__calculateAngles(theta_x, theta_y, azimuth)
        hmet = heightList[hmet]
        h1met = hmet*numpy.cos(zenith_arr[cmet])      #Corrected heights

        velEst = numpy.zeros((heightList.size,2))*numpy.nan

        for i in range(nHeights - 1):
            hmin = heightList[i]
            hmax = heightList[i + 1]

            thisH = (h1met>=hmin) & (h1met<hmax) & (cmet!=2) & (SNRmet>8) & (vmet<50) & (spcmet<10)
            indthisH = numpy.where(thisH)

            if numpy.size(indthisH) > 3:

                vel_aux = vmet[thisH]
                chan_aux = cmet[thisH]
                cosu_aux = dir_cosu[chan_aux]
                cosv_aux = dir_cosv[chan_aux]
                cosw_aux = dir_cosw[chan_aux]

                nch = numpy.size(numpy.unique(chan_aux))
                if  nch > 1:
                    A = self.__calculateMatA(cosu_aux, cosv_aux, cosw_aux, True)
                    velEst[i,:] = numpy.dot(A,vel_aux)

        return velEst

    def run(self, dataOut, technique, nHours=1, hmin=70, hmax=110, **kwargs):

        param = dataOut.data_param
        if dataOut.abscissaList != None:
            absc = dataOut.abscissaList[:-1]
        # noise = dataOut.noise
        heightList = dataOut.heightList
        SNR = dataOut.data_snr

        if technique == 'DBS':

            kwargs['velRadial'] = param[:,1,:] #Radial velocity
            kwargs['heightList'] = heightList
            kwargs['SNR'] = SNR

            dataOut.data_output, dataOut.heightList, dataOut.data_snr = self.techniqueDBS(kwargs) #DBS Function
            dataOut.utctimeInit = dataOut.utctime
            dataOut.outputInterval = dataOut.paramInterval

        elif technique == 'SA':

            #Parameters
#             position_x = kwargs['positionX']
#             position_y = kwargs['positionY']
#             azimuth = kwargs['azimuth']
#
#             if kwargs.has_key('crosspairsList'):
#                 pairs = kwargs['crosspairsList']
#             else:
#                 pairs = None
#
#             if kwargs.has_key('correctFactor'):
#                 correctFactor = kwargs['correctFactor']
#             else:
#                 correctFactor = 1

#             tau = dataOut.data_param
#             _lambda = dataOut.C/dataOut.frequency
#             pairsList = dataOut.groupList
#             nChannels = dataOut.nChannels

            kwargs['groupList'] = dataOut.groupList
            kwargs['tau'] = dataOut.data_param
            kwargs['_lambda'] = dataOut.C/dataOut.frequency
#             dataOut.data_output = self.techniqueSA(pairs, pairsList, nChannels, tau, azimuth, _lambda, position_x, position_y, absc, correctFactor)
            dataOut.data_output = self.techniqueSA(kwargs)
            dataOut.utctimeInit = dataOut.utctime
            dataOut.outputInterval = dataOut.timeInterval

        elif technique == 'Meteors':
            dataOut.flagNoData = True
            self.__dataReady = False

            if 'nHours' in kwargs:
                nHours = kwargs['nHours']
            else:
                nHours = 1

            if 'meteorsPerBin' in kwargs:
                meteorThresh = kwargs['meteorsPerBin']
            else:
                meteorThresh = 6

            if 'hmin' in kwargs:
                hmin = kwargs['hmin']
            else:   hmin = 70
            if 'hmax' in kwargs:
                hmax = kwargs['hmax']
            else:   hmax = 110

            dataOut.outputInterval = nHours*3600

            if self.__isConfig == False:
#                 self.__initime = dataOut.datatime.replace(minute = 0, second = 0, microsecond = 03)
                #Get Initial LTC time
                self.__initime = datetime.datetime.utcfromtimestamp(dataOut.utctime)
                self.__initime = (self.__initime.replace(minute = 0, second = 0, microsecond = 0) - datetime.datetime(1970, 1, 1)).total_seconds()

                self.__isConfig = True

            if self.__buffer is None:
                self.__buffer = dataOut.data_param
                self.__firstdata = copy.copy(dataOut)

            else:
                self.__buffer = numpy.vstack((self.__buffer, dataOut.data_param))

            self.__checkTime(dataOut.utctime, dataOut.paramInterval, dataOut.outputInterval) #Check if the buffer is ready

            if self.__dataReady:
                dataOut.utctimeInit = self.__initime

                self.__initime += dataOut.outputInterval #to erase time offset

                dataOut.data_output, dataOut.heightList = self.techniqueMeteors(self.__buffer, meteorThresh, hmin, hmax)
                dataOut.flagNoData = False
                self.__buffer = None

        elif technique == 'Meteors1':
            dataOut.flagNoData = True
            self.__dataReady = False

            if 'nMins' in kwargs:
                nMins = kwargs['nMins']
            else: nMins = 20
            if 'rx_location' in kwargs:
                rx_location = kwargs['rx_location']
            else: rx_location = [(0,1),(1,1),(1,0)]
            if 'azimuth' in kwargs:
                azimuth = kwargs['azimuth']
            else: azimuth = 51.06
            if 'dfactor' in kwargs:
                dfactor = kwargs['dfactor']
            if 'mode' in kwargs:
                mode = kwargs['mode']
            if 'theta_x' in kwargs:
                theta_x = kwargs['theta_x']
            if 'theta_y' in kwargs:
                theta_y = kwargs['theta_y']
            else: mode = 'SA'

            #Borrar luego esto
            if dataOut.groupList is None:
                dataOut.groupList = [(0,1),(0,2),(1,2)]
            groupList = dataOut.groupList
            C = 3e8
            freq = 50e6
            lamb = C/freq
            k = 2*numpy.pi/lamb

            timeList = dataOut.abscissaList
            heightList = dataOut.heightList

            if self.__isConfig == False:
                dataOut.outputInterval = nMins*60
#                 self.__initime = dataOut.datatime.replace(minute = 0, second = 0, microsecond = 03)
                #Get Initial LTC time
                initime = datetime.datetime.utcfromtimestamp(dataOut.utctime)
                minuteAux = initime.minute
                minuteNew = int(numpy.floor(minuteAux/nMins)*nMins)
                self.__initime = (initime.replace(minute = minuteNew, second = 0, microsecond = 0) - datetime.datetime(1970, 1, 1)).total_seconds()

                self.__isConfig = True

            if self.__buffer is None:
                self.__buffer = dataOut.data_param
                self.__firstdata = copy.copy(dataOut)

            else:
                self.__buffer = numpy.vstack((self.__buffer, dataOut.data_param))

            self.__checkTime(dataOut.utctime, dataOut.paramInterval, dataOut.outputInterval) #Check if the buffer is ready

            if self.__dataReady:
                dataOut.utctimeInit = self.__initime
                self.__initime += dataOut.outputInterval #to erase time offset

                metArray = self.__buffer
                if mode == 'SA':
                    dataOut.data_output = self.techniqueNSM_SA(rx_location=rx_location, groupList=groupList, azimuth=azimuth, dfactor=dfactor, k=k,metArray=metArray, heightList=heightList,timeList=timeList)
                elif mode == 'DBS':
                    dataOut.data_output = self.techniqueNSM_DBS(metArray=metArray,heightList=heightList,timeList=timeList, azimuth=azimuth, theta_x=theta_x, theta_y=theta_y)
                dataOut.data_output = dataOut.data_output.T
                dataOut.flagNoData = False
                self.__buffer = None

        return

class EWDriftsEstimation(Operation):

    def __init__(self):
        Operation.__init__(self)

    def __correctValues(self, heiRang, phi, velRadial, SNR):
        listPhi = phi.tolist()
        maxid = listPhi.index(max(listPhi))
        minid = listPhi.index(min(listPhi))

        rango = list(range(len(phi)))
   #     rango = numpy.delete(rango,maxid)

        heiRang1 = heiRang*math.cos(phi[maxid])
        heiRangAux = heiRang*math.cos(phi[minid])
        indOut = (heiRang1 < heiRangAux[0]).nonzero()
        heiRang1 = numpy.delete(heiRang1,indOut)

        velRadial1 = numpy.zeros([len(phi),len(heiRang1)])
        SNR1 = numpy.zeros([len(phi),len(heiRang1)])

        for i in rango:
            x = heiRang*math.cos(phi[i])
            y1 = velRadial[i,:]
            vali= (numpy.isfinite(y1)==True).nonzero()
            y1=y1[vali]
            x = x[vali]
            f1 = interpolate.interp1d(x,y1,kind = 'cubic',bounds_error=False)

            #heiRang1 = x*math.cos(phi[maxid])
            x1 = heiRang1
            y11 = f1(x1)

            y2 = SNR[i,:]
            #print 'snr ', y2
            x = heiRang*math.cos(phi[i])
            vali= (y2 != -1).nonzero()
            y2 = y2[vali]
            x = x[vali]
            #print 'snr  ',y2
            f2 = interpolate.interp1d(x,y2,kind = 'cubic',bounds_error=False)
            y21 = f2(x1)

            velRadial1[i,:] = y11
            SNR1[i,:] = y21

        return heiRang1, velRadial1, SNR1

          
  
    def run(self, dataOut, zenith, zenithCorrection):

        heiRang = dataOut.heightList
        velRadial = dataOut.data_param[:,3,:]
        velRadialm = dataOut.data_param[:,2:4,:]*-1

        rbufc=dataOut.data_paramC[:,:,0]
        ebufc=dataOut.data_paramC[:,:,1]
        SNR = dataOut.data_snr
        velRerr = dataOut.data_error[:,4,:]
        moments=numpy.vstack(([velRadialm[0,:]],[velRadialm[0,:]],[velRadialm[1,:]],[velRadialm[1,:]]))
        dataOut.moments=moments
        # Coherent
        smooth_wC = ebufc[0,:]
        p_w0C = rbufc[0,:]
        p_w1C = rbufc[1,:]
        w_wC = rbufc[2,:]*-1 #*radial_sign(radial EQ 1)
        t_wC = rbufc[3,:]
        my_nbeams = 2

        zenith = numpy.array(zenith)
        zenith -= zenithCorrection
        zenith *= numpy.pi/180
        if  zenithCorrection != 0 :
            heiRang1, velRadial1, SNR1 = self.__correctValues(heiRang, numpy.abs(zenith), velRadial, SNR)
        else :
            heiRang1 = heiRang
            velRadial1 = velRadial
            SNR1 = SNR
            
        alp = zenith[0]
        bet = zenith[1]

        w_w = velRadial1[0,:]
        w_e = velRadial1[1,:]
        w_w_err = velRerr[0,:]
        w_e_err = velRerr[1,:]

        val = (numpy.isfinite(w_w)==False).nonzero()
        val = val[0]
        bad = val
        if len(bad) > 0 :
            w_w[bad] = w_wC[bad]
            w_w_err[bad]= numpy.nan
        if my_nbeams == 2:
            smooth_eC=ebufc[4,:]
            p_e0C = rbufc[4,:]
            p_e1C = rbufc[5,:]
            w_eC = rbufc[6,:]*-1
            t_eC = rbufc[7,:]
            val = (numpy.isfinite(w_e)==False).nonzero()
            val = val[0]
            bad = val
            if len(bad) > 0 :
                w_e[bad] = w_eC[bad]
                w_e_err[bad]= numpy.nan
                
        w = (w_w*numpy.sin(bet) - w_e*numpy.sin(alp))/(numpy.cos(alp)*numpy.sin(bet) - numpy.cos(bet)*numpy.sin(alp))   
        u = (w_w*numpy.cos(bet) - w_e*numpy.cos(alp))/(numpy.sin(alp)*numpy.cos(bet) - numpy.sin(bet)*numpy.cos(alp))   

        w_err = numpy.sqrt((w_w_err*numpy.sin(bet))**2.+(w_e_err*numpy.sin(alp))**2.)/ numpy.absolute(numpy.cos(alp)*numpy.sin(bet)-numpy.cos(bet)*numpy.sin(alp))
        u_err = numpy.sqrt((w_w_err*numpy.cos(bet))**2.+(w_e_err*numpy.cos(alp))**2.)/ numpy.absolute(numpy.cos(alp)*numpy.sin(bet)-numpy.cos(bet)*numpy.sin(alp))
        
        winds = numpy.vstack((w,u))

        dataOut.heightList = heiRang1
        dataOut.data_output = winds

        snr1 = 10*numpy.log10(SNR1[0])
        dataOut.data_snr1 = numpy.reshape(snr1,(1,snr1.shape[0]))
        dataOut.utctimeInit = dataOut.utctime
        dataOut.outputInterval = dataOut.timeInterval
        
        hei_aver0 = 218 
        jrange = 450 #900 para HA drifts
        deltah = 15.0 #dataOut.spacing(0)
        h0 = 0.0 #dataOut.first_height(0)
        heights = dataOut.heightList
        nhei = len(heights)
         
        range1 = numpy.arange(nhei) * deltah + h0
 
        #jhei = WHERE(range1 GE hei_aver0 , jcount)
        jhei = (range1 >= hei_aver0).nonzero()
        if len(jhei[0]) > 0 :
            h0_index = jhei[0][0] # Initial height for getting averages 218km
 
        mynhei = 7
        nhei_avg = int(jrange/deltah)
        h_avgs = int(nhei_avg/mynhei)
        nhei_avg = h_avgs*(mynhei-1)+mynhei
 
        navgs = numpy.zeros(mynhei,dtype='float')
        delta_h = numpy.zeros(mynhei,dtype='float')
        range_aver = numpy.zeros(mynhei,dtype='float')
        for ih in range( mynhei-1 ):
            range_aver[ih] = numpy.sum(range1[h0_index+h_avgs*ih:h0_index+h_avgs*(ih+1)-0])/h_avgs
            navgs[ih] = h_avgs
            delta_h[ih] = deltah*h_avgs
         
        range_aver[mynhei-1] = numpy.sum(range1[h0_index:h0_index+6*h_avgs-0])/(6*h_avgs)
        navgs[mynhei-1] = 6*h_avgs
        delta_h[mynhei-1] = deltah*6*h_avgs
        
        wA = w[h0_index:h0_index+nhei_avg-0]
        wA_err = w_err[h0_index:h0_index+nhei_avg-0]

        for i in range(5) :
            vals = wA[i*h_avgs:(i+1)*h_avgs-0]
            errs = wA_err[i*h_avgs:(i+1)*h_avgs-0]
            avg = numpy.nansum(vals/errs**2.)/numpy.nansum(1./errs**2.)
            sigma = numpy.sqrt(1./numpy.nansum(1./errs**2.))
            wA[6*h_avgs+i] = avg
            wA_err[6*h_avgs+i] = sigma
         
 
        vals = wA[0:6*h_avgs-0]
        errs=wA_err[0:6*h_avgs-0]
        avg = numpy.nansum(vals/errs**2.)/numpy.nansum(1./errs**2)
        sigma = numpy.sqrt(1./numpy.nansum(1./errs**2.))
        wA[nhei_avg-1] = avg
        wA_err[nhei_avg-1] = sigma
 
        wA = wA[6*h_avgs:nhei_avg-0]
        wA_err=wA_err[6*h_avgs:nhei_avg-0]
        if my_nbeams == 2 :

            uA = u[h0_index:h0_index+nhei_avg]
            uA_err=u_err[h0_index:h0_index+nhei_avg]

            for i in range(5) :
                vals = uA[i*h_avgs:(i+1)*h_avgs-0]
                errs=uA_err[i*h_avgs:(i+1)*h_avgs-0]
                avg = numpy.nansum(vals/errs**2.)/numpy.nansum(1./errs**2.)
                sigma = numpy.sqrt(1./numpy.nansum(1./errs**2.))
                uA[6*h_avgs+i] = avg
                uA_err[6*h_avgs+i]=sigma

            vals = uA[0:6*h_avgs-0]
            errs = uA_err[0:6*h_avgs-0]
            avg = numpy.nansum(vals/errs**2.)/numpy.nansum(1./errs**2.)
            sigma = numpy.sqrt(1./numpy.nansum(1./errs**2.))
            uA[nhei_avg-1] = avg
            uA_err[nhei_avg-1] = sigma
            uA = uA[6*h_avgs:nhei_avg-0]
            uA_err = uA_err[6*h_avgs:nhei_avg-0]
        
        dataOut.drifts_avg = numpy.vstack((wA,uA))

        tini=time.localtime(dataOut.utctime)
        datefile= str(tini[0]).zfill(4)+str(tini[1]).zfill(2)+str(tini[2]).zfill(2)
        nfile = '/home/pcondor/Database/ewdriftsschain2019/jro'+datefile+'drifts_sch3.txt'

        f1 = open(nfile,'a')

        datedriftavg=str(tini[0])+' '+str(tini[1])+' '+str(tini[2])+' '+str(tini[3])+' '+str(tini[4])
        driftavgstr=str(dataOut.drifts_avg)

        numpy.savetxt(f1,numpy.column_stack([tini[0],tini[1],tini[2],tini[3],tini[4]]),fmt='%4i')
        numpy.savetxt(f1,dataOut.drifts_avg,fmt='%10.2f')
        f1.close()

        return dataOut

#---------------    Non Specular Meteor    ----------------

class NonSpecularMeteorDetection(Operation):

    def run(self, dataOut, mode, SNRthresh=8, phaseDerThresh=0.5, cohThresh=0.8, allData = False):
        data_acf = dataOut.data_pre[0]
        data_ccf = dataOut.data_pre[1]
        pairsList = dataOut.groupList[1]

        lamb = dataOut.C/dataOut.frequency
        tSamp = dataOut.ippSeconds*dataOut.nCohInt
        paramInterval = dataOut.paramInterval

        nChannels = data_acf.shape[0]
        nLags = data_acf.shape[1]
        nProfiles = data_acf.shape[2]
        nHeights = dataOut.nHeights
        nCohInt = dataOut.nCohInt
        sec = numpy.round(nProfiles/dataOut.paramInterval)
        heightList = dataOut.heightList
        ippSeconds = dataOut.ippSeconds*dataOut.nCohInt*dataOut.nAvg
        utctime = dataOut.utctime

        dataOut.abscissaList = numpy.arange(0,paramInterval+ippSeconds,ippSeconds)

        #------------------------    SNR    --------------------------------------
        power = data_acf[:,0,:,:].real
        noise = numpy.zeros(nChannels)
        SNR = numpy.zeros(power.shape)
        for i in range(nChannels):
            noise[i] = hildebrand_sekhon(power[i,:], nCohInt)
            SNR[i] = (power[i]-noise[i])/noise[i]
        SNRm = numpy.nanmean(SNR, axis = 0)
        SNRdB = 10*numpy.log10(SNR)

        if mode == 'SA':
            dataOut.groupList = dataOut.groupList[1]
            nPairs = data_ccf.shape[0]
            #----------------------    Coherence and Phase   --------------------------
            phase = numpy.zeros(data_ccf[:,0,:,:].shape)
#             phase1 = numpy.copy(phase)
            coh1 = numpy.zeros(data_ccf[:,0,:,:].shape)

            for p in range(nPairs):
                ch0 = pairsList[p][0]
                ch1 = pairsList[p][1]
                ccf = data_ccf[p,0,:,:]/numpy.sqrt(data_acf[ch0,0,:,:]*data_acf[ch1,0,:,:])
                phase[p,:,:] = ndimage.median_filter(numpy.angle(ccf), size = (5,1)) #median filter
#                 phase1[p,:,:] = numpy.angle(ccf) #median filter
                coh1[p,:,:] = ndimage.median_filter(numpy.abs(ccf), 5) #median filter
#                 coh1[p,:,:] = numpy.abs(ccf) #median filter
            coh = numpy.nanmax(coh1, axis = 0)
#             struc = numpy.ones((5,1))
#             coh = ndimage.morphology.grey_dilation(coh, size=(10,1))
            #----------------------    Radial Velocity    ----------------------------
            phaseAux = numpy.mean(numpy.angle(data_acf[:,1,:,:]), axis = 0)
            velRad = phaseAux*lamb/(4*numpy.pi*tSamp)

            if allData:
                boolMetFin = ~numpy.isnan(SNRm)
#                 coh[:-1,:] = numpy.nanmean(numpy.abs(phase[:,1:,:] - phase[:,:-1,:]),axis=0)
            else:
                #------------------------    Meteor mask    ---------------------------------
#                 #SNR mask
#                 boolMet = (SNRdB>SNRthresh)#|(~numpy.isnan(SNRdB))
#
#                 #Erase small objects
#                 boolMet1 = self.__erase_small(boolMet, 2*sec, 5)
#
#                 auxEEJ = numpy.sum(boolMet1,axis=0)
#                 indOver = auxEEJ>nProfiles*0.8  #Use this later
#                 indEEJ = numpy.where(indOver)[0]
#                 indNEEJ = numpy.where(~indOver)[0]
#
#                 boolMetFin = boolMet1
#
#                 if indEEJ.size > 0:
#                     boolMet1[:,indEEJ] = False  #Erase heights with EEJ
#
#                     boolMet2 = coh > cohThresh
#                     boolMet2 = self.__erase_small(boolMet2, 2*sec,5)
#
#                     #Final Meteor mask
#                     boolMetFin = boolMet1|boolMet2

                #Coherence mask
                boolMet1 = coh > 0.75
                struc = numpy.ones((30,1))
                boolMet1 = ndimage.morphology.binary_dilation(boolMet1, structure=struc)

                #Derivative mask
                derPhase = numpy.nanmean(numpy.abs(phase[:,1:,:] - phase[:,:-1,:]),axis=0)
                boolMet2 = derPhase < 0.2
#                 boolMet2 = ndimage.morphology.binary_opening(boolMet2)
#                 boolMet2 = ndimage.morphology.binary_closing(boolMet2, structure = numpy.ones((10,1)))
                boolMet2 = ndimage.median_filter(boolMet2,size=5)
                boolMet2 = numpy.vstack((boolMet2,numpy.full((1,nHeights), True, dtype=bool)))
#                 #Final mask
#                 boolMetFin = boolMet2
                boolMetFin = boolMet1&boolMet2
#                 boolMetFin = ndimage.morphology.binary_dilation(boolMetFin)
            #Creating data_param
            coordMet = numpy.where(boolMetFin)

            tmet = coordMet[0]
            hmet = coordMet[1]

            data_param = numpy.zeros((tmet.size, 6 + nPairs))
            data_param[:,0] = utctime
            data_param[:,1] = tmet
            data_param[:,2] = hmet
            data_param[:,3] = SNRm[tmet,hmet]
            data_param[:,4] = velRad[tmet,hmet]
            data_param[:,5] = coh[tmet,hmet]
            data_param[:,6:] = phase[:,tmet,hmet].T

        elif mode == 'DBS':
            dataOut.groupList = numpy.arange(nChannels)

            #Radial Velocities
            phase = numpy.angle(data_acf[:,1,:,:])
#             phase = ndimage.median_filter(numpy.angle(data_acf[:,1,:,:]), size = (1,5,1))
            velRad = phase*lamb/(4*numpy.pi*tSamp)

            #Spectral width
#             acf1 = ndimage.median_filter(numpy.abs(data_acf[:,1,:,:]), size = (1,5,1))
#             acf2 = ndimage.median_filter(numpy.abs(data_acf[:,2,:,:]), size = (1,5,1))
            acf1 = data_acf[:,1,:,:]
            acf2 = data_acf[:,2,:,:]

            spcWidth = (lamb/(2*numpy.sqrt(6)*numpy.pi*tSamp))*numpy.sqrt(numpy.log(acf1/acf2))
#             velRad = ndimage.median_filter(velRad, size = (1,5,1))
            if allData:
                boolMetFin = ~numpy.isnan(SNRdB)
            else:
                #SNR
                boolMet1 = (SNRdB>SNRthresh) #SNR mask
                boolMet1 = ndimage.median_filter(boolMet1, size=(1,5,5))

                #Radial velocity
                boolMet2 = numpy.abs(velRad) < 20
                boolMet2 = ndimage.median_filter(boolMet2, (1,5,5))

                #Spectral Width
                boolMet3 = spcWidth < 30
                boolMet3 = ndimage.median_filter(boolMet3, (1,5,5))
#                 boolMetFin = self.__erase_small(boolMet1, 10,5)
                boolMetFin = boolMet1&boolMet2&boolMet3

            #Creating data_param
            coordMet = numpy.where(boolMetFin)

            cmet = coordMet[0]
            tmet = coordMet[1]
            hmet = coordMet[2]

            data_param = numpy.zeros((tmet.size, 7))
            data_param[:,0] = utctime
            data_param[:,1] = cmet
            data_param[:,2] = tmet
            data_param[:,3] = hmet
            data_param[:,4] = SNR[cmet,tmet,hmet].T
            data_param[:,5] = velRad[cmet,tmet,hmet].T
            data_param[:,6] = spcWidth[cmet,tmet,hmet].T

#         self.dataOut.data_param = data_int
        if len(data_param) == 0:
            dataOut.flagNoData = True
        else:
            dataOut.data_param = data_param

    def __erase_small(self, binArray, threshX, threshY):
        labarray, numfeat = ndimage.measurements.label(binArray)
        binArray1 = numpy.copy(binArray)

        for i in range(1,numfeat + 1):
            auxBin = (labarray==i)
            auxSize = auxBin.sum()

            x,y = numpy.where(auxBin)
            widthX = x.max() - x.min()
            widthY = y.max() - y.min()

            #width X: 3 seg -> 12.5*3
            #width Y:

            if (auxSize < 50) or (widthX < threshX) or (widthY < threshY):
                binArray1[auxBin] = False

        return binArray1

#---------------    Specular Meteor    ----------------

class SMDetection(Operation):
    '''
        Function DetectMeteors()
            Project developed with paper:
            HOLDSWORTH ET AL. 2004

        Input:
            self.dataOut.data_pre

            centerReceiverIndex:      From the channels, which is the center receiver

            hei_ref:                  Height reference for the Beacon signal extraction
            tauindex:
            predefinedPhaseShifts:    Predefined phase offset for the voltge signals

            cohDetection:             Whether to user Coherent detection or not
            cohDet_timeStep:          Coherent Detection calculation time step
            cohDet_thresh:            Coherent Detection phase threshold to correct phases

            noise_timeStep:           Noise calculation time step
            noise_multiple:           Noise multiple to define signal threshold

            multDet_timeLimit:        Multiple Detection Removal time limit in seconds
            multDet_rangeLimit:       Multiple Detection Removal range limit in km

            phaseThresh:              Maximum phase difference between receiver to be consider a meteor
            SNRThresh:                Minimum SNR threshold of the meteor signal to be consider a meteor

            hmin:                     Minimum Height of the meteor to use it in the further wind estimations
            hmax:                     Maximum Height of the meteor to use it in the further wind estimations
            azimuth:                  Azimuth angle correction

        Affected:
            self.dataOut.data_param

        Rejection Criteria (Errors):
            0: No error; analysis OK
            1: SNR < SNR threshold
            2: angle of arrival (AOA) ambiguously determined
            3: AOA estimate not feasible
            4: Large difference in AOAs obtained from different antenna baselines
            5: echo at start or end of time series
            6: echo less than 5 examples long; too short for analysis
            7: echo rise exceeds 0.3s
            8: echo decay time less than twice rise time
            9: large power level before echo
            10: large power level after echo
            11: poor fit to amplitude for estimation of decay time
            12: poor fit to CCF phase variation for estimation of radial drift velocity
            13: height unresolvable echo: not valid height within 70 to 110 km
            14: height ambiguous echo: more then one possible height within 70 to 110 km
            15: radial drift velocity or projected horizontal velocity exceeds 200 m/s
            16: oscilatory echo, indicating event most likely not an underdense echo

            17: phase difference in meteor Reestimation

        Data Storage:
            Meteors for Wind Estimation   (8):
            Utc Time   |    Range    Height
            Azimuth    Zenith    errorCosDir
            VelRad    errorVelRad
            Phase0 Phase1 Phase2 Phase3
            TypeError

         '''

    def run(self, dataOut, hei_ref = None, tauindex = 0,
                      phaseOffsets = None,
                      cohDetection = False, cohDet_timeStep = 1, cohDet_thresh = 25,
                      noise_timeStep = 4, noise_multiple = 4,
                      multDet_timeLimit = 1, multDet_rangeLimit = 3,
                      phaseThresh = 20, SNRThresh = 5,
                      hmin = 50, hmax=150, azimuth = 0,
                      channelPositions = None) :


        #Getting Pairslist
        if channelPositions is None:
#             channelPositions = [(2.5,0), (0,2.5), (0,0), (0,4.5), (-2,0)]   #T
            channelPositions = [(4.5,2), (2,4.5), (2,2), (2,0), (0,2)]   #Estrella
        meteorOps = SMOperations()
        pairslist0, distances = meteorOps.getPhasePairs(channelPositions)
        heiRang = dataOut.heightList
        #Get Beacon signal - No Beacon signal anymore
#         newheis = numpy.where(self.dataOut.heightList>self.dataOut.radarControllerHeaderObj.Taus[tauindex])
#
#         if hei_ref != None:
#             newheis = numpy.where(self.dataOut.heightList>hei_ref)
#


        #****************REMOVING HARDWARE PHASE DIFFERENCES***************
        # see if the user put in pre defined phase shifts
        voltsPShift = dataOut.data_pre.copy()

#         if predefinedPhaseShifts != None:
#             hardwarePhaseShifts = numpy.array(predefinedPhaseShifts)*numpy.pi/180
#
# #         elif beaconPhaseShifts:
# #             #get hardware phase shifts using beacon signal
# #             hardwarePhaseShifts = self.__getHardwarePhaseDiff(self.dataOut.data_pre, pairslist, newheis, 10)
# #             hardwarePhaseShifts = numpy.insert(hardwarePhaseShifts,centerReceiverIndex,0)
#
#         else:
#             hardwarePhaseShifts = numpy.zeros(5)
#
#         voltsPShift = numpy.zeros((self.dataOut.data_pre.shape[0],self.dataOut.data_pre.shape[1],self.dataOut.data_pre.shape[2]), dtype = 'complex')
#         for i in range(self.dataOut.data_pre.shape[0]):
#             voltsPShift[i,:,:] = self.__shiftPhase(self.dataOut.data_pre[i,:,:], hardwarePhaseShifts[i])

        #******************END OF REMOVING HARDWARE PHASE DIFFERENCES*********

        #Remove DC
        voltsDC = numpy.mean(voltsPShift,1)
        voltsDC = numpy.mean(voltsDC,1)
        for i in range(voltsDC.shape[0]):
            voltsPShift[i] = voltsPShift[i] - voltsDC[i]

        #Don't considerate last heights, theyre used to calculate Hardware Phase Shift
#         voltsPShift = voltsPShift[:,:,:newheis[0][0]]

        #************ FIND POWER OF DATA W/COH OR NON COH DETECTION (3.4) **********
        #Coherent Detection
        if cohDetection:
            #use coherent detection to get the net power
            cohDet_thresh = cohDet_thresh*numpy.pi/180
            voltsPShift = self.__coherentDetection(voltsPShift, cohDet_timeStep, dataOut.timeInterval, pairslist0, cohDet_thresh)

        #Non-coherent detection!
        powerNet = numpy.nansum(numpy.abs(voltsPShift[:,:,:])**2,0)
        #********** END OF COH/NON-COH POWER CALCULATION**********************

        #********** FIND THE NOISE LEVEL AND POSSIBLE METEORS ****************
        #Get noise
        noise, noise1 = self.__getNoise(powerNet, noise_timeStep, dataOut.timeInterval)
#         noise = self.getNoise1(powerNet, noise_timeStep, self.dataOut.timeInterval)
        #Get signal threshold
        signalThresh = noise_multiple*noise
        #Meteor echoes detection
        listMeteors = self.__findMeteors(powerNet, signalThresh)
        #******* END OF NOISE LEVEL AND POSSIBLE METEORS CACULATION **********

        #************** REMOVE MULTIPLE DETECTIONS (3.5) ***************************
        #Parameters
        heiRange = dataOut.heightList
        rangeInterval = heiRange[1] - heiRange[0]
        rangeLimit = multDet_rangeLimit/rangeInterval
        timeLimit = multDet_timeLimit/dataOut.timeInterval
        #Multiple detection removals
        listMeteors1 = self.__removeMultipleDetections(listMeteors, rangeLimit, timeLimit)
        #************ END OF REMOVE MULTIPLE DETECTIONS **********************

        #*********************     METEOR REESTIMATION  (3.7, 3.8, 3.9, 3.10)   ********************
        #Parameters
        phaseThresh = phaseThresh*numpy.pi/180
        thresh = [phaseThresh, noise_multiple, SNRThresh]
        #Meteor reestimation  (Errors N 1, 6, 12, 17)
        listMeteors2, listMeteorsPower, listMeteorsVolts = self.__meteorReestimation(listMeteors1, voltsPShift, pairslist0, thresh, noise, dataOut.timeInterval, dataOut.frequency)
#         listMeteors2, listMeteorsPower, listMeteorsVolts = self.meteorReestimation3(listMeteors2, listMeteorsPower, listMeteorsVolts, voltsPShift, pairslist, thresh, noise)
        #Estimation of decay times (Errors N 7, 8, 11)
        listMeteors3 = self.__estimateDecayTime(listMeteors2, listMeteorsPower, dataOut.timeInterval, dataOut.frequency)
        #*******************     END OF METEOR REESTIMATION    *******************

        #*********************    METEOR PARAMETERS CALCULATION (3.11, 3.12, 3.13)    **************************
        #Calculating Radial Velocity (Error N 15)
        radialStdThresh = 10
        listMeteors4 = self.__getRadialVelocity(listMeteors3, listMeteorsVolts, radialStdThresh, pairslist0, dataOut.timeInterval)

        if len(listMeteors4) > 0:
            #Setting New Array
            date = dataOut.utctime
            arrayParameters = self.__setNewArrays(listMeteors4, date, heiRang)

            #Correcting phase offset
            if phaseOffsets != None:
                phaseOffsets = numpy.array(phaseOffsets)*numpy.pi/180
                arrayParameters[:,8:12] = numpy.unwrap(arrayParameters[:,8:12] + phaseOffsets)

            #Second Pairslist
            pairsList = []
            pairx = (0,1)
            pairy = (2,3)
            pairsList.append(pairx)
            pairsList.append(pairy)

            jph = numpy.array([0,0,0,0])
            h = (hmin,hmax)
            arrayParameters = meteorOps.getMeteorParams(arrayParameters, azimuth, h, pairsList, distances, jph)

#             #Calculate AOA (Error N 3, 4)
#             #JONES ET AL. 1998
#             error = arrayParameters[:,-1]
#             AOAthresh = numpy.pi/8
#             phases = -arrayParameters[:,9:13]
#             arrayParameters[:,4:7], arrayParameters[:,-1] = meteorOps.getAOA(phases, pairsList, error, AOAthresh, azimuth)
#
#             #Calculate Heights (Error N 13 and 14)
#             error = arrayParameters[:,-1]
#             Ranges = arrayParameters[:,2]
#             zenith = arrayParameters[:,5]
#             arrayParameters[:,3], arrayParameters[:,-1] = meteorOps.getHeights(Ranges, zenith, error, hmin, hmax)
#             error = arrayParameters[:,-1]
        #*********************    END OF PARAMETERS CALCULATION    **************************

        #***************************+     PASS DATA TO NEXT STEP    **********************
#             arrayFinal = arrayParameters.reshape((1,arrayParameters.shape[0],arrayParameters.shape[1]))
            dataOut.data_param = arrayParameters

            if arrayParameters is None:
                dataOut.flagNoData = True
        else:
            dataOut.flagNoData = True

        return

    def __getHardwarePhaseDiff(self, voltage0, pairslist, newheis, n):

        minIndex = min(newheis[0])
        maxIndex = max(newheis[0])

        voltage = voltage0[:,:,minIndex:maxIndex+1]
        nLength = voltage.shape[1]/n
        nMin = 0
        nMax = 0
        phaseOffset = numpy.zeros((len(pairslist),n))

        for i in range(n):
            nMax += nLength
            phaseCCF = -numpy.angle(self.__calculateCCF(voltage[:,nMin:nMax,:], pairslist, [0]))
            phaseCCF = numpy.mean(phaseCCF, axis = 2)
            phaseOffset[:,i] = phaseCCF.transpose()
            nMin = nMax
#         phaseDiff, phaseArrival = self.estimatePhaseDifference(voltage, pairslist)

        #Remove Outliers
        factor = 2
        wt = phaseOffset - signal.medfilt(phaseOffset,(1,5))
        dw = numpy.std(wt,axis = 1)
        dw = dw.reshape((dw.size,1))
        ind = numpy.where(numpy.logical_or(wt>dw*factor,wt<-dw*factor))
        phaseOffset[ind] = numpy.nan
        phaseOffset = stats.nanmean(phaseOffset, axis=1)

        return phaseOffset

    def __shiftPhase(self, data, phaseShift):
        #this will shift the phase of a complex number
        dataShifted = numpy.abs(data) * numpy.exp((numpy.angle(data)+phaseShift)*1j)
        return dataShifted

    def __estimatePhaseDifference(self, array, pairslist):
        nChannel = array.shape[0]
        nHeights = array.shape[2]
        numPairs = len(pairslist)
#         phaseCCF = numpy.zeros((nChannel, 5, nHeights))
        phaseCCF = numpy.angle(self.__calculateCCF(array, pairslist, [-2,-1,0,1,2]))

        #Correct phases
        derPhaseCCF = phaseCCF[:,1:,:] - phaseCCF[:,0:-1,:]
        indDer = numpy.where(numpy.abs(derPhaseCCF) > numpy.pi)

        if indDer[0].shape[0] > 0:
            for i in range(indDer[0].shape[0]):
                signo = -numpy.sign(derPhaseCCF[indDer[0][i],indDer[1][i],indDer[2][i]])
                phaseCCF[indDer[0][i],indDer[1][i]+1:,:] += signo*2*numpy.pi

#         for j in range(numSides):
#             phaseCCFAux = self.calculateCCF(arrayCenter, arraySides[j,:,:], [-2,1,0,1,2])
#             phaseCCF[j,:,:] = numpy.angle(phaseCCFAux)
#
        #Linear
        phaseInt = numpy.zeros((numPairs,1))
        angAllCCF = phaseCCF[:,[0,1,3,4],0]
        for j in range(numPairs):
            fit = stats.linregress([-2,-1,1,2],angAllCCF[j,:])
            phaseInt[j] = fit[1]
        #Phase Differences
        phaseDiff = phaseInt - phaseCCF[:,2,:]
        phaseArrival = phaseInt.reshape(phaseInt.size)

        #Dealias
        phaseArrival = numpy.angle(numpy.exp(1j*phaseArrival))
#         indAlias = numpy.where(phaseArrival > numpy.pi)
#         phaseArrival[indAlias] -= 2*numpy.pi
#         indAlias = numpy.where(phaseArrival < -numpy.pi)
#         phaseArrival[indAlias] += 2*numpy.pi

        return phaseDiff, phaseArrival

    def __coherentDetection(self, volts, timeSegment, timeInterval, pairslist, thresh):
        #this function will run the coherent detection used in Holdworth et al. 2004 and return the net power
        #find the phase shifts of each channel over 1 second intervals
        #only look at ranges below the beacon signal
        numProfPerBlock = numpy.ceil(timeSegment/timeInterval)
        numBlocks = int(volts.shape[1]/numProfPerBlock)
        numHeights = volts.shape[2]
        nChannel = volts.shape[0]
        voltsCohDet = volts.copy()

        pairsarray = numpy.array(pairslist)
        indSides = pairsarray[:,1]
#         indSides = numpy.array(range(nChannel))
#         indSides = numpy.delete(indSides, indCenter)
#
#         listCenter = numpy.array_split(volts[indCenter,:,:], numBlocks, 0)
        listBlocks = numpy.array_split(volts, numBlocks, 1)

        startInd = 0
        endInd = 0

        for i in range(numBlocks):
            startInd = endInd
            endInd = endInd + listBlocks[i].shape[1]

            arrayBlock = listBlocks[i]
#             arrayBlockCenter = listCenter[i]

            #Estimate the Phase Difference
            phaseDiff, aux = self.__estimatePhaseDifference(arrayBlock, pairslist)
            #Phase Difference RMS
            arrayPhaseRMS = numpy.abs(phaseDiff)
            phaseRMSaux = numpy.sum(arrayPhaseRMS < thresh,0)
            indPhase = numpy.where(phaseRMSaux==4)
            #Shifting
            if indPhase[0].shape[0] > 0:
                for j in range(indSides.size):
                    arrayBlock[indSides[j],:,indPhase] = self.__shiftPhase(arrayBlock[indSides[j],:,indPhase], phaseDiff[j,indPhase].transpose())
                voltsCohDet[:,startInd:endInd,:] = arrayBlock

        return voltsCohDet

    def __calculateCCF(self, volts, pairslist ,laglist):

        nHeights = volts.shape[2]
        nPoints = volts.shape[1]
        voltsCCF = numpy.zeros((len(pairslist), len(laglist), nHeights),dtype = 'complex')

        for i in range(len(pairslist)):
            volts1 = volts[pairslist[i][0]]
            volts2 = volts[pairslist[i][1]]

            for t in range(len(laglist)):
                idxT = laglist[t]
                if idxT >= 0:
                    vStacked = numpy.vstack((volts2[idxT:,:],
                                           numpy.zeros((idxT, nHeights),dtype='complex')))
                else:
                    vStacked = numpy.vstack((numpy.zeros((-idxT, nHeights),dtype='complex'),
                                           volts2[:(nPoints + idxT),:]))
                voltsCCF[i,t,:] = numpy.sum((numpy.conjugate(volts1)*vStacked),axis=0)

                vStacked = None
        return voltsCCF

    def __getNoise(self, power, timeSegment, timeInterval):
        numProfPerBlock = numpy.ceil(timeSegment/timeInterval)
        numBlocks = int(power.shape[0]/numProfPerBlock)
        numHeights = power.shape[1]

        listPower = numpy.array_split(power, numBlocks, 0)
        noise = numpy.zeros((power.shape[0], power.shape[1]))
        noise1 = numpy.zeros((power.shape[0], power.shape[1]))

        startInd = 0
        endInd = 0

        for i in range(numBlocks):             #split por canal
            startInd = endInd
            endInd = endInd + listPower[i].shape[0]

            arrayBlock = listPower[i]
            noiseAux = numpy.mean(arrayBlock, 0)
#             noiseAux = numpy.median(noiseAux)
#             noiseAux = numpy.mean(arrayBlock)
            noise[startInd:endInd,:] = noise[startInd:endInd,:] + noiseAux

            noiseAux1 = numpy.mean(arrayBlock)
            noise1[startInd:endInd,:] = noise1[startInd:endInd,:] + noiseAux1

        return noise, noise1

    def __findMeteors(self, power, thresh):
        nProf = power.shape[0]
        nHeights = power.shape[1]
        listMeteors = []

        for i in range(nHeights):
            powerAux = power[:,i]
            threshAux = thresh[:,i]

            indUPthresh = numpy.where(powerAux > threshAux)[0]
            indDNthresh = numpy.where(powerAux <= threshAux)[0]

            j = 0

            while (j < indUPthresh.size - 2):
                if (indUPthresh[j + 2] == indUPthresh[j] + 2):
                    indDNAux = numpy.where(indDNthresh > indUPthresh[j])
                    indDNthresh = indDNthresh[indDNAux]

                    if (indDNthresh.size > 0):
                        indEnd = indDNthresh[0] - 1
                        indInit = indUPthresh[j]

                        meteor = powerAux[indInit:indEnd + 1]
                        indPeak = meteor.argmax() + indInit
                        FLA = sum(numpy.conj(meteor)*numpy.hstack((meteor[1:],0)))

                        listMeteors.append(numpy.array([i,indInit,indPeak,indEnd,FLA])) #CHEQUEAR!!!!!
                        j = numpy.where(indUPthresh == indEnd)[0] + 1
                    else: j+=1
                else: j+=1

        return listMeteors

    def __removeMultipleDetections(self,listMeteors, rangeLimit, timeLimit):

        arrayMeteors = numpy.asarray(listMeteors)
        listMeteors1 = []

        while arrayMeteors.shape[0] > 0:
            FLAs = arrayMeteors[:,4]
            maxFLA = FLAs.argmax()
            listMeteors1.append(arrayMeteors[maxFLA,:])

            MeteorInitTime = arrayMeteors[maxFLA,1]
            MeteorEndTime = arrayMeteors[maxFLA,3]
            MeteorHeight = arrayMeteors[maxFLA,0]

            #Check neighborhood
            maxHeightIndex = MeteorHeight + rangeLimit
            minHeightIndex = MeteorHeight - rangeLimit
            minTimeIndex = MeteorInitTime - timeLimit
            maxTimeIndex = MeteorEndTime + timeLimit

            #Check Heights
            indHeight = numpy.logical_and(arrayMeteors[:,0] >= minHeightIndex, arrayMeteors[:,0] <= maxHeightIndex)
            indTime = numpy.logical_and(arrayMeteors[:,3] >= minTimeIndex, arrayMeteors[:,1] <= maxTimeIndex)
            indBoth = numpy.where(numpy.logical_and(indTime,indHeight))

            arrayMeteors = numpy.delete(arrayMeteors, indBoth, axis = 0)

        return listMeteors1

    def __meteorReestimation(self, listMeteors, volts, pairslist, thresh, noise, timeInterval,frequency):
        numHeights = volts.shape[2]
        nChannel = volts.shape[0]

        thresholdPhase = thresh[0]
        thresholdNoise = thresh[1]
        thresholdDB = float(thresh[2])

        thresholdDB1 = 10**(thresholdDB/10)
        pairsarray = numpy.array(pairslist)
        indSides = pairsarray[:,1]

        pairslist1 = list(pairslist)
        pairslist1.append((0,1))
        pairslist1.append((3,4))

        listMeteors1 = []
        listPowerSeries = []
        listVoltageSeries = []
        #volts has the war data

        if frequency == 30e6:
            timeLag = 45*10**-3
        else:
            timeLag = 15*10**-3
        lag = numpy.ceil(timeLag/timeInterval)

        for i in range(len(listMeteors)):

            ######################   3.6 - 3.7 PARAMETERS REESTIMATION    #########################
            meteorAux = numpy.zeros(16)

            #Loading meteor Data (mHeight, mStart, mPeak, mEnd)
            mHeight = listMeteors[i][0]
            mStart = listMeteors[i][1]
            mPeak = listMeteors[i][2]
            mEnd = listMeteors[i][3]

            #get the volt data between the start and end times of the meteor
            meteorVolts = volts[:,mStart:mEnd+1,mHeight]
            meteorVolts = meteorVolts.reshape(meteorVolts.shape[0], meteorVolts.shape[1], 1)

            #3.6. Phase Difference estimation
            phaseDiff, aux = self.__estimatePhaseDifference(meteorVolts, pairslist)

            #3.7. Phase difference removal & meteor start, peak and end times reestimated
            #meteorVolts0.- all Channels, all Profiles
            meteorVolts0 = volts[:,:,mHeight]
            meteorThresh = noise[:,mHeight]*thresholdNoise
            meteorNoise = noise[:,mHeight]
            meteorVolts0[indSides,:] = self.__shiftPhase(meteorVolts0[indSides,:], phaseDiff) #Phase Shifting
            powerNet0 = numpy.nansum(numpy.abs(meteorVolts0)**2, axis = 0)  #Power

            #Times reestimation
            mStart1 = numpy.where(powerNet0[:mPeak] < meteorThresh[:mPeak])[0]
            if mStart1.size > 0:
                mStart1 = mStart1[-1] + 1

            else:
                mStart1 = mPeak

            mEnd1 = numpy.where(powerNet0[mPeak:] < meteorThresh[mPeak:])[0][0] + mPeak - 1
            mEndDecayTime1 = numpy.where(powerNet0[mPeak:] < meteorNoise[mPeak:])[0]
            if mEndDecayTime1.size == 0:
                mEndDecayTime1 = powerNet0.size
            else:
                mEndDecayTime1 = mEndDecayTime1[0] + mPeak - 1
#             mPeak1 = meteorVolts0[mStart1:mEnd1 + 1].argmax()

            #meteorVolts1.- all Channels, from start to end
            meteorVolts1 = meteorVolts0[:,mStart1:mEnd1 + 1]
            meteorVolts2 = meteorVolts0[:,mPeak + lag:mEnd1 + 1]
            if meteorVolts2.shape[1] == 0:
                meteorVolts2 = meteorVolts0[:,mPeak:mEnd1 + 1]
            meteorVolts1 = meteorVolts1.reshape(meteorVolts1.shape[0], meteorVolts1.shape[1], 1)
            meteorVolts2 = meteorVolts2.reshape(meteorVolts2.shape[0], meteorVolts2.shape[1], 1)
            #####################    END PARAMETERS REESTIMATION    #########################

            #####################   3.8 PHASE DIFFERENCE REESTIMATION  ########################
#             if mEnd1 - mStart1 > 4:       #Error Number 6: echo less than 5 samples long; too short for analysis
            if meteorVolts2.shape[1] > 0:
                #Phase Difference re-estimation
                phaseDiff1, phaseDiffint = self.__estimatePhaseDifference(meteorVolts2, pairslist1)   #Phase Difference Estimation
#                 phaseDiff1, phaseDiffint = self.estimatePhaseDifference(meteorVolts2, pairslist)
                meteorVolts2 = meteorVolts2.reshape(meteorVolts2.shape[0], meteorVolts2.shape[1])
                phaseDiff11 = numpy.reshape(phaseDiff1, (phaseDiff1.shape[0],1))
                meteorVolts2[indSides,:] = self.__shiftPhase(meteorVolts2[indSides,:], phaseDiff11[0:4])     #Phase Shifting

                #Phase Difference RMS
                phaseRMS1 = numpy.sqrt(numpy.mean(numpy.square(phaseDiff1)))
                powerNet1 = numpy.nansum(numpy.abs(meteorVolts1[:,:])**2,0)
                #Data from Meteor
                mPeak1 = powerNet1.argmax() + mStart1
                mPeakPower1 = powerNet1.max()
                noiseAux = sum(noise[mStart1:mEnd1 + 1,mHeight])
                mSNR1 = (sum(powerNet1)-noiseAux)/noiseAux
                Meteor1 = numpy.array([mHeight, mStart1, mPeak1, mEnd1, mPeakPower1, mSNR1, phaseRMS1])
                Meteor1 = numpy.hstack((Meteor1,phaseDiffint))
                PowerSeries  = powerNet0[mStart1:mEndDecayTime1 + 1]
                #Vectorize
                meteorAux[0:7] = [mHeight, mStart1, mPeak1, mEnd1, mPeakPower1, mSNR1, phaseRMS1]
                meteorAux[7:11] = phaseDiffint[0:4]

                #Rejection Criterions
                if phaseRMS1 > thresholdPhase:  #Error Number 17: Phase variation
                    meteorAux[-1] = 17
                elif mSNR1 < thresholdDB1:      #Error Number 1: SNR < threshold dB
                    meteorAux[-1] = 1


            else:
                meteorAux[0:4] = [mHeight, mStart, mPeak, mEnd]
                meteorAux[-1] = 6 #Error Number 6: echo less than 5 samples long; too short for analysis
                PowerSeries = 0

            listMeteors1.append(meteorAux)
            listPowerSeries.append(PowerSeries)
            listVoltageSeries.append(meteorVolts1)

        return listMeteors1, listPowerSeries, listVoltageSeries

    def __estimateDecayTime(self, listMeteors, listPower, timeInterval, frequency):

        threshError = 10
        #Depending if it is 30 or 50 MHz
        if frequency == 30e6:
            timeLag = 45*10**-3
        else:
            timeLag = 15*10**-3
        lag = numpy.ceil(timeLag/timeInterval)

        listMeteors1 = []

        for i in range(len(listMeteors)):
            meteorPower = listPower[i]
            meteorAux = listMeteors[i]

            if meteorAux[-1] == 0:

                try:
                    indmax = meteorPower.argmax()
                    indlag = indmax + lag

                    y = meteorPower[indlag:]
                    x = numpy.arange(0, y.size)*timeLag

                    #first guess
                    a = y[0]
                    tau = timeLag
                    #exponential fit
                    popt, pcov = optimize.curve_fit(self.__exponential_function, x, y, p0 = [a, tau])
                    y1 = self.__exponential_function(x, *popt)
                    #error estimation
                    error = sum((y - y1)**2)/(numpy.var(y)*(y.size - popt.size))

                    decayTime = popt[1]
                    riseTime = indmax*timeInterval
                    meteorAux[11:13] = [decayTime, error]

                    #Table items 7, 8 and 11
                    if (riseTime > 0.3):            #Number 7: Echo rise exceeds 0.3s
                        meteorAux[-1] = 7
                    elif (decayTime < 2*riseTime) : #Number 8: Echo decay time less than than twice rise time
                        meteorAux[-1] = 8
                    if (error > threshError):       #Number 11: Poor fit to amplitude for estimation of decay time
                        meteorAux[-1] = 11


                except:
                    meteorAux[-1] = 11


            listMeteors1.append(meteorAux)

        return listMeteors1

    #Exponential Function

    def __exponential_function(self, x, a, tau):
        y = a*numpy.exp(-x/tau)
        return y

    def __getRadialVelocity(self, listMeteors, listVolts, radialStdThresh, pairslist,  timeInterval):

        pairslist1 = list(pairslist)
        pairslist1.append((0,1))
        pairslist1.append((3,4))
        numPairs = len(pairslist1)
        #Time Lag
        timeLag = 45*10**-3
        c = 3e8
        lag = numpy.ceil(timeLag/timeInterval)
        freq = 30e6

        listMeteors1 = []

        for i in range(len(listMeteors)):
            meteorAux = listMeteors[i]
            if meteorAux[-1] == 0:
                mStart = listMeteors[i][1]
                mPeak = listMeteors[i][2]
                mLag = mPeak - mStart + lag

                #get the volt data between the start and end times of the meteor
                meteorVolts = listVolts[i]
                meteorVolts = meteorVolts.reshape(meteorVolts.shape[0], meteorVolts.shape[1], 1)

                #Get CCF
                allCCFs = self.__calculateCCF(meteorVolts, pairslist1, [-2,-1,0,1,2])

                #Method 2
                slopes = numpy.zeros(numPairs)
                time = numpy.array([-2,-1,1,2])*timeInterval
                angAllCCF = numpy.angle(allCCFs[:,[0,1,3,4],0])

                #Correct phases
                derPhaseCCF = angAllCCF[:,1:] - angAllCCF[:,0:-1]
                indDer = numpy.where(numpy.abs(derPhaseCCF) > numpy.pi)

                if indDer[0].shape[0] > 0:
                    for i in range(indDer[0].shape[0]):
                        signo = -numpy.sign(derPhaseCCF[indDer[0][i],indDer[1][i]])
                        angAllCCF[indDer[0][i],indDer[1][i]+1:] += signo*2*numpy.pi

#                     fit = scipy.stats.linregress(numpy.array([-2,-1,1,2])*timeInterval, numpy.array([phaseLagN2s[i],phaseLagN1s[i],phaseLag1s[i],phaseLag2s[i]]))
                for j in range(numPairs):
                    fit = stats.linregress(time, angAllCCF[j,:])
                    slopes[j] = fit[0]

                #Remove Outlier
#                 indOut = numpy.argmax(numpy.abs(slopes - numpy.mean(slopes)))
#                 slopes = numpy.delete(slopes,indOut)
#                 indOut = numpy.argmax(numpy.abs(slopes - numpy.mean(slopes)))
#                 slopes = numpy.delete(slopes,indOut)

                radialVelocity = -numpy.mean(slopes)*(0.25/numpy.pi)*(c/freq)
                radialError = numpy.std(slopes)*(0.25/numpy.pi)*(c/freq)
                meteorAux[-2] = radialError
                meteorAux[-3] = radialVelocity

                #Setting Error
                #Number 15: Radial Drift velocity or projected horizontal velocity exceeds 200 m/s
                if numpy.abs(radialVelocity) > 200:
                    meteorAux[-1] = 15
                #Number 12: Poor fit to CCF variation for estimation of radial drift velocity
                elif radialError > radialStdThresh:
                    meteorAux[-1] = 12

            listMeteors1.append(meteorAux)
        return listMeteors1

    def __setNewArrays(self, listMeteors, date, heiRang):

        #New arrays
        arrayMeteors = numpy.array(listMeteors)
        arrayParameters = numpy.zeros((len(listMeteors), 13))

        #Date inclusion
#         date = re.findall(r'\((.*?)\)', date)
#         date = date[0].split(',')
#         date = map(int, date)
#
#         if len(date)<6:
#             date.append(0)
#
#         date = [date[0]*10000 + date[1]*100 + date[2], date[3]*10000 + date[4]*100 + date[5]]
#         arrayDate = numpy.tile(date, (len(listMeteors), 1))
        arrayDate = numpy.tile(date, (len(listMeteors)))

        #Meteor array
#         arrayMeteors[:,0] = heiRang[arrayMeteors[:,0].astype(int)]
#         arrayMeteors = numpy.hstack((arrayDate, arrayMeteors))

        #Parameters Array
        arrayParameters[:,0] = arrayDate #Date
        arrayParameters[:,1] = heiRang[arrayMeteors[:,0].astype(int)] #Range
        arrayParameters[:,6:8] = arrayMeteors[:,-3:-1] #Radial velocity and its error
        arrayParameters[:,8:12] = arrayMeteors[:,7:11]  #Phases
        arrayParameters[:,-1] = arrayMeteors[:,-1]  #Error


        return arrayParameters

class CorrectSMPhases(Operation):

    def run(self, dataOut, phaseOffsets, hmin = 50, hmax = 150, azimuth = 45, channelPositions = None):

        arrayParameters = dataOut.data_param
        pairsList = []
        pairx = (0,1)
        pairy = (2,3)
        pairsList.append(pairx)
        pairsList.append(pairy)
        jph = numpy.zeros(4)

        phaseOffsets = numpy.array(phaseOffsets)*numpy.pi/180
    #         arrayParameters[:,8:12] = numpy.unwrap(arrayParameters[:,8:12] + phaseOffsets)
        arrayParameters[:,8:12] = numpy.angle(numpy.exp(1j*(arrayParameters[:,8:12] + phaseOffsets)))

        meteorOps = SMOperations()
        if channelPositions is None:
    #             channelPositions = [(2.5,0), (0,2.5), (0,0), (0,4.5), (-2,0)]   #T
            channelPositions = [(4.5,2), (2,4.5), (2,2), (2,0), (0,2)]   #Estrella

        pairslist0, distances = meteorOps.getPhasePairs(channelPositions)
        h = (hmin,hmax)

        arrayParameters = meteorOps.getMeteorParams(arrayParameters, azimuth, h, pairsList, distances, jph)

        dataOut.data_param = arrayParameters
        return

class SMPhaseCalibration(Operation):

    __buffer = None

    __initime = None

    __dataReady = False

    __isConfig = False

    def __checkTime(self, currentTime, initTime, paramInterval, outputInterval):

        dataTime = currentTime + paramInterval
        deltaTime = dataTime - initTime

        if deltaTime >= outputInterval or deltaTime < 0:
            return True

        return False

    def __getGammas(self, pairs, d, phases):
        gammas = numpy.zeros(2)

        for i in range(len(pairs)):

            pairi = pairs[i]

            phip3 = phases[:,pairi[0]]
            d3 = d[pairi[0]]
            phip2 = phases[:,pairi[1]]
            d2 = d[pairi[1]]
            #Calculating gamma
#             jdcos = alp1/(k*d1)
#             jgamma = numpy.angle(numpy.exp(1j*(d0*alp1/d1 - alp0)))
            jgamma = -phip2*d3/d2 - phip3
            jgamma = numpy.angle(numpy.exp(1j*jgamma))
#             jgamma[jgamma>numpy.pi] -= 2*numpy.pi
#             jgamma[jgamma<-numpy.pi] += 2*numpy.pi

            #Revised distribution
            jgammaArray = numpy.hstack((jgamma,jgamma+0.5*numpy.pi,jgamma-0.5*numpy.pi))

            #Histogram
            nBins = 64
            rmin = -0.5*numpy.pi
            rmax = 0.5*numpy.pi
            phaseHisto = numpy.histogram(jgammaArray, bins=nBins, range=(rmin,rmax))

            meteorsY = phaseHisto[0]
            phasesX = phaseHisto[1][:-1]
            width = phasesX[1] - phasesX[0]
            phasesX += width/2

            #Gaussian aproximation
            bpeak = meteorsY.argmax()
            peak = meteorsY.max()
            jmin = bpeak - 5
            jmax = bpeak + 5 + 1

            if jmin<0:
                jmin = 0
                jmax = 6
            elif jmax > meteorsY.size:
                jmin = meteorsY.size - 6
                jmax = meteorsY.size

            x0 = numpy.array([peak,bpeak,50])
            coeff = optimize.leastsq(self.__residualFunction, x0, args=(meteorsY[jmin:jmax], phasesX[jmin:jmax]))

            #Gammas
            gammas[i] = coeff[0][1]

        return gammas

    def __residualFunction(self, coeffs, y, t):

        return y - self.__gauss_function(t, coeffs)

    def __gauss_function(self, t, coeffs):

        return coeffs[0]*numpy.exp(-0.5*((t - coeffs[1]) / coeffs[2])**2)

    def __getPhases(self, azimuth, h, pairsList, d, gammas, meteorsArray):
        meteorOps = SMOperations()
        nchan = 4
        pairx = pairsList[0] #x es 0
        pairy = pairsList[1] #y es 1
        center_xangle = 0
        center_yangle = 0
        range_angle = numpy.array([10*numpy.pi,numpy.pi,numpy.pi/2,numpy.pi/4])
        ntimes = len(range_angle)

        nstepsx = 20
        nstepsy = 20

        for iz in range(ntimes):
            min_xangle = -range_angle[iz]/2 + center_xangle
            max_xangle = range_angle[iz]/2 + center_xangle
            min_yangle = -range_angle[iz]/2 + center_yangle
            max_yangle = range_angle[iz]/2 + center_yangle

            inc_x = (max_xangle-min_xangle)/nstepsx
            inc_y = (max_yangle-min_yangle)/nstepsy

            alpha_y = numpy.arange(nstepsy)*inc_y + min_yangle
            alpha_x = numpy.arange(nstepsx)*inc_x + min_xangle
            penalty = numpy.zeros((nstepsx,nstepsy))
            jph_array = numpy.zeros((nchan,nstepsx,nstepsy))
            jph = numpy.zeros(nchan)

            # Iterations looking for the offset
            for iy in range(int(nstepsy)):
                for ix in range(int(nstepsx)):
                    d3 = d[pairsList[1][0]]
                    d2 = d[pairsList[1][1]]
                    d5 = d[pairsList[0][0]]
                    d4 = d[pairsList[0][1]]

                    alp2 = alpha_y[iy]  #gamma 1
                    alp4 = alpha_x[ix]  #gamma 0

                    alp3 = -alp2*d3/d2 - gammas[1]
                    alp5 = -alp4*d5/d4 - gammas[0]
#                     jph[pairy[1]] = alpha_y[iy]
#                     jph[pairy[0]] = -gammas[1] - alpha_y[iy]*d[pairy[1]]/d[pairy[0]]

#                     jph[pairx[1]] = alpha_x[ix]
#                     jph[pairx[0]] = -gammas[0] - alpha_x[ix]*d[pairx[1]]/d[pairx[0]]
                    jph[pairsList[0][1]] = alp4
                    jph[pairsList[0][0]] = alp5
                    jph[pairsList[1][0]] = alp3
                    jph[pairsList[1][1]] = alp2
                    jph_array[:,ix,iy] = jph
#                     d = [2.0,2.5,2.5,2.0]
                    #falta chequear si va a leer bien  los meteoros
                    meteorsArray1 = meteorOps.getMeteorParams(meteorsArray, azimuth, h, pairsList, d, jph)
                    error = meteorsArray1[:,-1]
                    ind1 = numpy.where(error==0)[0]
                    penalty[ix,iy] = ind1.size

            i,j = numpy.unravel_index(penalty.argmax(), penalty.shape)
            phOffset = jph_array[:,i,j]

            center_xangle = phOffset[pairx[1]]
            center_yangle = phOffset[pairy[1]]

        phOffset = numpy.angle(numpy.exp(1j*jph_array[:,i,j]))
        phOffset = phOffset*180/numpy.pi
        return phOffset


    def run(self, dataOut, hmin, hmax, channelPositions=None, nHours = 1):

        dataOut.flagNoData = True
        self.__dataReady = False
        dataOut.outputInterval = nHours*3600

        if self.__isConfig == False:
#                 self.__initime = dataOut.datatime.replace(minute = 0, second = 0, microsecond = 03)
            #Get Initial LTC time
            self.__initime = datetime.datetime.utcfromtimestamp(dataOut.utctime)
            self.__initime = (self.__initime.replace(minute = 0, second = 0, microsecond = 0) - datetime.datetime(1970, 1, 1)).total_seconds()

            self.__isConfig = True

        if self.__buffer is None:
            self.__buffer = dataOut.data_param.copy()

        else:
            self.__buffer = numpy.vstack((self.__buffer, dataOut.data_param))

        self.__dataReady = self.__checkTime(dataOut.utctime, self.__initime, dataOut.paramInterval, dataOut.outputInterval) #Check if the buffer is ready

        if self.__dataReady:
            dataOut.utctimeInit = self.__initime
            self.__initime += dataOut.outputInterval #to erase time offset

            freq = dataOut.frequency
            c = dataOut.C #m/s
            lamb = c/freq
            k = 2*numpy.pi/lamb
            azimuth = 0
            h = (hmin, hmax)
#             pairs = ((0,1),(2,3)) #Estrella
#             pairs = ((1,0),(2,3)) #T

            if channelPositions is None:
#             channelPositions = [(2.5,0), (0,2.5), (0,0), (0,4.5), (-2,0)]   #T
                channelPositions = [(4.5,2), (2,4.5), (2,2), (2,0), (0,2)]   #Estrella
            meteorOps = SMOperations()
            pairslist0, distances = meteorOps.getPhasePairs(channelPositions)

            #Checking correct order of pairs
            pairs = []
            if distances[1] > distances[0]:
                pairs.append((1,0))
            else:
                pairs.append((0,1))

            if distances[3] > distances[2]:
                pairs.append((3,2))
            else:
                pairs.append((2,3))
#             distances1 = [-distances[0]*lamb, distances[1]*lamb, -distances[2]*lamb, distances[3]*lamb]

            meteorsArray = self.__buffer
            error = meteorsArray[:,-1]
            boolError = (error==0)|(error==3)|(error==4)|(error==13)|(error==14)
            ind1 = numpy.where(boolError)[0]
            meteorsArray = meteorsArray[ind1,:]
            meteorsArray[:,-1] = 0
            phases = meteorsArray[:,8:12]

            #Calculate Gammas
            gammas = self.__getGammas(pairs, distances, phases)
#             gammas = numpy.array([-21.70409463,45.76935864])*numpy.pi/180
            #Calculate Phases
            phasesOff = self.__getPhases(azimuth, h, pairs, distances, gammas, meteorsArray)
            phasesOff = phasesOff.reshape((1,phasesOff.size))
            dataOut.data_output = -phasesOff
            dataOut.flagNoData = False
            self.__buffer = None


        return

class SMOperations():

    def __init__(self):

        return

    def getMeteorParams(self, arrayParameters0, azimuth, h, pairsList, distances, jph):

        arrayParameters = arrayParameters0.copy()
        hmin = h[0]
        hmax = h[1]

        #Calculate AOA (Error N 3, 4)
        #JONES ET AL. 1998
        AOAthresh = numpy.pi/8
        error = arrayParameters[:,-1]
        phases = -arrayParameters[:,8:12] + jph
#         phases = numpy.unwrap(phases)
        arrayParameters[:,3:6], arrayParameters[:,-1] = self.__getAOA(phases, pairsList, distances, error, AOAthresh, azimuth)

        #Calculate Heights (Error N 13 and 14)
        error = arrayParameters[:,-1]
        Ranges = arrayParameters[:,1]
        zenith = arrayParameters[:,4]
        arrayParameters[:,2], arrayParameters[:,-1] = self.__getHeights(Ranges, zenith, error, hmin, hmax)

        #----------------------- Get Final data    ------------------------------------
#         error = arrayParameters[:,-1]
#         ind1 = numpy.where(error==0)[0]
#         arrayParameters = arrayParameters[ind1,:]

        return arrayParameters

    def __getAOA(self, phases, pairsList, directions, error, AOAthresh, azimuth):

        arrayAOA = numpy.zeros((phases.shape[0],3))
        cosdir0, cosdir = self.__getDirectionCosines(phases, pairsList,directions)

        arrayAOA[:,:2] = self.__calculateAOA(cosdir, azimuth)
        cosDirError = numpy.sum(numpy.abs(cosdir0 - cosdir), axis = 1)
        arrayAOA[:,2] = cosDirError

        azimuthAngle = arrayAOA[:,0]
        zenithAngle = arrayAOA[:,1]

        #Setting Error
        indError = numpy.where(numpy.logical_or(error == 3, error == 4))[0]
        error[indError] = 0
        #Number 3: AOA not fesible
        indInvalid = numpy.where(numpy.logical_and((numpy.logical_or(numpy.isnan(zenithAngle), numpy.isnan(azimuthAngle))),error == 0))[0]
        error[indInvalid] = 3
        #Number 4: Large difference in AOAs obtained from different antenna baselines
        indInvalid = numpy.where(numpy.logical_and(cosDirError > AOAthresh,error == 0))[0]
        error[indInvalid] = 4
        return arrayAOA, error

    def __getDirectionCosines(self, arrayPhase, pairsList, distances):

        #Initializing some variables
        ang_aux = numpy.array([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])*2*numpy.pi
        ang_aux = ang_aux.reshape(1,ang_aux.size)

        cosdir = numpy.zeros((arrayPhase.shape[0],2))
        cosdir0 = numpy.zeros((arrayPhase.shape[0],2))


        for i in range(2):
            ph0 = arrayPhase[:,pairsList[i][0]]
            ph1 = arrayPhase[:,pairsList[i][1]]
            d0 = distances[pairsList[i][0]]
            d1 = distances[pairsList[i][1]]

            ph0_aux = ph0 + ph1
            ph0_aux = numpy.angle(numpy.exp(1j*ph0_aux))
#             ph0_aux[ph0_aux > numpy.pi] -= 2*numpy.pi
#             ph0_aux[ph0_aux < -numpy.pi] += 2*numpy.pi
            #First Estimation
            cosdir0[:,i] = (ph0_aux)/(2*numpy.pi*(d0 - d1))

            #Most-Accurate Second Estimation
            phi1_aux =  ph0 - ph1
            phi1_aux = phi1_aux.reshape(phi1_aux.size,1)
            #Direction Cosine 1
            cosdir1 = (phi1_aux + ang_aux)/(2*numpy.pi*(d0 + d1))

            #Searching the correct Direction Cosine
            cosdir0_aux = cosdir0[:,i]
            cosdir0_aux = cosdir0_aux.reshape(cosdir0_aux.size,1)
            #Minimum Distance
            cosDiff = (cosdir1 - cosdir0_aux)**2
            indcos = cosDiff.argmin(axis = 1)
            #Saving Value obtained
            cosdir[:,i] = cosdir1[numpy.arange(len(indcos)),indcos]

        return cosdir0, cosdir

    def __calculateAOA(self, cosdir, azimuth):
        cosdirX = cosdir[:,0]
        cosdirY = cosdir[:,1]

        zenithAngle = numpy.arccos(numpy.sqrt(1 - cosdirX**2 - cosdirY**2))*180/numpy.pi
        azimuthAngle = numpy.arctan2(cosdirX,cosdirY)*180/numpy.pi + azimuth#0 deg north, 90 deg east
        angles = numpy.vstack((azimuthAngle, zenithAngle)).transpose()

        return angles

    def __getHeights(self, Ranges, zenith, error, minHeight, maxHeight):

        Ramb = 375  #Ramb = c/(2*PRF)
        Re = 6371   #Earth Radius
        heights = numpy.zeros(Ranges.shape)

        R_aux = numpy.array([0,1,2])*Ramb
        R_aux = R_aux.reshape(1,R_aux.size)

        Ranges = Ranges.reshape(Ranges.size,1)

        Ri = Ranges + R_aux
        hi = numpy.sqrt(Re**2 + Ri**2 + (2*Re*numpy.cos(zenith*numpy.pi/180)*Ri.transpose()).transpose()) - Re

        #Check if there is a height between 70 and 110 km
        h_bool = numpy.sum(numpy.logical_and(hi > minHeight, hi < maxHeight), axis = 1)
        ind_h = numpy.where(h_bool == 1)[0]

        hCorr = hi[ind_h, :]
        ind_hCorr = numpy.where(numpy.logical_and(hi > minHeight, hi < maxHeight))

        hCorr = hi[ind_hCorr][:len(ind_h)]
        heights[ind_h] = hCorr

        #Setting Error
        #Number 13: Height unresolvable echo: not valid height within 70 to 110 km
        #Number 14: Height ambiguous echo: more than one possible height within 70 to 110 km
        indError = numpy.where(numpy.logical_or(error == 13, error == 14))[0]
        error[indError] = 0
        indInvalid2 = numpy.where(numpy.logical_and(h_bool > 1, error == 0))[0]
        error[indInvalid2] = 14
        indInvalid1 = numpy.where(numpy.logical_and(h_bool == 0, error == 0))[0]
        error[indInvalid1] = 13

        return heights, error

    def getPhasePairs(self, channelPositions):
        chanPos = numpy.array(channelPositions)
        listOper = list(itertools.combinations(list(range(5)),2))

        distances = numpy.zeros(4)
        axisX = []
        axisY = []
        distX = numpy.zeros(3)
        distY = numpy.zeros(3)
        ix = 0
        iy = 0

        pairX = numpy.zeros((2,2))
        pairY = numpy.zeros((2,2))

        for i in range(len(listOper)):
            pairi = listOper[i]

            posDif = numpy.abs(chanPos[pairi[0],:] - chanPos[pairi[1],:])

            if posDif[0] == 0:
                axisY.append(pairi)
                distY[iy] = posDif[1]
                iy += 1
            elif posDif[1] == 0:
                axisX.append(pairi)
                distX[ix] = posDif[0]
                ix += 1

        for i in range(2):
            if i==0:
                dist0 = distX
                axis0 = axisX
            else:
                dist0 = distY
                axis0 = axisY

            side = numpy.argsort(dist0)[:-1]
            axis0 = numpy.array(axis0)[side,:]
            chanC = int(numpy.intersect1d(axis0[0,:], axis0[1,:])[0])
            axis1 = numpy.unique(numpy.reshape(axis0,4))
            side = axis1[axis1 != chanC]
            diff1 = chanPos[chanC,i] - chanPos[side[0],i]
            diff2 = chanPos[chanC,i] - chanPos[side[1],i]
            if diff1<0:
                chan2 = side[0]
                d2 = numpy.abs(diff1)
                chan1 = side[1]
                d1 = numpy.abs(diff2)
            else:
                chan2 = side[1]
                d2 = numpy.abs(diff2)
                chan1 = side[0]
                d1 = numpy.abs(diff1)

            if i==0:
                chanCX = chanC
                chan1X = chan1
                chan2X = chan2
                distances[0:2] = numpy.array([d1,d2])
            else:
                chanCY = chanC
                chan1Y = chan1
                chan2Y = chan2
                distances[2:4] = numpy.array([d1,d2])
#         axisXsides = numpy.reshape(axisX[ix,:],4)
#
#         channelCentX = int(numpy.intersect1d(pairX[0,:], pairX[1,:])[0])
#         channelCentY = int(numpy.intersect1d(pairY[0,:], pairY[1,:])[0])
#
#         ind25X = numpy.where(pairX[0,:] != channelCentX)[0][0]
#         ind20X = numpy.where(pairX[1,:] != channelCentX)[0][0]
#         channel25X = int(pairX[0,ind25X])
#         channel20X = int(pairX[1,ind20X])
#         ind25Y = numpy.where(pairY[0,:] != channelCentY)[0][0]
#         ind20Y = numpy.where(pairY[1,:] != channelCentY)[0][0]
#         channel25Y = int(pairY[0,ind25Y])
#         channel20Y = int(pairY[1,ind20Y])

#         pairslist = [(channelCentX, channel25X),(channelCentX, channel20X),(channelCentY,channel25Y),(channelCentY, channel20Y)]
        pairslist = [(chanCX, chan1X),(chanCX, chan2X),(chanCY,chan1Y),(chanCY, chan2Y)]

        return pairslist, distances
#     def __getAOA(self, phases, pairsList, error, AOAthresh, azimuth):
#
#         arrayAOA = numpy.zeros((phases.shape[0],3))
#         cosdir0, cosdir = self.__getDirectionCosines(phases, pairsList)
#
#         arrayAOA[:,:2] = self.__calculateAOA(cosdir, azimuth)
#         cosDirError = numpy.sum(numpy.abs(cosdir0 - cosdir), axis = 1)
#         arrayAOA[:,2] = cosDirError
#
#         azimuthAngle = arrayAOA[:,0]
#         zenithAngle = arrayAOA[:,1]
#
#         #Setting Error
#         #Number 3: AOA not fesible
#         indInvalid = numpy.where(numpy.logical_and((numpy.logical_or(numpy.isnan(zenithAngle), numpy.isnan(azimuthAngle))),error == 0))[0]
#         error[indInvalid] = 3
#         #Number 4: Large difference in AOAs obtained from different antenna baselines
#         indInvalid = numpy.where(numpy.logical_and(cosDirError > AOAthresh,error == 0))[0]
#         error[indInvalid] = 4
#         return arrayAOA, error
#
#     def __getDirectionCosines(self, arrayPhase, pairsList):
#
#         #Initializing some variables
#         ang_aux = numpy.array([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])*2*numpy.pi
#         ang_aux = ang_aux.reshape(1,ang_aux.size)
#
#         cosdir = numpy.zeros((arrayPhase.shape[0],2))
#         cosdir0 = numpy.zeros((arrayPhase.shape[0],2))
#
#
#         for i in range(2):
#             #First Estimation
#             phi0_aux = arrayPhase[:,pairsList[i][0]] + arrayPhase[:,pairsList[i][1]]
#             #Dealias
#             indcsi = numpy.where(phi0_aux > numpy.pi)
#             phi0_aux[indcsi] -= 2*numpy.pi
#             indcsi = numpy.where(phi0_aux < -numpy.pi)
#             phi0_aux[indcsi] += 2*numpy.pi
#             #Direction Cosine 0
#             cosdir0[:,i] = -(phi0_aux)/(2*numpy.pi*0.5)
#
#             #Most-Accurate Second Estimation
#             phi1_aux =  arrayPhase[:,pairsList[i][0]] - arrayPhase[:,pairsList[i][1]]
#             phi1_aux = phi1_aux.reshape(phi1_aux.size,1)
#             #Direction Cosine 1
#             cosdir1 = -(phi1_aux + ang_aux)/(2*numpy.pi*4.5)
#
#             #Searching the correct Direction Cosine
#             cosdir0_aux = cosdir0[:,i]
#             cosdir0_aux = cosdir0_aux.reshape(cosdir0_aux.size,1)
#             #Minimum Distance
#             cosDiff = (cosdir1 - cosdir0_aux)**2
#             indcos = cosDiff.argmin(axis = 1)
#             #Saving Value obtained
#             cosdir[:,i] = cosdir1[numpy.arange(len(indcos)),indcos]
#
#         return cosdir0, cosdir
#
#     def __calculateAOA(self, cosdir, azimuth):
#         cosdirX = cosdir[:,0]
#         cosdirY = cosdir[:,1]
#
#         zenithAngle = numpy.arccos(numpy.sqrt(1 - cosdirX**2 - cosdirY**2))*180/numpy.pi
#         azimuthAngle = numpy.arctan2(cosdirX,cosdirY)*180/numpy.pi + azimuth #0 deg north, 90 deg east
#         angles = numpy.vstack((azimuthAngle, zenithAngle)).transpose()
#
#         return angles
#
#     def __getHeights(self, Ranges, zenith, error, minHeight, maxHeight):
#
#         Ramb = 375  #Ramb = c/(2*PRF)
#         Re = 6371   #Earth Radius
#         heights = numpy.zeros(Ranges.shape)
#
#         R_aux = numpy.array([0,1,2])*Ramb
#         R_aux = R_aux.reshape(1,R_aux.size)
#
#         Ranges = Ranges.reshape(Ranges.size,1)
#
#         Ri = Ranges + R_aux
#         hi = numpy.sqrt(Re**2 + Ri**2 + (2*Re*numpy.cos(zenith*numpy.pi/180)*Ri.transpose()).transpose()) - Re
#
#         #Check if there is a height between 70 and 110 km
#         h_bool = numpy.sum(numpy.logical_and(hi > minHeight, hi < maxHeight), axis = 1)
#         ind_h = numpy.where(h_bool == 1)[0]
#
#         hCorr = hi[ind_h, :]
#         ind_hCorr = numpy.where(numpy.logical_and(hi > minHeight, hi < maxHeight))
#
#         hCorr = hi[ind_hCorr]
#         heights[ind_h] = hCorr
#
#         #Setting Error
#         #Number 13: Height unresolvable echo: not valid height within 70 to 110 km
#         #Number 14: Height ambiguous echo: more than one possible height within 70 to 110 km
#
#         indInvalid2 = numpy.where(numpy.logical_and(h_bool > 1, error == 0))[0]
#         error[indInvalid2] = 14
#         indInvalid1 = numpy.where(numpy.logical_and(h_bool == 0, error == 0))[0]
#         error[indInvalid1] = 13
#
#         return heights, error



class IGRFModel(Operation):
    '''
    Written by R. Flores
    '''
    """Operation to calculate Geomagnetic parameters.

    Parameters:
    -----------
    None

    Example
    --------

    op = proc_unit.addOperation(name='IGRFModel', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.aux=1

    def run(self,dataOut):

        try:
            from schainpy.model.proc import mkfact_short_2020_2
        except:
            log.warning('You should install "mkfact_short_2020" module to process IGRF Model')

        if self.aux==1:

            #dataOut.TimeBlockSeconds_First_Time=time.mktime(time.strptime(dataOut.TimeBlockDate))
            #### we do not use dataOut.datatime.ctime() because it's the time of the second (next) block
            dataOut.TimeBlockSeconds_First_Time=dataOut.TimeBlockSeconds
            dataOut.bd_time=time.gmtime(dataOut.TimeBlockSeconds_First_Time)
            dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
            dataOut.ut=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0

            self.aux=0
            dh = dataOut.heightList[1]-dataOut.heightList[0]
            dataOut.h=numpy.arange(0.0,dh*dataOut.MAXNRANGENDT,dh,dtype='float32')
            dataOut.bfm=numpy.zeros(dataOut.MAXNRANGENDT,dtype='float32')
            dataOut.bfm=numpy.array(dataOut.bfm,order='F')
            dataOut.thb=numpy.zeros(dataOut.MAXNRANGENDT,dtype='float32')
            dataOut.thb=numpy.array(dataOut.thb,order='F')
            dataOut.bki=numpy.zeros(dataOut.MAXNRANGENDT,dtype='float32')
            dataOut.bki=numpy.array(dataOut.bki,order='F')
            mkfact_short_2020_2.mkfact(dataOut.year,dataOut.h,dataOut.bfm,dataOut.thb,dataOut.bki,dataOut.MAXNRANGENDT)


        return dataOut

class MergeProc(ProcessingUnit):

    def __init__(self):
        ProcessingUnit.__init__(self)

    def run(self, attr_data, attr_data_2 = None, attr_data_3 = None, attr_data_4 = None, attr_data_5 = None, mode=0):
        #print("*****************************Merge***************")

        self.dataOut = getattr(self, self.inputs[0])
        data_inputs = [getattr(self, attr) for attr in self.inputs]
        #print(data_inputs)
        #print("Run: ",self.dataOut.runNextUnit)
        #exit(1)
        #print(self.dataOut.nHeights)
        #exit(1)
        #print("a:", [getattr(data, attr_data) for data in data_inputs][1])
        #exit(1)
        if mode==0:
            data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs])
            setattr(self.dataOut, attr_data, data)

        if mode==1: #Hybrid
            #data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs],axis=1)
            #setattr(self.dataOut, attr_data, data)
            setattr(self.dataOut, 'dataLag_spc', [getattr(data, attr_data) for data in data_inputs][0])
            setattr(self.dataOut, 'dataLag_spc_LP', [getattr(data, attr_data) for data in data_inputs][1])
            setattr(self.dataOut, 'dataLag_cspc', [getattr(data, attr_data_2) for data in data_inputs][0])
            setattr(self.dataOut, 'dataLag_cspc_LP', [getattr(data, attr_data_2) for data in data_inputs][1])
            #setattr(self.dataOut, 'nIncohInt', [getattr(data, attr_data_3) for data in data_inputs][0])
            #setattr(self.dataOut, 'nIncohInt_LP', [getattr(data, attr_data_3) for data in data_inputs][1])
            '''
            print(self.dataOut.dataLag_spc_LP.shape)
            print(self.dataOut.dataLag_cspc_LP.shape)
            exit(1)
            '''

            #self.dataOut.dataLag_spc_LP = numpy.transpose(self.dataOut.dataLag_spc_LP[0],(2,0,1))
            #self.dataOut.dataLag_cspc_LP = numpy.transpose(self.dataOut.dataLag_cspc_LP,(3,1,2,0))
            '''
            print("Merge")
            print(numpy.shape(self.dataOut.dataLag_spc))
            print(numpy.shape(self.dataOut.dataLag_spc_LP))
            print(numpy.shape(self.dataOut.dataLag_cspc))
            print(numpy.shape(self.dataOut.dataLag_cspc_LP))
            exit(1)
            '''
            #print(numpy.sum(self.dataOut.dataLag_spc_LP[2,:,164])/128)
            #print(numpy.sum(self.dataOut.dataLag_cspc_LP[0,:,30,1])/128)
            #exit(1)
            #print(self.dataOut.NDP)
            #print(self.dataOut.nNoiseProfiles)

            #self.dataOut.nIncohInt_LP = 128
            self.dataOut.nProfiles_LP = 128#self.dataOut.nIncohInt_LP
            self.dataOut.nIncohInt_LP = self.dataOut.nIncohInt
            self.dataOut.NLAG = 16
            self.dataOut.NRANGE = 200
            self.dataOut.NSCAN = 128
            #print(numpy.shape(self.dataOut.data_spc))

            #exit(1)

        if mode==2: #HAE 2022
            data = numpy.sum([getattr(data, attr_data) for data in data_inputs],axis=0)
            setattr(self.dataOut, attr_data, data)

            self.dataOut.nIncohInt *= 2
            #meta = self.dataOut.getFreqRange(1)/1000.
            self.dataOut.freqRange = self.dataOut.getFreqRange(1)/1000.

            #exit(1)

        if mode==4: #Hybrid LP-SSheightProfiles
            #data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs],axis=1)
            #setattr(self.dataOut, attr_data, data)
            setattr(self.dataOut, 'dataLag_spc', getattr(data_inputs[0], attr_data)) #DP
            setattr(self.dataOut, 'dataLag_cspc', getattr(data_inputs[0], attr_data_2)) #DP
            setattr(self.dataOut, 'dataLag_spc_LP', getattr(data_inputs[1], attr_data_3)) #LP
            #setattr(self.dataOut, 'dataLag_cspc_LP', getattr(data_inputs[1], attr_data_4)) #LP
            #setattr(self.dataOut, 'data_acf', getattr(data_inputs[1], attr_data_5)) #LP
            setattr(self.dataOut, 'data_acf', getattr(data_inputs[1], attr_data_5)) #LP
            #print("Merge data_acf: ",self.dataOut.data_acf.shape)


            #self.dataOut.nIncohInt_LP = 128
            #self.dataOut.nProfiles_LP = 128#self.dataOut.nIncohInt_LP
            self.dataOut.nProfiles_LP = 16#28#self.dataOut.nIncohInt_LP
            self.dataOut.nProfiles_LP = self.dataOut.data_acf.shape[1]#28#self.dataOut.nIncohInt_LP
            self.dataOut.NSCAN = 128
            self.dataOut.nIncohInt_LP = self.dataOut.nIncohInt*self.dataOut.NSCAN
            #print("sahpi",self.dataOut.nIncohInt_LP)
            #exit(1)
            self.dataOut.NLAG = 16
            self.dataOut.NLAG = self.dataOut.data_acf.shape[1]
            self.dataOut.NRANGE = self.dataOut.data_acf.shape[-1]

            #print(numpy.shape(self.dataOut.data_spc))

            #exit(1)
        if mode==5:
            data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs])
            setattr(self.dataOut, attr_data, data)
            data = numpy.concatenate([getattr(data, attr_data_2) for data in data_inputs])
            setattr(self.dataOut, attr_data_2, data)

        if mode==6: #Hybrid Spectra-Voltage
            #data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs],axis=1)
            #setattr(self.dataOut, attr_data, data)
            setattr(self.dataOut, 'dataLag_spc', getattr(data_inputs[1], attr_data)) #DP
            setattr(self.dataOut, 'dataLag_cspc', getattr(data_inputs[1], attr_data_2)) #DP
            setattr(self.dataOut, 'output_LP_integrated', getattr(data_inputs[0], attr_data_3)) #LP
            #setattr(self.dataOut, 'dataLag_cspc_LP', getattr(data_inputs[1], attr_data_4)) #LP
            #setattr(self.dataOut, 'data_acf', getattr(data_inputs[1], attr_data_5)) #LP
            #setattr(self.dataOut, 'data_acf', getattr(data_inputs[1], attr_data_5)) #LP
            #print("Merge data_acf: ",self.dataOut.data_acf.shape)
            #print(self.dataOut.NSCAN)
            self.dataOut.nIncohInt = int(self.dataOut.NAVG * self.dataOut.nint)
            #print(self.dataOut.dataLag_spc.shape)
            self.dataOut.nProfiles = self.dataOut.nProfiles_DP = self.dataOut.dataLag_spc.shape[1]
            '''
            #self.dataOut.nIncohInt_LP = 128
            #self.dataOut.nProfiles_LP = 128#self.dataOut.nIncohInt_LP
            self.dataOut.nProfiles_LP = 16#28#self.dataOut.nIncohInt_LP
            self.dataOut.nProfiles_LP = self.dataOut.data_acf.shape[1]#28#self.dataOut.nIncohInt_LP
            self.dataOut.NSCAN = 128
            self.dataOut.nIncohInt_LP = self.dataOut.nIncohInt*self.dataOut.NSCAN
            #print("sahpi",self.dataOut.nIncohInt_LP)
            #exit(1)
            self.dataOut.NLAG = 16
            self.dataOut.NLAG = self.dataOut.data_acf.shape[1]
            self.dataOut.NRANGE = self.dataOut.data_acf.shape[-1]
            '''
            #print(numpy.shape(self.dataOut.data_spc))
            #print("*************************GOOD*************************")
            #exit(1)

        if mode==11: #MST ISR
            #data = numpy.concatenate([getattr(data, attr_data) for data in data_inputs],axis=1)
            #setattr(self.dataOut, attr_data, data)
            #setattr(self.dataOut, 'ph2', [getattr(data, attr_data) for data in data_inputs][1])
            #setattr(self.dataOut, 'dphi', [getattr(data, attr_data_2) for data in data_inputs][1])
            #setattr(self.dataOut, 'sdp2', [getattr(data, attr_data_3) for data in data_inputs][1])

            setattr(self.dataOut, 'ph2', getattr(data_inputs[1], attr_data)) #DP
            setattr(self.dataOut, 'dphi', getattr(data_inputs[1], attr_data_2)) #DP
            setattr(self.dataOut, 'sdp2', getattr(data_inputs[1], attr_data_3)) #DP

            print("MST Density", numpy.shape(self.dataOut.ph2))
            print("cf MST: ", self.dataOut.cf)
            #exit(1)
            #print("MST Density", self.dataOut.ph2[116:283])
            print("MST Density", self.dataOut.ph2[80:120])
            print("MST dPhi", self.dataOut.dphi[80:120])
            self.dataOut.ph2 *= self.dataOut.cf#0.0008136899
            #print("MST Density", self.dataOut.ph2[116:283])
            self.dataOut.sdp2 *= 0#self.dataOut.cf#0.0008136899
            #print("MST Density", self.dataOut.ph2[116:283])
            print("MST Density", self.dataOut.ph2[80:120])
            self.dataOut.NSHTS = int(numpy.shape(self.dataOut.ph2)[0])
            dH = self.dataOut.heightList[1]-self.dataOut.heightList[0]
            dH /= self.dataOut.windowOfFilter
            self.dataOut.heightList = numpy.arange(0,self.dataOut.NSHTS)*dH + dH
            #print("heightList: ", self.dataOut.heightList)
            self.dataOut.NDP = self.dataOut.NSHTS
            #exit(1)
            #print(self.dataOut.heightList)

class MST_Den_Conv(Operation):
    '''
    Written by R. Flores
    '''
    """Operation to calculate Geomagnetic parameters.

    Parameters:
    -----------
    None

    Example
    --------

    op = proc_unit.addOperation(name='MST_Den_Conv', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

    def run(self,dataOut):

        dataOut.PowDen = numpy.zeros((1,dataOut.NDP))
        dataOut.PowDen[0] = numpy.copy(dataOut.ph2[:dataOut.NDP])

        dataOut.FarDen = numpy.zeros((1,dataOut.NDP))
        dataOut.FarDen[0] = numpy.copy(dataOut.dphi[:dataOut.NDP])
        print("pow den shape", numpy.shape(dataOut.PowDen))
        print("far den shape", numpy.shape(dataOut.FarDen))
        return dataOut
