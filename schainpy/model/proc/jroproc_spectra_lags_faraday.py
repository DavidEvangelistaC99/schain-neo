# Copyright (c) 2012-2020 Jicamarca Radio Observatory
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
"""Spectra Lag processing Unit and operations

Here you will find the processing unit `SpectraLagProc` and several operations
to work with Spectra data type with lags
"""

import time
import itertools

import numpy

from schainpy.model.proc.jroproc_base import ProcessingUnit, MPDecorator, Operation
from schainpy.model.data.jrodata import Spectra
from schainpy.model.data.jrodata import hildebrand_sekhon
from schainpy.utils import log
from schainpy.model.data import _HS_algorithm

from schainpy.model.proc.jroproc_voltage import CleanCohEchoes

from time import time, mktime, strptime, gmtime, ctime


class SpectraLagProc(ProcessingUnit):
    '''
    Written by R. Flores
    '''
    def __init__(self):

        ProcessingUnit.__init__(self)

        self.buffer = None
        self.buffer_Lag = None
        self.firstdatatime = None
        self.profIndex = 0
        self.dataOut = Spectra()
        self.id_min = None
        self.id_max = None
        self.setupReq = False #Agregar a todas las unidades de proc

    def __updateSpecFromVoltage(self):

        self.dataOut.timeZone = self.dataIn.timeZone
        self.dataOut.dstFlag = self.dataIn.dstFlag
        self.dataOut.errorCount = self.dataIn.errorCount
        self.dataOut.useLocalTime = self.dataIn.useLocalTime
        try:
            self.dataOut.processingHeaderObj = self.dataIn.processingHeaderObj.copy()
        except:
            pass
        self.dataOut.radarControllerHeaderObj = self.dataIn.radarControllerHeaderObj.copy()
        self.dataOut.systemHeaderObj = self.dataIn.systemHeaderObj.copy()
        self.dataOut.channelList = self.dataIn.channelList
        self.dataOut.heightList = self.dataIn.heightList
        self.dataOut.dtype = numpy.dtype([('real', '<f4'), ('imag', '<f4')])
        self.dataOut.nProfiles = self.dataOut.nFFTPoints
        self.dataOut.flagDiscontinuousBlock = self.dataIn.flagDiscontinuousBlock
        self.dataOut.utctime = self.firstdatatime
        self.dataOut.flagDecodeData = self.dataIn.flagDecodeData
        self.dataOut.flagDeflipData = self.dataIn.flagDeflipData
        self.dataOut.flagShiftFFT = False
        self.dataOut.nCohInt = self.dataIn.nCohInt
        self.dataOut.nIncohInt = 1
        self.dataOut.windowOfFilter = self.dataIn.windowOfFilter
        self.dataOut.frequency = self.dataIn.frequency
        self.dataOut.realtime = self.dataIn.realtime
        self.dataOut.azimuth = self.dataIn.azimuth
        self.dataOut.zenith = self.dataIn.zenith
        self.dataOut.beam.codeList = self.dataIn.beam.codeList
        self.dataOut.beam.azimuthList = self.dataIn.beam.azimuthList
        self.dataOut.beam.zenithList = self.dataIn.beam.zenithList
        self.dataOut.runNextUnit = self.dataIn.runNextUnit
        try:
            self.dataOut.final_noise = self.dataIn.final_noise
        except:
            self.dataOut.final_noise = None

    def __getFft(self):
        """
        Convierte valores de Voltaje a Spectra

        Affected:
            self.dataOut.data_spc
            self.dataOut.data_cspc
            self.dataOut.data_dc
            self.dataOut.heightList
            self.profIndex
            self.buffer
            self.dataOut.flagNoData
        """
        #print(self.buffer[1,:,0])
        #exit(1)
        #print("buffer shape",self.buffer.shape)
        fft_volt = numpy.fft.fft(
            self.buffer, n=self.dataOut.nFFTPoints, axis=1)
        fft_volt = fft_volt.astype(numpy.dtype('complex'))
        dc = fft_volt[:, 0, :]

        # calculo de self-spectra
        fft_volt = numpy.fft.fftshift(fft_volt, axes=(1,))
        spc = fft_volt * numpy.conjugate(fft_volt)
        spc = spc.real

        blocksize = 0
        blocksize += dc.size
        blocksize += spc.size

        cspc = None
        pairIndex = 0
        if self.dataOut.pairsList != None:
            # calculo de cross-spectra
            #print("HERE")
            cspc = numpy.zeros(
                (self.dataOut.nPairs, self.dataOut.nFFTPoints, self.dataOut.nHeights), dtype='complex')
            for pair in self.dataOut.pairsList:
                if pair[0] not in self.dataOut.channelList:
                    raise ValueError("Error getting CrossSpectra: pair 0 of %s is not in channelList = %s" % (
                        str(pair), str(self.dataOut.channelList)))
                if pair[1] not in self.dataOut.channelList:
                    raise ValueError("Error getting CrossSpectra: pair 1 of %s is not in channelList = %s" % (
                        str(pair), str(self.dataOut.channelList)))

                cspc[pairIndex, :, :] = fft_volt[pair[0], :, :] * \
                    numpy.conjugate(fft_volt[pair[1], :, :])
                pairIndex += 1
            blocksize += cspc.size

        self.dataOut.data_spc = spc
        self.dataOut.data_cspc = cspc
        self.dataOut.data_dc = dc
        self.dataOut.blockSize = blocksize
        self.dataOut.flagShiftFFT = False

        #return spc,cspc,dc


    def VoltageType(self,nFFTPoints,nProfiles,ippFactor,pairsList):
        self.dataOut.flagNoData = True

        if nFFTPoints == None:
            raise ValueError("This SpectraProc.run() need nFFTPoints input variable")

        if nProfiles == None:
            nProfiles = nFFTPoints

        if ippFactor == None:
            self.dataOut.ippFactor = 1

        self.dataOut.nFFTPoints = nFFTPoints

        if self.buffer is None:
            self.buffer = numpy.zeros((self.dataIn.nChannels,
                                       nProfiles,
                                       self.dataIn.nHeights),
                                      dtype='complex')

        if self.dataIn.flagDataAsBlock:
            nVoltProfiles = self.dataIn.data.shape[1]

            if nVoltProfiles == nProfiles:
                self.buffer = self.dataIn.data.copy()
                self.profIndex = nVoltProfiles

            elif nVoltProfiles < nProfiles:

                if self.profIndex == 0:
                    self.id_min = 0
                    self.id_max = nVoltProfiles

                self.buffer[:, self.id_min:self.id_max,
                            :] = self.dataIn.data

                self.profIndex += nVoltProfiles
                self.id_min += nVoltProfiles
                self.id_max += nVoltProfiles
            else:
                raise ValueError("The type object %s has %d profiles, it should just has %d profiles" % (
                    self.dataIn.type, self.dataIn.data.shape[1], nProfiles))
                self.dataOut.flagNoData = True
        else:
            self.buffer[:, self.profIndex, :] = self.dataIn.data.copy()
            self.profIndex += 1

        if self.firstdatatime == None:
            self.firstdatatime = self.dataIn.utctime

        if self.profIndex == nProfiles:
            self.__updateSpecFromVoltage()
            if pairsList == None:
                self.dataOut.pairsList = [pair for pair in itertools.combinations(self.dataOut.channelList, 2)]
            else:
                self.dataOut.pairsList = pairsList
            #print(self.dataOut.pairsList)
            self.__getFft()
            self.dataOut.flagNoData = False
            self.firstdatatime = None
            #print(self.profIndex)
            self.profIndex = 0
            #input()

            '''
            if not self.dataOut.ByLags:
                pass
            else:
                return self.dataOut.data_spc,self.dataOut.data_cspc,self.dataOut.data_dc
                '''


    def run(self, nProfiles=None, nFFTPoints=None, pairsList=None, ippFactor=None, shift_fft=False, ByLags=False, LagPlot=0, nLags = None, runNextUnit = 0):

        self.dataIn.runNextUnit = runNextUnit
        self.dataOut.ByLags=ByLags
        self.dataOut.LagPlot=LagPlot

        #print(self.dataIn.data.shape)
        #exit(1)
        '''
        try:
            print(self.dataIn.data.shape)
        except:
            print("datalags",self.dataIn.datalags.shape)
        try:
            print("datalags",self.dataIn.datalags.shape)
        except:
            pass
            '''
        if self.dataIn.type == "Spectra":
            self.dataOut.copy(self.dataIn)
            if shift_fft:
                #desplaza a la derecha en el eje 2 determinadas posiciones
                shift = int(self.dataOut.nFFTPoints/2)
                self.dataOut.data_spc = numpy.roll(self.dataOut.data_spc, shift , axis=1)

                if self.dataOut.data_cspc is not None:
                    #desplaza a la derecha en el eje 2 determinadas posiciones
                    self.dataOut.data_cspc = numpy.roll(self.dataOut.data_cspc, shift, axis=1)
            if pairsList:
                self.__selectPairs(pairsList)

        elif self.dataIn.type == "Voltage":

            if not self.dataOut.ByLags:
                #self.dataOut.data = self.dataIn.data
                try:
                    self.dataOut.FlipChannels=self.dataIn.FlipChannels
                except: pass
                self.dataOut.TimeBlockSeconds=self.dataIn.TimeBlockSeconds
                self.VoltageType(nFFTPoints,nProfiles,ippFactor,pairsList)
            else:
                self.dataOut.nLags = nLags
                self.dataOut.DPL=self.dataIn.DPL
                #self.dataOut.NDP=self.dataIn.NDP
                self.dataOut.datalags=self.dataIn.datalags
                self.dataOut.dataLag_spc=[]
                self.dataOut.dataLag_cspc=[]
                self.dataOut.dataLag_dc=[]
                if self.buffer_Lag is None:
                    self.buffer_Lag = numpy.zeros((self.dataIn.nChannels,
                                               nProfiles,
                                               self.dataIn.nHeights,self.dataOut.nLags),
                                              dtype='complex')

                for i in range(self.dataOut.nLags):
                    self.dataOut.data=self.dataIn.data=self.dataIn.datalags[:,:,:,i]


                    if i>0 and self.id_min is not None:

                        self.profIndex -= self.dataIn.data.shape[1]
                        self.id_min -= self.dataIn.data.shape[1]
                        self.id_max -= self.dataIn.data.shape[1]

                    if self.profIndex>0 and self.id_min is not None:

                        self.buffer[:,:self.id_max-self.dataIn.data.shape[1],:]=self.buffer_Lag[:,:self.id_max-self.dataIn.data.shape[1],:,i]

                    self.VoltageType(nFFTPoints,nProfiles,ippFactor,pairsList)


                    if self.id_min is not None:

                        self.buffer_Lag[:,self.id_min-self.dataIn.data.shape[1]:self.id_max-self.dataIn.data.shape[1],:,i]=self.buffer[:,self.id_min-self.dataIn.data.shape[1]:self.id_max-self.dataIn.data.shape[1],:]

                    if not self.dataOut.flagNoData:
                        self.profIndex=nProfiles
                        self.firstdatatime = self.dataOut.utctime
                        if i==self.dataOut.nLags-1:
                            self.profIndex=0
                            self.firstdatatime = None
                        self.dataOut.dataLag_spc.append(self.dataOut.data_spc)
                        self.dataOut.dataLag_cspc.append(self.dataOut.data_cspc)
                        self.dataOut.dataLag_dc.append(self.dataOut.data_dc)


                if not self.dataOut.flagNoData:
                    self.dataOut.dataLag_spc=numpy.array(self.dataOut.dataLag_spc)
                    self.dataOut.dataLag_cspc=numpy.array(self.dataOut.dataLag_cspc)
                    self.dataOut.dataLag_dc=numpy.array(self.dataOut.dataLag_dc)
                    self.dataOut.dataLag_spc=self.dataOut.dataLag_spc.transpose(1,2,3,0)
                    self.dataOut.dataLag_cspc=self.dataOut.dataLag_cspc.transpose(1,2,3,0)
                    self.dataOut.dataLag_dc=self.dataOut.dataLag_dc.transpose(1,2,0)

                    self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,self.dataOut.LagPlot]
                    self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,self.dataOut.LagPlot]
                    self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,self.dataOut.LagPlot]

                    self.dataOut.TimeBlockSeconds=self.dataIn.TimeBlockSeconds
                    self.dataOut.flagDataAsBlock=self.dataIn.flagDataAsBlock
                    try:
                        self.dataOut.FlipChannels=self.dataIn.FlipChannels
                    except: pass
        else:
            raise ValueError("The type of input object '%s' is not valid".format(
                self.dataIn.type))

        #print("after",self.dataOut.data_spc[0,:,20])

class removeDCLag(Operation):
    '''
    Written by R. Flores
    '''
    def remover(self,mode):
        jspectra = self.dataOut.data_spc
        jcspectra = self.dataOut.data_cspc

        num_chan = jspectra.shape[0]
        num_hei = jspectra.shape[2]


        if jcspectra is not None:
            self.jcspectraExist = jcspectraExist = True
            num_pairs = jcspectra.shape[0]
        else:
            self.jcspectraExist = jcspectraExist = False
        #print(jcspectraExist)

        freq_dc = int(jspectra.shape[1] / 2)
        ind_vel = numpy.array([-2, -1, 1, 2]) + freq_dc
        ind_vel = ind_vel.astype(int)

        if ind_vel[0] < 0:
            ind_vel[list(range(0, 1))] = ind_vel[list(range(0, 1))] + self.num_prof

        if mode == 1:
            jspectra[:, freq_dc, :] = (
                jspectra[:, ind_vel[1], :] + jspectra[:, ind_vel[2], :]) / 2  # CORRECCION

            if jcspectraExist:
                jcspectra[:, freq_dc, :] = (
                    jcspectra[:, ind_vel[1], :] + jcspectra[:, ind_vel[2], :]) / 2

        if mode == 2:

            vel = numpy.array([-2, -1, 1, 2])
            xx = numpy.zeros([4, 4])

            for fil in range(4):
                xx[fil, :] = vel[fil]**numpy.asarray(list(range(4)))

            xx_inv = numpy.linalg.inv(xx)
            xx_aux = xx_inv[0, :]
            #print("inside")


            for ich in range(num_chan):
                yy = jspectra[ich, ind_vel, :]
                jspectra[ich, freq_dc, :] = numpy.dot(xx_aux, yy)
                #print(jspectra.shape)

                junkid = jspectra[ich, freq_dc, :] <= 0
                cjunkid = sum(junkid)

                if cjunkid.any():
                    jspectra[ich, freq_dc, junkid.nonzero()] = (
                        jspectra[ich, ind_vel[1], junkid] + jspectra[ich, ind_vel[2], junkid]) / 2
            #print(jspectra.shape)
            if jcspectraExist:
                for ip in range(num_pairs):
                    yy = jcspectra[ip, ind_vel, :]
                    jcspectra[ip, freq_dc, :] = numpy.dot(xx_aux, yy)
        #print(jspectra.shape)
        if not self.dataOut.ByLags:
            self.dataOut.data_spc = jspectra
            self.dataOut.data_cspc = jcspectra
        else:
            if jcspectraExist is True:
                return jspectra,jcspectra
            else:
                #print(jspectra.shape)
                return jspectra


    def run(self, dataOut, mode=2):

        self.dataOut = dataOut
        if not dataOut.ByLags:
            self.remover(mode)

        else:
            for i in range(self.dataOut.nLags):
                self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,i]
                if self.dataOut.dataLag_cspc is not None:
                    self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,i]
                else:
                    self.dataOut.data_cspc = None
                ##self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,i] Check!
                #print("HERE")
                if self.dataOut.dataLag_cspc is not None:
                    self.dataOut.dataLag_spc[:,:,:,i],self.dataOut.dataLag_cspc[:,:,:,i]=self.remover(mode)
                else:
                    self.dataOut.dataLag_spc[:,:,:,i]=self.remover(mode)

            #exit()
            self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,self.dataOut.LagPlot]
            if self.jcspectraExist is True:
                self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,self.dataOut.LagPlot]
            ##self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,self.dataOut.LagPlot] Check!





        return self.dataOut

class removeDCLagFlip(Operation):
    '''
    Written by R. Flores
    '''
    #CHANGES MADE ONLY FOR MODE 2 AND NOT CONSIDERING CSPC

    def remover(self,mode):
        jspectra = self.dataOut.data_spc
        jcspectra = self.dataOut.data_cspc

        num_chan = jspectra.shape[0]
        num_hei = jspectra.shape[2]

        if jcspectra is not None:
            jcspectraExist = True
            num_pairs = jcspectra.shape[0]
        else:
            jcspectraExist = False

        freq_dc = int(jspectra.shape[1] / 2)
        ind_vel = numpy.array([-2, -1, 1, 2]) + freq_dc
        ind_vel = ind_vel.astype(int)

        if ind_vel[0] < 0:
            ind_vel[list(range(0, 1))] = ind_vel[list(range(0, 1))] + self.num_prof

        if mode == 1:
            jspectra[:, freq_dc, :] = (
                jspectra[:, ind_vel[1], :] + jspectra[:, ind_vel[2], :]) / 2  # CORRECCION

            if jcspectraExist:
                jcspectra[:, freq_dc, :] = (
                    jcspectra[:, ind_vel[1], :] + jcspectra[:, ind_vel[2], :]) / 2

        if mode == 2:

            vel = numpy.array([-2, -1, 1, 2])
            xx = numpy.zeros([4, 4])

            for fil in range(4):
                xx[fil, :] = vel[fil]**numpy.asarray(list(range(4)))

            xx_inv = numpy.linalg.inv(xx)
            xx_aux = xx_inv[0, :]

            for ich in range(num_chan):
                if ich in self.dataOut.FlipChannels:


                    ind_freq_flip=[-1, -2, 1, 2]

                    yy = jspectra[ich, ind_freq_flip, :]

                    jspectra[ich, 0, :] = numpy.dot(xx_aux, yy)

                    junkid = jspectra[ich, 0, :] <= 0
                    cjunkid = sum(junkid)

                    if cjunkid.any():
                        jspectra[ich, 0, junkid.nonzero()] = (
                            jspectra[ich, ind_freq_flip[1], junkid] + jspectra[ich, ind_freq_flip[2], junkid]) / 2


                else:
                    yy = jspectra[ich, ind_vel, :]
                    jspectra[ich, freq_dc, :] = numpy.dot(xx_aux, yy)

                    junkid = jspectra[ich, freq_dc, :] <= 0
                    cjunkid = sum(junkid)

                    if cjunkid.any():
                        jspectra[ich, freq_dc, junkid.nonzero()] = (
                            jspectra[ich, ind_vel[1], junkid] + jspectra[ich, ind_vel[2], junkid]) / 2

            if jcspectraExist:
                for ip in range(num_pairs):
                    yy = jcspectra[ip, ind_vel, :]
                    jcspectra[ip, freq_dc, :] = numpy.dot(xx_aux, yy)

                    yy = jcspectra[ip, ind_freq_flip, :]
                    jcspectra[ip, 0, :] = numpy.dot(xx_aux, yy)

        if not self.dataOut.ByLags:
            self.dataOut.data_spc = jspectra
            self.dataOut.data_cspc = jcspectra
        else:
            return jspectra,jcspectra


    def run(self, dataOut, mode=2):
        #print("***********************************Remove DC***********************************")
        ##print(dataOut.FlipChannels)
        #exit(1)
        self.dataOut = dataOut
        if not dataOut.ByLags:
            self.remover(mode)
        else:
            for i in range(self.dataOut.DPL):
                self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,i]
                self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,i]
                self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,i]
                self.dataOut.dataLag_spc[:,:,:,i],self.dataOut.dataLag_cspc[:,:,:,i]=self.remover(mode)

            self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,self.dataOut.LagPlot]
            self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,self.dataOut.LagPlot]
            self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,self.dataOut.LagPlot]

        return self.dataOut

class removeHighValuesFreq(Operation):

    def removeByLag(self,nkill,nChannels,nHeights,data):

        for i in range(nChannels):
            for j in range(nHeights):
                buffer=numpy.copy(data[i,:,j])
                sortdata=sorted(buffer)
                avg=numpy.mean(sortdata[:-nkill])
                sortID=buffer.argsort()
                for k in list(sortID[-nkill:]):
                    buffer[k]=avg
                data[i,:,j]=numpy.copy(buffer)

    def run(self,dataOut,nkill=3):

        for i in range(dataOut.DPL):
            data=dataOut.dataLag_spc[:,:,:,i]

            self.removeByLag(nkill,dataOut.nChannels,dataOut.nHeights,data)
            dataOut.dataLag_spc[:,:,:,i]=data


        return dataOut

class removeInterferenceLag(Operation):

    def removeInterference2(self):

        cspc = self.dataOut.data_cspc
        spc = self.dataOut.data_spc
        Heights = numpy.arange(cspc.shape[2])
        realCspc = numpy.abs(cspc)

        for i in range(cspc.shape[0]):
            LinePower= numpy.sum(realCspc[i], axis=0)
            Threshold = numpy.amax(LinePower)-numpy.sort(LinePower)[len(Heights)-int(len(Heights)*0.1)]
            SelectedHeights = Heights[ numpy.where( LinePower < Threshold ) ]
            InterferenceSum = numpy.sum( realCspc[i,:,SelectedHeights], axis=0 )
            InterferenceThresholdMin = numpy.sort(InterferenceSum)[int(len(InterferenceSum)*0.98)]
            InterferenceThresholdMax = numpy.sort(InterferenceSum)[int(len(InterferenceSum)*0.99)]

            #InterferenceSum[0]*=2
            #print("sum",InterferenceSum)
            #print("min",InterferenceThresholdMin)


            InterferenceRange = numpy.where( ([InterferenceSum > InterferenceThresholdMin]))# , InterferenceSum < InterferenceThresholdMax]) )
            #InterferenceRange = numpy.where( ([InterferenceRange < InterferenceThresholdMax]))
            if len(InterferenceRange)<int(cspc.shape[1]*0.3):
                #print("profile",InterferenceRange)
                #print(cspc[i,InterferenceRange,:])
                cspc[i,InterferenceRange,:] = numpy.NaN
                #print(cspc[i,InterferenceRange,:])
                #print("profile",InterferenceRange)
                #exit()

        if not self.dataOut.ByLags:
            self.dataOut.data_cspc = cspc
        else:
            return cspc

    def removeInterference(self, interf, hei_interf, nhei_interf, offhei_interf):

        jspectra = self.dataOut.data_spc
        jcspectra = self.dataOut.data_cspc
        jnoise = self.dataOut.getNoise()
        num_incoh = self.dataOut.nIncohInt

        num_channel = jspectra.shape[0]
        num_prof = jspectra.shape[1]
        num_hei = jspectra.shape[2]

        # hei_interf
        if hei_interf is None:
            count_hei = int(num_hei / 2)
            hei_interf = numpy.asmatrix(list(range(count_hei))) + num_hei - count_hei
            hei_interf = numpy.asarray(hei_interf)[0]
        # nhei_interf

        if (nhei_interf == None):
            nhei_interf = 5
        if (nhei_interf < 1):
            nhei_interf = 1
        if (nhei_interf > count_hei):
            nhei_interf = count_hei
        if (offhei_interf == None):
            offhei_interf = 0


        ind_hei = list(range(num_hei))
#         mask_prof = numpy.asarray(range(num_prof - 2)) + 1
#         mask_prof[range(num_prof/2 - 1,len(mask_prof))] += 1
        mask_prof = numpy.asarray(list(range(num_prof)))
        num_mask_prof = mask_prof.size
        comp_mask_prof = [0, num_prof / 2]

        # noise_exist:    Determina si la variable jnoise ha sido definida y contiene la informacion del ruido de cada canal
        if (jnoise.size < num_channel or numpy.isnan(jnoise).any()):
            jnoise = numpy.nan
        noise_exist = jnoise[0] < numpy.Inf

        # Subrutina de Remocion de la Interferencia
        for ich in range(num_channel):
            # Se ordena los espectros segun su potencia (menor a mayor)
            #print(mask_prof)
            #exit()

            power = jspectra[ich, mask_prof, :]
            power = power[:, hei_interf]
            power = power.sum(axis=0)
            psort = power.ravel().argsort()


            # Se estima la interferencia promedio en los Espectros de Potencia empleando
            junkspc_interf = jspectra[ich, :, hei_interf[psort[list(range(
                offhei_interf, nhei_interf + offhei_interf))]]]


            #exit()

            if noise_exist:
                #    tmp_noise = jnoise[ich] / num_prof
                tmp_noise = jnoise[ich]
            junkspc_interf = junkspc_interf - tmp_noise
            #junkspc_interf[:,comp_mask_prof] = 0

            jspc_interf = junkspc_interf.sum(axis=0) / nhei_interf
            jspc_interf = jspc_interf.transpose()
            # Calculando el espectro de interferencia promedio
            noiseid = numpy.where(
                jspc_interf <= tmp_noise / numpy.sqrt(num_incoh))
            noiseid = noiseid[0]
            cnoiseid = noiseid.size
            interfid = numpy.where(
                jspc_interf > tmp_noise / numpy.sqrt(num_incoh))
            interfid = interfid[0]
            cinterfid = interfid.size

            if (cnoiseid > 0):
                jspc_interf[noiseid] = 0

            # Expandiendo los perfiles a limpiar
            if (cinterfid > 0):
                new_interfid = (
                    numpy.r_[interfid - 1, interfid, interfid + 1] + num_prof) % num_prof
                new_interfid = numpy.asarray(new_interfid)
                new_interfid = {x for x in new_interfid}
                new_interfid = numpy.array(list(new_interfid))
                new_cinterfid = new_interfid.size
            else:
                new_cinterfid = 0

            for ip in range(new_cinterfid):
                ind = junkspc_interf[:, new_interfid[ip]].ravel().argsort()
                jspc_interf[new_interfid[ip]
                            ] = junkspc_interf[ind[nhei_interf // 2], new_interfid[ip]]

            jspectra[ich, :, ind_hei] = jspectra[ich, :,
                                                 ind_hei] - jspc_interf  # Corregir indices

            # Removiendo la interferencia del punto de mayor interferencia
            ListAux = jspc_interf[mask_prof].tolist()
            maxid = ListAux.index(max(ListAux))

            if cinterfid > 0:
                for ip in range(cinterfid * (interf == 2) - 1):
                    ind = (jspectra[ich, interfid[ip], :] < tmp_noise *
                           (1 + 1 / numpy.sqrt(num_incoh))).nonzero()
                    cind = len(ind)

                    if (cind > 0):
                        jspectra[ich, interfid[ip], ind] = tmp_noise * \
                            (1 + (numpy.random.uniform(cind) - 0.5) /
                             numpy.sqrt(num_incoh))

                ind = numpy.array([-2, -1, 1, 2])
                xx = numpy.zeros([4, 4])

                for id1 in range(4):
                    xx[:, id1] = ind[id1]**numpy.asarray(list(range(4)))

                xx_inv = numpy.linalg.inv(xx)
                xx = xx_inv[:, 0]
                ind = (ind + maxid + num_mask_prof) % num_mask_prof
                yy = jspectra[ich, mask_prof[ind], :]
                jspectra[ich, mask_prof[maxid], :] = numpy.dot(
                    yy.transpose(), xx)

            indAux = (jspectra[ich, :, :] < tmp_noise *
                      (1 - 1 / numpy.sqrt(num_incoh))).nonzero()
            jspectra[ich, indAux[0], indAux[1]] = tmp_noise * \
                (1 - 1 / numpy.sqrt(num_incoh))

        # Remocion de Interferencia en el Cross Spectra
        if jcspectra is None:
            return jspectra, jcspectra
        num_pairs = int(jcspectra.size / (num_prof * num_hei))
        jcspectra = jcspectra.reshape(num_pairs, num_prof, num_hei)

        for ip in range(num_pairs):

            #-------------------------------------------

            cspower = numpy.abs(jcspectra[ip, mask_prof, :])
            cspower = cspower[:, hei_interf]
            cspower = cspower.sum(axis=0)

            cspsort = cspower.ravel().argsort()
            junkcspc_interf = jcspectra[ip, :, hei_interf[cspsort[list(range(
                offhei_interf, nhei_interf + offhei_interf))]]]
            junkcspc_interf = junkcspc_interf.transpose()
            jcspc_interf = junkcspc_interf.sum(axis=1) / nhei_interf

            ind = numpy.abs(jcspc_interf[mask_prof]).ravel().argsort()

            median_real = int(numpy.median(numpy.real(
                junkcspc_interf[mask_prof[ind[list(range(3 * num_prof // 4))]], :])))
            median_imag = int(numpy.median(numpy.imag(
                junkcspc_interf[mask_prof[ind[list(range(3 * num_prof // 4))]], :])))
            comp_mask_prof = [int(e) for e in comp_mask_prof]
            junkcspc_interf[comp_mask_prof, :] = numpy.complex(
                median_real, median_imag)

            for iprof in range(num_prof):
                ind = numpy.abs(junkcspc_interf[iprof, :]).ravel().argsort()
                jcspc_interf[iprof] = junkcspc_interf[iprof, ind[nhei_interf // 2]]

            # Removiendo la Interferencia
            jcspectra[ip, :, ind_hei] = jcspectra[ip,
                                                  :, ind_hei] - jcspc_interf

            ListAux = numpy.abs(jcspc_interf[mask_prof]).tolist()
            maxid = ListAux.index(max(ListAux))

            ind = numpy.array([-2, -1, 1, 2])
            xx = numpy.zeros([4, 4])

            for id1 in range(4):
                xx[:, id1] = ind[id1]**numpy.asarray(list(range(4)))

            xx_inv = numpy.linalg.inv(xx)
            xx = xx_inv[:, 0]

            ind = (ind + maxid + num_mask_prof) % num_mask_prof
            yy = jcspectra[ip, mask_prof[ind], :]
            jcspectra[ip, mask_prof[maxid], :] = numpy.dot(yy.transpose(), xx)

        # Guardar Resultados
        if not self.dataOut.ByLags:
            self.dataOut.data_spc = jspectra
            self.dataOut.data_cspc = jcspectra
        else:
            return jspectra,jcspectra

        return 1

    def run(self, dataOut, interf = 2,hei_interf = None, nhei_interf = None, offhei_interf = None, mode=1):

        self.dataOut = dataOut
        if not dataOut.ByLags:
            if mode == 1:
                self.removeInterference(interf = 2,hei_interf = None, nhei_interf = None, offhei_interf = None)
            elif mode == 2:
                self.removeInterference2()
        else:
            for i in range(self.dataOut.DPL):
                #print("BEFORE")
                self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,i]
                #print(i)
                #print(self.dataOut.dataLag_spc[0,0,0,i])
                #print("AFTER")
                self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,i]
                self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,i]
                if mode == 1:
                    #print(self.dataOut.dataLag_spc[0,:,22,0])
                    self.dataOut.dataLag_spc[:,:,:,i],self.dataOut.dataLag_cspc[:,:,:,i]=self.removeInterference(interf, hei_interf, nhei_interf, offhei_interf)
                    #print(self.dataOut.dataLag_spc[0,:,22,0])
                    #input()
                elif mode ==2:
                    self.dataOut.dataLag_cspc[:,:,:,i]=self.removeInterference2()

            self.dataOut.data_spc=self.dataOut.dataLag_spc[:,:,:,self.dataOut.LagPlot]
            self.dataOut.data_cspc=self.dataOut.dataLag_cspc[:,:,:,self.dataOut.LagPlot]
            self.dataOut.data_dc=self.dataOut.dataLag_dc[:,:,self.dataOut.LagPlot]

        return self.dataOut

class IntegrationFaradaySpectra(Operation):
    '''
    Written by R. Flores
    '''
    __profIndex = 0
    __withOverapping = False

    __byTime = False
    __initime = None
    __lastdatatime = None
    __integrationtime = None

    __buffer_spc = None
    __buffer_cspc = None
    __buffer_dc = None

    __dataReady = False

    __timeInterval = None

    n = None

    def __init__(self):

        Operation.__init__(self)

    def setup(self, n=None, timeInterval=None, overlapping=False):
        """
        Set the parameters of the integration class.

        Inputs:

            n        :    Number of coherent integrations
            timeInterval   :    Time of integration. If the parameter "n" is selected this one does not work
            overlapping    :

        """

        self.__initime = None
        self.__lastdatatime = 0

        self.__buffer_spc = []
        self.__buffer_cspc = []
        self.__buffer_dc = 0

        self.__profIndex = 0
        self.__dataReady = False
        self.__byTime = False

        if n is None and timeInterval is None:
            raise ValueError("n or timeInterval should be specified ...")

        if n is not None:
            self.n = int(n)
        else:

            self.__integrationtime = int(timeInterval)
            self.n = None
            self.__byTime = True

    def putData(self, data_spc, data_cspc, data_dc):
        """
        Add a profile to the __buffer_spc and increase in one the __profileIndex

        """

        self.__buffer_spc.append(data_spc)

        if data_cspc is None:
            self.__buffer_cspc = None
        else:
            self.__buffer_cspc.append(data_cspc)

        if data_dc is None:
            self.__buffer_dc = None
        else:
            self.__buffer_dc += data_dc

        self.__profIndex += 1

        return

    def hildebrand_sekhon_Integration(self,data,navg):

        sortdata = numpy.sort(data, axis=None)
        sortID=data.argsort()
        lenOfData = len(sortdata)
        nums_min = lenOfData*0.75
        if nums_min <= 5:
            nums_min = 5
        sump = 0.
        sumq = 0.
        j = 0
        cont = 1
        while((cont == 1)and(j < lenOfData)):
            sump += sortdata[j]
            sumq += sortdata[j]**2
            if j > nums_min:
                rtest = float(j)/(j-1) + 1.0/navg
                if ((sumq*j) > (rtest*sump**2)):
                    j = j - 1
                    sump = sump - sortdata[j]
                    sumq = sumq - sortdata[j]**2
                    cont = 0
            j += 1
        #lnoise = sump / j

        return j,sortID

    def pushData(self):
        """
        Return the sum of the last profiles and the profiles used in the sum.

        Affected:

        self.__profileIndex

        """
        bufferH=None
        buffer=None
        buffer1=None
        buffer_cspc=None
        self.__buffer_spc=numpy.array(self.__buffer_spc)
        self.__buffer_cspc=numpy.array(self.__buffer_cspc)
        freq_dc = int(self.__buffer_spc.shape[2] / 2)
        #print("FREQ_DC",freq_dc)
        #print(self.__buffer_spc[:,1,5,37,0])
        #lag_array=[0,2,4,6,8,10,12,14,16,18,20]
        for l in range(self.DPL):#dataOut.DPL):
            #breakFlag=False
            for k in range(7,self.nHeights):
                buffer_cspc=numpy.copy(self.__buffer_cspc[:,0,:,k,l])
                outliers_IDs_cspc=[]
                cspc_outliers_exist=False
                #indexmin_cspc=0
                for i in range(self.nChannels):#dataOut.nChannels):
                    if i==1 and k >= self.nHeights-2*l:
                        #breakFlag=True
                        continue
                        #pass
                    else:
                        buffer1=numpy.copy(self.__buffer_spc[:,i,:,k,l])
                        indexes=[]
                        #sortIDs=[]
                        outliers_IDs=[]
                        for j in range(self.nProfiles):
                            if i==0 and j==freq_dc: #NOT CONSIDERING DC PROFILE AT CHANNEL 0
                                continue
                            if i==1 and j==0: #NOT CONSIDERING DC PROFILE AT CHANNEL 1
                                continue
                            buffer=buffer1[:,j]
                            #index,sortID=self.hildebrand_sekhon_Integration(buffer,1)
                            index=int(_HS_algorithm.HS_algorithm(numpy.sort(buffer, axis=None),1))
                            sortID = buffer.argsort()
                            '''
                            if i==1 and l==0 and k==37:
                                print("j",j)
                                print("INDEX",index)
                                print(sortID[index:])
                                if j==5:
                                    aa=numpy.mean(buffer,axis=0)
                                    bb=numpy.sort(buffer)
                                    print(buffer)
                                    print(aa)
                                    print(bb[-1])
                                    '''
                            indexes.append(index)
                            #sortIDs.append(sortID)
                            outliers_IDs=numpy.append(outliers_IDs,sortID[index:])

                        outliers_IDs=numpy.array(outliers_IDs)
                        outliers_IDs=outliers_IDs.ravel()
                        outliers_IDs=numpy.unique(outliers_IDs)
                        outliers_IDs=outliers_IDs.astype(numpy.dtype('int64'))
                        indexes=numpy.array(indexes)
                        indexmin=numpy.min(indexes)

                        if indexmin != buffer1.shape[0]:
                            cspc_outliers_exist=True
                            ###sortdata=numpy.sort(buffer1,axis=0)
                            ###avg2=numpy.mean(sortdata[:indexmin,:],axis=0)
                            lt=outliers_IDs
                            avg=numpy.mean(buffer1[[t for t in range(buffer1.shape[0]) if t not in lt],:],axis=0)
                            '''
                            if k==37 and i==1 and l==0:
                                #cc=
                                print("index_min",indexmin)
                                print("outliers_ID",lt)
                                print("AVG",avg[5])
                                print("AVG_2",avg2[5])
                                '''

                            for p in list(outliers_IDs):
                                buffer1[p,:]=avg

                        self.__buffer_spc[:,i,:,k,l]=numpy.copy(buffer1)
                        ###cspc IDs
                        #indexmin_cspc+=indexmin_cspc
                        outliers_IDs_cspc=numpy.append(outliers_IDs_cspc,outliers_IDs)

                #if not breakFlag:
                outliers_IDs_cspc=outliers_IDs_cspc.astype(numpy.dtype('int64'))
                if cspc_outliers_exist:
                    #sortdata=numpy.sort(buffer_cspc,axis=0)
                    #avg=numpy.mean(sortdata[:indexmin_cpsc,:],axis=0)
                    lt=outliers_IDs_cspc

                    avg=numpy.mean(buffer_cspc[[t for t in range(buffer_cspc.shape[0]) if t not in lt],:],axis=0)
                    for p in list(outliers_IDs_cspc):
                        buffer_cspc[p,:]=avg

                self.__buffer_cspc[:,0,:,k,l]=numpy.copy(buffer_cspc)
                #else:
                    #break




        buffer=None
        bufferH=None
        buffer1=None
        buffer_cspc=None

        #print("cpsc",self.__buffer_cspc[:,0,0,0,0])
        #print(self.__profIndex)
        #exit()

        buffer=None
        #print(self.__buffer_spc[:,1,3,20,0])
        #print(self.__buffer_spc[:,1,5,37,0])
        data_spc = numpy.sum(self.__buffer_spc,axis=0)
        data_cspc = numpy.sum(self.__buffer_cspc,axis=0)

        #print(numpy.shape(data_spc))
        #data_spc[1,4,20,0]=numpy.nan

        #data_cspc = self.__buffer_cspc
        data_dc = self.__buffer_dc
        n = self.__profIndex

        self.__buffer_spc = []
        self.__buffer_cspc = []
        self.__buffer_dc = 0
        self.__profIndex = 0

        return data_spc, data_cspc, data_dc, n

    def byProfiles(self, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(*args)

        if self.__profIndex == self.n:

            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def byTime(self, datatime, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(*args)

        if (datatime - self.__initime) >= self.__integrationtime:
            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def integrate(self, datatime, *args):

        if self.__profIndex == 0:
            self.__initime = datatime

        if self.__byTime:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byTime(
                datatime, *args)
        else:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byProfiles(*args)

        if not self.__dataReady:
            return None, None, None, None

        return self.__initime, avgdata_spc, avgdata_cspc, avgdata_dc

    def run(self, dataOut, n=None, timeInterval=None, overlapping=False):
        if n == 1:
            dataOut.VelRange = dataOut.getVelRange(0)
            return dataOut
        #print("holo")
        dataOut.flagNoData = True

        if not self.isConfig:
            self.setup(n, timeInterval, overlapping)
            self.isConfig = True

        if not dataOut.ByLags:
            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.data_spc,
                                                                                dataOut.data_cspc,
                                                                                dataOut.data_dc)
        else:
            self.nProfiles=dataOut.nProfiles
            self.nChannels=dataOut.nChannels
            self.nHeights=dataOut.nHeights
            self.DPL=dataOut.DPL
            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.dataLag_spc,
                                                                                dataOut.dataLag_cspc,
                                                                                dataOut.dataLag_dc)

        if self.__dataReady:

            if not dataOut.ByLags:
                dataOut.data_spc = avgdata_spc
                dataOut.data_cspc = avgdata_cspc
                dataOut.data_dc = avgdata_dc
            else:
                dataOut.dataLag_spc = avgdata_spc
                dataOut.dataLag_cspc = avgdata_cspc
                dataOut.dataLag_dc = avgdata_dc

                dataOut.data_spc=dataOut.dataLag_spc[:,:,:,dataOut.LagPlot]
                dataOut.data_cspc=dataOut.dataLag_cspc[:,:,:,dataOut.LagPlot]
                dataOut.data_dc=dataOut.dataLag_dc[:,:,dataOut.LagPlot]

            dataOut.VelRange = dataOut.getVelRange(0)
            dataOut.nIncohInt *= self.n
            dataOut.utctime = avgdatatime
            dataOut.flagNoData = False

        return dataOut

class IntegrationFaradaySpectra2(Operation):
    '''
    Written by R. Flores
    '''
    __profIndex = 0
    __withOverapping = False

    __byTime = False
    __initime = None
    __lastdatatime = None
    __integrationtime = None

    __buffer_spc = None
    __buffer_cspc = None
    __buffer_dc = None

    __dataReady = False

    __timeInterval = None

    n = None

    def __init__(self):

        Operation.__init__(self)

    def setup(self, n=None, timeInterval=None, overlapping=False):
        """
        Set the parameters of the integration class.

        Inputs:

            n        :    Number of coherent integrations
            timeInterval   :    Time of integration. If the parameter "n" is selected this one does not work
            overlapping    :

        """

        self.__initime = None
        self.__lastdatatime = 0

        self.__buffer_spc = None
        self.__buffer_cspc = None
        self.__buffer_dc = 0

        self.__profIndex = 0
        self.__dataReady = False
        self.__byTime = False

        if n is None and timeInterval is None:
            raise ValueError("n or timeInterval should be specified ...")

        if n is not None:
            self.n = int(n)
        else:

            self.__integrationtime = int(timeInterval)
            self.n = None
            self.__byTime = True

    def putData(self, data_spc, data_cspc, data_dc):
        """
        Add a profile to the __buffer_spc and increase in one the __profileIndex

        """

        #print(numpy.shape(self.__buffer_spc))
        ##print(numpy.shape(data_spc))

        #self.__buffer_spc = numpy.insert(self.__buffer_spc,[],data_spc,axis=0)
        self.__buffer_spc[self.__profIndex,:]=data_spc[:]
        ##self.__buffer_spc.append(data_spc)
        #self.__buffer_spc = numpy.array(self.__buffer_spc)
        #print(numpy.shape(self.__buffer_spc))
        #print("bytes",sys.getsizeof(self.__buffer_spc))
        #print("bytes",asizeof(self.__buffer_spc))
        if data_cspc is None:
            self.__buffer_cspc = None

        else:
            self.__buffer_cspc[self.__profIndex,:]=data_cspc[:]

        if data_dc is None:
            self.__buffer_dc = None
        else:
            self.__buffer_dc += data_dc

        self.__profIndex += 1

        return

    def hildebrand_sekhon_Integration(self,data,navg):

        sortdata = numpy.sort(data, axis=None)
        sortID=data.argsort()
        lenOfData = len(sortdata)
        nums_min = lenOfData*0.75
        if nums_min <= 5:
            nums_min = 5
        sump = 0.
        sumq = 0.
        j = 0
        cont = 1
        while((cont == 1)and(j < lenOfData)):
            sump += sortdata[j]
            sumq += sortdata[j]**2
            if j > nums_min:
                rtest = float(j)/(j-1) + 1.0/navg
                if ((sumq*j) > (rtest*sump**2)):
                    j = j - 1
                    sump = sump - sortdata[j]
                    sumq = sumq - sortdata[j]**2
                    cont = 0
            j += 1
        #lnoise = sump / j

        return j,sortID

    def pushData_V0(self):
        """
        Return the sum of the last profiles and the profiles used in the sum.

        Affected:

        self.__profileIndex

        """
        bufferH=None
        buffer=None
        buffer1=None
        buffer_cspc=None
        self.__buffer_spc=numpy.array(self.__buffer_spc)
        if self.__buffer_cspc is not None:
            self.__buffer_cspc=numpy.array(self.__buffer_cspc)
        freq_dc = int(self.__buffer_spc.shape[2] / 2)
        #print("FREQ_DC",freq_dc)
        #print(self.__buffer_spc[:,1,5,37,0])
        #lag_array=[0,2,4,6,8,10,12,14,16,18,20]

        if self.nLags == 11:
            h0 = 7
        elif self.nLags == 16:
            h0 = 180


        '''
        import matplotlib.pyplot as plt
        plt.plot(self.__buffer_spc[:,0,freq_dc,33,0],marker='*')
        plt.ylim((0,700000))
        plt.show()
        import time
        time.sleep(60)
        exit(1)
        '''
        #'''
        import matplotlib.pyplot as plt
        #plt.plot(self.__buffer_spc[:,0,freq_dc-2,33,1],marker='*')
        plt.plot(sorted(self.__buffer_spc[:,0,freq_dc-2,33,1]),marker='*')
        plt.ylim((0,1.1*1.e6))
        plt.show()
        import time
        time.sleep(60)
        exit(1)
        #'''

        print(self.nLags)
        '''
        if self.nLags == 16:
            self.nLags = 0
            #exit(1)
        '''

        for l in range(self.nLags):#dataOut.DPL):
            #breakFlag=False
            for k in range(7,self.nHeights):
                if self.__buffer_cspc is not None:
                    buffer_cspc=numpy.copy(self.__buffer_cspc[:,0,:,k,l])
                    outliers_IDs_cspc=[]
                    cspc_outliers_exist=False
                #indexmin_cspc=0
                for i in range(2):
                #for i in range(self.nChannels):#dataOut.nChannels):
                    #if self.TrueLags:
                        #print("HERE")
                    if i==1 and k >= self.nHeights-2*l and self.TrueLags:
                        #breakFlag=True
                        continue

                        #pass
                    else:
                        buffer1=numpy.copy(self.__buffer_spc[:,i,:,k,l])
                        indexes=[]
                        #sortIDs=[]
                        outliers_IDs=[]
                        for j in range(self.nProfiles):
                            if i==0 and j==freq_dc: #NOT CONSIDERING DC PROFILE AT CHANNEL 0
                                continue
                            if self.FlipChannelsExist:
                                if i==1 and j==0: #NOT CONSIDERING DC PROFILE AT CHANNEL 1
                                    continue
                            else:
                                if i==1 and j==freq_dc: #NOT CONSIDERING DC PROFILE AT CHANNEL 1
                                    continue
                            #buffer=buffer1[:,j]
                            buffer=(buffer1[:,j]).real
                            '''
                            if self.nLags ==16 and l!=0:
                                print(buffer)
                                exit(1)
                                '''
                            #index,sortID=self.hildebrand_sekhon_Integration(buffer,1)
                            index=int(_HS_algorithm.HS_algorithm(numpy.sort(buffer, axis=None),1))
                            sortID = buffer.argsort()

                            indexes.append(index)
                            #sortIDs.append(sortID)
                            outliers_IDs=numpy.append(outliers_IDs,sortID[index:])

                        outliers_IDs=numpy.array(outliers_IDs)
                        outliers_IDs=outliers_IDs.ravel()
                        outliers_IDs=numpy.unique(outliers_IDs)
                        outliers_IDs=outliers_IDs.astype(numpy.dtype('int64'))
                        indexes=numpy.array(indexes)
                        indexmin=numpy.min(indexes)

                        if indexmin != buffer1.shape[0]:
                            cspc_outliers_exist=True
                            ###sortdata=numpy.sort(buffer1,axis=0)
                            ###avg2=numpy.mean(sortdata[:indexmin,:],axis=0)
                            lt=outliers_IDs
                            avg=numpy.mean(buffer1[[t for t in range(buffer1.shape[0]) if t not in lt],:],axis=0)


                            for p in list(outliers_IDs):
                                buffer1[p,:]=avg

                        self.__buffer_spc[:,i,:,k,l]=numpy.copy(buffer1)
                        ###cspc IDs
                        #indexmin_cspc+=indexmin_cspc
                        if self.__buffer_cspc is not None:
                            outliers_IDs_cspc=numpy.append(outliers_IDs_cspc,outliers_IDs)

                #if not breakFlag:
                #print(outliers_IDs_cspc)
                if self.__buffer_cspc is not None:
                    outliers_IDs_cspc=outliers_IDs_cspc.astype(numpy.dtype('int64'))
                    if cspc_outliers_exist:
                        #sortdata=numpy.sort(buffer_cspc,axis=0)
                        #avg=numpy.mean(sortdata[:indexmin_cpsc,:],axis=0)
                        lt=outliers_IDs_cspc

                        avg=numpy.mean(buffer_cspc[[t for t in range(buffer_cspc.shape[0]) if t not in lt],:],axis=0)
                        for p in list(outliers_IDs_cspc):
                            buffer_cspc[p,:]=avg

                    self.__buffer_cspc[:,0,:,k,l]=numpy.copy(buffer_cspc)

                #else:
                    #break
        #'''
        import matplotlib.pyplot as plt
        plt.plot(self.__buffer_spc[:,0,freq_dc-2,33,1],marker='*')
        plt.ylim((0,1.1*1.e6))
        plt.show()
        import time
        time.sleep(60)
        exit(1)
        #'''

        buffer=None
        bufferH=None
        buffer1=None
        buffer_cspc=None

        #print("cpsc",self.__buffer_cspc[:,0,0,0,0])
        #print(self.__profIndex)
        #exit()
        '''
        if self.nLags == 16:
            print(self.__buffer_spc[:,0,0,0,2])
            exit(1)
            '''

        buffer=None
        #print(self.__buffer_spc[:,1,3,20,0])
        #print(self.__buffer_spc[:,1,5,37,0])
        data_spc = numpy.sum(self.__buffer_spc,axis=0)

        if self.__buffer_cspc is not None:
            data_cspc = numpy.sum(self.__buffer_cspc,axis=0)
        else:
            data_cspc = None

        #print(numpy.shape(data_spc))
        #data_spc[1,4,20,0]=numpy.nan


        data_dc = self.__buffer_dc

        n = self.__profIndex

        self.__buffer_spc = None
        self.__buffer_cspc = None
        self.__buffer_dc = 0
        self.__profIndex = 0

        return data_spc, data_cspc, data_dc, n

    def pushData(self):
        """
        Return the sum of the last profiles and the profiles used in the sum.

        Affected:

        self.__profileIndex

        """
        bufferH=None
        buffer=None
        buffer1=None
        buffer_cspc=None
        self.__buffer_spc=numpy.array(self.__buffer_spc)
        if self.__buffer_cspc is not None:
            self.__buffer_cspc=numpy.array(self.__buffer_cspc)
        freq_dc = int(self.__buffer_spc.shape[2] / 2)
        #print("FREQ_DC",freq_dc)
        #print(self.__buffer_spc[:,1,5,37,0])
        #lag_array=[0,2,4,6,8,10,12,14,16,18,20]

        if self.nLags == 11:
            h0 = 7
        elif self.nLags == 16:
            h0 = 180



        '''
        import matplotlib.pyplot as plt
        #plt.plot(self.__buffer_spc[:,0,freq_dc-2,33,1],marker='*')
        aux = self.__buffer_spc[:,0,freq_dc-2,66,1]
        a,b=self.hildebrand_sekhon_Integration(numpy.abs(aux),1)
        print(a)
        plt.plot(sorted(aux),marker='*')
        plt.vlines(x=a,ymin=min(aux),ymax=max(aux))
        #plt.ylim((-35000,65000))
        plt.show()
        import time
        time.sleep(60)
        exit(1)
        '''

        #print(self.nLags)
        '''
        if self.nLags == 16:
            self.nLags = 3
            #exit(1)
        '''
        #print(self.nHeights)
        #exit(1)
        for l in range(self.nLags):#dataOut.DPL): #if DP --> nLags=11, elif HP --> nLags=16
            #breakFlag=False
            for k in range(7,self.nHeights):
                if self.__buffer_cspc is not None:
                    buffer_cspc=numpy.copy(self.__buffer_cspc[:,0,:,k,l])
                    outliers_IDs_cspc=[]
                    cspc_outliers_exist=False
                #indexmin_cspc=0
                for i in range(2): #Solo nos interesa los 2 primeros canales que son los canales con señal
                #for i in range(self.nChannels):#dataOut.nChannels):
                    #if self.TrueLags:
                        #print("HERE")
                    '''
                    if i==1 and k >= self.nHeights-2*l and self.TrueLags:
                        #breakFlag=True
                        print("here")
                        exit(1)
                        continue
                        '''

                        #pass
                    #else:
                    buffer1=numpy.copy(self.__buffer_spc[:,i,:,k,l])
                    indexes=[]
                    #sortIDs=[]
                    outliers_IDs=[]
                    for j in range(self.nProfiles):
                        if i==0 and j==freq_dc: #NOT CONSIDERING DC PROFILE AT CHANNEL 0
                            continue
                        if self.FlipChannelsExist:
                            if i==1 and j==0: #NOT CONSIDERING DC PROFILE AT CHANNEL 1
                                continue
                        else:
                            if i==1 and j==freq_dc: #NOT CONSIDERING DC PROFILE AT CHANNEL 1
                                continue
                        #buffer=buffer1[:,j]
                        buffer=(buffer1[:,j])
                        '''
                        if self.nLags ==16 and l!=0:
                            print(buffer)
                            exit(1)
                            '''
                        #index,sortID=self.hildebrand_sekhon_Integration(numpy.abs(buffer),1)
                        index=int(_HS_algorithm.HS_algorithm(numpy.sort(buffer, axis=None),1))
                        sortID = buffer.argsort()

                        indexes.append(index)
                        #sortIDs.append(sortID)
                        outliers_IDs=numpy.append(outliers_IDs,sortID[index:])

                        sortdata=numpy.sort(buffer,axis=0)
                        avg=numpy.mean(sortdata[:index],axis=0)
                        #lt=outliers_IDs
                        #avg=numpy.mean(buffer1[[t for t in range(buffer1.shape[0]) if t not in lt],:],axis=0)

                        if index != buffer.shape[0]:
                            for p in list(sortID[index:]):
                                buffer1[p,j]=avg


                        self.__buffer_spc[:,i,j,k,l]=numpy.copy(buffer1[:,j])
                        ###cspc IDs
                        #indexmin_cspc+=indexmin_cspc
                        if self.__buffer_cspc is not None:
                            outliers_IDs_cspc=numpy.append(outliers_IDs_cspc,outliers_IDs)

                #if not breakFlag:
                #print(outliers_IDs_cspc)
                if self.__buffer_cspc is not None:
                    outliers_IDs_cspc=outliers_IDs_cspc.astype(numpy.dtype('int64'))
                    if cspc_outliers_exist:
                        #sortdata=numpy.sort(buffer_cspc,axis=0)
                        #avg=numpy.mean(sortdata[:indexmin_cpsc,:],axis=0)
                        lt=outliers_IDs_cspc

                        avg=numpy.mean(buffer_cspc[[t for t in range(buffer_cspc.shape[0]) if t not in lt],:],axis=0)
                        for p in list(outliers_IDs_cspc):
                            buffer_cspc[p,:]=avg

                    self.__buffer_cspc[:,0,:,k,l]=numpy.copy(buffer_cspc)

                #else:
                    #break
        '''
        import matplotlib.pyplot as plt
        plt.plot(sorted(self.__buffer_spc[:,0,freq_dc-2,66,1]),marker='*')
        #plt.ylim((0,1.1*1.e6))
        plt.ylim((-30000,65000))
        plt.show()
        import time
        time.sleep(60)
        exit(1)
        '''

        buffer=None
        bufferH=None
        buffer1=None
        buffer_cspc=None

        #print("cpsc",self.__buffer_cspc[:,0,0,0,0])
        #print(self.__profIndex)
        #exit()
        '''
        if self.nLags == 16:
            print(self.__buffer_spc[:,0,0,0,2])
            exit(1)
            '''

        buffer=None
        #print(self.__buffer_spc[:,1,3,20,0])
        #print(self.__buffer_spc[:,1,5,37,0])
        data_spc = numpy.sum(self.__buffer_spc,axis=0)

        if self.__buffer_cspc is not None:
            data_cspc = numpy.sum(self.__buffer_cspc,axis=0)
        else:
            data_cspc = None

        #print(numpy.shape(data_spc))
        #data_spc[1,4,20,0]=numpy.nan


        data_dc = self.__buffer_dc

        n = self.__profIndex

        self.__buffer_spc = None
        self.__buffer_cspc = None
        self.__buffer_dc = 0
        self.__profIndex = 0

        return data_spc, data_cspc, data_dc, n

    def byProfiles(self, data_spc, data_cspc, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(data_spc, data_cspc, *args)

        if self.__profIndex == self.n:

            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def byTime(self, datatime, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(*args)

        if (datatime - self.__initime) >= self.__integrationtime:
            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def integrate(self, datatime, data_spc, data_cspc, *args):

        if self.__profIndex == 0:
            self.__initime = datatime
            #print(data_cspc.shape)

            #self.__buffer_spc = numpy.empty_like(data_spc,shape=(self.n,self.nChannels,self.nProfiles,self.nHeights,self.nLags))
            self.__buffer_spc = numpy.ones_like(data_spc,shape=(self.n,self.nChannels,self.nProfiles,self.nHeights,self.nLags))*numpy.NAN

            #print(self.__buffer_spc[0])
            #print(self.__buffer_spc.dtype)
            #print(data_spc.dtype)

            if data_cspc is not None:
                nLags = numpy.shape(data_cspc)[-1]
                nCrossChannels = numpy.shape(data_cspc)[0]

            #self.__buffer_cspc = numpy.empty_like(data_cspc,shape=(self.n,crossChannels,self.nProfiles,self.nHeights,self.nLags))
                self.__buffer_cspc = numpy.ones_like(data_cspc,shape=(self.n,nCrossChannels,self.nProfiles,self.nHeights,nLags))*numpy.NAN
            else:
                self.__buffer_cspc = None
            #print("HEREEEE")
            #print(self.__buffer_cspc.dtype)
            #print(data_cspc.dtype)
            #exit(1)
        if self.__byTime:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byTime(
                datatime, *args)
        else:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byProfiles(data_spc, data_cspc, *args)

        if not self.__dataReady:
            return None, None, None, None

        return self.__initime, avgdata_spc, avgdata_cspc, avgdata_dc

    def run(self, dataOut, n=None, timeInterval=None, overlapping=False,TrueLags=True):
        if n == 1:
            return dataOut

        dataOut.flagNoData = True

        if not self.isConfig:
            self.setup(n, timeInterval, overlapping)
            try:
                dataOut.FlipChannels
                self.FlipChannelsExist=1
            except:
                self.FlipChannelsExist=0
            self.isConfig = True

        self.nProfiles=dataOut.nProfiles
        self.nChannels=dataOut.nChannels
        self.nHeights=dataOut.nHeights
        if not dataOut.ByLags:
            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.data_spc,
                                                                                dataOut.data_cspc,
                                                                                dataOut.data_dc)
        else:
            #self.nProfiles=dataOut.nProfiles
            #self.nChannels=dataOut.nChannels
            #self.nHeights=dataOut.nHeights
            self.nLags=dataOut.nLags
            self.TrueLags=TrueLags



            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.dataLag_spc,
                                                                                dataOut.dataLag_cspc,
                                                                                dataOut.dataLag_dc)

        if self.__dataReady:

            if not dataOut.ByLags:
                dataOut.data_spc = avgdata_spc
                dataOut.data_cspc = avgdata_cspc
                dataOut.data_dc = avgdata_dc
            else:
                dataOut.dataLag_spc = avgdata_spc
                dataOut.dataLag_cspc = avgdata_cspc
                dataOut.dataLag_dc = avgdata_dc

                dataOut.data_spc=dataOut.dataLag_spc[:,:,:,dataOut.LagPlot].real

                if self.__buffer_cspc is not None:
                    dataOut.data_cspc=dataOut.dataLag_cspc[:,:,:,dataOut.LagPlot]
                dataOut.data_dc=dataOut.dataLag_dc[:,:,dataOut.LagPlot]


            dataOut.nIncohInt *= self.n
            dataOut.utctime = avgdatatime
            dataOut.flagNoData = False

        return dataOut

class HybridSelectSpectra(Operation):
    '''
    Written by R. Flores
    '''
    """Operation to rearange and use selected channels of spectra data and pairs of cross-spectra data for Hybrid Experiment.

    Parameters:
    -----------
    spc_channs : list
        Selected channels.

    Example
    --------

    op = proc_unit.addOperation(name='SelectSpectra', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None

    def select_spc(self,spc,spc_channs):

        buffer = spc[spc_channs]

        return buffer


    def run(self,dataOut,spc_channs=None,cspc_pairs=None):
        #print("HERE")
        if spc_channs != None:
            channelIndexList = []
            for channel in spc_channs:
                if channel not in dataOut.channelList:
                    raise ValueError("Channel %d is not in %s" %(channel, str(dataOut.channelList)))

                index = dataOut.channelList.index(channel)
                channelIndexList.append(index)
            #print(dataOut.dataLag_spc.shape)
            dataOut.dataLag_spc = self.select_spc(dataOut.dataLag_spc,channelIndexList)
            aux = dataOut.nChannels
            dataOut.channelList = range(dataOut.nLags)
            dataOut.nLags = aux
            #dataOut.nLags = len(spc_channs)
            dataOut.dataLag_spc = numpy.transpose(dataOut.dataLag_spc,(3,1,2,0))
            #print(dataOut.dataLag_spc.shape)
            #exit(1)
        dataOut.dataLag_cspc = numpy.transpose(dataOut.dataLag_cspc,(3,1,2,0))

        dataOut.dataLag_spc = numpy.concatenate((dataOut.dataLag_spc,dataOut.dataLag_cspc),axis=-1)
        dataOut.dataLag_cspc = None

        dataOut.data_spc = dataOut.dataLag_spc[0].real
        #print(dataOut.getNoise())
        #print(dataOut.data_spc)
        #exit(1)
        dataOut.data_cspc = None

        return dataOut

class IncohIntLag(Operation):
    '''
    Written by R. Flores
    '''
    __profIndex = 0
    __withOverapping = False

    __byTime = False
    __initime = None
    __lastdatatime = None
    __integrationtime = None

    __buffer_spc = None
    __buffer_cspc = None
    __buffer_dc = None

    __dataReady = False

    __timeInterval = None

    n = None

    def __init__(self):

        Operation.__init__(self)

    def setup(self, n=None, timeInterval=None, overlapping=False):
        """
        Set the parameters of the integration class.

        Inputs:

            n        :    Number of coherent integrations
            timeInterval   :    Time of integration. If the parameter "n" is selected this one does not work
            overlapping    :

        """

        self.__initime = None
        self.__lastdatatime = 0

        self.__buffer_spc = 0
        self.__buffer_cspc = 0
        self.__buffer_dc = 0

        self.__profIndex = 0
        self.__dataReady = False
        self.__byTime = False

        if n is None and timeInterval is None:
            raise ValueError("n or timeInterval should be specified ...")

        if n is not None:
            self.n = int(n)
        else:

            self.__integrationtime = int(timeInterval)
            self.n = None
            self.__byTime = True

    def putData(self, data_spc, data_cspc, data_dc):
        """
        Add a profile to the __buffer_spc and increase in one the __profileIndex

        """

        self.__buffer_spc += data_spc

        if data_cspc is None:
            self.__buffer_cspc = None
        else:
            self.__buffer_cspc += data_cspc

        if data_dc is None:
            self.__buffer_dc = None
        else:
            self.__buffer_dc += data_dc

        self.__profIndex += 1

        return

    def pushData(self):
        """
        Return the sum of the last profiles and the profiles used in the sum.

        Affected:

        self.__profileIndex

        """

        data_spc = self.__buffer_spc
        data_cspc = self.__buffer_cspc
        data_dc = self.__buffer_dc
        n = self.__profIndex

        self.__buffer_spc = 0
        self.__buffer_cspc = 0
        self.__buffer_dc = 0
        self.__profIndex = 0

        return data_spc, data_cspc, data_dc, n

    def byProfiles(self, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(*args)

        if self.__profIndex == self.n:

            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def byTime(self, datatime, *args):

        self.__dataReady = False
        avgdata_spc = None
        avgdata_cspc = None
        avgdata_dc = None

        self.putData(*args)

        if (datatime - self.__initime) >= self.__integrationtime:
            avgdata_spc, avgdata_cspc, avgdata_dc, n = self.pushData()
            self.n = n
            self.__dataReady = True

        return avgdata_spc, avgdata_cspc, avgdata_dc

    def integrate(self, datatime, *args):

        if self.__profIndex == 0:
            self.__initime = datatime

        if self.__byTime:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byTime(
                datatime, *args)
        else:
            avgdata_spc, avgdata_cspc, avgdata_dc = self.byProfiles(*args)

        if not self.__dataReady:
            return None, None, None, None

        return self.__initime, avgdata_spc, avgdata_cspc, avgdata_dc

    def run(self, dataOut, n=None, timeInterval=None, overlapping=False):
        if n == 1:
            return dataOut

        dataOut.flagNoData = True
        #print("incohint")
        #print("IncohInt",dataOut.data_spc.shape)
        #print("IncohInt",dataOut.data_cspc.shape)
        if not self.isConfig:
            self.setup(n, timeInterval, overlapping)
            self.isConfig = True

        if not dataOut.ByLags:
            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.data_spc,
                                                                                dataOut.data_cspc,
                                                                                dataOut.data_dc)
        else:
            '''
            print(numpy.sum(dataOut.dataLag_cspc[0,:,20,0].real)/32)
            print(numpy.sum(dataOut.dataLag_cspc[0,:,20,0].imag)/32)
            exit(1)
            '''
            avgdatatime, avgdata_spc, avgdata_cspc, avgdata_dc = self.integrate(dataOut.utctime,
                                                                                dataOut.dataLag_spc,
                                                                                dataOut.dataLag_cspc,
                                                                                dataOut.dataLag_dc)
        #print("Incoh Int: ",self.__profIndex,n)
        if self.__dataReady:

            if not dataOut.ByLags:
                dataOut.data_spc = avgdata_spc
                dataOut.data_cspc = avgdata_cspc
                dataOut.data_dc = avgdata_dc
            else:
                dataOut.dataLag_spc = avgdata_spc
                dataOut.dataLag_cspc = avgdata_cspc
                dataOut.dataLag_dc = avgdata_dc

                #print(dataOut.LagPlot)
                #print(dataOut.dataLag_spc[1,:,100,2])
                #print(numpy.sum(dataOut.dataLag_spc[1,:,100,2]))
                #exit(1)

                #print("INCOH INT DONE")
                #exit(1)
                '''
                print(numpy.sum(dataOut.dataLag_spc[0,:,20,10])/32)
                print(numpy.sum(dataOut.dataLag_spc[1,:,20,10])/32)
                #exit(1)
                '''
                '''
                print(numpy.sum(dataOut.dataLag_cspc[0,:,20,0].real)/32)
                print(numpy.sum(dataOut.dataLag_cspc[0,:,20,0].imag)/32)
                exit(1)
                '''
                dataOut.data_spc=dataOut.dataLag_spc[:,:,:,dataOut.LagPlot].real#*numpy.NaN

                #print("done")
                #print(dataOut.dataLag_spc[0,0,0,2])
                if dataOut.dataLag_cspc is not None:
                    dataOut.data_cspc=dataOut.dataLag_cspc[:,:,:,dataOut.LagPlot]
                dataOut.data_dc=dataOut.dataLag_dc[:,:,dataOut.LagPlot]


            dataOut.nIncohInt *= self.n
            dataOut.utctime = avgdatatime
            dataOut.flagNoData = False


        #print("done")
        #print(dataOut.data_spc[0,0,0])
        #print("ut",dataOut.ut)
        return dataOut

class SnrFaraday(Operation):
    '''
    Written by R. Flores
    '''
    """Operation to use get SNR in Faraday processing.

    Parameters:
    -----------

    Example
    --------

    op = proc_unit.addOperation(name='SnrFaraday', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

    def run(self,dataOut):

        noise = dataOut.getNoise()
        maxdB = 16

        #dataOut.data_snr = (dataOut.data_spc.sum(axis=1)-noise[:,None])/(noise[:,None]*dataOut.normFactor)
        print("normFactor: ",dataOut.normFactor)
        print("nFFTPoints: ",dataOut.nFFTPoints)
        normFactor = 24
        print("Power: ",dataOut.data_spc.sum(axis=1)/dataOut.nFFTPoints)
        print("Noise: ",noise)
        print("Power dB: ",10*numpy.log10(dataOut.data_spc.sum(axis=1)/dataOut.nFFTPoints))
        print("Noise dB: ",10*numpy.log10(noise))
        #dataOut.data_snr = (dataOut.data_spc.sum(axis=1))/(noise[:,None]*dataOut.normFactor)
        dataOut.data_snr = (dataOut.data_spc.sum(axis=1))/(noise[:,None]*dataOut.nFFTPoints)
        snr_dB = 10*numpy.log10(dataOut.data_snr)
        print("Snr: ",snr_dB)
        '''
        for nch in range(dataOut.data_snr.shape[0]):
            for i in range(dataOut.data_snr.shape[1]):
                if snr_dB[nch,i] > maxdB:
                    dataOut.data_spc[nch,:,i] = numpy.nan
                    dataOut.data_snr[nch,i] = numpy.nan
                    '''

        return dataOut

class SpectraDataToFaraday(Operation): #ISR MODE
    '''
    Written by R. Flores
    '''
    """Operation to use spectra data in Faraday processing.

    Parameters:
    -----------
    nint : int
        Number of integrations.

    Example
    --------

    op = proc_unit.addOperation(name='SpectraDataToFaraday', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None



    def ConvertData(self,dataOut):

        dataOut.TimeBlockSeconds_for_dp_power=dataOut.utctime
        dataOut.bd_time=gmtime(dataOut.TimeBlockSeconds_for_dp_power)
        dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
        dataOut.ut_Faraday=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0

        '''
        tmpx=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_a2=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_b2=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_abr=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_abi=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        '''
        #print("DPL",dataOut.DPL)
        #print("NDP",dataOut.NDP)
        tmpx=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
        tmpx_a2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
        tmpx_b2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
        tmpx_abr=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
        tmpx_abi=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
        dataOut.kabxys_integrated=[tmpx,tmpx,tmpx,tmpx,tmpx_a2,tmpx,tmpx_b2,tmpx,tmpx_abr,tmpx,tmpx_abi,tmpx,tmpx,tmpx]
        '''
        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        for l in range(dataOut.DPL):
            if(l==0 or (l>=3 and l <=6)):
                dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)
            else:
                dataOut.rnint2[l]=2*(1.0/(dataOut.nIncohInt*dataOut.nProfiles))
                '''
        #for l in range(dataOut.DPL):
            #dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)#*dataOut.nProfiles


        self.dataLag_spc=(dataOut.dataLag_spc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)
        self.dataLag_cspc=(dataOut.dataLag_cspc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)


        '''
        self.dataLag_spc=(dataOut.dataLag_spc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)
        self.dataLag_cspc=(dataOut.dataLag_cspc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)
        #self.dataLag_dc=dataOut.dataLag_dc.sum(axis=1)/dataOut.rnint2[0]
        '''

        dataOut.kabxys_integrated[4][:,:,0]=self.dataLag_spc[0,:,:].real
        #dataOut.kabxys_integrated[5][:,:,0]+=self.dataLag_spc[0,:,:].imag
        dataOut.kabxys_integrated[6][:,:,0]=self.dataLag_spc[1,:,:].real
        #dataOut.kabxys_integrated[7][:,:,0]+=self.dataLag_spc[1,:,:].imag

        dataOut.kabxys_integrated[8][:,:,0]=self.dataLag_cspc[0,:,:].real
        dataOut.kabxys_integrated[10][:,:,0]=self.dataLag_cspc[0,:,:].imag

        '''
        print(dataOut.kabxys_integrated[4][:,0,0])
        print(dataOut.kabxys_integrated[6][:,0,0])
        print("times 12")
        print(dataOut.kabxys_integrated[4][:,0,0]*dataOut.nProfiles)
        print(dataOut.kabxys_integrated[6][:,0,0]*dataOut.nProfiles)
        print(dataOut.rnint2[0])
        input()
        '''

    def normFactor(self,dataOut):
        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        for l in range(dataOut.DPL):
            dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)

    def noise(self,dataOut):

        dataOut.noise_lag = numpy.zeros((dataOut.nChannels,dataOut.DPL),'float32')
        #print("Lags")
        '''
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=46)
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
            '''
        #print(dataOut.NDP)
        #exit(1)
        #Channel B
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            max_hei_id = dataOut.NDP - 2*lag
            #if lag < 6:
            dataOut.noise_lag[1,lag] = dataOut.getNoise(ymin_index=53,ymax_index=max_hei_id)[1]
            #else:
                #dataOut.noise_lag[1,lag] = numpy.mean(dataOut.noise_lag[1,:6])
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
        #Channel A
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[0,lag] = dataOut.getNoise(ymin_index=53)[0]

        nanindex = numpy.argwhere(numpy.isnan(numpy.log10(dataOut.noise_lag[1,:])))
        i1 = nanindex[0][0]
        dataOut.noise_lag[1,i1:] = numpy.mean(dataOut.noise_lag[1,:i1]) #El ruido de lags contaminados se
                                                                        #determina a partir del promedio del
                                                                        #ruido de los lags limpios
        '''
        dataOut.noise_lag[1,:] = dataOut.noise_lag[1,0] #El ruido de los lags diferentes de cero para
                                                        #el canal B es contaminado por el Tx y EEJ
                                                        #del siguiente perfil, por ello se asigna el ruido
                                                        #del lag 0 a todos los lags
                                                        '''
        #print("Noise lag: ", 10*numpy.log10(dataOut.noise_lag/dataOut.normFactor))
        #exit(1)
        '''
        dataOut.tnoise = dataOut.getNoise(ymin_index=46)
        dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]
        dataOut.pbn = dataOut.tnoise[1]
        '''

        dataOut.tnoise = dataOut.noise_lag/float(dataOut.nProfiles*dataOut.nIncohInt)
        #dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]
        dataOut.pbn = dataOut.tnoise[1]

    def get_eej_index_V0(self,data_to_remov_eej,dataOut):

        dataOut.data_spc = data_to_remov_eej
        #print(dataOut.data_spc)
        data_eej = dataOut.getPower()[1]
        print("data_eej: ", data_eej)
        #exit(1)
        index_eej = CleanCohEchoes.mad_based_outlier(self,data_eej[:20])
        aux_eej = numpy.array(index_eej.nonzero()).ravel()

        index2 = CleanCohEchoes.mad_based_outlier(self,data_eej[aux_eej[-1]+1:aux_eej[-1]+1+20])
        aux2 = numpy.array(index2.nonzero()).ravel()
        if aux2.size > 0:
          #print(aux2)
          #print(aux2[-1])
          #print(arr[aux[-1]+aux2[-1]+1])
            dataOut.min_id_eej = aux_eej[-1]+aux2[-1]+1
        else:
            dataOut.min_id_eej = aux_eej[-1]


        print(dataOut.min_id_eej)
        exit(1)

    def get_eej_index_V1(self,data_to_remov_eej,dataOut):

        dataOut.data_spc = data_to_remov_eej
        outliers_IDs = []
        #print(dataOut.data_spc)
        for ich in range(dataOut.nChannels):

            data_eej = dataOut.getPower()[ich]
            #print("data_eej: ", data_eej)
            #exit(1)
            index_eej = CleanCohEchoes.mad_based_outlier(self,data_eej[:20])
            aux_eej = numpy.array(index_eej.nonzero()).ravel()

            #index2 = CleanCohEchoes.mad_based_outlier(self,data_eej[aux_eej[-1]+1:aux_eej[-1]+1+20])
            index2 = CleanCohEchoes.mad_based_outlier(self,data_eej[aux_eej[-1]+1:aux_eej[-1]+1+10],thresh=1.)
            aux2 = numpy.array(index2.nonzero()).ravel()
            if aux2.size > 0:
                #min_id_eej = aux_eej[-1]+aux2[-1]+1
                ids = numpy.concatenate((aux_eej,aux2+aux_eej[-1]+1))
            else:
                ids = aux_eej

            outliers_IDs=numpy.append(outliers_IDs,ids)

        outliers_IDs=numpy.array(outliers_IDs)
        outliers_IDs=outliers_IDs.astype(numpy.dtype('int64'))

        (uniq, freq) = (numpy.unique(outliers_IDs, return_counts=True))
        aux_arr = numpy.column_stack((uniq,freq))

        final_index = []
        for i in range(aux_arr.shape[0]):
            if aux_arr[i,1] == 2:
                final_index.append(aux_arr[i,0])

        if final_index != []:
            dataOut.min_id_eej = final_index[-1]
        else:
            print("CHECKKKKK!!!!!!!!!!!!!!!")

        print(dataOut.min_id_eej)
        exit(1)

    def get_eej_index(self,data_to_remov_eej,dataOut):

        dataOut.data_spc = data_to_remov_eej

        data_eej = dataOut.getPower()[0]
        #print(data_eej)
        index_eej = CleanCohEchoes.mad_based_outlier(self,data_eej[:17])
        aux_eej = numpy.array(index_eej.nonzero()).ravel()
        print("aux_eej: ", aux_eej)
        if aux_eej != []:
            dataOut.min_id_eej = aux_eej[-1]
        else:
            dataOut.min_id_eej = 12


        #print("min_id_eej: ", dataOut.min_id_eej)
        #exit(1)

    def run(self,dataOut):
        #print(dataOut.nIncohInt)
        #exit(1)
        dataOut.paramInterval=dataOut.nIncohInt*2*2#nIncohInt*numero de fft/nprofiles*segundos de cada muestra
        dataOut.lat=-11.95
        dataOut.lon=-76.87

        data_to_remov_eej = dataOut.dataLag_spc[:,:,:,0]

        self.normFactor(dataOut)
        #print(dataOut.NDP)
        dataOut.NDP=dataOut.nHeights
        dataOut.NR=len(dataOut.channelList)
        dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        dataOut.H0=int(dataOut.heightList[0])

        self.ConvertData(dataOut)
        #print(dataOut.NDP)
        #exit(1)
        dataOut.NAVG=16#dataOut.rnint2[0] #CHECK THIS!
        if hasattr(dataOut, 'NRANGE'):
            dataOut.MAXNRANGENDT = max(dataOut.NRANGE,dataOut.NDT)
        else:
            dataOut.MAXNRANGENDT = dataOut.NDP


        #if hasattr(dataOut, 'HP'):
            #pass
        #else:
        self.noise(dataOut)

        '''
        if not hasattr(dataOut, 'tnoise'):
            print("noise")
            self.noise(dataOut)
        else:
            delattr(dataOut, 'tnoise')
            '''
        #dataOut.pan = numpy.mean(dataOut.pan)
        #dataOut.pbn = numpy.mean(dataOut.pbn)
        #print(dataOut.pan)
        #print(dataOut.pbn)
        #exit(1)

        #print("Noise: ",dataOut.tnoise)
        #print("Noise dB: ",10*numpy.log10(dataOut.tnoise))
        #exit(1)
        #dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        if gmtime(dataOut.utctime).tm_hour >= 21. or gmtime(dataOut.utctime).tm_hour < 13.:
            self.get_eej_index(data_to_remov_eej,dataOut)
        print("done")
        #exit(1)
        return dataOut

class SpectraDataToFaraday_MST(Operation): #MST MODE
    """Operation to use spectra data in Faraday processing.

    Parameters:
    -----------
    nint : int
        Number of integrations.

    Example
    --------

    op = proc_unit.addOperation(name='SpectraDataToFaraday', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None

    def noise_original(self,dataOut):

        dataOut.noise_lag = numpy.zeros((dataOut.nChannels,dataOut.DPL),'float32')
        #print("Lags")
        '''
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=46)
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
            '''
        #print(dataOut.NDP)
        #exit(1)
        #Channel B
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            max_hei_id = dataOut.NDP - 2*lag
            #if lag < 6:
            dataOut.noise_lag[1,lag] = dataOut.getNoise(ymin_index=53,ymax_index=max_hei_id)[1]
            #else:
                #dataOut.noise_lag[1,lag] = numpy.mean(dataOut.noise_lag[1,:6])
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
        #Channel A
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[0,lag] = dataOut.getNoise(ymin_index=53)[0]

        nanindex = numpy.argwhere(numpy.isnan(numpy.log10(dataOut.noise_lag[1,:])))
        i1 = nanindex[0][0]
        dataOut.noise_lag[1,i1:] = numpy.mean(dataOut.noise_lag[1,:i1]) #El ruido de lags contaminados se
                                                                        #determina a partir del promedio del
                                                                        #ruido de los lags limpios
        '''
        dataOut.noise_lag[1,:] = dataOut.noise_lag[1,0] #El ruido de los lags diferentes de cero para
                                                        #el canal B es contaminado por el Tx y EEJ
                                                        #del siguiente perfil, por ello se asigna el ruido
                                                        #del lag 0 a todos los lags
                                                        '''
        #print("Noise lag: ", 10*numpy.log10(dataOut.noise_lag/dataOut.normFactor))
        #exit(1)
        '''
        dataOut.tnoise = dataOut.getNoise(ymin_index=46)
        dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]
        dataOut.pbn = dataOut.tnoise[1]
        '''

        dataOut.tnoise = dataOut.noise_lag/float(dataOut.nProfiles*dataOut.nIncohInt)
        #dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]
        dataOut.pbn = dataOut.tnoise[1]

    def noise(self,dataOut,minIndex,maxIndex):

        dataOut.noise_lag = numpy.zeros((dataOut.nChannels),'float32')
        #print("Lags")
        '''
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=46)
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
            '''
        #print(dataOut.NDP)
        #exit(1)
        #Channel B

            #print(lag)
        dataOut.data_spc = dataOut.dataLag_spc[:,:,:,0]
        max_hei_id = dataOut.NDP - 2*0
        #if lag < 6:
        #dataOut.noise_lag[1] = dataOut.getNoise(ymin_index=80,ymax_index=106)[1]
        if dataOut.flagDecodeData:
            #dataOut.noise_lag[1] = dataOut.getNoise(ymin_index=150,ymax_index=200)[1]
            dataOut.noise_lag[1] = dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[1]
        else:
            dataOut.noise_lag[1] = dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[1]
            #else:
                #dataOut.noise_lag[1,lag] = numpy.mean(dataOut.noise_lag[1,:6])
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
        #Channel A

            #print(lag)
        dataOut.data_spc = dataOut.dataLag_spc[:,:,:,0]
        if dataOut.flagDecodeData:
            #dataOut.noise_lag[0] = dataOut.getNoise(ymin_index=150,ymax_index=200)[0]
            dataOut.noise_lag[0] = dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[0]
        else:
            dataOut.noise_lag[0] = dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[0]

        dataOut.tnoise = dataOut.noise_lag/float(dataOut.nProfiles*dataOut.nIncohInt)
        #dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]#*.98
        dataOut.pbn = dataOut.tnoise[1]#*.98

    def ConvertData(self,dataOut):

        dataOut.TimeBlockSeconds_for_dp_power=dataOut.utctime
        dataOut.bd_time=gmtime(dataOut.TimeBlockSeconds_for_dp_power)
        dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
        dataOut.ut_Faraday=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0


        tmpx=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_a2=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_b2=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_abr=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        tmpx_abi=numpy.zeros((dataOut.nHeights,dataOut.DPL,2),'float32')
        dataOut.kabxys_integrated=[tmpx,tmpx,tmpx,tmpx,tmpx_a2,tmpx,tmpx_b2,tmpx,tmpx_abr,tmpx,tmpx_abi,tmpx,tmpx,tmpx]

        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        for l in range(dataOut.DPL):
            dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)#*dataOut.nProfiles

        #try:
            #dataOut.rint2 /= dataOut.nCohInt*dataOut.windowOfFilter
        #except: pass
        '''
        if hasattr(dataOut,'flagDecodeData'):
            if dataOut.flagDecodeData:
                print("decode",numpy.sum(dataOut.code[0]**2))
                dataOut.rnint2 /= numpy.sum(dataOut.code[0]**2)
            else:
                print("widnow")
                dataOut.rnint2 /= dataOut.windowOfFilter
        else:
            print("widnow")
            dataOut.rint2 = dataOut.windowOfFilter
            '''
        self.dataLag_spc=(dataOut.dataLag_spc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)
        self.dataLag_cspc=(dataOut.dataLag_cspc.sum(axis=1))*(dataOut.rnint2[0]/dataOut.nProfiles)
        #self.dataLag_dc=dataOut.dataLag_dc.sum(axis=1)/dataOut.rnint2[0]

        dataOut.kabxys_integrated[4][:,:,0]=self.dataLag_spc[0,:,:].real
        #dataOut.kabxys_integrated[5][:,:,0]+=self.dataLag_spc[0,:,:].imag
        dataOut.kabxys_integrated[6][:,:,0]=self.dataLag_spc[1,:,:].real
        #dataOut.kabxys_integrated[7][:,:,0]+=self.dataLag_spc[1,:,:].imag

        dataOut.kabxys_integrated[8][:,:,0]=self.dataLag_cspc[0,:,:].real
        dataOut.kabxys_integrated[10][:,:,0]=self.dataLag_cspc[0,:,:].imag

        #print("power: ", numpy.sum(dataOut.kabxys_integrated[4][:16,0,0]))
        #print("power: ", numpy.sum(dataOut.kabxys_integrated[4][16:32,0,0]))
        #exit(1)
        '''
        print(dataOut.kabxys_integrated[4][:,0,0])
        print(dataOut.kabxys_integrated[6][:,0,0])
        print("times 12")
        print(dataOut.kabxys_integrated[4][:,0,0]*dataOut.nProfiles)
        print(dataOut.kabxys_integrated[6][:,0,0]*dataOut.nProfiles)
        print(dataOut.rnint2[0])
        input()
        '''

    def run(self,dataOut,ymin_noise = None,ymax_noise = None):

        dataOut.paramInterval=0#int(dataOut.nint*dataOut.header[7][0]*2 )
        dataOut.lat=-11.95
        dataOut.lon=-76.87

        dataOut.NDP=dataOut.nHeights
        dataOut.NR=len(dataOut.channelList)
        dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        dataOut.H0=int(dataOut.heightList[0])
        '''
        if dataOut.flagDecodeData:
            print("flagDecodeData")
            dataOut.data_spc /= numpy.sum(dataOut.code[0]**2)
            dataOut.data_cspc /= numpy.sum(dataOut.code[0]**2)
            dataOut.data_spc /= numpy.sum(dataOut.code[0]**2)
            dataOut.data_cspc /= numpy.sum(dataOut.code[0]**2)
        else:
            print("windowOfFilter")
            dataOut.data_spc /= dataOut.windowOfFilter
            dataOut.data_cspc /= dataOut.windowOfFilter
            dataOut.data_spc /= dataOut.windowOfFilter
            dataOut.data_cspc /= dataOut.windowOfFilter
            '''
        #print("dataOut.data_spc.shape: ", dataOut.data_spc.shape)
        #print("dataOut.data_cspc.shape: ", dataOut.data_cspc.shape)
        #print("*****************Sum: ", numpy.sum(dataOut.data_spc[0]))
        #print("*******************normFactor: *******************", dataOut.normFactor)
        dataOut.dataLag_spc = numpy.stack((dataOut.data_spc, dataOut.data_spc), axis=-1)
        dataOut.dataLag_cspc = numpy.stack((dataOut.data_cspc, dataOut.data_cspc), axis=-1)
        #print(dataOut.dataLag_spc.shape)
        dataOut.DPL = numpy.shape(dataOut.dataLag_spc)[-1]

        #exit(1)
        self.ConvertData(dataOut)

        inda = numpy.where(dataOut.heightList >= ymin_noise)
        indb = numpy.where(dataOut.heightList <= ymax_noise)

        minIndex = inda[0][0]
        maxIndex = indb[0][-1]

        #print("ymin_noise: ", dataOut.heightList[minIndex])
        #print("ymax_noise: ", dataOut.heightList[maxIndex])

        self.noise(dataOut,minIndex,maxIndex)
        dataOut.NAVG=16#dataOut.rnint2[0] #CHECK THIS!
        dataOut.MAXNRANGENDT=dataOut.NDP
        #'''
        if 0:
            #print(dataOut.kabxys_integrated[4][:,0,0])
            #print("dataOut.heightList: ", dataOut.heightList)
            #print("dataOut.pbn: ", dataOut.pbn)
            print("INSIDE")
            import matplotlib.pyplot as plt
            #print("dataOut.getPower(): ", dataOut.getPower())
            plt.plot(10*numpy.log10(dataOut.kabxys_integrated[4][:,0,0]),dataOut.heightList)
            #plt.plot(10**((dataOut.getPower()[1])/10),dataOut.heightList)
            #plt.plot(dataOut.getPower()[0],dataOut.heightList)
            #plt.plot(dataOut.dataLag_spc[:,:,:,0],dataOut.heightList)
            plt.axvline(10*numpy.log10(dataOut.pan))
            #print(dataOut.nProfiles)
            #plt.axvline(10*numpy.log10(1*dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[0]/dataOut.normFactor))
            #print("10*numpy.log10(dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[1]/dataOut.normFactor): ", 10*numpy.log10(dataOut.getNoise(ymin_index=minIndex,ymax_index=maxIndex)[1]/dataOut.normFactor))
            #plt.xlim(1,25000)
            #plt.xlim(15,20)
            #plt.ylim(30,90)
            plt.grid()
            plt.show()
            #'''
        dataOut.DPL = 1
        return dataOut

class SpectraDataToHybrid(SpectraDataToFaraday):
    '''
    Written by R. Flores
    '''
    """Operation to use spectra data in Faraday processing.

    Parameters:
    -----------
    nint : int
        Number of integrations.

    Example
    --------

    op = proc_unit.addOperation(name='SpectraDataToFaraday', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None
        self.dataLag_spc_LP=None
        self.dataLag_cspc_LP=None
        self.dataLag_dc_LP=None

    def noise(self,dataOut):
        '''
        for i in range(dataOut.NR):
            dataOut.pnoise[i]=0.0
            for k in range(dataOut.DPL):
                dataOut.pnoise[i]+= dataOut.getNoise()
                '''
        #print(dataOut.dataLag_spc_LP[:,:,:,0])
        dataOut.data_spc = dataOut.dataLag_spc_LP[:,:,:,0].real
        dataOut.tnoise = dataOut.getNoise()
        #print(dataOut.tnoise)
        #exit(1)
        dataOut.tnoise[0]*=0.995#0.976
        dataOut.tnoise[1]*=0.995
        #print(dataOut.nProfiles)
        dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        #print("pan: ",dataOut.pan)
        #print("pbn: ",dataOut.pbn)
        #print(numpy.shape(dataOut.pnoise))
        #exit(1)



    def ConvertDataLP_V0(self,dataOut):

        #print(dataOut.dataLag_spc[:,:,:,1]/dataOut.data_spc)
        #exit(1)
        normfactor=1.0/(dataOut.nIncohInt_LP*dataOut.nProfiles_LP)#*dataOut.nProfiles

        buffer = self.dataLag_spc_LP=(dataOut.dataLag_spc_LP.sum(axis=1))*(1./dataOut.nProfiles_LP)
        ##self.dataLag_cspc_LP=(dataOut.dataLag_cspc_LP.sum(axis=1))*(1./dataOut.nProfiles_LP)
        #self.dataLag_dc=dataOut.dataLag_dc.sum(axis=1)/dataOut.rnint2[0]
        #aux=numpy.expand_dims(self.dataLag_spc_LP, axis=2)
        #print(aux.shape)
        ##buffer = numpy.concatenate((numpy.expand_dims(self.dataLag_spc_LP, axis=2),self.dataLag_cspc_LP),axis=2)
        dataOut.output_LP_integrated = numpy.transpose(buffer,(2,1,0))
        #print("lP",numpy.shape(dataOut.output_LP_integrated))
        #exit(1)
        #print(numpy.shape(dataOut.output_LP_integrated))

        #exit(1)

    def ConvertDataLP(self,dataOut):

        #print(dataOut.dataLag_spc[:,:,:,1]/dataOut.data_spc)
        #exit(1)
        normfactor=1.0/(dataOut.nIncohInt_LP*dataOut.nProfiles_LP)#*dataOut.nProfiles

        buffer = self.dataLag_spc_LP=dataOut.dataLag_spc_LP
        ##self.dataLag_cspc_LP=(dataOut.dataLag_cspc_LP.sum(axis=1))*(1./dataOut.nProfiles_LP)
        #self.dataLag_dc=dataOut.dataLag_dc.sum(axis=1)/dataOut.rnint2[0]
        #aux=numpy.expand_dims(self.dataLag_spc_LP, axis=2)
        #print(aux.shape)
        ##buffer = numpy.concatenate((numpy.expand_dims(self.dataLag_spc_LP, axis=2),self.dataLag_cspc_LP),axis=2)
        dataOut.output_LP_integrated = numpy.transpose(buffer,(1,2,0))

    def normFactor(self,dataOut):
        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        for l in range(dataOut.DPL):
            if(l==0 or (l>=3 and l <=6)):
                dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)
            else:
                dataOut.rnint2[l]=2*(1.0/(dataOut.nIncohInt*dataOut.nProfiles))

    def run(self,dataOut):

        dataOut.paramInterval=0#int(dataOut.nint*dataOut.header[7][0]*2 )
        dataOut.lat=-11.95
        dataOut.lon=-76.87

        dataOut.NDP=dataOut.nHeights
        dataOut.NR=len(dataOut.channelList)
        dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        dataOut.H0=int(dataOut.heightList[0])
        #print(numpy.shape(dataOut.dataLag_spc))
        #print("a",numpy.sum(dataOut.dataLag_spc[0,:,20,10]))
        #print(numpy.sum(dataOut.dataLag_spc[1,:,20,10]))
        self.normFactor(dataOut)

        self.ConvertDataLP(dataOut)

        dataOut.output_LP_integrated[:,:,3] *= float(dataOut.NSCAN/22)#(dataOut.nNoiseProfiles) #Corrects the zero padding

        dataOut.nis=dataOut.NSCAN*dataOut.nIncohInt_LP*10

        #print(dataOut.output_LP_integrated[0,30,1])
        #exit(1)

        self.ConvertData(dataOut)

        dataOut.kabxys_integrated[4][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[6][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[8][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[10][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        hei = 2
        '''
        for hei in range(67):
            print("hei",hei)
            print(dataOut.kabxys_integrated[8][hei,:,0])#+dataOut.kabxys_integrated[11][53,6,0])
            print(dataOut.kabxys_integrated[10][hei,:,0])#+dataOut.kabxys_integrated[11][53,9,0])

        exit(1)
        '''
        #print(dataOut.dataLag_spc_LP.shape)
        #exit(1)
        #[:,:,:,0]

        self.noise(dataOut)


        hei = 53
        lag = 0
        '''
        print("b",dataOut.kabxys_integrated[4][hei,lag,0])
        print(dataOut.kabxys_integrated[6][hei,lag,0])

        print("c",dataOut.kabxys_integrated[8][hei,lag,0])
        print(dataOut.kabxys_integrated[10][hei,lag,0])
        exit(1)
        '''
        #'''
        #print(dataOut.tnoise)
        #print(dataOut.pbn)
        #exit(1)
        #'''
        #'''

        #print(dataOut.pan)
        #print(dataOut.pbn)
        #print(dataOut.tnoise[0])
        #dataOut.pan = 143.91122436523438
        #dataOut.pbn = 249.5623575846354

        #dataOut.tnoise[0] = 8.8419056e+05


        #'''

        dataOut.NAVG=1#dataOut.rnint2[0] #CHECK THIS!
        dataOut.nint=dataOut.nIncohInt
        dataOut.MAXNRANGENDT=dataOut.NRANGE

        #exit(1)

        return dataOut

class SpectraDataToHybrid_V2(SpectraDataToFaraday):
    '''
    Written by R. Flores
    '''
    """Operation to use spectra data in Faraday processing.

    Parameters:
    -----------
    nint : int
        Number of integrations.

    Example
    --------

    op = proc_unit.addOperation(name='SpectraDataToFaraday', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None
        self.dataLag_spc_LP=None
        self.dataLag_cspc_LP=None
        self.dataLag_dc_LP=None

    def noise_v0(self,dataOut):

        dataOut.data_spc = dataOut.dataLag_spc_LP.real
        #print(dataOut.dataLag_spc.shape)
        #exit(1)
        #dataOut.data_spc = dataOut.dataLag_spc[:,:,:,0].real
        #print("spc noise shape: ",dataOut.data_spc.shape)
        dataOut.tnoise = dataOut.getNoise(ymin_index=100,ymax_index=166)
        #print("Noise LP: ",10*numpy.log10(dataOut.tnoise))
        #exit(1)
        #dataOut.tnoise[0]*=0.995#0.976
        #dataOut.tnoise[1]*=0.995
        #print(dataOut.nProfiles)
        #dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        #dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)
        dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)
        ##dataOut.pan=dataOut.tnoise[0]*float(self.normfactor_LP)
        ##dataOut.pbn=dataOut.tnoise[1]*float(self.normfactor_LP)
        #print("pan: ",10*numpy.log10(dataOut.pan))
        #print("pbn: ",dataOut.pbn)
        #print(numpy.shape(dataOut.pnoise))
        #exit(1)
        #print("pan: ",dataOut.pan)
        #print("pbn: ",dataOut.pbn)
        #exit(1)

    def noise_v0_aux(self,dataOut):

        dataOut.data_spc = dataOut.dataLag_spc
        #print(dataOut.dataLag_spc.shape)
        #exit(1)
        #dataOut.data_spc = dataOut.dataLag_spc[:,:,:,0].real
        #print("spc noise shape: ",dataOut.data_spc.shape)
        tnoise = dataOut.getNoise(ymin_index=100,ymax_index=166)
        #print("Noise LP: ",10*numpy.log10(dataOut.tnoise))
        #exit(1)
        #dataOut.tnoise[0]*=0.995#0.976
        #dataOut.tnoise[1]*=0.995
        #print(dataOut.nProfiles)
        #dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        #dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        dataOut.pan=tnoise[0]/float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pbn=tnoise[1]/float(dataOut.nProfiles*dataOut.nIncohInt)

    def noise(self,dataOut):

        dataOut.noise_lag = numpy.zeros((dataOut.nChannels,dataOut.DPL),'float32')
        #print("Lags")
        '''
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=46)
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
            '''
        #print(dataOut.NDP)
        #exit(1)
        #Channel B
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            max_hei_id = dataOut.NDP - 2*lag
            #if lag < 6:
            dataOut.noise_lag[1,lag] = dataOut.getNoise(ymin_index=53,ymax_index=max_hei_id)[1]
            #else:
                #dataOut.noise_lag[1,lag] = numpy.mean(dataOut.noise_lag[1,:6])
            #dataOut.noise_lag[:,lag] = dataOut.getNoise(ymin_index=33,ymax_index=46)
        #Channel A
        for lag in range(dataOut.DPL):
            #print(lag)
            dataOut.data_spc = dataOut.dataLag_spc[:,:,:,lag]
            dataOut.noise_lag[0,lag] = dataOut.getNoise(ymin_index=53)[0]

        nanindex = numpy.argwhere(numpy.isnan(numpy.log10(dataOut.noise_lag[1,:])))
        i1 = nanindex[0][0]
        dataOut.noise_lag[1,(1,2,7,8,9,10)] *= 2 #Correction LP
        dataOut.noise_lag[1,i1:] = numpy.mean(dataOut.noise_lag[1,:i1]) #El ruido de lags contaminados se
                                                                        #determina a partir del promedio del
                                                                        #ruido de los lags limpios
        '''
        dataOut.noise_lag[1,:] = dataOut.noise_lag[1,0] #El ruido de los lags diferentes de cero para
                                                        #el canal B es contaminado por el Tx y EEJ
                                                        #del siguiente perfil, por ello se asigna el ruido
                                                        #del lag 0 a todos los lags
                                                        '''
        #print("Noise lag: ", 10*numpy.log10(dataOut.noise_lag/dataOut.normFactor))
        #exit(1)
        '''
        dataOut.tnoise = dataOut.getNoise(ymin_index=46)
        dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = dataOut.tnoise[0]
        dataOut.pbn = dataOut.tnoise[1]
        '''
        #print("i1: ", i1)
        #exit(1)
        tnoise = dataOut.noise_lag/float(dataOut.nProfiles*dataOut.nIncohInt)
        #dataOut.tnoise /= float(dataOut.nProfiles*dataOut.nIncohInt)
        dataOut.pan = tnoise[0]
        dataOut.pbn = tnoise[1]

    def noise_LP(self,dataOut):

        dataOut.data_spc = dataOut.dataLag_spc_LP.real
        #print(dataOut.dataLag_spc.shape)
        #exit(1)
        #dataOut.data_spc = dataOut.dataLag_spc[:,:,:,0].real
        #print("spc noise shape: ",dataOut.data_spc.shape)
        dataOut.tnoise = dataOut.getNoise(ymin_index=100,ymax_index=166)
        #print("Noise LP: ",10*numpy.log10(dataOut.tnoise))
        #exit(1)
        #dataOut.tnoise[0]*=0.995#0.976
        #dataOut.tnoise[1]*=0.995
        #print(dataOut.nProfiles)
        #dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        #dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt)
        ######dataOut.pan=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)
        ######dataOut.pbn=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)
        dataOut.pan_LP=dataOut.tnoise[0]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)
        dataOut.pbn_LP=dataOut.tnoise[1]/float(dataOut.nProfiles_LP*dataOut.nIncohInt_LP)

    def ConvertDataLP(self,dataOut):
        #print(numpy.shape(dataOut.data_acf))
        #print(dataOut.dataLag_spc[:,:,:,1]/dataOut.data_spc)
        #exit(1)
        self.normfactor_LP=1.0/(dataOut.nIncohInt_LP*dataOut.nProfiles_LP)#*dataOut.nProfiles
        #print("acf: ",dataOut.data_acf[0,0,100])
        #print("Power: ",numpy.mean(dataOut.dataLag_spc_LP[0,:,100]))
        #buffer = dataOut.data_acf*(1./(normfactor*dataOut.nProfiles_LP))
        #buffer = dataOut.data_acf*(1./(normfactor))
        buffer = dataOut.data_acf#*(self.normfactor_LP) #nChannels x nProfiles (nLags) x nHeights
        #print("acf: ",numpy.sum(buffer))

        dataOut.output_LP_integrated = numpy.transpose(buffer,(1,2,0)) #nProfiles (nLags) x nHeights x nChannels

    def normFactor(self,dataOut):
        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        for l in range(dataOut.DPL):
            if(l==0 or (l>=3 and l <=6)):
                dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)
            else:
                dataOut.rnint2[l]=2*(1.0/(dataOut.nIncohInt*dataOut.nProfiles))

    def run(self,dataOut):

        dataOut.paramInterval=0#int(dataOut.nint*dataOut.header[7][0]*2 )
        dataOut.lat=-11.95
        dataOut.lon=-76.87

        dataOut.NDP=dataOut.nHeights
        dataOut.NR=len(dataOut.channelList)
        dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        dataOut.H0=int(dataOut.heightList[0])

        self.normFactor(dataOut)

        #Probar sin comentar lo siguiente y comentando
        #dataOut.data_acf *= 16 #Corrects the zero padding
        #dataOut.dataLag_spc_LP *= 16 #Corrects the zero padding
        self.ConvertDataLP(dataOut)
        #dataOut.dataLag_spc_LP *= 2
        #dataOut.output_LP_integrated[:,:,3] *= float(dataOut.NSCAN/22)#(dataOut.nNoiseProfiles) #Corrects the zero padding

        dataOut.nis=dataOut.NSCAN*dataOut.nIncohInt_LP*10
        #print("nis/10: ", dataOut.NSCAN,dataOut.nIncohInt_LP,dataOut.nProfiles_LP)
        dataOut.nis=dataOut.NSCAN*dataOut.nIncohInt_LP*dataOut.nProfiles_LP*10
        dataOut.nis=dataOut.nIncohInt_LP*dataOut.nProfiles_LP*10 #Removemos NSCAN debido a que está incluido en nProfiles_LP

        self.ConvertData(dataOut)

        dataOut.kabxys_integrated[4][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[6][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[8][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[10][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        hei = 2

        self.noise(dataOut) #Noise for DP Profiles
        dataOut.pan[[1,2,7,8,9,10]] *= 2 #Corrects the zero padding
        #dataOut.pbn[[1,2,7,8,9,10]] *= 2 #Corrects the zero padding #Chequear debido a que se están mezclando lags en self.noise()
        self.noise_LP(dataOut) #Noise for LP Profiles

        print("pan: , pan_LP: ",dataOut.pan,dataOut.pan_LP)
        print("pbn: , pbn_LP: ",dataOut.pbn,dataOut.pbn_LP)



        dataOut.NAVG=1#dataOut.rnint2[0] #CHECK THIS!
        dataOut.nint=dataOut.nIncohInt
        dataOut.MAXNRANGENDT=dataOut.output_LP_integrated.shape[1]

        '''
        range_aux=numpy.zeros(dataOut.MAXNRANGENDT,order='F',dtype='float32')
        range_aux_dp=numpy.zeros(dataOut.NDT,order='F',dtype='float32')
        for i in range(dataOut.MAXNRANGENDT):
            range_aux[i]=dataOut.H0 + i*dataOut.DH
        for i in range(dataOut.NDT):
            range_aux_dp[i]=dataOut.H0 + i*dataOut.DH
        import matplotlib.pyplot as plt
        #plt.plot(10*numpy.log10(dataOut.output_LP_integrated.real[0,:,0]),range_aux)
        plt.plot(10*numpy.log10(dataOut.output_LP_integrated.real[0,:,0]),range_aux_dp)
        #plt.plot(10*numpy.log10(dataOut.output_LP_integrated.real[0,:,0]/dataOut.nProfiles_LP),dataOut.range1)
        plt.axvline(10*numpy.log10(dataOut.tnoise[0]),color='k',linestyle='dashed')
        plt.grid()
        plt.xlim(20,100)
        plt.show()
        exit(1)
        '''
        return dataOut

class SpcVoltageDataToHybrid(SpectraDataToFaraday):
    '''
    Written by R. Flores
    '''
    """Operation to use spectra data in Faraday processing.

    Parameters:
    -----------
    nint : int
        Number of integrations.

    Example
    --------

    op = proc_unit.addOperation(name='SpcVoltageDataToHybrid', optype='other')

    """

    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        self.dataLag_spc=None
        self.dataLag_cspc=None
        self.dataLag_dc=None

    def normFactor(self,dataOut):
        dataOut.rnint2=numpy.zeros(dataOut.DPL,'float32')
        #print(dataOut.nIncohInt,dataOut.nProfiles)
        for l in range(dataOut.DPL):
            if(l==0 or (l>=3 and l <=6)):
                dataOut.rnint2[l]=1.0/(dataOut.nIncohInt*dataOut.nProfiles)
            else:
                dataOut.rnint2[l]=2*(1.0/(dataOut.nIncohInt*dataOut.nProfiles))

    def run(self,dataOut):

        dataOut.paramInterval=0#int(dataOut.nint*dataOut.header[7][0]*2 )
        dataOut.lat=-11.95
        dataOut.lon=-76.87
        #print(numpy.shape(dataOut.dataLag_spc))
        #exit(1)
        data_to_remov_eej = dataOut.dataLag_spc[:,:,:,0]
        #dataOut.NDP=dataOut.nHeights
        #dataOut.NR=len(dataOut.channelList)
        #dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        #dataOut.H0=int(dataOut.heightList[0])

        self.normFactor(dataOut)

        #dataOut.nis=dataOut.NSCAN*dataOut.NAVG*dataOut.nint*10
        #print(dataOut.nHeights)
        #exit(1)
        #dataOut.NDP=dataOut.nHeights
        self.ConvertData(dataOut)

        dataOut.kabxys_integrated[4][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[6][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[8][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        dataOut.kabxys_integrated[10][:,(1,2,7,8,9,10),0] *= 2 #Corrects the zero padding
        #print(numpy.sum(dataOut.kabxys_integrated[4][:,1,0]))

        if hasattr(dataOut, 'NRANGE'):
            dataOut.MAXNRANGENDT = max(dataOut.NRANGE,dataOut.NDT)
        else:
            dataOut.MAXNRANGENDT = dataOut.NDP

        #dataOut.MAXNRANGENDT = max(dataOut.NRANGE,dataOut.NDP)
        #print(dataOut.rnint2)
        dataOut.DH=dataOut.heightList[1]-dataOut.heightList[0]
        dataOut.H0=int(dataOut.heightList[0])
        #print(dataOut.nis)
        #exit(1)
        #self.noise(dataOut)

        if gmtime(dataOut.utctime).tm_hour >= 22. or gmtime(dataOut.utctime).tm_hour < 12.:
            self.get_eej_index(data_to_remov_eej,dataOut)

        return dataOut
