'''
Created on Jun 9, 2020

@author: Roberto Flores
'''

import os
import sys
import time

import struct


import datetime

import numpy


import schainpy.admin
from schainpy.model.io.jroIO_base import LOCALTIME, Reader
from schainpy.model.data.jroheaderIO import BasicHeader, SystemHeader, RadarControllerHeader, ProcessingHeader
from schainpy.model.proc.jroproc_base import ProcessingUnit, Operation, MPDecorator
from schainpy.model.data.jrodata import Voltage, Parameters
from schainpy.utils import log


class DatReader(Reader, ProcessingUnit):

    def __init__(self):

        ProcessingUnit.__init__(self)
        self.basicHeaderObj = BasicHeader(LOCALTIME)
        self.systemHeaderObj = SystemHeader()
        self.radarControllerHeaderObj = RadarControllerHeader()
        self.processingHeaderObj = ProcessingHeader()
        self.dataOut = Parameters()
        #print(self.basicHeaderObj.timezone)
        #self.counter_block=0
        self.format='dat'
        self.flagNoMoreFiles = 0
        self.filename = None
        self.intervals = set()
        #self.datatime = datetime.datetime(1900,1,1)

        self.filefmt = "***%Y%m%d*******"

        self.padding=numpy.zeros(1,'int32')
        self.hsize=numpy.zeros(1,'int32')
        self.bufsize=numpy.zeros(1,'int32')
        self.nr=numpy.zeros(1,'int32')
        self.ngates=numpy.zeros(1,'int32') ###  ###  ### 2
        self.time1=numpy.zeros(1,'uint64') # pos 3
        self.time2=numpy.zeros(1,'uint64') # pos 4
        self.lcounter=numpy.zeros(1,'int32')
        self.groups=numpy.zeros(1,'int32')
        self.system=numpy.zeros(4,'int8') # pos 7
        self.h0=numpy.zeros(1,'float32')
        self.dh=numpy.zeros(1,'float32')
        self.ipp=numpy.zeros(1,'float32')
        self.process=numpy.zeros(1,'int32')
        self.tx=numpy.zeros(1,'int32')

        self.ngates1=numpy.zeros(1,'int32')  ###  ###  ### 13
        self.time0=numpy.zeros(1,'uint64') # pos 14
        self.nlags=numpy.zeros(1,'int32')
        self.nlags1=numpy.zeros(1,'int32')
        self.txb=numpy.zeros(1,'float32')   ###  ###  ### 17
        self.time3=numpy.zeros(1,'uint64') # pos 18
        self.time4=numpy.zeros(1,'uint64') # pos 19
        self.h0_=numpy.zeros(1,'float32')
        self.dh_=numpy.zeros(1,'float32')
        self.ipp_=numpy.zeros(1,'float32')
        self.txa_=numpy.zeros(1,'float32')

        self.pad=numpy.zeros(100,'int32')

        self.nbytes=numpy.zeros(1,'int32')
        self.limits=numpy.zeros(1,'int32')
        self.ngroups=numpy.zeros(1,'int32') ###  ###  ### 27
        #Make the header list
        #header=[hsize,bufsize,nr,ngates,time1,time2,lcounter,groups,system,h0,dh,ipp,process,tx,padding,ngates1,time0,nlags,nlags1,padding,txb,time3,time4,h0_,dh_,ipp_,txa_,pad,nbytes,limits,padding,ngroups]
        self.header=[self.hsize,self.bufsize,self.nr,self.ngates,self.time1,self.time2,self.lcounter,self.groups,self.system,self.h0,self.dh,self.ipp,self.process,self.tx,self.ngates1,self.padding,self.time0,self.nlags,self.nlags1,self.padding,self.txb,self.time3,self.time4,self.h0_,self.dh_,self.ipp_,self.txa_,self.pad,self.nbytes,self.limits,self.padding,self.ngroups]



    def setup(self, **kwargs):

        self.set_kwargs(**kwargs)


        if self.path is None:
            raise ValueError('The path is not valid')

        self.open_file = open
        self.open_mode = 'rb'



        if self.format is None:
            raise ValueError('The format is not valid')
        elif self.format.lower() in ('dat'):
            self.ext = '.dat'
        elif self.format.lower() in ('out'):
            self.ext = '.out'


        log.log("Searching files in {}".format(self.path), self.name)
        self.filenameList = self.searchFilesOffLine(self.path, self.startDate,
            self.endDate, self.expLabel, self.ext, self.walk, self.filefmt, self.folderfmt)
        #print(self.path)
        #print(self.filenameList)
        #input()


        self.setNextFile()

    def readFirstHeader(self):
        '''Read header and data'''

        #self.flag_same_file=1
        self.counter_block=0
        self.parseHeader()
        self.parseData()
        self.blockIndex = 0

        return

    def parseHeader(self):
        '''
        '''

        for i in range(len(self.header)):
            for j in range(len(self.header[i])):
                #print("len(header[i]) ",len(header[i]))
                #input()
                temp=self.fp.read(int(self.header[i].itemsize))
                if isinstance(self.header[i][0], numpy.int32):
                    #print(struct.unpack('i', temp)[0])
                    self.header[i][0]=struct.unpack('i', temp)[0]
                if isinstance(self.header[i][0], numpy.uint64):
                    self.header[i][0]=struct.unpack('q', temp)[0]
                if isinstance(self.header[i][0], numpy.int8):
                    self.header[i][0]=struct.unpack('B', temp)[0]
                if isinstance(self.header[i][0], numpy.float32):
                    self.header[i][0]=struct.unpack('f', temp)[0]

        self.fp.seek(0,0)
        if int(self.header[1][0])==int(81864):
            self.experiment='DP'
            
        elif int(self.header[1][0])==int(185504):
            self.experiment='HP'


        self.total_blocks=os.stat(self.filename).st_size//self.header[1][0]


    def parseData(self):
        '''
        '''
        if self.experiment=='DP':
            self.header[15][0]=66
            self.header[18][0]=16
            self.header[17][0]=11
            self.header[2][0]=2


        self.noise=numpy.zeros(self.header[2][0],'float32') #self.header[2][0]
        #tmpx=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kax=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kay=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kbx=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kby=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kax2=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kay2=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kbx2=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kby2=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kaxbx=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kaxby=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kaybx=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kayby=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kaxay=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.kbxby=numpy.zeros((self.header[15][0],self.header[17][0],2),'float32')
        self.output_LP_real=numpy.zeros((self.header[18][0],200,self.header[2][0]),'float32')
        self.output_LP_imag=numpy.zeros((self.header[18][0],200,self.header[2][0]),'float32')
        self.final_cross_products=[self.kax,self.kay,self.kbx,self.kby,self.kax2,self.kay2,self.kbx2,self.kby2,self.kaxbx,self.kaxby,self.kaybx,self.kayby,self.kaxay,self.kbxby]
        #self.final_cross_products=[tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx]

        #print("pos: ",self.fp.tell())


    def readNextBlock(self):

        while True:
            self.flagDiscontinuousBlock = 0
            #print(os.stat(self.filename).st_size)
            #print(os.stat(self.filename).st_size//self.header[1][0])
            #os.stat(self.fp)
            if self.counter_block == self.total_blocks:

                self.setNextFile()

            self.readBlock()
            #self.counter_block+=1

            if (self.datatime < datetime.datetime.combine(self.startDate, self.startTime)) or \
               (self.datatime > datetime.datetime.combine(self.endDate, self.endTime)):

                #print(self.datatime)
                #print(datetime.datetime.combine(self.startDate, self.startTime))
                #print(datetime.datetime.combine(self.endDate, self.endTime))
                #print("warning")
                log.warning(
                    'Reading Block No. {}/{} -> {} [Skipping]'.format(
                        self.counter_block,
                        self.total_blocks,
                        self.datatime.ctime()),
                    'DATReader')
                continue
            break

        log.log(
            'Reading Block No. {}/{} -> {}'.format(
                self.counter_block,
                self.total_blocks,
                self.datatime.ctime()),
            'DATReader')

        return 1

    def readBlock(self):
        '''
        '''

        self.npos=self.counter_block*self.header[1][0]
        #print(self.counter_block)
        self.fp.seek(self.npos, 0)
        self.counter_block+=1
        #print("fpos1: ",self.fp.tell())

        self.read_header()

        #put by hand because old files didn't save it in the header
        if self.experiment=='DP':
            self.header[15][0]=66
            self.header[18][0]=16
            self.header[17][0]=11
            self.header[2][0]=2
        #########################################

        if self.experiment=="HP":
            self.long_pulse_products()

        self.read_cross_products()


        self.read_noise()


        return



    def read_header(self):


        for i in range(len(self.header)):
            for j in range(len(self.header[i])):
                #print("len(header[i]) ",len(header[i]))
                #input()
                temp=self.fp.read(int(self.header[i].itemsize))
                #if(b''==temp):
                #    self.setNextFile()
                #    self.flag_same_file=0
                if isinstance(self.header[i][0], numpy.int32):
                    #print(struct.unpack('i', temp)[0])
                    self.header[i][0]=struct.unpack('i', temp)[0]
                if isinstance(self.header[i][0], numpy.uint64):
                    self.header[i][0]=struct.unpack('q', temp)[0]
                if isinstance(self.header[i][0], numpy.int8):
                    self.header[i][0]=struct.unpack('B', temp)[0]
                if isinstance(self.header[i][0], numpy.float32):
                    self.header[i][0]=struct.unpack('f', temp)[0]
            #else:
            #    continue
            #self.fp.seek(self.npos_aux, 0)
        #    break

        #print("fpos2: ",self.fp.tell())
        #log.success('Parameters found: {}'.format(self.parameters),
        #            'DATReader')
        #print("Success")
        #self.TimeBlockSeconds_for_dp_power = self.header[4][0]#-((self.dataOut.nint-1)*self.dataOut.NAVG*2)
        #print(dataOut.TimeBlockSeconds_for_dp_power)

        #self.datatime=datetime.datetime.fromtimestamp(self.header[4][0]).strftime("%Y-%m-%d %H:%M:%S")
        #print(self.header[4][0])
        self.datatime=datetime.datetime.fromtimestamp(self.header[4][0])
        #print(self.header[1][0])

    def long_pulse_products(self):
        temp=self.fp.read(self.header[18][0]*self.header[2][0]*200*8)
        ii=0

        for l in range(self.header[18][0]): #lag
            for r in range(self.header[2][0]): # channels
                for k in range(200): #RANGE## generalizar
                    self.output_LP_real[l,k,r]=struct.unpack('f', temp[ii:ii+4])[0]
                    ii=ii+4
                    self.output_LP_imag[l,k,r]=struct.unpack('f', temp[ii:ii+4])[0]
                    ii=ii+4

        #print(self.output_LP_real[1,1,1])
        #print(self.output_LP_imag[1,1,1])
    def read_cross_products(self):

        for ind in range(len(self.final_cross_products)): #final cross products
            temp=self.fp.read(self.header[17][0]*2*self.header[15][0]*4) #*4 bytes
            #if(b''==temp):
            #    self.setNextFile()
            #    self.flag_same_file=0
            ii=0
            #print("kabxys.shape ",kabxys.shape)
            #print(kabxys)
            #print("fpos3: ",self.fp.tell())
            for l in range(self.header[17][0]): #lag
                #print("fpos3: ",self.fp.tell())
                for fl in range(2): # unflip and flip
                    for k in range(self.header[15][0]): #RANGE
                        #print("fpos3: ",self.fp.tell())
                        self.final_cross_products[ind][k,l,fl]=struct.unpack('f', temp[ii:ii+4])[0]
                        ii=ii+4
                        #print("fpos2: ",self.fp.tell())



    def read_noise(self):

        temp=self.fp.read(self.header[2][0]*4) #*4 bytes    self.header[2][0]
        for ii in range(self.header[2][0]): #self.header[2][0]
            self.noise[ii]=struct.unpack('f', temp[ii*4:(ii+1)*4])[0]

        #print("fpos5: ",self.fp.tell())



    def set_output(self):
        '''
        Storing data from buffer to dataOut object
        '''
        #print("fpos2: ",self.fp.tell())
        ##self.dataOut.header = self.header
        #this is put by hand because it isn't saved in the header
        if self.experiment=='DP':
            self.dataOut.NRANGE=0
            self.dataOut.NSCAN=132
            self.dataOut.heightList=self.header[10][0]*(numpy.arange(self.header[15][0]))
        elif self.experiment=='HP':
            self.dataOut.output_LP=self.output_LP_real+1.j*self.output_LP_imag
            self.dataOut.NRANGE=200
            self.dataOut.NSCAN=128
            self.dataOut.heightList=self.header[10][0]*(numpy.arange(90)) #NEEEDS TO BE GENERALIZED
        #########################################
        #print(self.dataOut.output_LP[1,1,1])
        self.dataOut.MAXNRANGENDT=self.header[3][0]
        self.dataOut.NDP=self.header[15][0]
        self.dataOut.DPL=self.header[17][0]
        self.dataOut.DH=self.header[10][0]
        self.dataOut.NAVG=self.header[7][0]
        self.dataOut.H0=self.header[9][0]
        self.dataOut.NR=self.header[2][0]
        self.dataOut.NLAG=self.header[18][0]
        #self.dataOut.tmpx=self.tmpx
        #self.dataOut.timeZone = 5
        #self.dataOut.final_cross_products=self.final_cross_products
        self.dataOut.kax=self.kax
        #print(self.dataOut.kax[1,1,1])
        self.dataOut.kay=self.kay
        self.dataOut.kbx=self.kbx
        self.dataOut.kby=self.kby
        self.dataOut.kax2=self.kax2
        self.dataOut.kay2=self.kay2
        self.dataOut.kbx2=self.kbx2
        self.dataOut.kby2=self.kby2
        self.dataOut.kaxbx=self.kaxbx
        self.dataOut.kaxby=self.kaxby
        self.dataOut.kaybx=self.kaybx
        self.dataOut.kayby=self.kayby
        self.dataOut.kaxay=self.kaxay
        self.dataOut.kbxby=self.kbxby
        self.dataOut.noise_final=self.noise
        #print("NOISE",self.noise)


        self.dataOut.useLocalTime=True

        #self.dataOut.experiment=self.experiment
        #print(self.datatime)
        #print(self.dataOut.datatime)


        #self.dataOut.utctime = (self.datatime - datetime.datetime(1970, 1, 1)).total_seconds()
        #self.dataOut.utctimeInit = self.dataOut.utctime



        self.dataOut.lt=self.datatime.hour


        #print(RadarControllerHeader().ippSeconds)
        #print(RadarControllerHeader().ipp)
        #self.dataOut.utctime=time.gmtime(self.header[4][0])- datetime.datetime(1970, 1, 1)
        #self.dataOut.utctime=self.dataOut.utctime.total_seconds()
        #time1 = self.header[4][0] # header.time1
        #print("time1: ",time1)
        #print(self.header[4][0])
        #date = time.ctime(time1)
        #print("DADSADA",time.strptime(date))
        #print("date_before: ",date)
        #bd_time=time.gmtime(time1)
        #print(time.mktime(bd_time))
        #self.dataOut.utctime=time.mktime(bd_time)
        self.dataOut.utctime = self.header[4][0]
        #self.dataOut.datatime=a
        #print(datetime.datetime.utcfromtimestamp(self.dataOut.utctime))
        #self.dataOut.TimeBlockDate=self.datatime.ctime()
        self.dataOut.TimeBlockSeconds=time.mktime(time.strptime(self.dataOut.datatime.ctime()))

        #self.dataOut.heightList = self.ranges
        #self.dataOut.utctime = (self.datatime - datetime.datetime(1970, 1, 1)).total_seconds()
        #self.dataOut.utctimeInit = self.dataOut.utctime
        #self.dataOut.paramInterval = min(self.intervals)
        #self.dataOut.useLocalTime = False
        self.dataOut.flagNoData = False
        self.dataOut.flagDiscontinuousBlock = self.flagDiscontinuousBlock
        #print(self.dataOut.channelIndexList)
        self.dataOut.channelList=list(range(0,self.header[2][0]))
        #print(self.dataOut.channelList)
        #print(self.datatime)
        #print(self.dataOut.final_cross_products[0])


        #self.dataOut.heightList=self.header[10][0]*(numpy.arange(self.header[15][0]))

        #print(numpy.shape(self.dataOut.heightList))


    def getData(self):
        '''
        Storing data from databuffer to dataOut object
        '''

        if not self.readNextBlock():
            self.dataOut.flagNoData = True
            return 0

        self.set_output()

        return 1

    def run(self, **kwargs):

        if not(self.isConfig):
            self.setup(**kwargs)
            self.isConfig = True
        #print("fpos1: ",self.fp.tell())
        self.getData()

        return

@MPDecorator
class DatWriter(Operation):


    def __init__(self):

        Operation.__init__(self)
        #self.dataOut = Voltage()
        self.counter = 0
        self.path = None
        self.fp = None
        return
        #self.ext= '.dat'

    def run(self, dataOut, path, format='dat', experiment=None, **kwargs):
        print(dataOut.flagNoData)
        print(dataOut.datatime.ctime())
        print(dataOut.TimeBlockDate)
        input()
        #if dataOut.flag_save:
        self.experiment=experiment
        self.path=path
        if self.experiment=='DP':
            dataOut.header[1][0]=81864
        elif self.experiment=='HP':
            dataOut.header[1][0]=185504#173216
        #dataOut.header[1][0]=bufsize
        self.dataOut = dataOut
        #print(self.dataOut.nint)
        #self.bufsize=bufsize
        if format == 'dat':
            self.ext = '.dat'
        if format == 'out':
            self.ext = '.out'
        self.putData()

        return



    def setFile(self):
        '''
        Create new out file object
        '''

        #self.dataOut.TimeBlockSeconds=time.mktime(time.strptime(self.dataOut.TimeBlockDate))
        date = datetime.datetime.fromtimestamp(self.dataOut.TimeBlockSeconds)

        #print("date",date)

        filename = '{}{}{}'.format('jro',
                                   date.strftime('%Y%m%d_%H%M%S'),
                                   self.ext)
        #print(filename)
        #print(self.path)

        self.fullname = os.path.join(self.path, filename)

        if os.path.isfile(self.fullname) :
            log.warning(
                'Destination file {} already exists, previous file deleted.'.format(
                    self.fullname),
                'DatWriter')
            os.remove(self.fullname)

        try:
            log.success(
                'Creating file: {}'.format(self.fullname),
                'DatWriter')
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            #self.fp = madrigal.cedar.MadrigalCedarFile(self.fullname, True)
            self.fp = open(self.fullname,'wb')

        except ValueError as e:
            log.error(
                'Impossible to create *.out file',
                'DatWriter')
            return

        return 1

    def writeBlock(self):

        #self.dataOut.paramInterval=2
        #startTime = datetime.datetime.utcfromtimestamp(self.dataOut.utctime)
        #print(startTime)
        #endTime = startTime + datetime.timedelta(seconds=self.dataOut.paramInterval)

        self.dataOut.header[0].astype('int32').tofile(self.fp)
        self.dataOut.header[1].astype('int32').tofile(self.fp)
        self.dataOut.header[2].astype('int32').tofile(self.fp)
        self.dataOut.header[3].astype('int32').tofile(self.fp)
        self.dataOut.header[4].astype('uint64').tofile(self.fp)
        self.dataOut.header[5].astype('uint64').tofile(self.fp)
        self.dataOut.header[6].astype('int32').tofile(self.fp)
        self.dataOut.header[7].astype('int32').tofile(self.fp)
        #print(dataOut.header[7])
        self.dataOut.header[8].astype('int8').tofile(self.fp)
        self.dataOut.header[9].astype('float32').tofile(self.fp)
        self.dataOut.header[10].astype('float32').tofile(self.fp)
        self.dataOut.header[11].astype('float32').tofile(self.fp)
        self.dataOut.header[12].astype('int32').tofile(self.fp)
        self.dataOut.header[13].astype('int32').tofile(self.fp)
        self.dataOut.header[14].astype('int32').tofile(self.fp)
        self.dataOut.header[15].astype('int32').tofile(self.fp)
        self.dataOut.header[16].astype('uint64').tofile(self.fp)
        self.dataOut.header[17].astype('int32').tofile(self.fp)
        self.dataOut.header[18].astype('int32').tofile(self.fp)
        self.dataOut.header[19].astype('int32').tofile(self.fp)
        self.dataOut.header[20].astype('float32').tofile(self.fp)
        self.dataOut.header[21].astype('uint64').tofile(self.fp)
        self.dataOut.header[22].astype('uint64').tofile(self.fp)
        self.dataOut.header[23].astype('float32').tofile(self.fp)
        self.dataOut.header[24].astype('float32').tofile(self.fp)
        self.dataOut.header[25].astype('float32').tofile(self.fp)
        self.dataOut.header[26].astype('float32').tofile(self.fp)
        self.dataOut.header[27].astype('int32').tofile(self.fp)
        self.dataOut.header[28].astype('int32').tofile(self.fp)
        self.dataOut.header[29].astype('int32').tofile(self.fp)
        self.dataOut.header[30].astype('int32').tofile(self.fp)
        self.dataOut.header[31].astype('int32').tofile(self.fp)
        #print("tell before 1 ",self.fp.tell())
        #input()

        if self.experiment=="HP":
            #print("INSIDE")
            #tmp=numpy.zeros(1,dtype='complex64')
            #print("tmp ",tmp)
            #input()
            #print(dataOut.NLAG)
            #print(dataOut.NR)
            #print(dataOut.NRANGE)
            for l in range(self.dataOut.NLAG): #lag
                for r in range(self.dataOut.NR): # unflip and flip
                    for k in range(self.dataOut.NRANGE): #RANGE
                        self.dataOut.output_LP.real[l,k,r].astype('float32').tofile(self.fp)
                        self.dataOut.output_LP.imag[l,k,r].astype('float32').tofile(self.fp)


                #print("tell before 2 ",self.outputfile.tell())





            #print(self.dataOut.output_LP[1,1,1])

        #print(self.dataOut.kax)
        final_cross_products=[self.dataOut.kax,self.dataOut.kay,self.dataOut.kbx,self.dataOut.kby,
                            self.dataOut.kax2,self.dataOut.kay2,self.dataOut.kbx2,self.dataOut.kby2,
                            self.dataOut.kaxbx,self.dataOut.kaxby,self.dataOut.kaybx,self.dataOut.kayby,
                            self.dataOut.kaxay,self.dataOut.kbxby]

        #print(self.dataOut.kax)
        #print("tell before crossp saving ",self.outputfile.tell())
        for kabxys in final_cross_products:

            for l in range(self.dataOut.DPL): #lag
                            for fl in range(2): # unflip and flip
                                for k in range(self.dataOut.NDT): #RANGE
                                    kabxys[k,l,fl].astype('float32').tofile(self.fp)


        #print("tell before noise saving ",self.outputfile.tell())


        for nch in range(self.dataOut.NR):
            self.dataOut.noise_final[nch].astype('float32').tofile(self.fp)

        #print("tell before noise saving ",self.fp.tell())
        #input()




        log.log(
            'Writing {} blocks'.format(
                self.counter+1),
            'DatWriter')






    def putData(self):
        #print("flagNoData",self.dataOut.flagNoData)
        #print("flagDiscontinuousBlock",self.dataOut.flagDiscontinuousBlock)
        #print(self.dataOut.flagNoData)

        if self.dataOut.flagNoData:
            return 0

        if self.dataOut.flagDiscontinuousBlock:

            self.counter = 0

        if self.counter == 0:
            self.setFile()
        #if self.experiment=="HP":
            #if self.dataOut.debris_activated==0:
                #self.writeBlock()
                #self.counter += 1
        #else:
        self.writeBlock()
        self.counter += 1

    def close(self):

        if self.counter > 0:
            self.fp.close()
            log.success('Closing file {}'.format(self.fullname), 'DatWriter')
