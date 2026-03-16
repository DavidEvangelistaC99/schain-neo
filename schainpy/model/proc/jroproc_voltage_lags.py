
import matplotlib.pyplot as plt



import numpy
import time
import math

from datetime import datetime

from schainpy.utils import log

import struct
import os

import sys

from ctypes import *

from schainpy.model.io.jroIO_voltage import VoltageReader,JRODataReader
from schainpy.model.proc.jroproc_base import ProcessingUnit, Operation, MPDecorator
from schainpy.model.data.jrodata import Voltage




@MPDecorator
class VoltageLagsProc(ProcessingUnit):

    def __init__(self):

        ProcessingUnit.__init__(self)

        self.dataOut = Voltage()
        self.bcounter=0
        self.dataOut.kax=None
        self.dataOut.kay=None
        self.dataOut.kbx=None
        self.dataOut.kby=None
        self.dataOut.kax2=None
        self.dataOut.kay2=None
        self.dataOut.kbx2=None
        self.dataOut.kby2=None
        self.dataOut.kaxbx=None
        self.dataOut.kaxby=None
        self.dataOut.kaybx=None
        self.dataOut.kayby=None
        self.dataOut.kaxay=None
        self.dataOut.kbxby=None
        self.aux=1

        self.LP_products_aux=0
        self.lag_products_LP_median_estimates_aux=0

        #self.dataOut.input_dat_type=0 #06/04/2020

    def get_products_cabxys(self):


        if self.aux==1:



            self.dataOut.read_samples=int(self.dataOut.systemHeaderObj.nSamples/self.dataOut.OSAMP)
            if self.dataOut.experiment=="DP":
                self.dataOut.nptsfft1=132  #30/03/2020
                self.dataOut.nptsfft2=140  #30/03/2020
            if self.dataOut.experiment=="HP":
                self.dataOut.nptsfft1=128  #30/03/2020
                self.dataOut.nptsfft2=150  #30/03/2020


            #self.dataOut.noise_final_list=[]  #30/03/2020

            padding=numpy.zeros(1,'int32')

            hsize=numpy.zeros(1,'int32')
            bufsize=numpy.zeros(1,'int32')
            nr=numpy.zeros(1,'int32')
            ngates=numpy.zeros(1,'int32') ###  ###  ### 2
            time1=numpy.zeros(1,'uint64') # pos 3
            time2=numpy.zeros(1,'uint64') # pos 4
            lcounter=numpy.zeros(1,'int32')
            groups=numpy.zeros(1,'int32')
            system=numpy.zeros(4,'int8') # pos 7
            h0=numpy.zeros(1,'float32')
            dh=numpy.zeros(1,'float32')
            ipp=numpy.zeros(1,'float32')
            process=numpy.zeros(1,'int32')
            tx=numpy.zeros(1,'int32')

            ngates1=numpy.zeros(1,'int32')  ###  ###  ### 13
            time0=numpy.zeros(1,'uint64') # pos 14
            nlags=numpy.zeros(1,'int32')
            nlags1=numpy.zeros(1,'int32')
            txb=numpy.zeros(1,'float32')   ###  ###  ### 17
            time3=numpy.zeros(1,'uint64') # pos 18
            time4=numpy.zeros(1,'uint64') # pos 19
            h0_=numpy.zeros(1,'float32')
            dh_=numpy.zeros(1,'float32')
            ipp_=numpy.zeros(1,'float32')
            txa_=numpy.zeros(1,'float32')

            pad=numpy.zeros(100,'int32')

            nbytes=numpy.zeros(1,'int32')
            limits=numpy.zeros(1,'int32')
            ngroups=numpy.zeros(1,'int32') ###  ###  ### 27


            self.dataOut.header=[hsize,bufsize,nr,ngates,time1,time2,
                    lcounter,groups,system,h0,dh,ipp,
                    process,tx,ngates1,padding,time0,nlags,
                    nlags1,padding,txb,time3,time4,h0_,dh_,
                    ipp_,txa_,pad,nbytes,limits,padding,ngroups]

            if self.dataOut.experiment == "DP":
                self.dataOut.header[1][0]=81864
            if self.dataOut.experiment == "HP":
                self.dataOut.header[1][0]=173216

            self.dataOut.header[3][0]=max(self.dataOut.NRANGE,self.dataOut.NDT)
            self.dataOut.header[7][0]=self.dataOut.NAVG
            self.dataOut.header[9][0]=int(self.dataOut.heightList[0])
            self.dataOut.header[10][0]=self.dataOut.DH
            self.dataOut.header[17][0]=self.dataOut.DPL
            self.dataOut.header[18][0]=self.dataOut.NLAG
            #self.header[5][0]=0
            self.dataOut.header[15][0]=self.dataOut.NDP
            self.dataOut.header[2][0]=self.dataOut.NR
            #time.mktime(time.strptime()




            self.aux=0









        if self.dataOut.experiment=="DP":


            self.dataOut.lags_array=[x / self.dataOut.DH for x in self.dataOut.flags_array]
            self.cax=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cay=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cbx=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cby=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cax2=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cay2=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cbx2=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cby2=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.caxbx=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.caxby=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.caybx=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cayby=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.caxay=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))
            self.cbxby=numpy.zeros((self.dataOut.NDP,self.dataOut.nlags_array,2))

            for i in range(2):
                for j in range(self.dataOut.NDP):
                    for k in range(int(self.dataOut.NSCAN/2)):
                        n=k%self.dataOut.nlags_array
                        ax=self.dataOut.data[0,2*k+i,j].real
                        ay=self.dataOut.data[0,2*k+i,j].imag
                        if j+self.dataOut.lags_array[n]<self.dataOut.NDP:
                            bx=self.dataOut.data[1,2*k+i,j+int(self.dataOut.lags_array[n])].real
                            by=self.dataOut.data[1,2*k+i,j+int(self.dataOut.lags_array[n])].imag
                        else:
                            if k+1<int(self.dataOut.NSCAN/2):
                                bx=self.dataOut.data[1,2*(k+1)+i,(self.dataOut.NRANGE+self.dataOut.NCAL+j+int(self.dataOut.lags_array[n]))%self.dataOut.NDP].real
                                by=self.dataOut.data[1,2*(k+1)+i,(self.dataOut.NRANGE+self.dataOut.NCAL+j+int(self.dataOut.lags_array[n]))%self.dataOut.NDP].imag

                            if k+1==int(self.dataOut.NSCAN/2):
                                bx=self.dataOut.data[1,2*k+i,(self.dataOut.NRANGE+self.dataOut.NCAL+j+int(self.dataOut.lags_array[n]))%self.dataOut.NDP].real
                                by=self.dataOut.data[1,2*k+i,(self.dataOut.NRANGE+self.dataOut.NCAL+j+int(self.dataOut.lags_array[n]))%self.dataOut.NDP].imag

                        if(k<self.dataOut.nlags_array):
                            self.cax[j][n][i]=ax
                            self.cay[j][n][i]=ay
                            self.cbx[j][n][i]=bx
                            self.cby[j][n][i]=by
                            self.cax2[j][n][i]=ax*ax
                            self.cay2[j][n][i]=ay*ay
                            self.cbx2[j][n][i]=bx*bx
                            self.cby2[j][n][i]=by*by
                            self.caxbx[j][n][i]=ax*bx
                            self.caxby[j][n][i]=ax*by
                            self.caybx[j][n][i]=ay*bx
                            self.cayby[j][n][i]=ay*by
                            self.caxay[j][n][i]=ax*ay
                            self.cbxby[j][n][i]=bx*by
                        else:
                            self.cax[j][n][i]+=ax
                            self.cay[j][n][i]+=ay
                            self.cbx[j][n][i]+=bx
                            self.cby[j][n][i]+=by
                            self.cax2[j][n][i]+=ax*ax
                            self.cay2[j][n][i]+=ay*ay
                            self.cbx2[j][n][i]+=bx*bx
                            self.cby2[j][n][i]+=by*by
                            self.caxbx[j][n][i]+=ax*bx
                            self.caxby[j][n][i]+=ax*by
                            self.caybx[j][n][i]+=ay*bx
                            self.cayby[j][n][i]+=ay*by
                            self.caxay[j][n][i]+=ax*ay
                            self.cbxby[j][n][i]+=bx*by



        #return self.cax,self.cay,self.cbx,self.cby,self.cax2,self.cay2,self.cbx2,self.cby2,self.caxbx,self.caxby,self.caybx,self.cayby,self.caxay,self.cbxby

        if self.dataOut.experiment=="HP":

            #lagind=[0,1,2,3,4,5,6,7,0,3,4,5,6,8,9,10]
            #lagfirst=[1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1]

            self.cax=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))# hp:67x11x2  dp: 66x11x2
            self.cay=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cbx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cax2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cay2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cbx2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cby2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.caxbx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.caxby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.caybx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cayby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.caxay=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            self.cbxby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2))
            for i in range(2):   # flipped and unflipped
                for j in range(self.dataOut.NDP): # loop over true ranges # 67
                    for k in range(int(self.dataOut.NSCAN)): # 128
                        #print("flip ",i,"  NDP ",j, "  NSCAN ",k)
                        #print("cdata ",cdata[i:NSCAN:2][k][:,0])
                        n=self.dataOut.lagind[k%self.dataOut.nlags_array] # 128=16x8
                        #print("n ",n)
                        #ind1=nrx*(j+ngates_2*i+ngates_2*2*k)# scan has flip or unflip
                        #ind2=ind1+(1)+nrx*lags_array[n]#jump  each lagged
                        #ax=cdata[i:NSCAN:2][k][:,0][NRANGE+NCAL+j].real #cdata[ind1].r
                        #ay=cdata[i:NSCAN:2][k][:,0][NRANGE+NCAL+j].imag #cdata[ind1].i
                        #input()
                        ##ax=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,0][NRANGE+NCAL+j].real #cdata[ind1].r
                        ##ay=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,0][NRANGE+NCAL+j].imag #cdata[ind1].i

                        ax=self.dataOut.data[0,k,self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT].real
                        ay=self.dataOut.data[0,k,self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT].imag

                        #print("ax ",ax,"  ay",ay)
                        if self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n<self.dataOut.read_samples:
                            #bx=cdata[i:NSCAN:2][k][:,1][NRANGE+NCAL+j+n].real #cdata[ind2].r
                            #by=cdata[i:NSCAN:2][k][:,1][NRANGE+NCAL+j+n].imag #cdata[ind2].i
                            ##bx=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,1][NRANGE+NCAL+j+2*n].real #cdata[ind2].r
                            ##by=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,1][NRANGE+NCAL+j+2*n].imag #cdata[ind2].i

                            bx=self.dataOut.data[1,k,self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n].real
                            by=self.dataOut.data[1,k,self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n].imag

                            #bx=self.dataOut.data[0:NSCAN][k][:,1][NRANGE+NCAL+j+i*NDT+2*n].real #cdata[ind2].r
                            #by=self.dataOut.data[0:NSCAN][k][:,1][NRANGE+NCAL+j+i*NDT+2*n].imag #cdata[ind2].i
                            #print("bx ",bx, "  by ",by)
                            #input()
                        else:
                            #print("n ",n,"  k ",k,"  j ",j," i ",i, " n ",n)
                            #input()
                            if k+1<int(self.dataOut.NSCAN):
                                #print("k+1 ",k+1)
                                #print("int(NSCAN/2) ",int(NSCAN/2))
                                #bx=cdata[i:NSCAN:2][k+1][:,1][(NRANGE+NCAL+j+n)%NDP].real#np.nan
                                #by=cdata[i:NSCAN:2][k+1][:,1][(NRANGE+NCAL+j+n)%NDP].imag#np.nan
                                ##bx=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k+1][:,1][(NRANGE+NCAL+j+2*n)%NDP].real#np.nan
                                ##by=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k+1][:,1][(NRANGE+NCAL+j+2*n)%NDP].imag#np.nan
                                #bx=self.dataOut.data[0:NSCAN][k+1][:,1][(NRANGE+NCAL+j+i*NDT+2*n)%NDP].real#np.nan
                                #by=self.dataOut.data[0:NSCAN][k+1][:,1][(NRANGE+NCAL+j+i*NDT+2*n)%NDP].imag#np.nan
                                bx=self.dataOut.data[1,k+1,(self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n)%self.dataOut.NDP].real
                                by=self.dataOut.data[1,k+1,(self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n)%self.dataOut.NDP].imag

                                #print("n ",n,"  k ",k,"  j ",j," i ",i, " lags_array[n] ",lags_array[n])
                                #print("bx ",bx, "  by ",by)
                                #input()
                            if k+1==int(self.dataOut.NSCAN):## ESTO ES UN PARCHE PUES NO SE TIENE EL SIGUIENTE BLOQUE
                                #bx=cdata[i:NSCAN:2][k][:,1][(NRANGE+NCAL+j+n)%NDP].real#np.nan
                                #by=cdata[i:NSCAN:2][k][:,1][(NRANGE+NCAL+j+n)%NDP].imag#np.nan
                                ##bx=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,1][(NRANGE+NCAL+j+2*n)%NDP].real#np.nan
                                ##by=cdata[int(i*NSCAN):int((i+1)*NSCAN)][k][:,1][(NRANGE+NCAL+j+2*n)%NDP].imag#np.nan
                                #print("****n ",n,"  k ",k,"  j ",j," i ",i, " lags_array[n] ",lags_array[n])
                                #bx=self.dataOut.data[0:NSCAN][k][:,1][(NRANGE+NCAL+j+i*NDT+2*n)%NDP].real#np.nan
                                #by=self.dataOut.data[0:NSCAN][k][:,1][(NRANGE+NCAL+j+i*NDT+2*n)%NDP].imag#np.nan
                                bx=self.dataOut.data[1,k,(self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n)%self.dataOut.NDP].real
                                by=self.dataOut.data[1,k,(self.dataOut.NRANGE+self.dataOut.NCAL+j+i*self.dataOut.NDT+2*n)%self.dataOut.NDP].imag

                                #print("bx ",bx, "  by ",by)
                                #input()

                        #print("i ",i," j ",j," k ",k," n ",n," ax ",ax)
                        #input()
                        #ip1=j+NDP*(i+2*n)
                        #ip2=ip1*navg+iavg
                        ##if(k<11): # PREVIOUS
                        if(k<self.dataOut.nlags_array and self.dataOut.lagfirst[k%self.dataOut.nlags_array]==1):# if(k<16 && lagfirst[k%16]==1)
                            self.cax[j][n][i]=ax#[int(k/nlags_array)*nlags_array+n]
                            self.cay[j][n][i]=ay#[int(k/nlags_array)*nlags_array+n]
                            self.cbx[j][n][i]=bx#[int(k/nlags_array)*nlags_array+n]
                            self.cby[j][n][i]=by#[int(k/nlags_array)*nlags_array+n]
                            self.cax2[j][n][i]=ax*ax#np.multiply(ax,ax)[int(k/nlags_array)*nlags_array+n]
                            self.cay2[j][n][i]=ay*ay#np.multiply(ay,ay)[int(k/nlags_array)*nlags_array+n]
                            self.cbx2[j][n][i]=bx*bx#np.multiply(bx,bx)[int(k/nlags_array)*nlags_array+n]
                            self.cby2[j][n][i]=by*by#np.multiply(by,by)[int(k/nlags_array)*nlags_array+n]
                            self.caxbx[j][n][i]=ax*bx#np.multiply(ax,bx)[int(k/nlags_array)*nlags_array+n]
                            self.caxby[j][n][i]=ax*by#np.multiply(ax,by)[int(k/nlags_array)*nlags_array+n]
                            self.caybx[j][n][i]=ay*bx#np.multiply(ay,bx)[int(k/nlags_array)*nlags_array+n]
                            self.cayby[j][n][i]=ay*by#np.multiply(ay,by)[int(k/nlags_array)*nlags_array+n]
                            self.caxay[j][n][i]=ax*ay#np.multiply(ax,ay)[int(k/nlags_array)*nlags_array+n]
                            self.cbxby[j][n][i]=bx*by#np.multiply(bx,by)[int(k/nlags_array)*nlags_array+n]
                        else:
                            self.cax[j][n][i]+=ax#[int(k/nlags_array)*nlags_array+n]
                            self.cay[j][n][i]+=ay#[int(k/nlags_array)*nlags_array+n]
                            self.cbx[j][n][i]+=bx#[int(k/nlags_array)*nlags_array+n]
                            self.cby[j][n][i]+=by#[int(k/nlags_array)*nlags_array+n]
                            self.cax2[j][n][i]+=ax*ax#np.multiply(ax,ax)[int(k/nlags_array)*nlags_array+n]
                            self.cay2[j][n][i]+=ay*ay#np.multiply(ay,ay)[int(k/nlags_array)*nlags_array+n]
                            self.cbx2[j][n][i]+=bx*bx#np.multiply(bx,bx)[int(k/nlags_array)*nlags_array+n]
                            self.cby2[j][n][i]+=by*by#np.multiply(by,by)[int(k/nlags_array)*nlags_array+n]
                            self.caxbx[j][n][i]+=ax*bx#np.multiply(ax,bx)[int(k/nlags_array)*nlags_array+n]
                            self.caxby[j][n][i]+=ax*by#np.multiply(ax,by)[int(k/nlags_array)*nlags_array+n]
                            self.caybx[j][n][i]+=ay*bx#np.multiply(ay,bx)[int(k/nlags_array)*nlags_array+n]
                            self.cayby[j][n][i]+=ay*by#np.multiply(ay,by)[int(k/nlags_array)*nlags_array+n]
                            self.caxay[j][n][i]+=ax*ay#np.multiply(ax,ay)[int(k/nlags_array)*nlags_array+n]
                            self.cbxby[j][n][i]+=bx*by#np.multiply(bx,by)[int(k/nlags_array)*nlags_array+n]









    def medi(self,data_navg):
        sorts=sorted(data_navg)
        rsorts=numpy.arange(self.dataOut.NAVG)
        result=0.0
        for k in range(self.dataOut.NAVG):
            if k>=self.dataOut.nkill/2 and k<self.dataOut.NAVG-self.dataOut.nkill/2:
                result+=sorts[k]*float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill)
        return result



        '''
    def range(self):
        Range=numpy.arange(0,990,self.DH)
        return Range
        '''






    def cabxys_navg(self):

        #print("blocknow",self.dataOut.CurrentBlock)
        #bcounter=0
        #print("self.bcounter",self.bcounter)
        self.get_products_cabxys()

        self.dataOut.header[5][0]=time.mktime(time.strptime(self.dataOut.TimeBlockDate))

        #if salf.dataOut.CurrentBlock<NAVG:
        if self.bcounter==0:

            self.dataOut.header[4][0]=self.dataOut.header[5][0]
            if self.dataOut.CurrentBlock==1:
                self.dataOut.header[16][0]=self.dataOut.header[5][0]

            self.cax_navg=[]
            self.cay_navg=[]
            self.cbx_navg=[]
            self.cby_navg=[]
            self.cax2_navg=[]
            self.cay2_navg=[]
            self.cbx2_navg=[]
            self.cby2_navg=[]
            self.caxbx_navg=[]
            self.caxby_navg=[]
            self.caybx_navg=[]
            self.cayby_navg=[]
            self.caxay_navg=[]
            self.cbxby_navg=[]
            self.dataOut.kax=None
            self.dataOut.kay=None
            self.dataOut.kbx=None
            self.dataOut.kby=None
            self.dataOut.kax2=None
            self.dataOut.kay2=None
            self.dataOut.kbx2=None
            self.dataOut.kby2=None
            self.dataOut.kaxbx=None
            self.dataOut.kaxby=None
            self.dataOut.kaybx=None
            self.dataOut.kayby=None
            self.dataOut.kaxay=None
            self.dataOut.kbxby=None

            self.dataOut.noisevector=numpy.zeros((self.dataOut.read_samples,self.dataOut.NR,self.dataOut.NAVG),'float32')  #30/03/2020
            self.dataOut.noisevector_=numpy.zeros((self.dataOut.read_samples,self.dataOut.NR,self.dataOut.NAVG),'float32')
            self.dataOut.dc=numpy.zeros(self.dataOut.NR,dtype=numpy.complex_)  #30/03/2020
            #self.dataOut.noisevector=numpy.zeros((self.dataOut.read_samples,2,self.dataOut.NAVG),'float32')  #31/03/2020
            #self.dataOut.noisevector_=numpy.zeros((self.dataOut.read_samples,2,self.dataOut.NAVG),'float32')  #31/03/2020

            #self.dataOut.dc=numpy.zeros(2,dtype=numpy.complex_)  #31/03/2020
            #self.dataOut.processingHeaderObj.profilesPerBlock
        if self.dataOut.experiment=="DP":
            self.noisevectorizer(self.dataOut.nptsfft1,self.dataOut.nptsfft2)   #30/03/2020
        if self.dataOut.experiment=="HP":
            self.noisevectorizer(self.dataOut.nptsfft1,self.dataOut.nptsfftx1)   #31/03/2020
        #print(self.dataOut.noisevector[:,:,:])
        #print("·················································")
        #print("CAX: ",self.cax)
        self.cax_navg.append(self.cax)
        self.cay_navg.append(self.cay)
        self.cbx_navg.append(self.cbx)
        self.cby_navg.append(self.cby)
        self.cax2_navg.append(self.cax2)
        self.cay2_navg.append(self.cay2)
        self.cbx2_navg.append(self.cbx2)
        self.cby2_navg.append(self.cby2)
        self.caxbx_navg.append(self.caxbx)
        self.caxby_navg.append(self.caxby)
        self.caybx_navg.append(self.caybx)
        self.cayby_navg.append(self.cayby)
        self.caxay_navg.append(self.caxay)
        self.cbxby_navg.append(self.cbxby)
        self.bcounter+=1

        #self.dataOut.data=None
            #print("bcounter",bcounter)
        #/#/#/#if self.bcounter==NAVG:

            #/#/#/#print("cax_navg: ",self.cax_navg)
            #/#/#/#self.bcounter=0
        #print("blocknow",self.dataOut.current)



    def kabxys(self,NAVG,nkill):#,NRANGE,NCAL,NDT):

        self.dataOut.NAVG=NAVG
        self.dataOut.nkill=nkill
        #print("bcounter_before: ",self.bcounter)
        #print("kabxys")

        #if self.dataOut.input_dat_type==0:
        #self.dataOut.NDP=NDP
        #self.dataOut.nlags_array=nlags_array
        #self.dataOut.NSCAN=NSCAN
        #self.dataOut.DH=float(DH)
        #self.dataOut.flags_array=flags_array

        #self.dataOut.DPL=DPL
        #self.dataOut.NRANGE=NRANGE
        #self.dataOut.NCAL=NCAL

        #self.dataOut.NDT=NDT
        #self.lag_products_LP()
        ####self.cabxys_navg(NDP,nlags_array,NSCAN,flags_array)
        self.cabxys_navg()
        #self.dataOut.kshape=numpy.zeros((numpy.shape(self.cax_navg[0])[0],numpy.shape(self.cax_navg[0])[1],numpy.shape(self.cax_navg[0])[2]))
        #print("Shape cavg",numpy.shape(self.cax_navg[0])[0])
        self.dataOut.flag_save=0
        #self.dataOut.flagNoData =  True # new 1


        if self.bcounter==self.dataOut.NAVG:

            #self.dataOut.flagNoData =  False # new 2
            self.dataOut.flag_save=1
            #self.dataOut.kax=None


            self.dataOut.noise_final=numpy.zeros(self.dataOut.NR,'float32')  #30/03/2020
            #self.dataOut.noise_final=numpy.zeros(2,'float32')  #31/03/2020

            #print("self.dataOut.nChannels: ",self.dataOut.systemHeaderObj.nChannels)
            self.kax=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kay=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kbx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kax2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kay2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kbx2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kby2=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kaxbx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kaxby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kaybx=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kayby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kaxay=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            self.kbxby=numpy.zeros((self.dataOut.NDP,self.dataOut.DPL,2),'float32')
            #print("Shape K",numpy.shape(self.kax))
            for i in range(self.cax_navg[0].shape[0]):
                        for j in range(self.cax_navg[0].shape[1]):
                            for k in range(self.cax_navg[0].shape[2]):
                                data_navg=[item[i,j,k] for item in self.cax_navg]
                                self.kax[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cay_navg]
                                self.kay[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cbx_navg]
                                self.kbx[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cby_navg]
                                self.kby[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cax2_navg]
                                self.kax2[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cay2_navg]
                                self.kay2[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cbx2_navg]
                                self.kbx2[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cby2_navg]
                                self.kby2[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.caxbx_navg]
                                self.kaxbx[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.caxby_navg]
                                self.kaxby[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.caybx_navg]
                                self.kaybx[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cayby_navg]
                                self.kayby[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.caxay_navg]
                                self.kaxay[i,j,k]=self.medi(data_navg)
                                data_navg=[item[i,j,k] for item in self.cbxby_navg]
                                self.kbxby[i,j,k]=self.medi(data_navg)
            #self.bcounter=0
            #print("KAX",self.kax)
            #self.__buffer=self.kax
            #print("CurrentBlock: ", self.dataOut.CurrentBlock)

            self.dataOut.kax=self.kax
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
            self.bcounter=0

            #print("before: ",self.dataOut.noise_final)

            self.noise_estimation4x()  #30/03/2020

            #print("after: ", self.dataOut.noise_final)
            #print(numpy.shape(self.dataOut.data))
            #input()
            #self.dataOut.noise_final_list.append(self.dataOut.noise_final[0])  #30/03/2020


            '''
            print("hsize[0] ",self.dataOut.header[0])
            print("bufsize[1] ",self.dataOut.header[1])
            print("nr[2] ",self.dataOut.header[2])
            print("ngates[3] ",self.dataOut.header[3])
            print("time1[4] ",self.dataOut.header[4])
            print("time2[5] ",self.dataOut.header[5])
            print("lcounter[6] ",self.dataOut.header[6])
            print("groups[7] ",self.dataOut.header[7])
            print("system[8] ",self.dataOut.header[8])
            print("h0[9] ",self.dataOut.header[9])
            print("dh[10] ",self.dataOut.header[10])
            print("ipp[11] ",self.dataOut.header[11])
            print("process[12] ",self.dataOut.header[12])
            print("tx[13] ",self.dataOut.header[13])
            print("padding[14] ",self.dataOut.header[14])
            print("ngates1[15] ",self.dataOut.header[15])
            print("header[16] ",self.dataOut.header[16])
            print("header[17] ",self.dataOut.header[17])
            print("header[18] ",self.dataOut.header[18])
            print("header[19] ",self.dataOut.header[19])
            print("header[20] ",self.dataOut.header[20])
            print("header[21] ",self.dataOut.header[21])
            print("header[22] ",self.dataOut.header[22])
            print("header[23] ",self.dataOut.header[23])
            print("header[24] ",self.dataOut.header[24])
            print("header[25] ",self.dataOut.header[25])
            print("header[26] ",self.dataOut.header[26])
            print("header[27] ",self.dataOut.header[27])
            print("header[28] ",self.dataOut.header[28])
            print("header[29] ",self.dataOut.header[29])
            print("header[30] ",self.dataOut.header[30])
            print("header[31] ",self.dataOut.header[31])
            '''



            #print("CurrentBlock: ",self.dataOut.CurrentBlock)
            ##print("KAX: ",self.dataOut.kax)


            '''
            plt.plot(self.kaxby[:,0,0],self.range(),'m',linewidth=2.0)
            plt.xlim(min(self.kaxby[12::,0,0]), max(self.kaxby[12::,0,0]))
            plt.show()
            '''





            #/#/#/#print("CurrentBlock: ",self.dataOut.CurrentBlock)
            ####self.newdataOut=self.kax
            #print("shapedataout",numpy.shape(self.dataOut.data))
            #print("kax",numpy.shape(self.kax))
##        return 1


    ####def NewData(self):
        ####print("NewData",self.dataOut.kaxby)
        ####print("CurrentBlock: ",self.dataOut.CurrentBlock)


    '''
    def PlotVoltageLag(self):

        plt.plot(self.dataOut.data[:,0,0],self.range(),'m',linewidth=2.0)
        plt.xlim(min(self.dataOut.data[12::,0,0]), max(self.dataOut.data[12::,0,0]))
        plt.show()


        if self.bcounter==self.NAVG:
           #print("shapedataout",self.dataOut.data)
           print("CurrentBlock: ",self.dataOut.CurrentBlock)
           self.bcounter=0
            '''

        #print("Newdataout",self.dataOut.data)
##


    #30/03/2020:
    def noisevectorizer(self,nptsfft1,nptsfft2):

        rnormalizer= 1./float(nptsfft2 - nptsfft1)
        for i in range(self.dataOut.NR):
            for j in range(self.dataOut.read_samples):
                for k in range(nptsfft1,nptsfft2):
                    #TODO:integrate just 2nd quartile gates
                    if k==nptsfft1:
                        self.dataOut.noisevector[j][i][self.bcounter]=(abs(self.dataOut.data[i][k][j]-self.dataOut.dc[i])**2)*rnormalizer
                        ##noisevector[j][i][iavg]=(abs(cdata[k][j][i])**2)*rnormalizer
                    else:
                        self.dataOut.noisevector[j][i][self.bcounter]+=(abs(self.dataOut.data[i][k][j]-self.dataOut.dc[i])**2)*rnormalizer

    #30/03/2020:
    def noise_estimation4x(self):
        snoise=numpy.zeros((self.dataOut.NR,self.dataOut.NAVG),'float32')
        nvector1=numpy.zeros((self.dataOut.NR,self.dataOut.NAVG,self.dataOut.read_samples),'float32')
        for i in range(self.dataOut.NR):
            self.dataOut.noise_final[i]=0.0
            for k in range(self.dataOut.NAVG):
                snoise[i][k]=0.0
                for j in range(self.dataOut.read_samples):
                    nvector1[i][k][j]= self.dataOut.noisevector[j][i][k];
                snoise[i][k]=self.noise_hs4x(self.dataOut.read_samples, nvector1[i][k])
            self.dataOut.noise_final[i]=self.noise_hs4x(self.dataOut.NAVG, snoise[i])


    #30/03/2020:
    def  noise_hs4x(self, ndatax, datax):
        #print("datax ",datax)
        divider=10#divider was originally 10
        noise=0.0
        data=numpy.zeros(ndatax,'float32')
        ndata1=int(ndatax/4)
        ndata2=int(2.5*(ndatax/4.))
        ndata=int(ndata2-ndata1)
        sorts=sorted(datax)
        for k in range(ndata2): # select just second quartile
            data[k]=sorts[k+ndata1]
        nums_min= int(ndata/divider)
        if(int(ndata/divider)> 2):
            nums_min= int(ndata/divider)
        else:
            nums_min=2
        sump=0.0
        sumq=0.0
        j=0
        cont=1
        while ( (cont==1) and (j<ndata)):
            sump+=data[j]
            sumq+= data[j]*data[j]
            j=j+1
            if (j> nums_min):
                rtest= float(j/(j-1)) +1.0/ndata
                if( (sumq*j) > (rtest*sump*sump ) ):
                    j=j-1
                    sump-= data[j]
                    sumq-=data[j]*data[j]
                    cont= 0
        noise= (sump/j)

        return noise



    def test(self):

        #print("LP_init")
        #self.dataOut.flagNoData=1
        buffer=self.dataOut.data
        #self.dataOut.flagNoData=0
        if self.LP_products_aux==0:

            #self.dataOut.nptsfft2=150
            self.cnorm=float((self.dataOut.nptsfft2LP-self.dataOut.NSCAN)/self.dataOut.NSCAN)


            #print("self.bcounter",self.bcounter)
            self.lagp0=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp1=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp2=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp3=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp4=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.LP_products_aux=1

        #print(self.dataOut.data[0,0,0])
        #self.dataOut.flagNoData =False
        for i in range(self.dataOut.NR-1):
            #print("inside i",i)
            buffer_dc=self.dataOut.dc[i]
            for j in range(self.dataOut.NRANGE):
                #print("inside j",j)
                #print(self.dataOut.read_samples)
                #input()
                range_for_n=numpy.min((self.dataOut.NRANGE-j,self.dataOut.NLAG))
                for k in range(self.dataOut.nptsfft2LP):
                    #print(self.dataOut.data[i][k][j])
                    #input()
                    #print(self.dataOut.dc)
                    #input()
                    #aux_ac=0
                    buffer_aux=numpy.conj(buffer[i][k][j]-buffer_dc)
                    #self.dataOut.flagNoData=0
                    for n in range(range_for_n):


                    #for n in range(numpy.min((self.dataOut.NRANGE-j,self.dataOut.NLAG))):
                        #print(numpy.shape(self.dataOut.data))
                        #input()
                        #pass
                        #self.dataOut.flagNoData=1
                        #c=2*buffer_aux
                        #c=(self.dataOut.data[i][k][j]-self.dataOut.dc[i])*(numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))
                        #c=(buffer[i][k][j]-buffer_dc)*(numpy.conj(buffer[i][k][j+n])-buffer_dc)

                        c=(buffer_aux)*(buffer[i][k][j+n]-buffer_dc)
                        #c=(buffer[i][k][j])*(buffer[i][k][j+n])
                        #print("first: ",self.dataOut.data[i][k][j]-self.dataOut.dc[i])
                        #print("second: ",numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))

                        #print("c: ",c)
                        #input()
                        #print("n: ",n)
                        #print("aux_ac",aux_ac)
                        #print("data1:",self.dataOut.data[i][k][j])
                        #print("data2:",self.dataOut.data[i][k][j+n])
                        #print("dc: ",self.dataOut.dc[i])
                        #if aux_ac==2:
                            #input()
                        #aux_ac+=1
                        #print("GG")
                        #print("inside n",n)
                        #pass

                        if k<self.dataOut.NSCAN:
                            if k==0:

                                while True:
                                    if i==0:
                                        self.lagp0[n][j][self.bcounter-1]=c
                                        break
                                    elif i==1:
                                        self.lagp1[n][j][self.bcounter-1]=c
                                        break
                                    elif i==2:
                                        self.lagp2[n][j][self.bcounter-1]=c
                                        break
                                    else:
                                        break

                            else:

                                while True:
                                    if i==0:
                                        self.lagp0[n][j][self.bcounter-1]=c+self.lagp0[n][j][self.bcounter-1]
                                        break
                                    elif i==1:
                                        self.lagp1[n][j][self.bcounter-1]=c+self.lagp1[n][j][self.bcounter-1]
                                        break
                                    elif i==2:
                                        self.lagp2[n][j][self.bcounter-1]=c+self.lagp2[n][j][self.bcounter-1]
                                        break
                                    else:
                                        break

                        else:
                            #c=c/self.cnorm
                            if i==0:
                                c=c/self.cnorm
                                if k==self.dataOut.NSCAN:
                                    #if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c
                                        #print("n: ",n,"j: ",j,"iavg: ",self.bcounter-1)
                                        #print("lagp3_inside: ",self.lagp3[n][j][self.bcounter-1])
                                else:
                                    #if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c+self.lagp3[n][j][self.bcounter-1]




        #print("lagp2: ",self.lagp2[:,0,0])
        self.lagp0[:,:,self.bcounter-1]=numpy.conj(self.lagp0[:,:,self.bcounter-1])
        self.lagp1[:,:,self.bcounter-1]=numpy.conj(self.lagp1[:,:,self.bcounter-1])
        self.lagp2[:,:,self.bcounter-1]=numpy.conj(self.lagp2[:,:,self.bcounter-1])
        self.lagp3[:,:,self.bcounter-1]=numpy.conj(self.lagp3[:,:,self.bcounter-1])
        #self.dataOut.flagNoData=0
        #print(self.bcounter-1)
        #print("lagp2_conj: ",self.lagp2[:,0,self.bcounter-1])
        #input()
        #self.dataOut.lagp3=self.lagp3
        print("TEST")



    def lag_products_LP(self):

        #print("LP_init")
        #self.dataOut.flagNoData=1
        buffer=self.dataOut.data
        #self.dataOut.flagNoData=0
        if self.LP_products_aux==0:

            #self.dataOut.nptsfft2=150
            self.cnorm=float((self.dataOut.nptsfft2LP-self.dataOut.NSCAN)/self.dataOut.NSCAN)


            #print("self.bcounter",self.bcounter)
            self.lagp0=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp1=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp2=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp3=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp4=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.LP_products_aux=1

        #print(self.dataOut.data[0,0,0])
        #self.dataOut.flagNoData =False
        for i in range(self.dataOut.NR):
            #print("inside i",i)
            buffer_dc=self.dataOut.dc[i]
            for j in range(self.dataOut.NRANGE):
                #print("inside j",j)
                #print(self.dataOut.read_samples)
                #input()
                range_for_n=numpy.min((self.dataOut.NRANGE-j,self.dataOut.NLAG))
                for k in range(self.dataOut.nptsfft2LP):
                    #print(self.dataOut.data[i][k][j])
                    #input()
                    #print(self.dataOut.dc)
                    #input()
                    #aux_ac=0
                    buffer_aux=numpy.conj(buffer[i][k][j]-buffer_dc)
                    #self.dataOut.flagNoData=0
                    for n in range(range_for_n):
                    #for n in range(numpy.min((self.dataOut.NRANGE-j,self.dataOut.NLAG))):
                        #print(numpy.shape(self.dataOut.data))
                        #input()
                        #pass
                        #self.dataOut.flagNoData=1
                        #c=2*buffer_aux
                        #c=(self.dataOut.data[i][k][j]-self.dataOut.dc[i])*(numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))
                        #c=(buffer[i][k][j]-buffer_dc)*(numpy.conj(buffer[i][k][j+n])-buffer_dc)

                        c=(buffer_aux)*(buffer[i][k][j+n]-buffer_dc)
                        #c=(buffer[i][k][j])*(buffer[i][k][j+n])
                        #print("first: ",self.dataOut.data[i][k][j]-self.dataOut.dc[i])
                        #print("second: ",numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))

                        #print("c: ",c)
                        #input()
                        #print("n: ",n)
                        #print("aux_ac",aux_ac)
                        #print("data1:",self.dataOut.data[i][k][j])
                        #print("data2:",self.dataOut.data[i][k][j+n])
                        #print("dc: ",self.dataOut.dc[i])
                        #if aux_ac==2:
                            #input()
                        #aux_ac+=1
                        #print("GG")
                        #print("inside n",n)
                        #pass

                        if k<self.dataOut.NSCAN:
                            if k==0:
                                if i==0:
                                    self.lagp0[n][j][self.bcounter-1]=c
                                elif i==1:
                                    self.lagp1[n][j][self.bcounter-1]=c
                                elif i==2:
                                    self.lagp2[n][j][self.bcounter-1]=c
                            else:
                                if i==0:
                                    self.lagp0[n][j][self.bcounter-1]=c+self.lagp0[n][j][self.bcounter-1]
                                elif i==1:
                                    self.lagp1[n][j][self.bcounter-1]=c+self.lagp1[n][j][self.bcounter-1]
                                elif i==2:
                                    self.lagp2[n][j][self.bcounter-1]=c+self.lagp2[n][j][self.bcounter-1]

                        else:
                            c=c/self.cnorm
                            if k==self.dataOut.NSCAN:
                                if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c
                                    #print("n: ",n,"j: ",j,"iavg: ",self.bcounter-1)
                                    #print("lagp3_inside: ",self.lagp3[n][j][self.bcounter-1])
                            else:
                                if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c+self.lagp3[n][j][self.bcounter-1]



        #print("lagp2: ",self.lagp2[:,0,0])
        self.lagp0[:,:,self.bcounter-1]=numpy.conj(self.lagp0[:,:,self.bcounter-1])
        self.lagp1[:,:,self.bcounter-1]=numpy.conj(self.lagp1[:,:,self.bcounter-1])
        self.lagp2[:,:,self.bcounter-1]=numpy.conj(self.lagp2[:,:,self.bcounter-1])
        self.lagp3[:,:,self.bcounter-1]=numpy.conj(self.lagp3[:,:,self.bcounter-1])
        #self.dataOut.flagNoData=0
        #print(self.bcounter-1)
        #print("lagp2_conj: ",self.lagp2[:,0,self.bcounter-1])
        #input()
        #self.dataOut.lagp3=self.lagp3
        print("LP")


    def test_2(self):

        #print("LP_init")
        #self.dataOut.flagNoData=1

        #self.dataOut.flagNoData=0
        if self.LP_products_aux==0:

            #self.dataOut.nptsfft2=150
            self.cnorm=float((self.dataOut.nptsfft2LP-self.dataOut.NSCAN)/self.dataOut.NSCAN)


            #print("self.bcounter",self.bcounter)
            self.lagp0=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp1=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp2=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp3=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.lagp4=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NAVG),'complex64')
            self.LP_products_aux=1

        #print(self.dataOut.data[0,0,0])
        #self.dataOut.flagNoData =False
        for i in range(self.dataOut.NR):
            #print("inside i",i)

            for j in range(self.dataOut.NRANGE):
                #print("inside j",j)
                #print(self.dataOut.read_samples)
                #input()

                for k in range(self.dataOut.nptsfft2LP):
                    #print(self.dataOut.data[i][k][j])
                    #input()
                    #print(self.dataOut.dc)
                    #input()
                    #aux_ac=0

                    #self.dataOut.flagNoData=0

                    for n in range(numpy.min((self.dataOut.NRANGE-j,self.dataOut.NLAG))):
                        #print(numpy.shape(self.dataOut.data))
                        #input()
                        #pass
                        #self.dataOut.flagNoData=1
                        #c=2*buffer_aux
                        c=(self.dataOut.data[i][k][j]-self.dataOut.dc[i])*(numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))
                        #c=(buffer[i][k][j]-buffer_dc)*(numpy.conj(buffer[i][k][j+n])-buffer_dc)


                        #c=(buffer[i][k][j])*(buffer[i][k][j+n])
                        #print("first: ",self.dataOut.data[i][k][j]-self.dataOut.dc[i])
                        #print("second: ",numpy.conj(self.dataOut.data[i][k][j+n]-self.dataOut.dc[i]))

                        #print("c: ",c)
                        #input()
                        #print("n: ",n)
                        #print("aux_ac",aux_ac)
                        #print("data1:",self.dataOut.data[i][k][j])
                        #print("data2:",self.dataOut.data[i][k][j+n])
                        #print("dc: ",self.dataOut.dc[i])
                        #if aux_ac==2:
                            #input()
                        #aux_ac+=1
                        #print("GG")
                        #print("inside n",n)
                        #pass

                        if k<self.dataOut.NSCAN:
                            if k==0:
                                if i==0:
                                    self.lagp0[n][j][self.bcounter-1]=c
                                elif i==1:
                                    self.lagp1[n][j][self.bcounter-1]=c
                                elif i==2:
                                    self.lagp2[n][j][self.bcounter-1]=c
                            else:
                                if i==0:
                                    self.lagp0[n][j][self.bcounter-1]=c+self.lagp0[n][j][self.bcounter-1]
                                elif i==1:
                                    self.lagp1[n][j][self.bcounter-1]=c+self.lagp1[n][j][self.bcounter-1]
                                elif i==2:
                                    self.lagp2[n][j][self.bcounter-1]=c+self.lagp2[n][j][self.bcounter-1]

                        else:
                            c=c/self.cnorm
                            if k==self.dataOut.NSCAN:
                                if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c
                                    #print("n: ",n,"j: ",j,"iavg: ",self.bcounter-1)
                                    #print("lagp3_inside: ",self.lagp3[n][j][self.bcounter-1])
                            else:
                                if i==0:
                                    self.lagp3[n][j][self.bcounter-1]=c+self.lagp3[n][j][self.bcounter-1]



        #print("lagp2: ",self.lagp2[:,0,0])

        #self.dataOut.flagNoData=0
        #print(self.bcounter-1)
        #print("lagp2_conj: ",self.lagp2[:,0,self.bcounter-1])
        #input()
        #self.dataOut.lagp3=self.lagp3
        print("LP")


    def LP_median_estimates(self):
        #print("lagp3: ",self.lagp3[:,0,0])
        #print("self.bcounter: ",self.bcounter)
        if self.dataOut.flag_save==1:

            #print("lagp1: ",self.lagp1[0,0,:])
            #input()

            if self.lag_products_LP_median_estimates_aux==0:
                self.output=numpy.zeros((self.dataOut.NLAG,self.dataOut.NRANGE,self.dataOut.NR),'complex64')
                #sorts=numpy.zeros(128,'float32')
                #self.dataOut.output_LP=None
                self.lag_products_LP_median_estimates_aux=1


            for i in range(self.dataOut.NLAG):
                for j in range(self.dataOut.NRANGE):
                    for l in range(4): #four outputs
                        '''
                        for k in range(self.dataOut.NAVG):
                            #rsorts[k]=float(k)
                            if l==0:
                                #sorts[k]=self.lagp0[i,j,k].real
                                self.lagp0[i,j,k].real=sorted(self.lagp0[i,j,k].real)
                            if l==1:
                                #sorts[k]=self.lagp1[i,j,k].real
                                self.lagp1[i,j,k].real=sorted(self.lagp1[i,j,k].real)
                            if l==2:
                                #sorts[k]=self.lagp2[i,j,k].real
                                self.lagp2[i,j,k].real=sorted(self.lagp2[i,j,k].real)
                            if l==3:
                                #sorts[k]=self.lagp3[i,j,k].real
                                self.lagp3[i,j,k].real=sorted(self.lagp3[i,j,k].real)
                                '''

                    #sorts=sorted(sorts)
                            #self.lagp0[i,j,k].real=sorted(self.lagp0[i,j,k].real)
                            #self.lagp1[i,j,k].real=sorted(self.lagp1[i,j,k].real)
                            #self.lagp2[i,j,k].real=sorted(self.lagp2[i,j,k].real)
                            #self.lagp3[i,j,k].real=sorted(self.lagp3[i,j,k].real)

                        for k in range(self.dataOut.NAVG):



                            if k==0:
                                self.output[i,j,l]=0.0+0.j

                                if l==0:
                                    self.lagp0[i,j,:]=sorted(self.lagp0[i,j,:], key=lambda x: x.real)  #sorted(self.lagp0[i,j,:].real)

                                if l==1:
                                    self.lagp1[i,j,:]=sorted(self.lagp1[i,j,:], key=lambda x: x.real)  #sorted(self.lagp1[i,j,:].real)
                                if l==2:
                                    self.lagp2[i,j,:]=sorted(self.lagp2[i,j,:], key=lambda x: x.real)  #sorted(self.lagp2[i,j,:].real)
                                if l==3:
                                    self.lagp3[i,j,:]=sorted(self.lagp3[i,j,:], key=lambda x: x.real)  #sorted(self.lagp3[i,j,:].real)


                            if k>=self.dataOut.nkill/2 and k<self.dataOut.NAVG-self.dataOut.nkill/2:
                                if l==0:

                                    self.output[i,j,l]=self.output[i,j,l]+((float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill))*self.lagp0[i,j,k])
                                if l==1:
                                    #print("lagp1: ",self.lagp1[0,0,:])
                                    #input()
                                    self.output[i,j,l]=self.output[i,j,l]+((float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill))*self.lagp1[i,j,k])
                                    #print("self.lagp1[i,j,k]: ",self.lagp1[i,j,k])
                                    #input()
                                if l==2:
                                    self.output[i,j,l]=self.output[i,j,l]+((float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill))*self.lagp2[i,j,k])
                                if l==3:
                                    #print(numpy.shape(output))
                                    #print(numpy.shape(self.lagp3))
                                    #print("i: ",i,"j: ",j,"k: ",k)

                                    #a=((float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill))*self.lagp3[i,j,k])
                                    #print("self.lagp3[i,j,k]: ",self.lagp3[i,j,k])
                                    #input()
                                    self.output[i,j,l]=self.output[i,j,l]+((float(self.dataOut.NAVG)/(float)(self.dataOut.NAVG-self.dataOut.nkill))*self.lagp3[i,j,k])
                                    #print(a)
                                    #print("output[i,j,l]: ",output[i,j,l])
                                    #input()


            self.dataOut.output_LP=self.output
            #print(numpy.shape(sefl.dataOut.output_LP))
            #input()
            #print("output: ",self.dataOut.output_LP[:,0,0])
            #input()


    def remove_debris_LP(self):

        if self.dataOut.flag_save==1:
            debris=numpy.zeros(self.dataOut.NRANGE,'float32')
            #self.dataOut.debris_activated=0
            for j in range(0,3):
                for i in range(self.dataOut.NRANGE):
                    if j==0:
                        debris[i]=10*numpy.log10(numpy.abs(self.dataOut.output_LP[j,i,0]))
                    else:
                        debris[i]+=10*numpy.log10(numpy.abs(self.dataOut.output_LP[j,i,0]))

            '''
            debris=10*numpy.log10(numpy.abs(self.dataOut.output_LP[0,:,0]))

            for j in range(1,3):
                for i in range(self.dataOut.NRANGE):
                    debris[i]+=debris[i]
                    '''

            thresh=8.0+4+4+4
            for i in range(47,100):
                if ((debris[i-2]+debris[i-1]+debris[i]+debris[i+1])>
                    ((debris[i-12]+debris[i-11]+debris[i-10]+debris[i-9]+
                    debris[i+12]+debris[i+11]+debris[i+10]+debris[i+9])/2.0+
                    thresh)):

                    self.dataOut.debris_activated=1
                    #print("LP debris",i)


            #print("self.debris",debris)


    def remove_debris_DP(self):

        if self.dataOut.flag_save==1:
            debris=numpy.zeros(self.dataOut.NDP,dtype='float32')
            Range=numpy.arange(0,3000,15)
            for k in range(2): #flip
                for i in range(self.dataOut.NDP): #
                    debris[i]+=numpy.sqrt((self.dataOut.kaxbx[i,0,k]+self.dataOut.kayby[i,0,k])**2+(self.dataOut.kaybx[i,0,k]-self.dataOut.kaxby[i,0,k])**2)

            #print("debris: ",debris)

            if time.gmtime(self.dataOut.utctime).tm_hour > 11:
                for i in range(2,self.dataOut.NDP-2):
                    if (debris[i]>3.0*debris[i-2] and
                        debris[i]>3.0*debris[i+2] and
                        Range[i]>200.0 and Range[i]<=540.0):

                        self.dataOut.debris_activated=1
                        #print("DP debris")






    def run(self, experiment="", nlags_array=None, NLAG=None, NR=None, NRANGE=None, NCAL=None, DPL=None,
            NDN=None, NDT=None, NDP=None, NLP=None, NSCAN=None, HDR_SIZE=None, DH=15, H0=None, LPMASK=None,
            flags_array=None,
            NPROFILE1=None, NPROFILE2=None, NPROFILES=None, NPROFILE=None,
            lagind=None, lagfirst=None,
            nptsfftx1=None):

        #self.dataOut.input_dat_type=input_dat_type

        self.dataOut.experiment=experiment

        #print(self.dataOut.experiment)
        self.dataOut.nlags_array=nlags_array
        self.dataOut.NLAG=NLAG
        self.dataOut.NR=NR
        self.dataOut.NRANGE=NRANGE
        #print(self.dataOut.NRANGE)
        self.dataOut.NCAL=NCAL
        self.dataOut.DPL=DPL
        self.dataOut.NDN=NDN
        self.dataOut.NDT=NDT
        self.dataOut.NDP=NDP
        self.dataOut.NLP=NLP
        self.dataOut.NSCAN=NSCAN
        self.dataOut.HDR_SIZE=HDR_SIZE
        self.dataOut.DH=float(DH)
        self.dataOut.H0=H0
        self.dataOut.LPMASK=LPMASK
        self.dataOut.flags_array=flags_array

        self.dataOut.NPROFILE1=NPROFILE1
        self.dataOut.NPROFILE2=NPROFILE2
        self.dataOut.NPROFILES=NPROFILES
        self.dataOut.NPROFILE=NPROFILE
        self.dataOut.lagind=lagind
        self.dataOut.lagfirst=lagfirst
        self.dataOut.nptsfftx1=nptsfftx1


        self.dataOut.copy(self.dataIn)
        #print(self.dataOut.datatime)
        #print(self.dataOut.ippSeconds_general)
        #print("Data: ",numpy.shape(self.dataOut.data))
        #print("Data_after: ",self.dataOut.data[0,0,1])
                                ## (4, 150, 334)
        #print(self.dataOut.channelIndexList)

        #print(self.dataOut.timeInterval)

        ###NEWWWWWWW
        self.dataOut.lat=-11.95
        self.dataOut.lon=-7687
        self.dataOut.debris_activated=0

        #print(time.gmtime(self.dataOut.utctime).tm_hour)
        #print(numpy.shape(self.dataOut.heightList))



class NewData(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)





    def run(self,dataOut):

        #print("SHAPE",numpy.shape(dataOut.kaxby))
        print("CurrentBlock",dataOut.CurrentBlock)
        #print("DATAOUT",dataOut.kaxby)
        #print("TRUE OR FALSE",numpy.shape(dataOut.kaxby)==())
        #print("SHAPE",numpy.shape(dataOut.kaxby))
        if numpy.shape(dataOutF.kax)!=():    ############VER SI SE PUEDE TRABAJAR CON dataOut.kaxby==None  ##Puede ser cualquier k...

            print("NEWDATA",dataOut.kaxby)





        return dataOut











'''

class PlotVoltageLag(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)



self.kax=numpy.zeros((self.NDP,self.nlags_array,2),'float32')
    def range(self,DH):
        Range=numpy.arange(0,990,DH)
        return Range



    def run(self,dataOut):



        #plt.subplot(1, 4, 1)
        plt.plot(kax[:,0,0],Range,'r',linewidth=2.0)
        plt.xlim(min(limit_min_plot1[12::,0,0]), max(limit_max_plot1[12::,0,0]))
        plt.show()

self.kax=numpy.zeros((self.NDP,self.nlags_array,2),'float32')

        return dataOut
'''





























class Integration(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)



        self.counter=0
        self.aux=0
        self.aux2=1

    def run(self,dataOut,nint=None):

        dataOut.nint=nint
        dataOut.AUX=0
        dataOut.paramInterval=dataOut.nint*dataOut.header[7][0]*2   #GENERALIZAR EL 2
        #print("CurrentBlock: ",dataOut.CurrentBlock)
        #print("date: ",dataOut.datatime)
        #print("self.aux: ",self.aux)
        #print("CurrentBlockAAAAAA: ",dataOut.CurrentBlock)
        #print(dataOut.input_dat_type)
        #print(dataOut.heightList)

        #print(dataOut.blocktime.ctime())
        '''
        if  dataOut.input_dat_type: #when .dat data is read
            #print(dataOut.realtime)
            #print("OKODOKO")
            #dataOut.flagNoData = False
            #print(dataOut.flagNoData)
            if self.aux2:

                self.noise=numpy.zeros(dataOut.NR,'float32')


                padding=numpy.zeros(1,'int32')

                hsize=numpy.zeros(1,'int32')
                bufsize=numpy.zeros(1,'int32')
                nr=numpy.zeros(1,'int32')
                ngates=numpy.zeros(1,'int32') ###  ###  ### 2
                time1=numpy.zeros(1,'uint64') # pos 3
                time2=numpy.zeros(1,'uint64') # pos 4
                lcounter=numpy.zeros(1,'int32')
                groups=numpy.zeros(1,'int32')
                system=numpy.zeros(4,'int8') # pos 7
                h0=numpy.zeros(1,'float32')
                dh=numpy.zeros(1,'float32')
                ipp=numpy.zeros(1,'float32')
                process=numpy.zeros(1,'int32')
                tx=numpy.zeros(1,'int32')

                ngates1=numpy.zeros(1,'int32')  ###  ###  ### 13
                time0=numpy.zeros(1,'uint64') # pos 14
                nlags=numpy.zeros(1,'int32')
                nlags1=numpy.zeros(1,'int32')
                txb=numpy.zeros(1,'float32')   ###  ###  ### 17
                time3=numpy.zeros(1,'uint64') # pos 18
                time4=numpy.zeros(1,'uint64') # pos 19
                h0_=numpy.zeros(1,'float32')
                dh_=numpy.zeros(1,'float32')
                ipp_=numpy.zeros(1,'float32')
                txa_=numpy.zeros(1,'float32')

                pad=numpy.zeros(100,'int32')

                nbytes=numpy.zeros(1,'int32')
                limits=numpy.zeros(1,'int32')
                ngroups=numpy.zeros(1,'int32') ###  ###  ### 27
                #Make the header list
                #header=[hsize,bufsize,nr,ngates,time1,time2,lcounter,groups,system,h0,dh,ipp,process,tx,padding,ngates1,time0,nlags,nlags1,padding,txb,time3,time4,h0_,dh_,ipp_,txa_,pad,nbytes,limits,padding,ngroups]
                dataOut.header=[hsize,bufsize,nr,ngates,time1,time2,lcounter,groups,system,h0,dh,ipp,process,tx,ngates1,padding,time0,nlags,nlags1,padding,txb,time3,time4,h0_,dh_,ipp_,txa_,pad,nbytes,limits,padding,ngroups]



                dataOut.kax=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kay=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbx=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kby=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kax2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kay2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbx2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kby2=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxbx=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxby=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaybx=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kayby=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxay=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbxby=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')

                self.dataOut.final_cross_products=[dataOut.kax,dataOut.kay,dataOut.kbx,dataOut.kby,dataOut.kax2,dataOut.kay2,dataOut.kbx2,dataOut.kby2,dataOut.kaxbx,dataOut.kaxby,dataOut.kaybx,dataOut.kayby,dataOut.kaxay,dataOut.kbxby]

                self.inputfile_DP = open(dataOut.fname,"rb")

                ## read header the header first time
                for i in range(len(dataOut.header)):
                    for j in range(len(dataOut.header[i])):
                        #print("len(header[i]) ",len(header[i]))
                        #input()
                        temp=self.inputfile_DP.read(int(dataOut.header[i].itemsize))
                        if isinstance(dataOut.header[i][0], numpy.int32):
                            #print(struct.unpack('i', temp)[0])
                            dataOut.header[i][0]=struct.unpack('i', temp)[0]
                        if isinstance(dataOut.header[i][0], numpy.uint64):
                            dataOut.header[i][0]=struct.unpack('q', temp)[0]
                        if isinstance(dataOut.header[i][0], numpy.int8):
                            dataOut.header[i][0]=struct.unpack('B', temp)[0]
                        if isinstance(dataOut.header[i][0], numpy.float32):
                            dataOut.header[i][0]=struct.unpack('f', temp)[0]




                self.activator_No_Data=1

                self.inputfile_DP.seek(0,0)

                #print("Repositioning to",self.npos," bytes, bufsize ", self.header[1][0])
                #self.inputfile.seek(self.npos, 0)
                #print("inputfile.tell() ",self.inputfile.tell() ,"   npos : ", self.npos)

                self.npos=0

                #if dataOut.nint < 0:
                #    dataOut.nint=-dataOut.nint
                #    sfile=os.stat(dataOut.fname)
                #    if (os.path.exists(dataOut.fname)==0):
                #        print("ERROR on STAT file: %s\n", dataOut.fname)
                #    self.npos=sfile.st_size - dataOut.nint*dataOut.header[1][0]# sfile.st_size - nint*header.bufsize

                self.start_another_day=False
                if dataOut.new_time_date!=" ":
                    self.start_another_day=True


                if self.start_another_day:
                    #print("Starting_at_another_day")
                    #new_time_date = "16/08/2013 09:51:43"
                    #new_time_seconds=time.mktime(time.strptime(new_time_date))
                    #dataOut.new_time_date = "04/12/2019 09:21:21"
                    d = datetime.strptime(dataOut.new_time_date, "%d/%m/%Y %H:%M:%S")
                    new_time_seconds=time.mktime(d.timetuple())

                    d_2 = datetime.strptime(dataOut.new_ending_time, "%d/%m/%Y %H:%M:%S")
                    self.new_ending_time_seconds=time.mktime(d_2.timetuple())
                    #print("new_time_seconds: ",new_time_seconds)
                    #input()
                    jumper=0

                    #if jumper>0 and nint>0:
                    while True:
                        sfile=os.stat(dataOut.fname)

                        if (os.path.exists(dataOut.fname)==0):
                            print("ERROR on STAT file: %s\n",dataOut.fname)
                        self.npos=jumper*dataOut.nint*dataOut.header[1][0] #jump_blocks*header,bufsize
                        self.npos_next=(jumper+1)*dataOut.nint*dataOut.header[1][0]
                        self.inputfile_DP.seek(self.npos, 0)
                        jumper+=1
                        for i in range(len(dataOut.header)):
                            for j in range(len(dataOut.header[i])):
                                #print("len(header[i]) ",len(header[i]))
                                #input()
                                temp=self.inputfile_DP.read(int(dataOut.header[i].itemsize))
                                if isinstance(dataOut.header[i][0], numpy.int32):
                                    #print(struct.unpack('i', temp)[0])
                                    dataOut.header[i][0]=struct.unpack('i', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.uint64):
                                    dataOut.header[i][0]=struct.unpack('q', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.int8):
                                    dataOut.header[i][0]=struct.unpack('B', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.float32):
                                    dataOut.header[i][0]=struct.unpack('f', temp)[0]

                        if self.npos==0:
                            if new_time_seconds<dataOut.header[4][0]:
                                break
                                #dataOut.flagNoData=1
                                #return dataOut.flagNoData

                            self.npos_aux=sfile.st_size - dataOut.nint*dataOut.header[1][0]
                            self.inputfile_DP.seek(self.npos_aux, 0)

                            for i in range(len(dataOut.header)):
                                for j in range(len(dataOut.header[i])):
                                    #print("len(header[i]) ",len(header[i]))
                                    #input()
                                    temp=self.inputfile_DP.read(int(dataOut.header[i].itemsize))
                                    if isinstance(dataOut.header[i][0], numpy.int32):
                                        #print(struct.unpack('i', temp)[0])
                                        dataOut.header[i][0]=struct.unpack('i', temp)[0]
                                    if isinstance(dataOut.header[i][0], numpy.uint64):
                                        dataOut.header[i][0]=struct.unpack('q', temp)[0]
                                    if isinstance(dataOut.header[i][0], numpy.int8):
                                        dataOut.header[i][0]=struct.unpack('B', temp)[0]
                                    if isinstance(dataOut.header[i][0], numpy.float32):
                                        dataOut.header[i][0]=struct.unpack('f', temp)[0]

                            if new_time_seconds>dataOut.header[4][0]:
                                print("No Data")
                                self.inputfile_DP.close()
                                sys.exit(1)

                            self.inputfile_DP.seek(self.npos, 0)




                        if new_time_seconds==dataOut.header[4][0]:
                            #print("EQUALS")
                            break

                        self.inputfile_DP.seek(self.npos_next, 0)

                        for i in range(len(dataOut.header)):
                            for j in range(len(dataOut.header[i])):
                                #print("len(header[i]) ",len(header[i]))
                                #input()
                                temp=self.inputfile_DP.read(int(dataOut.header[i].itemsize))
                                if isinstance(dataOut.header[i][0], numpy.int32):
                                    #print(struct.unpack('i', temp)[0])
                                    dataOut.header[i][0]=struct.unpack('i', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.uint64):
                                    dataOut.header[i][0]=struct.unpack('q', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.int8):
                                    dataOut.header[i][0]=struct.unpack('B', temp)[0]
                                if isinstance(dataOut.header[i][0], numpy.float32):
                                    dataOut.header[i][0]=struct.unpack('f', temp)[0]


                        if new_time_seconds<dataOut.header[4][0]:
                            break








                #print("Repositioning to",self.npos," bytes, bufsize ", dataOut.header[1][0])
                self.inputfile_DP.seek(self.npos, 0)
                #print("inputfile.tell() ",self.inputfile_DP.tell() ,"   npos : ", self.npos)

                self.aux2=0


            for ii in range(len(dataOut.header)):
                    for j in range(len(dataOut.header[ii])):
                        temp=self.inputfile_DP.read(int(dataOut.header[ii].itemsize))

                        if(b''==temp):# sizeof(header)
                            dataOut.flagDiscontinuousBlock=1
                            #print("EOF \n\n\n\n")
                            #log.success("")
                            #self.inputfile_DP.close()
                            dataOut.error = True
                            #dataOut.flagNoData = True
                            #dataOut.stop=True
                            #return dataOut
                            #dataOut.
                            return dataOut

                            #return dataOut.flagNoData
                            #writedb_head()
                            #outputfile.close()
                            #sys.exit(0)
                            #THE PROGRAM SHOULD END HERE

                        if isinstance(dataOut.header[ii][0], numpy.int32):
                            #print(struct.unpack('i', temp)[0])
                            dataOut.header[ii][0]=struct.unpack('i', temp)[0]
                        if isinstance(dataOut.header[ii][0], numpy.uint64):
                            dataOut.header[ii][0]=struct.unpack('q', temp)[0]
                        if isinstance(dataOut.header[ii][0], numpy.int8):
                            dataOut.header[ii][0]=struct.unpack('B', temp)[0]
                        if isinstance(dataOut.header[ii][0], numpy.float32):
                            dataOut.header[ii][0]=struct.unpack('f', temp)[0]


            if self.start_another_day:

                    if dataOut.header[4][0]>self.new_ending_time_seconds:
                        print("EOF \n")
                        if self.activator_No_Data:
                            print("No Data")
                        self.inputfile_DP.close()
                        #sys.exit(0)
                        dataOut.error = True
                        return dataOut
            #print(self.activator_No_Data)
            self.activator_No_Data=0
            #dataOut.TimeBlockDate_for_dp_power=dataOut.TimeBlockDate
            #dataOut.TimeBlockSeconds_for_dp_power=time.mktime(time.strptime(dataOut.TimeBlockDate_for_dp_power))
            dataOut.TimeBlockSeconds_for_dp_power = dataOut.header[4][0]-((dataOut.nint-1)*dataOut.NAVG*2)
            #print(dataOut.TimeBlockSeconds_for_dp_power)
            dataOut.TimeBlockDate_for_dp_power=datetime.fromtimestamp(dataOut.TimeBlockSeconds_for_dp_power).strftime("%a %b  %-d %H:%M:%S %Y")
            #print("Date: ",dataOut.TimeBlockDate_for_dp_power)
            #print("Seconds: ",dataOut.TimeBlockSeconds_for_dp_power)
            dataOut.bd_time=time.gmtime(dataOut.TimeBlockSeconds_for_dp_power)
            dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
            dataOut.ut=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0


            if dataOut.experiment=="HP": # NRANGE*NLAG*NR # np.zeros([total_samples*nprofiles],dtype='complex64')
                    temp=self.inputfile_DP.read(dataOut.NLAG*dataOut.NR*176*8)
                    ii=0
                    for l in range(dataOut.NLAG): #lag
                        for r in range(dataOut.NR): # unflip and flip
                            for k in range(176): #RANGE## generalizar
                                struct.unpack('q', temp[ii:ii+8])[0]
                                ii=ii+8



            #print("A: ",dataOut.kax)
            for ind in range(len(self.dataOut.final_cross_products)): #final cross products
                temp=self.inputfile_DP.read(dataOut.DPL*2*dataOut.NDT*4) #*4 bytes
                ii=0
                #print("kabxys.shape ",kabxys.shape)
                #print(kabxys)
                for l in range(dataOut.DPL): #lag
                    for fl in range(2): # unflip and flip
                        for k in range(dataOut.NDT): #RANGE
                            self.dataOut.final_cross_products[ind][k,l,fl]=struct.unpack('f', temp[ii:ii+4])[0]
                            ii=ii+4
            #print("DPL*2*NDT*4 es: ", DPL*2*NDT*4)
            #print("B: ",dataOut.kax)
            ## read noise
            temp=self.inputfile_DP.read(dataOut.NR*4) #*4 bytes
            for ii in range(dataOut.NR):
                self.noise[ii]=struct.unpack('f', temp[ii*4:(ii+1)*4])[0]
            #print("NR*4 es: ", NR*4)


################################END input_dat_type################################
        '''

        #if dataOut.input_dat_type==0:

        if self.aux==1:
            #print("CurrentBlockBBBBB: ",dataOut.CurrentBlock)
            #print(dataOut.datatime)
            dataOut.TimeBlockDate_for_dp_power=dataOut.TimeBlockDate

            #print("Date: ",dataOut.TimeBlockDate_for_dp_power)
            dataOut.TimeBlockSeconds_for_dp_power=time.mktime(time.strptime(dataOut.TimeBlockDate_for_dp_power))
            #print("Seconds: ",dataOut.TimeBlockSeconds_for_dp_power)
            dataOut.bd_time=time.gmtime(dataOut.TimeBlockSeconds_for_dp_power)
            dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
            dataOut.ut=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0
            #print("date: ", dataOut.TimeBlockDate)
            self.aux=0

        if numpy.shape(dataOut.kax)!=():
            #print("SELFCOUNTER",self.counter)
            #dataOut.flagNoData =True
            if self.counter==0:
                '''
                dataOut.kax_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kay_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kax2_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kay2_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbx_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kby_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbx2_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kby2_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxbx_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxby_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaybx_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kayby_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kaxay_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                dataOut.kbxby_integrated=numpy.zeros((dataOut.NDP,dataOut.DPL,2),'float32')
                '''

                tmpx=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0],2),'float32')
                dataOut.kabxys_integrated=[tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx,tmpx]
                #self.final_cross_products=[dataOut.kax,dataOut.kay,dataOut.kbx,dataOut.kby,dataOut.kax2,dataOut.kay2,dataOut.kbx2,dataOut.kby2,dataOut.kaxbx,dataOut.kaxby,dataOut.kaybx,dataOut.kayby,dataOut.kaxay,dataOut.kbxby]

                #print(numpy.shape(tmpx))
            if self.counter < dataOut.nint:
                #if dataOut.input_dat_type==0:
                dataOut.final_cross_products=[dataOut.kax,dataOut.kay,dataOut.kbx,dataOut.kby,dataOut.kax2,dataOut.kay2,dataOut.kbx2,dataOut.kby2,dataOut.kaxbx,dataOut.kaxby,dataOut.kaybx,dataOut.kayby,dataOut.kaxay,dataOut.kbxby]

                '''
                dataOut.kax_integrated=dataOut.kax_integrated+dataOut.kax
                dataOut.kay_integrated=dataOut.kay_integrated+dataOut.kay
                dataOut.kax2_integrated=dataOut.kax2_integrated+dataOut.kax2
                dataOut.kay2_integrated=dataOut.kay2_integrated+dataOut.kay2
                dataOut.kbx_integrated=dataOut.kbx_integrated+dataOut.kbx
                dataOut.kby_integrated=dataOut.kby_integrated+dataOut.kby
                dataOut.kbx2_integrated=dataOut.kbx2_integrated+dataOut.kbx2
                dataOut.kby2_integrated=dataOut.kby2_integrated+dataOut.kby2
                dataOut.kaxbx_integrated=dataOut.kaxbx_integrated+dataOut.kaxbx
                dataOut.kaxby_integrated=dataOut.kaxby_integrated+dataOut.kaxby
                dataOut.kaybx_integrated=dataOut.kaybx_integrated+dataOut.kaybx
                dataOut.kayby_integrated=dataOut.kayby_integrated+dataOut.kayby
                dataOut.kaxay_integrated=dataOut.kaxay_integrated+dataOut.kaxbx
                dataOut.kbxby_integrated=dataOut.kbxby_integrated+dataOut.kbxby
                #print("KAX_BEFORE: ",self.kax_integrated)
                '''
                #print("self.final_cross_products[0]: ",self.final_cross_products[0])

                for ind in range(len(dataOut.kabxys_integrated)): #final cross products
                    dataOut.kabxys_integrated[ind]=dataOut.kabxys_integrated[ind]+dataOut.final_cross_products[ind]
                #print("ataOut.kabxys_integrated[0]: ",dataOut.kabxys_integrated[0])

                self.counter+=1
                if self.counter==dataOut.nint-1:
                    self.aux=1
                    #dataOut.TimeBlockDate_for_dp_power=dataOut.TimeBlockDate
                if self.counter==dataOut.nint:

                    #dataOut.flagNoData =False

                    self.counter=0
                    dataOut.AUX=1
                    #self.aux=1
                    #print("KAXBY_INTEGRATED: ",dataOut.kaxby_integrated)

            '''
            else :
                #dataOut.kax_integrated=self.kax_integrated
                self.counter=0


            #print("CurrentBlock: ", dataOut.CurrentBlock)
                print("KAX_INTEGRATED: ",self.kax_integrated)
            #print("nint: ",nint)
            '''

            ##print("CurrentBlock: ", dataOut.CurrentBlock)
            ##print("KAX_INTEGRATED: ",dataOut.kax_integrated)


        return dataOut








class SumLagProducts_Old(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)
        #dataOut.rnint2=numpy.zeros(dataOut.nlags_array,'float32')


    def run(self,dataOut):

        if dataOut.AUX: #Solo cuando ya hizo la intregacion se ejecuta


            dataOut.rnint2=numpy.zeros(dataOut.header[17][0],'float32')
            #print(dataOut.experiment)
            if dataOut.experiment=="DP":
                for l in range(dataOut.header[17][0]):
                    dataOut.rnint2[l]=1.0/(dataOut.nint*dataOut.header[7][0]*12.0)


            if dataOut.experiment=="HP":
                for l in range(dataOut.header[17][0]):
                    if(l==0 or (l>=3 and l <=6)):
                        dataOut.rnint2[l]=0.5/float(dataOut.nint*dataOut.header[7][0]*16.0)
                    else:
                        dataOut.rnint2[l]=0.5/float(dataOut.nint*dataOut.header[7][0]*8.0)
            #print(dataOut.rnint2)
            for l in range(dataOut.header[17][0]):

                dataOut.kabxys_integrated[4][:,l,0]=(dataOut.kabxys_integrated[4][:,l,0]+dataOut.kabxys_integrated[4][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[5][:,l,0]=(dataOut.kabxys_integrated[5][:,l,0]+dataOut.kabxys_integrated[5][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[6][:,l,0]=(dataOut.kabxys_integrated[6][:,l,0]+dataOut.kabxys_integrated[6][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[7][:,l,0]=(dataOut.kabxys_integrated[7][:,l,0]+dataOut.kabxys_integrated[7][:,l,1])*dataOut.rnint2[l]

                dataOut.kabxys_integrated[8][:,l,0]=(dataOut.kabxys_integrated[8][:,l,0]-dataOut.kabxys_integrated[8][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[9][:,l,0]=(dataOut.kabxys_integrated[9][:,l,0]-dataOut.kabxys_integrated[9][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[10][:,l,0]=(dataOut.kabxys_integrated[10][:,l,0]-dataOut.kabxys_integrated[10][:,l,1])*dataOut.rnint2[l]
                dataOut.kabxys_integrated[11][:,l,0]=(dataOut.kabxys_integrated[11][:,l,0]-dataOut.kabxys_integrated[11][:,l,1])*dataOut.rnint2[l]


            #print("Final Integration: ",dataOut.kabxys_integrated[4][:,l,0])






        return dataOut









class BadHeights_Old(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)



    def run(self,dataOut):


        if dataOut.AUX==1:
            dataOut.ibad=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]),'int32')

            for j in range(dataOut.header[15][0]):
                for l in range(dataOut.header[17][0]):
                    ip1=j+dataOut.header[15][0]*(0+2*l)
                    if( (dataOut.kabxys_integrated[5][j,l,0] <= 0.) or (dataOut.kabxys_integrated[4][j,l,0] <= 0.) or (dataOut.kabxys_integrated[7][j,l,0] <= 0.) or (dataOut.kabxys_integrated[6][j,l,0] <= 0.)):
                        dataOut.ibad[j][l]=1
                    else:
                        dataOut.ibad[j][l]=0
            #print("ibad: ",dataOut.ibad)



        return dataOut
















class NoisePower_old(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

    def hildebrand(self,dataOut,data):
            #print("data ",data )
        divider=10 # divider was originally 10
        noise=0.0
        n1=0
        n2=int(dataOut.header[15][0]/2)
        sorts= sorted(data)

        nums_min= dataOut.header[15][0]/divider
        if((dataOut.header[15][0]/divider)> 2):
            nums_min= int(dataOut.header[15][0]/divider)
        else:
            nums_min=2
        sump=0.0
        sumq=0.0
        j=0
        cont=1
        while( (cont==1) and (j<n2)):
            sump+=sorts[j+n1]
            sumq+= sorts[j+n1]*sorts[j+n1]
            t3= sump/(j+1)
            j=j+1
            if(j> nums_min):
                rtest= float(j/(j-1)) +1.0/dataOut.header[7][0]
                t1= (sumq*j)
                t2=(rtest*sump*sump)
                if( (t1/t2) > 0.990):
                    j=j-1
                    sump-= sorts[j+n1]
                    sumq-=sorts[j+n1]*sorts[j+n1]
                    cont= 0

        noise= sump/j
        stdv=numpy.sqrt((sumq- noise*noise)/(j-1))
        return noise

    def run(self,dataOut):

        if dataOut.AUX==1:

        #print("ax2 shape ",ax2.shape)
            p=numpy.zeros((dataOut.header[2][0],dataOut.header[15][0],dataOut.header[17][0]),'float32')
            av=numpy.zeros(dataOut.header[15][0],'float32')
            dataOut.pnoise=numpy.zeros(dataOut.header[2][0],'float32')

            p[0,:,:]=dataOut.kabxys_integrated[4][:,:,0]+dataOut.kabxys_integrated[5][:,:,0] #total power for channel 0, just  pulse with non-flip
            p[1,:,:]=dataOut.kabxys_integrated[6][:,:,0]+dataOut.kabxys_integrated[7][:,:,0] #total power for channel 1

            #print("p[0,:,:] ",p[0,:,:])
            #print("p[1,:,:] ",p[1,:,:])

            for i in range(dataOut.header[2][0]):
                dataOut.pnoise[i]=0.0
                for k in range(dataOut.header[17][0]):
                    dataOut.pnoise[i]+= self.hildebrand(dataOut,p[i,:,k])
                    #print("dpl ",k, "pnoise[",i,"] ",pnoise[i] )
                dataOut.pnoise[i]=dataOut.pnoise[i]/dataOut.header[17][0]


            #print("POWERNOISE: ",dataOut.pnoise)
            dataOut.pan=1.0*dataOut.pnoise[0] # weights could change
            dataOut.pbn=1.0*dataOut.pnoise[1] # weights could change
            #print("dataOut.pan ",dataOut.pan, "   dataOut.pbn ",dataOut.pbn)
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

            #print("POWERNOISE: ",dataOut.pnoise)


        return dataOut








class double_pulse_ACFs(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)
        self.aux=1

    def run(self,dataOut):
        dataOut.pairsList=None
        if dataOut.AUX==1:
            dataOut.igcej=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]),'int32')

            if self.aux==1:
                dataOut.rhor=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)
                dataOut.rhoi=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)
                dataOut.sdp=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)
                dataOut.sd=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)
                #dataOut.igcej=numpy.zeros((dataOut.NDP,dataOut.nlags_array),'int32')
                dataOut.p=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)
                dataOut.alag=numpy.zeros(dataOut.header[15][0],'float32')
                for l in range(dataOut.header[17][0]):
                    dataOut.alag[l]=l*dataOut.header[10][0]*2.0/150.0
                self.aux=0
            sn4=dataOut.pan*dataOut.pbn
            rhorn=0
            rhoin=0
            #p=np.zeros((ndt,dpl), dtype=float)
            panrm=numpy.zeros((dataOut.header[15][0],dataOut.header[17][0]), dtype=float)


            for i in range(dataOut.header[15][0]):
                for j in range(dataOut.header[17][0]):
                    #################  Total power
                    pa=numpy.abs(dataOut.kabxys_integrated[4][i,j,0]+dataOut.kabxys_integrated[5][i,j,0])
                    pb=numpy.abs(dataOut.kabxys_integrated[6][i,j,0]+dataOut.kabxys_integrated[7][i,j,0])
                    #print("PA",pb)
                    st4=pa*pb
                    dataOut.p[i,j]=pa+pb-(dataOut.pan+dataOut.pbn)
                    dataOut.sdp[i,j]=2*dataOut.rnint2[j]*((pa+pb)*(pa+pb))
                    ## ACF
                    rhorp=dataOut.kabxys_integrated[8][i,j,0]+dataOut.kabxys_integrated[11][i,j,0]
                    rhoip=dataOut.kabxys_integrated[10][i,j,0]-dataOut.kabxys_integrated[9][i,j,0]
                    if ((pa>dataOut.pan)&(pb>dataOut.pbn)):
                        #print("dataOut.pnoise[0]: ",dataOut.pnoise[0])
                        #print("dataOut.pnoise[1]: ",dataOut.pnoise[1])
                        #print("OKKKKKKKKKKKKKKK")
                        ss4=numpy.abs((pa-dataOut.pan)*(pb-dataOut.pbn))
                        #print("ss4: ",ss4)
                        #print("OKKKKKKKKKKKKKKK")
                        panrm[i,j]=math.sqrt(ss4)
                        rnorm=1/panrm[i,j]
                        #print("rnorm: ",rnorm)get_number_density
                        #print("OKKKKKKKKKKKKKKK")

                        ##  ACF
                        dataOut.rhor[i,j]=rhorp*rnorm
                        dataOut.rhoi[i,j]=rhoip*rnorm
                        #print("rhoi: ",dataOut.rhoi)
                        #print("OKKKKKKKKKKKKKKK")
                        #############  Compute standard error for ACF
                        stoss4=st4/ss4
                        snoss4=sn4/ss4
                        rp2=((rhorp*rhorp)+(rhoip*rhoip))/st4
                        rn2=((rhorn*rhorn)+(rhoin*rhoin))/sn4
                        rs2=(dataOut.rhor[i,j]*dataOut.rhor[i,j])+(dataOut.rhoi[i,j]*dataOut.rhoi[i,j])
                        st=1.0+rs2*(stoss4-(2*math.sqrt(stoss4*snoss4)))
                        stn=1.0+rs2*(snoss4-(2*math.sqrt(stoss4*snoss4)))
                        dataOut.sd[i,j]=((stoss4*((1.0+rp2)*st+(2.0*rp2*rs2*snoss4)-4.0*math.sqrt(rs2*rp2)))+(0.25*snoss4*((1.0+rn2)*stn+(2.0*rn2*rs2*stoss4)-4.0*math.sqrt(rs2*rn2))))*dataOut.rnint2[j]
                        dataOut.sd[i,j]=numpy.abs(dataOut.sd[i,j])
                        #print("sd: ",dataOut.sd)
                        #print("OKKKKKKKKKKKKKKK")
                    else: #default values for bad points
                        rnorm=1/math.sqrt(st4)
                        dataOut.sd[i,j]=1.e30
                        dataOut.ibad[i,j]=4
                        dataOut.rhor[i,j]=rhorp*rnorm
                        dataOut.rhoi[i,j]=rhoip*rnorm
                    if ((pa/dataOut.pan-1.0)>2.25*(pb/dataOut.pbn-1.0)):
                        dataOut.igcej[i,j]=1

            #print("sdp",dataOut.sdp)

        return dataOut







class faraday_angle_and_power_double_pulse(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)
        self.aux=1

    def run(self,dataOut):
        #dataOut.NRANGE=NRANGE
        #dataOut.H0=H0
        #########     H0 Y NRANGE SON PARAMETROS?

        if dataOut.AUX==1:
            if self.aux==1:
                dataOut.h2=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.range1=numpy.zeros(dataOut.header[15][0],order='F',dtype='float32')
                dataOut.sdn2=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.ph2=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.sdp2=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.ibd=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.phi=numpy.zeros(dataOut.header[15][0],'float32')
                self.aux=0
            #print("p: ",dataOut.p)


            for i in range(dataOut.header[15][0]):
                dataOut.range1[i]=dataOut.header[9][0] + i*dataOut.header[10][0] # (float) header.h0 + (float)i * header.dh
                dataOut.h2[i]=dataOut.range1[i]**2

            #print("sd: ",dataOut.sd)
            #print("OIKKKKKKKKKKKKKKK")
            #print("ibad: ",dataOut.ibad)
            #print("igcej: ",dataOut.igcej)
            for j in range(dataOut.header[15][0]):
                dataOut.ph2[j]=0.
                dataOut.sdp2[j]=0.
                ri=dataOut.rhoi[j][0]/dataOut.sd[j][0]
                rr=dataOut.rhor[j][0]/dataOut.sd[j][0]
                dataOut.sdn2[j]=1./dataOut.sd[j][0]
                #print("sdn2: ",dataOut.sdn2)
                #print("OIKKKKKKKKKKKKKKK")
                pt=0.# // total power
                st=0.# // total signal
                ibt=0# // bad lags
                ns=0#  // no. good lags
                for l in range(dataOut.header[17][0]):
                     #add in other lags if outside of e-jet contamination
                    if( (dataOut.igcej[j][l] == 0) and (dataOut.ibad[j][l] ==  0) ):
                        #print("dataOut.p[j][l]: ",dataOut.p[j][l])
                        dataOut.ph2[j]+=dataOut.p[j][l]/dataOut.sdp[j][l]
                        dataOut.sdp2[j]=dataOut.sdp2[j]+1./dataOut.sdp[j][l]
                        ns+=1

                    pt+=dataOut.p[j][l]/dataOut.sdp[j][l]
                    st+=1./dataOut.sdp[j][l]
                    ibt|=dataOut.ibad[j][l];
                    #print("pt: ",pt)
                    #print("st: ",st)
                if(ns!= 0):
                    dataOut.ibd[j]=0
                    dataOut.ph2[j]=dataOut.ph2[j]/dataOut.sdp2[j]
                    dataOut.sdp2[j]=1./dataOut.sdp2[j]
                else:
                    dataOut.ibd[j]=ibt
                    dataOut.ph2[j]=pt/st
                    #print("ph2: ",dataOut.ph2)
                    dataOut.sdp2[j]=1./st
                #print("ph2: ",dataOut.ph2)
                dataOut.ph2[j]=dataOut.ph2[j]*dataOut.h2[j]
                dataOut.sdp2[j]=numpy.sqrt(dataOut.sdp2[j])*dataOut.h2[j]
                rr=rr/dataOut.sdn2[j]
                ri=ri/dataOut.sdn2[j]
                #rm[j]=np.sqrt(rr*rr + ri*ri) it is not used in c program
                dataOut.sdn2[j]=1./(dataOut.sdn2[j]*(rr*rr + ri*ri))
                if( (ri == 0.) and (rr == 0.) ):
                    dataOut.phi[j]=0.
                else:
                    dataOut.phi[j]=math.atan2( ri , rr )

            #print("ph2: ",dataOut.ph2)
            #print("sdp2: ",dataOut.sdp2)
            #print("sdn2",dataOut.sdn2)


        return dataOut






class get_number_density(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)
        self.aux=1

    def run(self,dataOut,NSHTS=None,RATE=None):
        dataOut.NSHTS=NSHTS
        dataOut.RATE=RATE
        if dataOut.AUX==1:
            #dataOut.TimeBlockSeconds=time.mktime(time.strptime(dataOut.TimeBlockDate))
            #dataOut.bd_time=time.gmtime(dataOut.TimeBlockSeconds)
            #dataOut.ut=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0
            if self.aux==1:
                dataOut.dphi=numpy.zeros(dataOut.header[15][0],'float32')
                dataOut.sdn1=numpy.zeros(dataOut.header[15][0],'float32')
                self.aux=0
            theta=numpy.zeros(dataOut.header[15][0],dtype=numpy.complex_)
            thetai=numpy.zeros(dataOut.header[15][0],dtype=numpy.complex_)
            # use complex numbers for phase
            for i in range(dataOut.NSHTS):
                theta[i]=math.cos(dataOut.phi[i])+math.sin(dataOut.phi[i])*1j
                thetai[i]=-math.sin(dataOut.phi[i])+math.cos(dataOut.phi[i])*1j

            # differentiate and convert to number density
            ndphi=dataOut.NSHTS-4
            #print("dataOut.dphiBEFORE: ",dataOut.dphi)
            for i in range(2,dataOut.NSHTS-2):
                fact=(-0.5/(dataOut.RATE*dataOut.header[10][0]))*dataOut.bki[i]
                #four-point derivative, no phase unwrapping necessary
                dataOut.dphi[i]=((((theta[i+1]-theta[i-1])+(2.0*(theta[i+2]-theta[i-2])))/thetai[i])).real/10.0
                #print("dataOut.dphi[i]AFTER: ",dataOut.dphi[i])
                dataOut.dphi[i]=abs(dataOut.dphi[i]*fact)
                dataOut.sdn1[i]=(4.*(dataOut.sdn2[i-2]+dataOut.sdn2[i+2])+dataOut.sdn2[i-1]+dataOut.sdn2[i+1])
                dataOut.sdn1[i]=numpy.sqrt(dataOut.sdn1[i])*fact
            '''
            #print("date: ",dataOut.TimeBlockDate)
            #print("CurrentBlock: ", dataOut.CurrentBlock)
            #print("NSHTS: ",dataOut.NSHTS)
            print("phi: ",dataOut.phi)
            #print("header[10][0]: ",dataOut.DH)
            print("bkibki: ",dataOut.bki)
            #print("RATE: ",dataOut.RATE)
            print("sdn2: ",dataOut.sdn2)
            print("dphi: ",dataOut.dphi)
            print("sdn1: ",dataOut.sdn1)
            print("ph2: ",dataOut.ph2)
            print("sdp2: ",dataOut.sdp2)
            print("sdn1: ",dataOut.sdn1)
            '''

            '''
            Al finallllllllllllllllllllllllllllllllllllllllllllllllllllllllll
            for i in range(dataOut.NSHTS):
                dataOut.ph2[i]=(max(1.0, dataOut.ph2[i]))
                dataOut.dphi[i]=(max(1.0, dataOut.dphi[i]))
                #print("dphi ",dphi)
                # threshold - values less than 10⁴
            for i in range(dataOut.NSHTS):
                if dataOut.ph2[i]<10000:
                    dataOut.ph2[i]=10000

            # threshold values more than 10⁷
            for i in range(dataOut.NSHTS):
                if dataOut.ph2[i]>10000000:#
                    dataOut.ph2[i]=10000000

            ## filter for errors
            for i in range(dataOut.NSHTS):
                if dataOut.sdp2[i]>100000:#
                    dataOut.ph2[i]=10000
                    '''




        return dataOut











class normalize_dp_power2(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)
        self.aux=1

    def normal(self,a,b,n,m):
        chmin=1.0e30
        chisq=numpy.zeros(150,'float32')
        temp=numpy.zeros(150,'float32')

        for i in range(2*m-1):
            an=al=be=chisq[i]=0.0
            for j in range(int(n/m)):
                k=int(j+i*n/(2*m))
                if(a[k]>0.0 and b[k]>0.0):
                    al+=a[k]*b[k]
                    be+=b[k]*b[k]

            if(be>0.0):
                temp[i]=al/be
            else:
                temp[i]=1.0

            for j in range(int(n/m)):
                k=int(j+i*n/(2*m))
                if(a[k]>0.0 and b[k]>0.0):
                    chisq[i]+=(numpy.log10(b[k]*temp[i]/a[k]))**2
                    an=an+1

            if(chisq[i]>0.0):
                chisq[i]/=an


        for i in range(int(2*m-1)):
            if(chisq[i]<chmin and chisq[i]>1.0e-6):
                chmin=chisq[i]
                cf=temp[i]
        return cf



    def run(self,dataOut,cut0=None,cut1=None):
        dataOut.cut0=float(cut0)
        dataOut.cut1=float(cut1)
        if dataOut.AUX==1:
            #print("dateBefore: ",dataOut.TimeBlockDate_for_dp_power)
            #print("dateNow: ",dataOut.TimeBlockDate)
            if self.aux==1:
                dataOut.cf=numpy.zeros(1,'float32')
                dataOut.cflast=numpy.zeros(1,'float32')
                self.aux=0

            night_first=300.0
            night_first1= 310.0
            night_end= 450.0
            day_first=250.0
            day_end=400.0
            day_first_sunrise=190.0
            day_end_sunrise=280.0

            if(dataOut.ut>4.0 and dataOut.ut<11.0): #early
                i2=(night_end-dataOut.range1[0])/dataOut.header[10][0]
                i1=(night_first -dataOut.range1[0])/dataOut.header[10][0]
            elif (dataOut.ut>0.0 and dataOut.ut<4.0): #night
                i2=(night_end-dataOut.range1[0])/dataOut.header[10][0]
                i1=(night_first1 -dataOut.range1[0])/dataOut.header[10][0]
            elif (dataOut.ut>=11.0 and dataOut.ut<13.5): #sunrise
                i2=( day_end_sunrise-dataOut.range1[0])/dataOut.header[10][0]
                i1=(day_first_sunrise - dataOut.range1[0])/dataOut.header[10][0]
            else:
                i2=(day_end-dataOut.range1[0])/dataOut.header[10][0]
                i1=(day_first -dataOut.range1[0])/dataOut.header[10][0]

            i1=int(i1)
            i2=int(i2)
            #print("ph2: ",dataOut.ph2)
            dataOut.cf=self.normal(dataOut.dphi[i1::], dataOut.ph2[i1::], i2-i1, 1)

            #print("n in:",i1,"(",dataOut.range1[i1],"), i2=",i2,"(",dataOut.range1[i2],"), ut=",dataOut.ut,", cf=",dataOut.cf,", cf_last=",
                  #dataOut.cflast)
            #  in case of spread F, normalize much higher
            if(dataOut.cf<dataOut.cflast[0]/10.0):
                i1=(night_first1+100.-dataOut.range1[0])/dataOut.header[10][0]
                i2=(night_end+100.0-dataOut.range1[0])/dataOut.header[10][0]
                i1=int(i1)
                i2=int(i2)
                #print("normal over: ",i1,"(",dataOut.range1[i1],") ",i2,"(",dataOut.range1[i2],") => cf: ",dataOut.cf,"   cflast: ", dataOut.cflast)
                dataOut.cf=self.normal(dataOut.dphi[int(i1)::], dataOut.ph2[int(i1)::], int(i2-i1), 1)
                dataOut.cf=dataOut.cflast[0]

            #print(">>>i1=",i1,"(",dataOut.range1[i1],"), i2=",i2,"(",dataOut.range1[i2],"), ut=",dataOut.ut,", cf=",dataOut.cf,", cf_last=",
                 # dataOut.cflast," (",dataOut.cf/dataOut.cflast,"), cut=",dataOut.cut0," ",dataOut.cut1)
            dataOut.cflast[0]=dataOut.cf

            ## normalize double pulse power and error bars to Faraday
            for i in range(dataOut.NSHTS):
                dataOut.ph2[i]*=dataOut.cf
                dataOut.sdp2[i]*=dataOut.cf
            #print("******* correction factor: ",dataOut.cf)

            #print(dataOut.ph2)

            for i in range(dataOut.NSHTS):
                dataOut.ph2[i]=(max(1.0, dataOut.ph2[i]))
                dataOut.dphi[i]=(max(1.0, dataOut.dphi[i]))
                #print("dphi ",dphi)
                # threshold - values less than 10⁴

                '''
            for i in range(dataOut.NSHTS):
                if dataOut.ph2[i]<10000:
                    dataOut.ph2[i]=10000

            # threshold values more than 10⁷
            for i in range(dataOut.NSHTS):
                if dataOut.ph2[i]>10000000:#
                    dataOut.ph2[i]=10000000

            ## filter for errors
            for i in range(dataOut.NSHTS):
                if dataOut.sdp2[i]>100000:#
                    dataOut.ph2[i]=10000
                        '''





            '''
            #print("date: ",dataOut.TimeBlockDate)
            #print("CurrentBlock: ", dataOut.CurrentBlock)
            #print("NSHTS: ",dataOut.NSHTS)
            print("phi: ",dataOut.phi)
            #print("header[10][0]: ",dataOut.DH)
            print("bkibki: ",dataOut.bki)
            #print("RATE: ",dataOut.RATE)
            print("sdn2: ",dataOut.sdn2)
            print("dphi: ",dataOut.dphi)
            print("sdn1: ",dataOut.sdn1)
            print("ph2: ",dataOut.ph2)
            print("sdp2: ",dataOut.sdp2)
            print("sdn1: ",dataOut.sdn1)
            '''







        return dataOut















'''
from ctypes import *
class IDATE(Structure):
    _fields_ = [
        ("year", c_int),
        ("moda", c_int),
        ("hrmn", c_int),
        ("sec", c_int),
        ("secs", c_int),
    ]
#typedef struct IDATE {int year,moda,hrmn,sec,secs;} idate;
'''




'''
class get_number_density(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)

        #self.aux=1
'''

'''
    def IDATE(Structure):

        _fields_ = [
            ("year", c_int),
            ("moda", c_int),
            ("hrmn", c_int),
            ("sec", c_int),
            ("secs", c_int),
        ]

        '''




'''
    def run(self,dataOut):
'''
'''
        if dataOut.CurrentBlock==1 and self.aux==1:

            #print("CurrentBlock: ",dataOut.CurrentBlock)

            dataOut.TimeBlockSeconds=time.mktime(time.strptime(dataOut.TimeBlockDate))
            #print("time1: ",dataOut.TimeBlockSeconds)

            #print("date: ",dataOut.TimeBlockDate)
            dataOut.bd_time=time.gmtime(dataOut.TimeBlockSeconds)
            #print("bd_time: ",dataOut.bd_time)
            dataOut.year=dataOut.bd_time.tm_year+(dataOut.bd_time.tm_yday-1)/364.0
            #print("year: ",dataOut.year)
            dataOut.ut=dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min/60.0+dataOut.bd_time.tm_sec/3600.0
            #print("ut: ",dataOut.ut)
            self.aux=0




            '''
        #print("CurrentBlock: ",dataOut.CurrentBlock)
        #print("date: ",dataOut.firsttime)
        #print("bd_time: ",time.strptime(dataOut.datatime.ctime()))
        #mkfact_short.mkfact(year,h,bfm,thb,bki,dataOut.NDP)
        #print("CurrentBlock: ",dataOut.CurrentBlock)
'''
        if dataOut.AUX==1:
            '''
'''
            #begin=IDATE()
            #begin.year=dataOut.bd_time.tm_year
            #begin.moda=100*(dataOut.bd_time.tm_mon)+dataOut.bd_time.tm_mday
            #begin.hrmn=100*dataOut.bd_time.tm_hour+dataOut.bd_time.tm_min
            #begin.sec=dataOut.bd_time.tm_sec
            #begin.secs=dataOut.bd_time.tm_sec+60*(dataOut.bd_time.tm_min+60*(dataOut.bd_time.tm_hour+24*(dataOut.bd_time.tm_yday-1)))
            h=numpy.arange(0.0,15.0*dataOut.NDP,15.0,dtype='float32')
            bfm=numpy.zeros(dataOut.NDP,dtype='float32')
            bfm=numpy.array(bfm,order='F')
            thb=numpy.zeros(dataOut.NDP,dtype='float32')
            thb=numpy.array(thb,order='F')
            bki=numpy.zeros(dataOut.NDP,dtype='float32')
            bki=numpy.array(thb,order='F')
            #yearmanually=2019.9285714285713
            #print("year manually: ",yearmanually)
            #print("year: ",dataOut.year)
            mkfact_short.mkfact(dataOut.year,h,bfm,thb,bki,dataOut.NDP)
            #print("tm ",tm)
            '''
'''
            print("year ",dataOut.year)
            print("h ", dataOut.h)
            print("bfm ", dataOut.bfm)
            print("thb ", dataOut.thb)
            print("bki ", dataOut.bki)
'''




'''
            print("CurrentBlock: ",dataOut.CurrentBlock)










        return dataOut
'''






class test(Operation):
    def __init__(self, **kwargs):

        Operation.__init__(self, **kwargs)




    def run(self,dataOut,tt=10):

        print("tt: ",tt)



        return dataOut
