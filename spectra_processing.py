###---Spectra Processing Test---###

import os, sys
import datetime
import time
import numpy as np 
import matplotlib.pyplot as plt
from schainpy.controller import Project
import modFreq as modf

path = '/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata/'
figpath = '/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata'

controllerObj = Project()
controllerObj.setup(id = '002', name='Test_002', description='Processing Test')

#######################################################################
########################## PLOTTING RANGE #############################
#######################################################################

# Plot parameters
dBmin = -90
dBmax = 120
xmin = '0'
xmax = '24'
ymin = '0'
ymax = '60'

#######################################################################
########################### READING UNIT ##############################
#######################################################################

readUnitConfObj = controllerObj.addReadUnit(datatype='DigitalRFReader',
                                            path=path,
                                            startDate='2025/01/01',
                                            endDate='2025/12/31',
                                            startTime='00:00:00',
                                            endTime='23:59:59',
                                            ippKm = 60,
                                            walk=1,
                                            getByBlock = 1,
                                            nProfileBlocks = 500,
                                            )

opObj11 = readUnitConfObj.addOperation(name='printInfo')

procUnitConfObjA = controllerObj.addProcUnit(datatype='VoltageProc', inputId=readUnitConfObj.getId())

op =  procUnitConfObjA.addOperation(name='setAttribute')
op.addParameter(name='frequency', value='9.345e9', format='float')

op1 = procUnitConfObjA.addOperation(name='ProfileSelector')
op1.addParameter(name='profileRangeList', value='0,499')

# Chirp parameters 
A = 1.0
ipp = 400.0e-6
dc = 12.0
sr_tx = 20.0e6
sr_rx = 2.5e6
fc = 0.0e6
bw = 1.0e6
      
_, chirp_tx_1 = modf.chirpMod(A, ipp, dc, sr_rx, sr_rx, fc, bw, t_d = 0, window = 'B', mode_f = 0)
code = [chirp_tx_1]

op2 = procUnitConfObjA.addOperation(name='Decoder', optype='other')
op2.addParameter(name='code', value=code)
op2.addParameter(name='nCode', value=len(code), format='int')
op2.addParameter(name='nBaud', value=len(code[0]), format='int')

# op3 = procUnitConfObjA.addOperation(name='CohInt', optype='other') 
# op3.addParameter(name='n', value=2, format='int')

#######################################################################
################### FREQUENCY DOMAIN OPERATIONS #######################
#######################################################################

procUnitConfObjSousySpectra = controllerObj.addProcUnit(datatype='SpectraProc', inputId=procUnitConfObjA.getId())
procUnitConfObjSousySpectra.addParameter(name='nFFTPoints', value='500', format='int')
procUnitConfObjSousySpectra.addParameter(name='nProfiles', value='500', format='int')

# DC remove

# opObj13 = procUnitConfObjSousySpectra.addOperation(name='removeDC')
# opObj13.addParameter(name='mode', value='2', format='int')

#######################################################################
###################### FREQUENCY DOMAIN PLOTTING ######################
#######################################################################

# Spectra Plot
opObj11 = procUnitConfObjSousySpectra.addOperation(name='SpectraPlot', optype='external')
opObj11.addParameter(name='id', value='1', format='int')
opObj11.addParameter(name='wintitle', value='Spectra NEW', format='str')
opObj11.addParameter(name='zmin', value=dBmin)
opObj11.addParameter(name='zmax', value=dBmax)
opObj11.addParameter(name='ymin', value=ymin, format='int')
opObj11.addParameter(name='ymax', value=ymax, format='int')
opObj11.addParameter(name='showprofile', value='1', format='int')
opObj11.addParameter(name='save', value=figpath, format='str')
opObj11.addParameter(name='xaxis', value='velocity', format='str')

controllerObj.start()
