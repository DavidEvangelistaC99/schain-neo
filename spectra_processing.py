# TEST_BOLITA_COBRE_4

import modFreq as modf

import os, sys
import datetime
import time
from schainpy.controller import Project
import numpy as np 
import matplotlib.pyplot as plt

desc = "USRP_test"
filename = "USRP_processing.xml"

path = '/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata/'
figpath = '/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata'

## REVISION ##
## 1 ##
controllerObj = Project()
controllerObj.setup(id = '192', name='Test_USRP', description="Hola Mundo")

#######################################################################
######################### RANGO DE PLOTEO #############################
#######################################################################

# Parametros de graficos
## 2 ##
dBmin = 0
dBmax = 120
xmin = '0'
xmax = '24'
ymin = '0'
ymax = '60'

#######################################################################
######################## UNIDAD DE LECTURA ############################
#######################################################################

readUnitConfObj = controllerObj.addReadUnit(datatype='DigitalRFReader',
                                            path=path,
                                            startDate="2025/01/01",
                                            endDate="2025/12/30",
                                            startTime='00:00:00',
                                            endTime='23:59:59',
                                            delay=0,
                                            
                                            set=0,
                                            # online=0,
                                            # walk=1,


                                            getByBlock = 0,
                                            #nProfileBlocks = 500,


                                            # Importante para el uso con el radar SOPHy
                                            ippKm = 60)

opObj11 = readUnitConfObj.addOperation(name='printInfo')
# opObj11 = readUnitConfObj.addOperation(name='printNumberOfBlock')

# voltage -> procUnitConfObjA 
# Añadir una unidad de procesamiento

## 3 ##
procUnitConfObjA = controllerObj.addProcUnit(datatype='VoltageProc', inputId=readUnitConfObj.getId())

op =  procUnitConfObjA.addOperation(name='setAttribute')
op.addParameter(name='frequency', value='9.345e9', format='float')

op1 = procUnitConfObjA.addOperation(name='ProfileSelector')
# Cambio del valor del numero de perfiles por uno constante (nProfiles)
op1.addParameter(name='profileRangeList', value='0,499')
# op1.addParameter(name='profileRangeList', value='250,499')

# Parameters
A = 1.0
ipp = 400.0e-6
dc = 12.0
# Consideramos el SR RX
sr_tx = 20.0e6
sr_rx = 2.5e6
fc = 0.0e6
bw = 1.0e6
      
chirp_tx_1, _ = modf.chirpMod(A, ipp, dc, sr_rx, sr_rx, fc, bw, t_d = 0, window = 'B', mode_f = 0)
# chirp_tx_1 = modf.chirpModUnion(ipp, sr_rx, sr_rx, A, A, 14.6, 0.4, 0.75e6, 0.0, 1.5e6, 0.0, 'B', 'R')

code_ = chirp_tx_1
code = [code_]

## 4 ##
op2 = procUnitConfObjA.addOperation(name='Decoder', optype='other')

## 5 ##
op2.addParameter(name='code', value=code)
op2.addParameter(name='nCode', value=len(code), format='int')
op2.addParameter(name='nBaud', value=len(code[0]), format='int')

# Minimo integrar 2 perfiles por ser codigo complementario, para el Chirp no es necesario por no ser continua y no tener dos códigos
# op3 = procUnitConfObjA.addOperation(name='CohInt', optype='other') 
# op3.addParameter(name='n', value=2, format='int')

#######################################################################
############## OPERACIONES DOMINIO DE LA FRECUENCIA ###################
#######################################################################

procUnitConfObjSousySpectra = controllerObj.addProcUnit(datatype='SpectraProc', inputId=procUnitConfObjA.getId())
procUnitConfObjSousySpectra.addParameter(name='nFFTPoints', value='500', format='int')
procUnitConfObjSousySpectra.addParameter(name='nProfiles', value='500', format='int')

# Remocion DC

opObj13 = procUnitConfObjSousySpectra.addOperation(name='removeDC')
opObj13.addParameter(name='mode', value='2', format='int')

#######################################################################
################## PLOTEO DOMINIO DE LA FRECUENCIA ####################
#######################################################################

# SpectraPlot

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
# opObj11.addParameter(name='save_period', value=2, format='int')

#######################################################################
############################ RTI PLOT #################################
#######################################################################

'''
opObj11 = procUnitConfObjSousySpectra.addOperation(name='RTIPlot', optype='external')
opObj11.addParameter(name='id', value='2', format='int')
opObj11.addParameter(name='wintitle', value='RTIPlot', format='str')
opObj11.addParameter(name='ymin', value=ymin, format='int')
opObj11.addParameter(name='ymax', value=ymax, format='int')
opObj11.addParameter(name='xmin', value=10, format='int')
opObj11.addParameter(name='xmax', value=30, format='int')
opObj11.addParameter(name='showprofile', value='1', format='int')
opObj11.addParameter(name='save', value=figpath, format='str')
opObj11.addParameter(name='save_period', value=5, format='int')
'''

controllerObj.start()
