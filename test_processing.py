###---Processing Test---###

import os, sys
import datetime
import time
from schainpy.controller import Project
import numpy as np
import matplotlib.pyplot as plt

desc = "Processing Test By Profiles"

path = '/home/david/Documents/DATA_2/CHIRP_TFM@2026-01-27T20-36-02/rawdata/'
# path = '/home/david/Documents/DATA_2/HYO@2025-12-11T14-46-59/rawdata/'

## REVISION ##
## 1 ##
controllerObj = Project()
controllerObj.setup(id = '192', name='Test_USRP', description="Spectra Test Processing")

N = int(500.0)

#######################################################################
############################ READING UNIT #############################
#######################################################################

# Working only Read Unit
## 2 ##
readUnitConfObj = controllerObj.addReadUnit(datatype='DigitalRFReader',
                                            # Digital RF Data TFM 5 MHz was found from 2026-01-27 15:36:14 to 2026-01-27 15:43:55 (Time: 88.50s)
                                            # Digital RF Data HYO 2 MHz was found from 2025-12-11 09:47:12 to 2025-12-11 09:53:04 (Time: 38.15s)
                                            # TFM 1 min of data - 11.5 s
                                            # HYO 1 min of data - 6.5 s
                                            path=path,
                                            startDate='2026/01/01',
                                            endDate='2026/12/31',
                                            startTime='00:00:00',
                                            endTime='23:59:59',
                                            ippKm = 60,
                                            walk=1,
                                            getByBlock = 1,
                                            nProfileBlocks = N,
                                            )

controllerObj.start()
