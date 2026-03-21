###---Processing Test---###

import os, sys
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from schainpy.controller import Project

path = '/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata/'

controllerObj = Project()
controllerObj.setup(id = '001', name='Test_001', description='Processing Test')

#######################################################################
############################ READING UNIT #############################
#######################################################################

# Working only Read Unit
# Parameters send to ReadUnitConf()
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

controllerObj.start()
