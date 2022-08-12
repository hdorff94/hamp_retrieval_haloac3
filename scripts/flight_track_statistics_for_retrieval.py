# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:52:53 2022

@author: u300737
"""
import os
import glob
import sys
    
actual_working_path=os.getcwd()
airborne_retrieval_module_path=actual_working_path+"/retrieval/"
#airborne_processing_module_path=actual_working_path+"/scripts/"
airborne_plotting_module_path=actual_working_path+"/plotting/"
major_retrieval_path=os.getcwd()+"/../"
#os.chdir(airborne_processing_module_path)

sys.path.insert(1,major_retrieval_path)
sys.path.insert(2,airborne_plotting_module_path)
import retrieval_paths
working_path=retrieval_paths.main()
airborne_data_importer_path=working_path+"/Work/GIT_Repository/"+\
                                "hamp_processing_py/"+\
                                    "hamp_processing_python/" # This is also the major path where your data will be stored
sys.path.insert(3,airborne_data_importer_path)
    
import config_handler
import campaign_time
    
#import performance
import numpy as np
import pandas as pd
import xarray as xr
    
import radar_attitude
import radar_masks
import unified_grid as unigrid
    
try:
    import Flight_Campaign as Campaign
except:
        print("Module Flight Campaign is not listed in the path",
              "Flights need to be defined manually.")
Flight_Dates={}
Flight_Dates["EUREC4A"]={"RF01":"20200119","RF02":"20200122",
                                 "RF03":"20200124","RF04":"20200126",
                                 "RF05":"20200128","RF06":"20200130",
                                 "RF07":"20200131","RF08":"20200202",
                                 "RF09":"20200205","RF10":"20200207",
                                 "RF11":"20200209","RF12":"20200211",
                                 "RF13":"20200213","RF14":"20200215",
                                 "RF15":"20200218"}
        
Flight_Dates["HALO_AC3"]={"RF00":"20220225",
                              "RF01":"20220311", # if this is the transfer flight
                              "RF02":"20220312",
                              "RF03":"20220313",
                              "RF04":"20220314",
                              "RF05":"20220315",
                              "RF06":"20220316",
                              "RF07":"20220320",
                              "RF08":"20220321",
                              "RF09":"20220328",
                              "RF10":"20220329",
                              "RF11":"20220330",
                              "RF12":"20220401",
                              "RF13":"20220404",
                              "RF14":"20220407",
                              "RF15":"20220408",
                              "RF16":"20220410",
                              "RF17":"20220411",
                              "RF18":"20220412"}
    
