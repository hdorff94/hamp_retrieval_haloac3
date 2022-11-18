# Python PAMTRA for ICON generator
# Version: Adapted for HOPE SmallScale simulation of ICON LES
# v1: Command line version to generate cloud radar and MW simulations from bash script
# v2: Corrected script to use right surface variables and all 151 height levels
# v2: Included vertical wind, horizontal wind and EDR for doppler spectra velocity simulations
# v3: Using 3D input fields as desired by PAMTRA
# v4: Cleaned code and optimized structure, introduced 3D input fields as of v3
# v5: Tested with corrected height levels, but PAMTRA wants to get 151 and 150 level structure
# _HATPRO - 18/07/2019: Calculation of all 14 HATPRO Channels, especially the Oxygen 50 GHz channels

from __future__ import division
import pyPamtra
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#import pandas as pn
from netCDF4 import Dataset
from sys import argv
from datetime import datetime, date, timedelta
from dateutil.relativedelta import *
import time

# Check for active or passive simulation and descriptor file
sim_type  = argv[1]

# Debug
#sim_type  = 'active'

# Optional parameter, if desired
#desc_file = argv[2]

# Generate input and output filename with locations
in_file  = argv[2]
out_file = argv[3]
station_id = argv[4]

# Debug options
#in_file  = '/home/u233139/ICON_DATA/METEOGRAM_HOPE/METEOGRAM_patch003_08-16UTC_26042013.nc'
#out_file = '/home/u233139/ICON_DATA/METEOGRAM_HOPE/out_v8_airturb_Spec_METEOGRAM_patch003_08-16UTC_26042013.nc'
#station_id = 2

# Station List
# LACROS station-ID: c3 -> 2

# Has to be modified, because currently python has to be started from meteoSI runtime directory
import imp
meteoSI_func = imp.load_source('meteoSI_func', '/home/zmaw/u233139/pamtra/python/pyPamtra/meteoSI.py')
from meteoSI_func import q2rh

# User Imputs
#desc_pamtra_file = '/home/u233139/pamtra/descriptorfiles/descriptor_file_COSMO_1mom.txt'
#desc_pamtra_file = '/home/zmaw/u233139/pamtra/descriptorfiles/descriptor_file_2m_crystal.txt'
desc_pamtra_file = '/home/zmaw/u233139/pamtra/descriptorfiles/descriptor_file_2m_icon.txt'
#in_file          = '/home/u233139/ICON_DATA/1d_vars_DOM03_20130426T000000Z_20130427T000000Z.nc'

# Initialize PyPAMTRA instance
pam = pyPamtra.pyPamtra()

# Load COSMO 2-moments descriptor file
pam.df.readFile(desc_pamtra_file)

# Load and Modify Namelist file
pam.nmlSet

# Turn on active or passive forward calculations depending on command line
if sim_type == 'active':
    pam.nmlSet["passive"] = False  # Activate this for active cloud radar simulation
    pam.nmlSet["radar_mode"] = "spectrum"
elif sim_type == 'passive':
    pam.nmlSet["active"] = False    # Activate this for Microwave radiometer simulation
else:
    print('Please select simulation type!')

#show some messages
pam.set["verbose"] = 0
pam.set["pyVerbose"] = 1

# Load ICON file for one station
# Open NetCDF connection to file in read-only mode
fh_ICON = Dataset(in_file, mode='r')

# Read and set variables
# LACROS station-ID: c3 -> 2
st_id = int(station_id)

# Read date
#date = fh_ICON.variables["date"]

# Read static variables
st_lon = fh_ICON.variables["station_lon"][st_id]
st_lat = fh_ICON.variables["station_lat"][st_id]
st_alt = fh_ICON.variables["station_hsurf"][st_id]
st_frland = fh_ICON.variables["station_frland"][st_id]

# Read main variable
st_varname = fh_ICON.variables["var_long_name"]
st_sfcvarname = fh_ICON.variables["sfcvar_long_name"]
st_varshortname = fh_ICON.variables["var_name"]
st_sfcvarshortname = fh_ICON.variables["sfcvar_name"]

st_values = fh_ICON.variables["values"][:,:,:,st_id]
st_sfcvalues = fh_ICON.variables["sfcvalues"][:,:,st_id]

st_height = fh_ICON.variables["heights"][:,0,st_id].squeeze()
num_steps = np.size(st_values[:,0,0])
num_levs  = 151

# Combine station surface altitude to overall altitudes
st_height[150] = st_alt

# Create static variables by repeating values by number of timesteps
icon_lon = np.array([st_lon,]*num_steps)
icon_lat = np.array([st_lat,]*num_steps)
icon_alt = np.array([st_alt,]*num_steps)
icon_frland = np.array([st_frland,]*num_steps)
icon_height = np.array([st_height,]*num_steps)
obs_height  = np.array([0.0,]*num_steps)

# Initialize unix timestamp vector and convert datetime object to unix time
time_step = fh_ICON.variables['time_step'][:]
time_unix_1970_sec = np.empty(time_step.size)
for dt in np.arange(0,(time_step.size) ):
    time_unix_1970_sec[dt] = time.mktime( (datetime.strptime( ''.join(fh_ICON.variables['date'][dt]) , '%Y%m%dT%H%M%SZ')).timetuple() )
#time_unix_1970_sec = np.arange(1366927200.0, 1366948809.0, 9.0)

# Close NetCDF file connection
fh_ICON.close()

# Variable description
# 0: Pressure
# 1: Temperature
# 5: U-Wind
# 6: V-Wind
# 8: QV
# 9: QC
# 10: QI
# 11: QR
# 12: QS
# 13: REL-HUM
# 14: QG
# 15: QH
# 16: QNI
# 17: QNS
# 18: QNR
# 19: QNG
# 20: QNH
# 21: QNC
# 29: PHALF

# Combine hydrometeors to one array
hydro_cmpl = np.zeros([num_steps,(num_levs-1),6])
hydro_cmpl[:,:,0] = np.fliplr( st_values[:,0:150, 9] ) # QC
hydro_cmpl[:,:,1] = np.fliplr( st_values[:,0:150,10] ) # QI
hydro_cmpl[:,:,2] = np.fliplr( st_values[:,0:150,11] ) # QR
hydro_cmpl[:,:,3] = np.fliplr( st_values[:,0:150,12] ) # QS
hydro_cmpl[:,:,4] = np.fliplr( st_values[:,0:150,14] ) # QG
hydro_cmpl[:,:,5] = np.fliplr( st_values[:,0:150,15] ) # QH

hydro_num_cmpl = np.zeros([num_steps,(num_levs-1),6])
hydro_num_cmpl[:,:,0] = np.fliplr( st_values[:,0:150,21] ) # QNCLOUD
hydro_num_cmpl[:,:,1] = np.fliplr( st_values[:,0:150,16] ) # QNICE
hydro_num_cmpl[:,:,2] = np.fliplr( st_values[:,0:150,18] ) # QNRAIN
hydro_num_cmpl[:,:,3] = np.fliplr( st_values[:,0:150,17] ) # QNSNOW
hydro_num_cmpl[:,:,4] = np.fliplr( st_values[:,0:150,19] ) # QNGRAUPEL
hydro_num_cmpl[:,:,5] = np.fliplr( st_values[:,0:150,20] ) # QNHAIL

# Set time frame - overall 4800
timeframe = np.arange(0,np.size(time_unix_1970_sec))
#timeframe = np.arange(0,50)

# Combine temperature field with 2 m temperature
st_values[timeframe,150,1] = st_sfcvalues[timeframe,23]
# Add 0.1 % relative humidity to lowest level, no RH available at lowest ICON level
st_values[timeframe,150,13] = st_values[timeframe,149,13] + 0.1

# Calculate wind speed and Eddy dissipation rate (EDR)
windspd_3d  = np.fliplr(np.sqrt( (st_values[:,0:150,5]**2) + (st_values[:,0:150,6]**2) ))
wind_uv_arr = (windspd_3d[timeframe,:]).reshape(-1, 1, 150)
#EDR_3d     = np.ones((3200, 150))*0.2 # Only for debugging purposes

# Calculate of turbulent kinetic energy (TKE)
TKVM   = st_values[:,0:150,27]    # assumed to be same as K-parameter
turlen = 60.0 # (m)
tur_c  = 2.0   # Constant (tuning parameter for spectral width)
e_tke  = (TKVM**2)/(turlen**2)

# Calculate eddy dissipation rate
edr_tur = np.fliplr( (tur_c * ((e_tke)**(3/2)) ) / turlen )
turb_edr_arr   = (edr_tur[timeframe,:]).reshape(-1, 1, 150)

# Calculate spectral broadening for higher moments of radar spectra
beamwidth_deg = 0.5
integration_time = 10.0
frequency = 35.0
#pam.addSpectralBroadening(EDR, wind_uv, beamwidth_deg, integration_time, frequency, kolmogorov=0.5)


# Generate PAMTRA data dictonary
# Change v4: Added 3D arrays as required by PAMTRA routines
pamData = dict()
pamData["press_lev"]  = np.fliplr( st_values[timeframe,:,29] ) # Pressure
pamData["relhum_lev"] = np.fliplr( st_values[timeframe,:,13] ) # Relative Humidity

# Copy data to PAMTRA dictonary
pamData["timestamp"] = time_unix_1970_sec[timeframe]
pamData["lat"] = (icon_lat[timeframe])
pamData["lon"] = (icon_lon[timeframe])
#pamData["obs_height"] = obs_height[timeframe]
pamData["lfrac"] = (icon_frland[timeframe])
pamData["wind10u"] = (st_sfcvalues[timeframe,25])
pamData["wind10v"] = (st_sfcvalues[timeframe,26])
#pamData["wind_w"] = (np.fliplr( st_values[timeframe,0:150,7] )).reshape(-1, 1, 150)
pamData["hgt_lev"] = (np.fliplr(icon_height[timeframe,:]))
#data["press"] = PS
pamData["temp_lev"] = (np.fliplr( st_values[timeframe,:,1] ))
#data["relhum"] = timestamp
pamData["hydro_q"] = (hydro_cmpl[timeframe,:,:])
pamData["hydro_n"] = (hydro_num_cmpl[timeframe,:,:])
pamData["groundtemp"] = (st_sfcvalues[timeframe,10])
#data["press_lev"] = timestamp
# Turbulence parameters
#pamData["wind_uv"] = (windspd_3d[timeframe,:]).reshape(-1, 1, 150)
#pamData["turb_edr"] = (edr_tur[timeframe,:]).reshape(-1, 1, 150)
#pamData["airturb"] = (EDR_3d[timeframe,:]).reshape(-1, 1, 150)

# Add them to pamtra object and create profile
pam.createProfile(**pamData)

# Add spectral broadening by air_turb parameter
#pam.addSpectralBroadening(EDR=turb_edr_arr, wind_uv=wind_uv_arr,
#    beamwidth_deg=beamwidth_deg, integration_time=integration_time, frequency=frequency, kolmogorov=0.5)

# Update PAMTRA profile with airturb parameter calculated before
#pam.createProfile(**pamData)


# Delete large ICON values variable
del st_values

if sim_type == 'active':
    ### MIRA Simulation Part ############
    # Execute PAMTRA for MIRA 35 GHz
    pam.runPamtra(35)
    # Write output to NetCDF4 file
    pam.writeResultsToNetCDF(out_file)
    #pam.writeResultsToNetCDF('example_ICON_radar_12-00_v190417_2mom.nc')
    ### End of MIRA Simulation Part #####
else:
    ### HATPRO Simulation Part ##########
    ##!freqs = np.array([22.2400,23.0400, 23.8400,25.4400,26.2400,27.8400, 31.4000, 58.0000])
    freqs = np.array([22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,31.4000, 51.2600,52.2800,53.8600,54.9400,56.6600,57.3000,58.0000])
    #freqs = np.array([22.2400])
    #freqs = np.array([22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,
    # 31.4000,58.0000])
    #freqs = np.array([22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,
    #    31.4000,51.2600,52.2800,53.8600,54.9400,56.6600,57.3000,58.0000])
    pam.runPamtra(freqs)
    # Write output to NetCDF4 file
    pam.writeResultsToNetCDF(out_file)
    #pam.writeResultsToNetCDF('example_ICON_MWR_12-00_v190417_2mom_part14.nc')
    ### End of HATPRO Simulation Part ###

print('Simulation finished, starting next simulation...')
