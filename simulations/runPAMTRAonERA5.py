#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyPamtra
import datetime
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# this creates the input for readERA5
from cdo import *

def runCDO(area=[-30,50,65,89],yyyy=2019,mm=4,dd=1,timestep=1,threads="32",outPath='/tmp/'):
    
    cdo = Cdo()
    yyyy = str(yyyy)
    mm = "%02d" % (mm,)
    dd = "%02d" % (dd,)
    outtime = "%02d" % (timestep-1,)
    area = ','.join([str(coord) for coord in area])
    era5_datetime = yyyy+mm+dd+"_"+outtime

    cdo_string = "-sp2gpl -seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/ml00_1H/"+yyyy+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_130"
    cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_ml_"+era5_datetime+"_130.nc", options='-f nc -P ' + threads)
        
    for var in ['075', '076', '133', '246', '247']:
        cdo_string = "-seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/ml00_1H/"+yyyy+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
    
    for var in ['031','034','129','134','137','151','165','166','172','235']:
        cdo_string = "-seltimestep," + str(timestep) +" -selday,"+dd+" -setgridtype,regular "+"/pool/data/ERA5/sf00_1H/"+yyyy+"/E5sf00_1H_"+yyyy+"-"+mm+"_"+var
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_sf_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)

    return era5_datetime


# In[4]:


def readERA5(era5_datetime,descriptorFile,inPath='/tmp/',debug=False,verbosity=0,step=1,cut_levels=None):
    """
    import ERA5 full model output from netcdf files
    
    netcdf files have been created with cdo by extracting desired timestep and lat/lon region for all variables needed and 
    converting the temperature from spherical harmonics to Gaussian linear grid. 

    2d vars: 031 034 129 134 137 151 165 166 172 235
    cdo -f nc -P $threads -seltimestep,$timestep -selday,$dd -sellonlatbox,$minlon,$maxlon,$minlat,$maxlat -setgridtype,regular gribfile ncfile
    3d vars: 075 076 133 246 247 130
    cdo -f nc -P $threads -sellonlatbox,$minlon,$maxlon,$minlat,$maxlat [-setgridtype,regular|-sp2gpl] -seltimestep,$timestep gribfile ncfile

    era5_datetime: yyyymmmdd_hh of the model output
    descriptorfile: ECMWF descriptorfile
    debug: switch on debugging
    verbosity: increase verbosity
    step: reduce ERA5 grid to nth point in lat/lon
    cut_levels: cut atmosphere from top. This is necessary, cause PAMTRA can not calculate gas absorption for pressure below 3hPa. 
                A sufficiently good value for cut_levels is 5.
    """

    if debug: import pdb;pdb.set_trace()

    R_d = 287.0597
    R_v = 461.5250
    g = 9.80665
    R = 6371229

    # read constant file for pressure level calculation
    dataC = np.genfromtxt('/home/b/b380702/pamtra/data/era5_ecmwf_akbk.csv',usecols=[1,2],skip_header=1,delimiter=',')
    ak, bk = dataC[-1:cut_levels:-1,0],dataC[-1:cut_levels:-1,1]

    # define and get 2d vars
    vals2D = dict()
    vals2D_params = {'031':'ci','034':'sst','129':'z','134':'sp','151':'msl','165':'10u','166':'10v','172':'lsm','235':'skt'}

    for key,value in vals2D_params.items():
        tmp_ds = xr.open_dataset(inPath+'reduced_sf_'+era5_datetime+'_'+key+'.nc')
        vals2D[value] = np.squeeze(np.swapaxes(tmp_ds['var'+str(int(key))].values,1,2))[0::step,0::step]

    # define and get 3d vars
    vals3D = dict()
    vals3D_params = {'075':'crwc','076':'cswc','130':'t','133':'q','246':'clwc','247':'ciwc'}

    for key,value in vals3D_params.items():
        tmp_ds = xr.open_dataset(inPath+'reduced_ml_'+era5_datetime+'_'+key+'.nc')
        vals3D[value] = np.squeeze(np.swapaxes(tmp_ds[value].values,1,3)[...,-1:cut_levels:-1])[0::step,0::step,:]

    # set grid size for the data
    (Nx,Ny,Nz) = vals3D['t'].shape
    nHydro = 4 # ERA5 has 4 hydrometeor classes

    shape2D = (Nx,Ny)
    shape3D = (Nx,Ny,Nz)
    shape3Dplus = (Nx,Ny,Nz+1)
    shape4D = (Nx,Ny,Nz,nHydro)

    # time in seconds since 1970 UTC
    unixtime = np.zeros(shape2D)
    unixtime[:] = tmp_ds['time'][0].astype(int)/ 10**9

    # create empty pamData dict
    pamData = dict()

    pamData['timestamp'] = unixtime

    # create latitude and longitude grid
    pamData['lat'] = np.tile(tmp_ds['lat'][0::step].values,(Nx,1))
    pamData['lon'] = np.tile(tmp_ds['lon'][0::step].values,(Ny,1)).T

    # create temperature field
    pamData['temp'] = vals3D['t']
    pamData['temp_lev'] = np.empty(shape3Dplus)
    pamData['temp_lev'][...,1:-1] = (pamData['temp'][...,1:] + pamData['temp'][...,0:-1])*0.5
    pamData['temp_lev'][...,-1] = pamData['temp_lev'][...,-2]+ (pamData['temp_lev'][...,-2] - pamData['temp_lev'][...,-3])*0.5
    pamData['temp_lev'][...,0] = vals2D['skt'][...]

    # surface geopotential
    z_sfc = vals2D['z'][:,:]

    # height and pressure grid
    pamData['hgt'] = np.empty(shape3D)
    pamData['hgt_lev'] = np.empty(shape3Dplus)
    pamData['press'] = np.empty(shape3D)
    pamData['press_lev'] = np.empty(shape3Dplus)

    pamData['hgt_lev'][...,0] = z_sfc/g
    # pamData['hgt_lev'][...,0] = z_sfc/g*R/(R-z_sfc/g)

    sfc_press = vals2D['sp']
    msl_press = vals2D['msl']

    q = vals3D['q']

    for i in range(Nz+1):
        pamData['press_lev'][...,i] = sfc_press*bk[i]+ak[i]

    pamData['press'][...,:] = (pamData['press_lev'][...,1:] + pamData['press_lev'][...,0:-1])*0.5

    pamData['hydro_q'] = np.zeros(shape4D) + np.nan
    pamData['hydro_q'][...,0] = vals3D['clwc']
    pamData['hydro_q'][...,1] = vals3D['ciwc']
    pamData['hydro_q'][...,2] = vals3D['crwc']
    pamData['hydro_q'][...,3] = vals3D['cswc']

    qh = np.zeros(shape3D)
    qh = np.sum(pamData['hydro_q'],axis=3)

    z = np.zeros(shape2D)
    t_v = np.zeros(shape3D)
    t_v = pamData['temp'][...] * (1+((R_v/R_d)-1)*q)
    pdlog = np.zeros(shape3D)
    pdlog = np.log(pamData['press_lev'][...,0:-1]/pamData['press_lev'][...,1:])
    for k in range(shape3Dplus[2]):
        z[:,:] = 0
        for k2 in range(0,k):
            z[:,:] += R_d*t_v[:,:,k2]*pdlog[:,:,k2]
        z[:,:] = z[:,:] + z_sfc
        pamData['hgt_lev'][:,:,k] = z[:,:]/g
    # pamData['hgt_lev'][...,k] =z/g*R/(R-z/g)

    # create relative humidity field
    pamData['relhum'] = np.empty(shape3D)

    pamData['relhum'][:,:,:] = (pyPamtra.meteoSI.q2rh(q,pamData['temp'][:,:,:],pamData['press'][:,:,:]) * 100.)

    # fill remaining vars that need no conversion
    varPairs = [['10u','wind10u'],['10v','wind10v'],['skt','groundtemp'],['lsm','sfc_slf'],['ci','sfc_sif']]

    for era5Var,pamVar in varPairs:
        pamData[pamVar] = np.zeros(shape2D)
        pamData[pamVar][:,:] = vals2D[era5Var][:,:]

    # surface properties
    pamData['sfc_type'] = np.around(pamData['sfc_slf']).astype('int32')
    pamData['sfc_model'] = np.zeros(shape2D, dtype='int32')
    pamData['sfc_refl'] = np.chararray(shape2D,unicode=True)
    pamData['sfc_refl'][:] = 'F'
    pamData['sfc_refl'][pamData['sfc_type'] > 0] = 'S'

    # sea ice is taken from telsem2 and defined to be Lambertian
    ice_idx = (pamData['sfc_sif'] > 0)
    pamData['sfc_type'][ice_idx] = 1
    pamData['sfc_model'][ice_idx] = 0
    pamData['sfc_refl'][ice_idx] = 'L'

    # create pyPamtra object
    pam = pyPamtra.pyPamtra()
    pam.set['pyVerbose']= verbosity

    # read descriptorfile
    if isinstance(descriptorFile, str):
        pam.df.readFile(descriptorFile)
    else:
        for df in descriptorFile:
            pam.df.addHydrometeor(df)

    # create pam profiles
    pam.createProfile(**pamData)

    pam.addIntegratedValues()

    return pam


# In[5]:


def runPAMTRA(pam, era5_datetime, mode='passive'):
    if mode == 'active':
        # General settings for the Doppler spectra simulations

        pam.nmlSet['active'] = True
        pam.nmlSet['radar_mode'] = 'spectrum' # 'spectrum'
        pam.nmlSet['passive'] = False # Passive is time consuming
        pam.set['verbose'] = 0 # set verbosity levels
        pam.set['pyVerbose'] = 0 # change to 2 if you want to see job progress number on the output
        # pam.p['turb_edr'][:] = 1.0e-4
        pam.nmlSet['radar_airmotion'] = True
        #pam.nmlSet['radar_airmotion_vmin'] = 0.0 # workaround to potential bug in radar_spectrum
        pam.nmlSet['radar_airmotion_model'] = 'constant'

        # Instrument specific settings (W-band radar Joyrad94)
        pam.nmlSet['radar_fwhr_beamwidth_deg']=0.5
        pam.nmlSet['radar_integration_time']=1.0
        pam.nmlSet['radar_max_v']=6.8
        pam.nmlSet['radar_min_v']=-6.8
        pam.nmlSet['radar_nfft']=512
        pam.nmlSet['radar_no_ave']=17
        pam.nmlSet['radar_pnoise0']=-100#-54.0
        pam.runParallelPamtra(np.array([35.5, 94.0, 155.5, 167., 174.8]), pp_deltaX=1000, pp_deltaY=1, pp_deltaF=1)
        pam.writeResultsToNetCDF('/scratch/b/b380702/'+campaign+'_'+flight+'_'+date+'_'+pam.nmlSet["radar_mode"][:3]+'.nc',xarrayCompatibleOutput=True)
    elif mode == 'passive':
        pam.nmlSet['active'] = False
        pam.nmlSet['passive'] = True # Passive is time consuming
        pam.set['verbose'] = 0 # set verbosity levels
        pam.set['pyVerbose'] = 0 # change to 2 if you want to see job progress number on the output
        pam.p['noutlevels'] = 71
        pam.p['obs_height'] = np.zeros((pam._shape2D[0],pam._shape2D[1],pam.p['noutlevels']))
        pam.p['obs_height'][:,:,:] = [833000., 12000., 11900., 11800., 11700., 11600., 11500., 11400., 11300., 11200.,
           11100., 11000., 10900., 10800., 10700., 10600., 10500., 10400.,
           10300., 10200., 10100., 10000.,  9900.,  9800.,  9700.,  9600.,
            9500.,  9400.,  9300.,  9200.,  9100.,  9000., 8900., 8800., 8700., 8600., 8500., 5200., 5100., 5000., 4900., 4800., 4700., 4600., 4500., 4400., 4300., 4200., 4100., 4000., 3900., 3800., 3700., 3600.,
           3500., 3400., 3300., 3200., 3100., 3000., 2900., 2800., 2700.,
           2600., 2500., 2400., 2300., 2200., 2100., 2000., 0.]

        freqs = np.array([22.24,23.04,23.84,25.44,26.24,27.84,31.4,50.3,51.76,52.8,53.75,54.94,56.66,58.,89.,90.,
          110.25,114.55,116.45,117.35,120.15,121.05,122.95,127.25,155.5,167.,170.81,174.8,175.81,178.31,179.81,180.81,181.81,182.71,
          183.91,184.81,185.81,186.81,188.31,190.81,195.81,243.,340.])
        #freqs = np.array([89.])
        pam.runParallelPamtra(freqs, pp_deltaX=1, pp_deltaY=1, pp_deltaF=1,pp_local_workers=64)
        pam.addIntegratedValues()
        pam.writeResultsToNetCDF('/scratch/b/b380702/'+era5_datetime+'_passive.nc',xarrayCompatibleOutput=True,wpNames=['cwp','iwp','rwp','swp'])
    else:
        pam.addIntegratedValues()
        print('Just reading')
    return pam


# In[6]:


era5_datetime = runCDO(yyyy=2017,mm=5,dd=27,timestep=13,outPath='/scratch/b/b380702/')


# In[7]:


pam = readERA5(era5_datetime,'/home/b/b380702/pamtra/descriptorfiles/descriptor_file_ecmwf.txt',inPath='/scratch/b/b380702/',step=4,cut_levels=5)


# In[8]:


pam._shape2D


# In[9]:

# reduce to just ocean grid points
filter = np.empty(pam._shape2D,dtype=bool)
filter[:,:] = False
filter[pam.p['sfc_type'] == 0] = True
pam.filterProfiles(filter)


# In[10]:


pam._shape2D


# In[ ]:


runPAMTRA(pam,era5_datetime)

# In[7]:


pam.r['tb'].shape


# In[11]:


def plotDataHyd(lon,lat,data):

    data[data < 0.05] = np.nan

    map_proj=ccrs.Mollweide(central_longitude=-30)
    data_proj=ccrs.PlateCarree()

    ax = plt.subplot(221,projection=map_proj)
    ax.coastlines()

    plt.pcolormesh(lon,lat,data[:,:,0],transform=data_proj,cmap='jet')
    plt.colorbar()

    ax = plt.subplot(222,projection=map_proj)
    ax.coastlines()

    plt.pcolormesh(lon,lat,data[:,:,1],transform=data_proj,cmap='jet')
    plt.colorbar()

    ax = plt.subplot(223,projection=map_proj)
    ax.coastlines()

    plt.pcolormesh(lon,lat,data[:,:,2],transform=data_proj,cmap='jet')
    plt.colorbar()

    ax = plt.subplot(224,projection=map_proj)
    ax.coastlines()

    plt.pcolormesh(lon,lat,data[:,:,3],transform=data_proj,cmap='jet')
    plt.colorbar()

    plt.show()

    return

def plotMap(lon,lat,data):
    proj = ccrs.NorthPolarStereo(central_longitude=10)
    data_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.coastlines()
    plt.pcolormesh(lon,lat,data[:,:],transform=data_crs,cmap='jet')
    plt.colorbar()

    plt.show()

    return


# In[22]:


#plt.plot(pam.r['tb'][:,0,0,0,0])
plotMap(pam.p['lon'],pam.p['lat'],pam.r['tb'][:,:,0,6,0,0])
# plotDataHyd(pam.p['lon'],pam.p['lat'],pam.p['hydro_wp'])




