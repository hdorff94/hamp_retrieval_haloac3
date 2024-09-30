"""
This script contains two classes primarily needed for calculating Brightness Temperatures TB for radiometers using ERA5 data.
ERA5_Preprocessing is used to prepare the ERA5 data for usage in the Forward Simulator PAMTRA.
PAMTRA_Handler is a class in order to handle and conduct the actual forward simulations with the preprocessed ERA5 data.

"""

###
"""
notes pamtra reupdate/install on levante":
module load python3/2023.01-gcc-11.2.0
spack load gcc@11.2.0%gcc@11.2.0
spack load /fnfhvr6
spack load openblas%gcc
module load netcdf-fortran/4.5.3-gcc-11.2.0
sed 's/-llapack//g' Makefile > Makefile.levante
sed -i 's/-lblas/-lopenblas/g' Makefile.levante
make -f Makefile.levante clean
make -f Makefile.levante
make pyinstall -f Makefile.levante
ls -l $HOME/lib/python
"""

import datetime

import os

import pickle
import numpy as np

import pyPamtra
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
get_ipython().run_line_magic('matplotlib', 'inline')

# this creates the input for readERA5
from cdo import *

class ERA5_Preprocessing():
    
    def __init__(self,yyyy,mm,dd,descriptor_file,timestep=1,outPath="/tmp/",area=[-30,50,65,89],campaign="HALO_AC3"):
        self.threads              = "32"
        self.outPath              = outPath
        self.yyyy                 = yyyy
        self.mm                   = mm
        self.dd                   = dd
        self.descriptor_file      = descriptor_file
        self.area                 = area
        self.timestep             = timestep
        self.era_ml_path          = "/pool/data/ERA5/E5/ml/an/1H/" # old path "/pool/data/ERA5/ml00_1H/"
        self.era_sfc_path         = "/pool/data/ERA5/E5/sf/an/1H/"# old path "/pool/data/ERA5/sf00_1H/"
        yyyy                      = str(self.yyyy)
        mm                        = "%02d" % (self.mm,)
        dd                        = "%02d" % (self.dd,)
        
        self.date                 = yyyy+mm+dd
        
        self.outtime              = "%02d" % (self.timestep-1,)
        self.campaign             = campaign
        self.era5_datetime        = self.date+"_"+self.outtime
        print("Version of 2023-05-16 runned")
    
    def runCDO(self, run_although_already_processed=False): 
        """
        the default ERA5 files are in grib format and need to be slightly processed 
        before transforming to netCDF for later handling
        """
        # first check if file is already being processed
        
        yyyy = str(self.yyyy)
        mm = "%02d" % (self.mm,)
        dd = "%02d" % (self.dd,)
        
        cdo = Cdo()
        area = ','.join([str(coord) for coord in self.area])
        # Run the CDO commands
        base_file=self.outPath+"reduced_ml_"+yyyy+mm+dd+"_"+self.outtime+"_130.nc"
        print("File to check:",base_file)
        
        if not os.path.exists(base_file): 
            for var in ["130","152"]:
                cdo_string = "-sp2gpl -seltimestep," + str(self.timestep) +\
                        " -setgridtype,regular "+self.era_ml_path+\
                            var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
                cdo.sellonlatbox(area,input=cdo_string, output=self.outPath+"reduced_ml_"+self.era5_datetime+"_"+var+".nc",
                         options='-f nc -P ' + self.threads)
        ##############################################################################################################
        # ERA Variables are labelled with numbers
        # 129 variable has changed from surface level to model level
        # Model level files
        new_ml_file_counter=0
        for var in ['075', '076','133', '246', '247']:
            var_ml_file=self.outPath+"reduced_ml_"+yyyy+mm+dd+"_"+self.outtime+"_"+var+".nc"
            if not os.path.exists(var_ml_file):
                # old --> this was the old levante data storage configuration
                #cdo_string = "-seltimestep," + str(self.timestep) +" -setgridtype,regular "+self.era_ml_path+\
                #                yyyy+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var
                # new --> /data/ERA5/E5/ml/an/1H/135/E5ml00_1H_1979-03-23_135.grb
                cdo_string= "-seltimestep," + str(self.timestep) +" -setgridtype,regular "+self.era_ml_path+\
                            var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
                cdo.sellonlatbox(area,input=cdo_string, output=self.outPath+"reduced_ml_"+self.era5_datetime+"_"+var+".nc",
                             options='-f nc -P ' + self.threads)
                new_ml_file_counter+=1
            else:
                pass
        
        if new_ml_file_counter==0:
            print("All ml files already calculated for ",yyyy+mm+dd)
        
        # Surface level files
        new_sf_file_counter=0
        for var in ['031','034','134','137','151','165','166','235']:
            var_sf_file=self.outPath+"reduced_sf_"+yyyy+mm+dd+"_"+self.outtime+"_"+var+".nc"
            if not os.path.exists(var_sf_file):
                file_to_look_for=self.era_sfc_path+"/"+var+"/E5sf00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
                cdo_string = "-seltimestep," + str(self.timestep) +" -selday,"+dd+" -setgridtype,regular "+file_to_look_for
                cdo.sellonlatbox(area,input=cdo_string, output=self.outPath+"reduced_sf_"+self.era5_datetime+"_"+var+".nc", 
                             options='-f nc -P ' + self.threads)
                new_sf_file_counter+=1
        
        if new_sf_file_counter==0:
            print("All sf files already calculated for ",yyyy+mm+dd)
        else:
            print("All sf files now calculated")
        # Invariant data
        for var in ['129','172']:
            cdo_string = " -setgridtype,regular "+"/pool/data/ERA5/E5/sf/an/IV/"+var+\
              "/E5sf00_IV_2000-01-01_"+var+".grb"
            cdo.sellonlatbox(area,input=cdo_string, output=self.outPath+"reduced_sf_"+self.era5_datetime+"_"+var+".nc", 
                             options='-f nc -P ' + self.threads)
        print("all IV files calculated")
        
    def get_sonde_ColumnfromERA5(self,point=[10.,78.],minute=0):
        """
        Interpolate to a column from ERA5 model output, selects timestep, transforms grid, and stores as netcdf files.
    
        point: array integer giving the point of choice [lon,lat]
        yyyy: integer year
        mm:   integer month
        dd:   integer day of month
        hour: integer starting with 0 = 0 UTC, 12 = 12 UTC
        threads: integer giving threads to be used for the cdo process
        outPath: string giving the path where to store the output
        """
        yyyy    = str(self.yyyy)
        mm      = "%02d" % (self.mm,)
        dd      = "%02d" % (self.dd,)
        outhour = self.outtime 
        outmin  = "%02d" % (minute,)
        point   = "lon="+"_lat".join([str(coord) for coord in point])
        profile_output="/scratch/u/u300737/col_%s.nc" % (outFile)
        cdo     = Cdo()
        
        infiles = list()
        
        for var in ["130","152"]:
            cdo_string = " -sp2gpl -seltimestep," + str(self.timestep) +\
                            " -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+\
                                var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
            cdo.remapnn(point,input=cdo_string,output=self.outPath+"col_ml_"+self.era5_datetime+\
                        "_"+var+".nc",options="-f nc -P" + self.threads)
            infiles.append("/scratch/u/u300737/"+self.campaign+"/col_ml_"+self.era5_datetime+"_"+var+".nc")
        for var in ['075', '076', '133', '246', '247']:
            cdo_string = "-seltimestep," + str(self.timestep) +" -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+\
                            var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
            cdo.remapnn(point,input=cdo_string, 
                        output=self.outPath+"col_ml_"+self.era5_datetime+"_"+var+".nc", 
                        options='-f nc -P ' + self.threads)
            infiles.append("/scratch/u/u300737/"+self.campaign+"/col_ml_"+\
                           self.era5_datetime+"_"+var+".nc")
    
        for var in ['031','034','129','134','137','151','165','166','172','235']:
            cdo_string = "-seltimestep," + str(self.timestep) +" -selday,"+dd+" -setgridtype,regular "+\
            "/pool/data/ERA5/E5/sf/an/1H/"+var+"/E5sf00_1H_"+yyyy+"-"+mm+"_"+var+".grb"
            cdo.remapnn(point,input=cdo_string, 
                        output=self.outPath+"col_sf_"+self.era5_datetime+"_"+var+".nc",
                        options='-f nc -P ' + self.threads)
            infiles.append()
            #infiles.append("/scratch/b/b380702/"+self.campaign+"/col_sf_"+self.era5_datetime+"_"+var+".nc")

        cdo.merge(input=infiles, output=profile_output)
        
    def readERA5(self,inPath='/tmp/',debug=False,verbosity=0,step=1,cut_levels=5):
        """
        import ERA5 full model output from netcdf files
    
        netcdf files have been created with cdo by extracting desired timestep and lat/lon region 
        for all variables needed and converting the temperature from spherical harmonics to Gaussian linear grid. 

        2d vars: 031 034 129 134 137 151 165 166 172 235
        cdo -f nc -P $threads -seltimestep,$timestep -selday,$dd -sellonlatbox,
            $minlon,$maxlon,$minlat,$maxlat -setgridtype,regular gribfile ncfile
        3d vars: 075 076 133 246 247 130
        cdo -f nc -P $threads -sellonlatbox,$minlon,$maxlon,$minlat,$maxlat
            [-setgridtype,regular|-sp2gpl] -seltimestep,$timestep gribfile ncfile

        era5_datetime: yyyymmmdd_hh of the model output
        descriptorfile: ECMWF descriptorfile
        debug: switch on debugging
        verbosity: increase verbosity
        step: reduce ERA5 grid to nth point in lat/lon
        cut_levels: cut atmosphere from top. This is necessary, 
                        cause PAMTRA can not calculate gas absorption for pressure below 3hPa. 
                        A sufficiently good value for cut_levels is 5.
        """

        if debug: import pdb;pdb.set_trace()
        # parameters
        self.R_d = 287.0597
        self.R_v = 461.5250
        self.g = 9.80665
        self.R = 6371229

        # read constant file for pressure level calculation
        dataC = np.genfromtxt('/home/b/b380702/pamtra/data/era5_ecmwf_akbk.csv',
                              usecols=[1,2],skip_header=1,delimiter=',')
        self.ak, self.bk = dataC[-1:cut_levels:-1,0],dataC[-1:cut_levels:-1,1]

        # define and get 2d vars
        self.vals2D = dict()
        vals2D_params = {'031':'ci','034':'sst',
                         '129':'z','134':'sp',
                         '151':'msl','165':'10u',
                         '166':'10v','172':'lsm',
                         '235':'skt'}

        for key,value in vals2D_params.items():
            self.tmp_ds = xr.open_dataset(inPath+'reduced_sf_'+self.era5_datetime+'_'+key+'.nc')
            self.vals2D[value] = np.squeeze(np.swapaxes(self.tmp_ds['var'+str(int(key))].values,1,2))[0::step,0::step]
        
        # get logarithm of surface presure, a_coef and b_coef
        self.tmp_ds             = xr.open_dataset(inPath+"reduced_ml_"+self.era5_datetime+"_152.nc")
        self.vals2D["lnsp"]     = np.squeeze(np.swapaxes(self.tmp_ds["lnsp"].values,2,3))[0::step,0::step]
        self.a_coef             = self.tmp_ds["hyai"].values[cut_levels:]
        self.b_coef             = self.tmp_ds["hybi"].values[cut_levels:]
        
        # define and get 3d vars
        self.vals3D = dict()
        vals3D_params = {'075':'crwc','076':'cswc',
                         '130':'t','133':'q',
                         '246':'clwc','247':'ciwc'}

        for key,value in vals3D_params.items():
            self.tmp_ds = xr.open_dataset(inPath+'reduced_ml_'+self.era5_datetime+'_'+key+'.nc')
            self.vals3D[value] = np.squeeze(np.swapaxes(self.tmp_ds[value].values,1,3)[...,-1:cut_levels:-1])[0::step,0::step,:]

        # set grid size for the data
        (self.Nx,self.Ny,self.Nz) = self.vals3D['t'].shape
        self.nHydro = 4 # ERA5 has 4 hydrometeor classes

        self.shape2D = (self.Nx,self.Ny)
        self.shape3D = (self.Nx,self.Ny,self.Nz)
        self.shape3Dplus = (self.Nx,self.Ny,self.Nz+1)
        self.shape4D = (self.Nx,self.Ny,self.Nz,self.nHydro)

        # time in seconds since 1970 UTC
        self.unixtime = np.zeros(self.shape2D)
        self.unixtime[:] = self.tmp_ds['time'][0].astype(int)/ 10**9
    
    def create_pyPamtra_obj(self,verbosity=0):
        self.pam = pyPamtra.pyPamtra()
        self.pam.set['pyVerbose']= verbosity

        # read descriptorfile
        if isinstance(self.descriptor_file, str):
            self.pam.df.readFile(self.descriptor_file)
        else:
            for df in self.descriptorFile:
                self.pam.df.addHydrometeor(df)

        # create pam profiles
        self.pam.createProfile(**self.pamData)
        self.pam.addIntegratedValues()
    
    def compute_z_level(self,t,q,z_h,p_lev,p_levpo,lev):
        '''Compute z at half- and full-level for the given level, based on t/q/p'''

        # compute moist temperature
        t_v = t * (1. + 0.609133 * q)

        if lev == 0:
            dlog_p = np.log(p_levpo / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(p_levpo / p_lev)
            alpha = 1. - ((p_lev / (p_levpo - p_lev)) * dlog_p)

        t_v = t_v * self.R_d

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        z_f = z_h + (t_v * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (t_v * dlog_p)

        return z_h, z_f

    def create_pamData_dict(self,step=1):
        pamData = dict()

        pamData['timestamp'] = self.unixtime

        # create latitude and longitude grid
        pamData['lat'] = np.tile(self.tmp_ds['lat'][0::step].values,(self.Nx,1))
        pamData['lon'] = np.tile(self.tmp_ds['lon'][0::step].values,(self.Ny,1)).T

        # create temperature field
        pamData['temp'] = self.vals3D['t']
        pamData['temp_lev'] = np.empty(self.shape3Dplus)
        pamData['temp_lev'][...,1:-1] = (pamData['temp'][...,1:] + pamData['temp'][...,0:-1])*0.5
        pamData['temp_lev'][...,-1] = pamData['temp_lev'][...,-2]+\
                                    (pamData['temp_lev'][...,-2] - pamData['temp_lev'][...,-3])*0.5
        pamData['temp_lev'][...,0] = self.vals2D['skt'][...]

        # surface geopotential
        z_sfc = self.vals2D['z'][:,:]

        # height and pressure grid
        pamData['hgt'] = np.empty(self.shape3D)
        pamData['hgt_lev'] = np.empty(self.shape3Dplus)
        pamData['press'] = np.empty(self.shape3D)
        pamData['press_lev'] = np.empty(self.shape3Dplus)
        pamData['hgt_lev'][...,0] = z_sfc/self.g
        # pamData['hgt_lev'][...,0] = z_sfc/g*R/(R-z_sfc/g)

        sfc_press = self.vals2D['sp']
        msl_press = self.vals2D['msl']

        q = self.vals3D['q']
        
        # Calc p levels. They are now updated with the surface pressure field
        pamData["press_lev"][:,:,-1] = np.exp(self.vals2D["lnsp"][:,:])
        pamData["press_lev"][...,0]  = self.a_coef[0] + (self.b_coef[0]*pamData["press_lev"][:,:,-1]) 
        for i in range(self.Nz+1):
            pamData['press_lev'][...,i] = self.a_coef[i]+(self.b_coef[i]*np.exp(self.vals2D['lnsp']))
            pamData['press'][...,i-1] = (pamData['press_lev'][...,i-1] + pamData['press_lev'][...,i])*0.5
        
        # Z levels
        z      = np.zeros(self.shape2D)
        t_v    = np.zeros(self.shape3D)
        t_v    = pamData['temp'][...] * (1+((self.R_v/self.R_d)-1)*q)
        z_h    = self.vals2D["z"][:,:]
        tmp_t  = pamData["temp"][...,::-1]
        tmp_q  = self.vals3D["q"][...,::-1]
        pamData["hgt_lev"][...,-1] = z_h/self.g
        
        for i in sorted(range(self.Nz),reverse=True):
            z_h, z_f= self.compute_z_level(tmp_t[...,i],tmp_q[...,i],z_h,
                                           pamData["press_lev"][0,0,i],
                                           pamData["press_lev"][0,0,i+1],i)
            pamData["hgt"][...,i]     = z_f/self.g
            pamData["hgt_lev"][...,i] = z_h/self.g
        # reverse levels of pressure and height variables    
        for height_var in ["press","press_lev","hgt","hgt_lev"]:
            pamData[height_var]=pamData[height_var][...,::-1]
        #pdlog = np.zeros(self.shape3D)
        #pdlog = np.log(pamData['press_lev'][...,0:-1]/pamData['press_lev'][...,1:])
        ### -----> change
        #for k in range(self.shape3Dplus[2]):
        #    z[:,:] = 0
        #    for k2 in range(0,k):
        #        z[:,:] += self.R_d*t_v[:,:,k2]*pdlog[:,:,k2]
        #    z[:,:] = z[:,:] + z_sfc
        #    pamData['hgt_lev'][:,:,k] = z[:,:]/self.g
        # pamData['hgt_lev'][...,k] =z/g*R/(R-z/g)
        #pamData['hgt'] = (pamData['hgt_lev'][...,1:] + pamData['hgt_lev'][...,:-1])*0.5
        
        # hydrometeors
        pamData['hydro_q'] = np.zeros(self.shape4D) + np.nan
        pamData['hydro_q'][...,0] = self.vals3D['clwc']
        pamData['hydro_q'][...,1] = self.vals3D['ciwc']
        pamData['hydro_q'][...,2] = self.vals3D['crwc']
        pamData['hydro_q'][...,3] = self.vals3D['cswc']
        
        # create relative humidity field
        qh = np.zeros(self.shape3D)
        qh = np.sum(pamData['hydro_q'],axis=3)

        
        pamData['relhum'] = np.empty(self.shape3D)

        pamData['relhum'][:,:,:] = (pyPamtra.meteoSI.q2rh(q,pamData['temp'][:,:,:],
                                                          pamData['press'][:,:,:]) * 100.)

        # fill remaining vars that need no conversion
        varPairs = [['10u','wind10u'],
                    ['10v','wind10v'],
                    ['skt','groundtemp'],
                    ['lsm','sfc_slf'],
                    ['ci','sfc_sif']]
        
        for era5Var,pamVar in varPairs:
            pamData[pamVar] = np.zeros(self.shape2D)
            pamData[pamVar][:,:] = self.vals2D[era5Var][:,:]
            
        # surface properties
        pamData['sfc_type'] = np.around(pamData['sfc_slf']).astype('int32')
        pamData['sfc_model'] = np.zeros(self.shape2D, dtype='int32')
        pamData['sfc_refl'] = np.chararray(self.shape2D,unicode=True)
        pamData['sfc_refl'][:] = 'F'
        pamData['sfc_refl'][pamData['sfc_type'] > 0] = 'S'

        # sea ice is taken from telsem2 and defined to be Lambertian
        ice_idx = (pamData['sfc_sif'] > 0)
        pamData['sfc_type'][ice_idx] = 1
        pamData['sfc_model'][ice_idx] = 0
        pamData['sfc_refl'][ice_idx] = 'L'
        self.pamData=pamData
        self.create_pyPamtra_obj()
    
    def checkfor_era5_pamtra_files(self):
        era5_file_there=False
        pamtra_file_there=False
        #era5 file
        era5_fname=self.outPath+"era5_"+self.date+"_"+self.outtime+"_atmos.nc"
        if os.path.exists(era5_fname):
            era5_file_there=True
        pamtra_fname=self.outPath+"pamtra_hamp_"+self.date+"_"+self.outtime+".nc"
        if os.path.exists(pamtra_fname):
            pamtra_file_there=True
        return era5_file_there,pamtra_file_there
#####################################################################################################################################################
#####################################################################################################################################################
class PAMTRA_Handler(ERA5_Preprocessing):
    def __init__(self,ERA5_Preprocessing,
        output_heights=[833000.,16000., 
                        15400.,15200.,
                        15000.,14800.,
                        14600.,14400.,
                        14200.,14000.,
                        13800.,13600.,
                        13400,13200,
                        13000,12800,
                        12600,12400,
                        12200,12000., 
                        11800.,11600.,
                        11400.,11200.,
                        11000.,10800.,
                        10600.,10400.,
                        10200.,10000.,
                        9800.,9600.,
                        9400.,9200.,
                        9000., 8800.,
                        8600.,8400.,
                        8200.,8000.,
                        7800.,7600.,
                        7400.,7200.,
                        7000.]):
        #         [833000.,16000, 15500,15000,14500,14000,13000,12500,12000.,11900., 11800., 11700., 11600., 11500., 
        #                11400., 11300., 11200.,11100., 11000.,10900., 10800., 10700., 10600., 10500., 10400.,10300.,
        #                10200., 10100., 10000., 9900.,  9800.,  9700.,  9600., 9500.,  9400.,  9300.,  9200.,  9100.,
        #                9000., 8900., 8800.,8700., 8600., 8500.,#8400,8300,8200,8100,8000,
        #                0.]):
        self.Grid_Data_cls=ERA5_Preprocessing
        if hasattr(self.Grid_Data_cls,"pam"):
            self.pam=self.Grid_Data_cls.pam
        else:
            self.pam = pyPamtra.pyPamtra()
                
        self.output_heights=output_heights
        self.outPath=ERA5_Preprocessing.outPath
        self.era5_datetime=self.Grid_Data_cls.era5_datetime
        
    def runPAMTRA(self, mode='passive'):
        print(mode)
        if mode == 'active':
            pass # so far unused
            """
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
            """
        elif mode == 'passive':
            # these output levels can be easily changed. Accordingly, the dimension are changed
            # old default
            #[833000.,16000, 15500,15000,14500,14000,13000,12500,12000., 11900., 11800., 11700., 11600., 11500., 11400., 11300., 11200.,
            #                               11100., 11000., 10900., 10800., 10700., 10600., 10500., 10400.,
            #                               10300., 10200., 10100., 10000.,  9900.,  9800.,  9700.,  9600.,
            #                               9500.,  9400.,  9300.,  9200.,  9100.,  9000., 8900., 8800., 8700., 8600., 8500.,
#                                           5200., 5100., 5000., 4900., 4800., 4700., 4600., 4500., 4400., 4300., 4200., 4100., 4000.,
#                                           3900., 3800., 3700., 3600.,3500., 3400., 3300., 3200., 3100., 3000., 2900., 2800., 2700.,
#                                           2600., 2500., 2400., 2300., 2200., 2100., 2000., 
            #                                   0.]
            self.pam.nmlSet['active'] = False
            self.pam.nmlSet['passive'] = True # Passive is time consuming
            self.pam.set['verbose'] = 0 # set verbosity levels
            self.pam.set['pyVerbose'] = 0 # change to 2 if you want to see job progress number on the output
            self.pam.p['noutlevels'] = len(self.output_heights)
            
            #print(self.output_heights)
            #print("noutlevels",self.pam.p['noutlevels'])
            
            self.pam.p['obs_height'] = np.zeros((self.pam._shape2D[0],self.pam._shape2D[1],self.pam.p['noutlevels']))
            self.pam.p['obs_height'][:,:,:] = self.output_heights
            freqs = np.array([22.24, 23.04, 23.84,
                              25.44, 26.24, 27.84,
                              31.4,   50.3, 51.76,
                              52.8,  53.75,  54.94,
                              56.66, 58.,  89.,
                              90.,110.25,114.55,
                              116.45,117.35,120.15,
                              121.05,122.95,127.25,
                              155.5, 167., 170.81, 
                              174.8,175.81,178.31,
                              179.81, 180.81, 181.81,
                              182.71, 183.91,184.81,
                              185.81,186.81,188.31,
                              190.81,195.81,243.,
                              340.])
            
            #freqs = np.array([89.])
            print("OUTPUT path:",self.outPath)
            self.pam.runParallelPamtra(freqs, pp_deltaX=1, pp_deltaY=1,
                              pp_deltaF=1,pp_local_workers=64)
            self.pam.addIntegratedValues()
            #'/scratch/b/b380702/'+self.Grid_Data_cls.era5_datetime+\
            outFile=self.Grid_Data_cls.era5_datetime+'_passive.nc'
            self.pam.writeResultsToNetCDF(self.outPath+outFile,
                                          xarrayCompatibleOutput=True,wpNames=['cwp','iwp','rwp','swp'])
            print("PAMTRA Simulation stored as:",self.outPath+outFile)
            pickle.dump(self.pam, open(self.outPath+self.Grid_Data_cls.era5_datetime+"_pam.pkl", "wb" ))
        else:
            self.pam.addIntegratedValues()
            print('Just reading')

    def collectERA5(self,inPath='/scratch/u/u300737/',step=4,cut_levels=5, applyFilter=False):
        """
        Collects the ERA5 data from nc files generated by runCDO, adds integrated values, and dumps everything to nc-files.

        applyFilter  if True, only selects ocean pixels
        """
        def addIntegratedValues():
            """
            Sums up hydrometeors and integrated  water vapor from bottom to all observation heights 
            in pam object. This results in dataset of size (x,y,nout).
            Temperature, pressure and humidity are stored as full grid.
            """
            def calcMoistRho():
                """
                Calculates the wet density of gridbox.
                """
                self.pam._helperP = dict()
                self.pam._helperP['dz'] = self.pam.p['hgt_lev'][...,1::]-self.pam.p['hgt_lev'][...,0:-1]
                self.pam._helperP['vapor'] = rh2q(self.pam.p['relhum']/100.,self.pam.p['temp'],pam.p['press'])
                self.pam._helperP['sum_hydro_q'] = np.nansum(self.pam.p['hydro_q'],axis=-1)
                self.pam._helperP['rho_moist']=\
                moist_rho_rh(self.pam.p['press'],self.pam.p['temp'],self.pam.p['relhum']/100.,self.pam._helperP['sum_hydro_q'])

                return

            self.pam._shape4Dret = (self.pam.p["ngridx"],self.pam.p["ngridy"],self.pam.p['noutlevels'],self.pam.df.nhydro)
            self.pam._shape3Dret = (self.pam.p["ngridx"],self.pam.p["ngridy"],self.pam.p['noutlevels'])
            self.pam.p['hydro_wp'] = np.zeros(self.pam._shape4Dret)
            self.pam._calcMoistRho() # provies as well dz, sum_hydro_q, and q within dict() self._helperP
            self.pam.p['iwv'] = np.zeros(self.pam._shape3Dret)
            self.pam.p['noutlevels'] = len(self.output_heights)
            for h in range(self.pam.p['noutlevels']):
                target_height = self.pam.p['obs_height'][0,0,h]
                #print("Target height",target_height)
                for ix in range(self.pam.p["ngridx"]):
                    zgrid = self.pam.p['hgt_lev'][ix,0,:]
                    z_diffs = np.absolute(zgrid - target_height)
                    z_index = np.argmin(z_diffs)+1
                    self.pam.p['iwv'][...,h] = np.nansum(self.pam._helperP['vapor'][...,0:z_index]*\
                                                         self.pam._helperP["rho_moist"][...,0:z_index]*\
                                                         self.pam._helperP["dz"][...,0:z_index],axis=-1)
                    #nothing to do without hydrometeors:
                    if np.all(self.pam.p['hydro_q']==0):
                        self.p['hydro_wp'] = np.zeros(self.pam._shape4Dret)
                    else:
                        for i in range(self.pam.df.nhydro):
                            self.pam.p['hydro_wp'][...,h,i] = np.nansum(self.pam.p['hydro_q'][...,0:z_index,i]*\
                                                                        self.pam._helperP["rho_moist"][...,0:z_index]*\
                                                                        self.pam._helperP["dz"][...,0:z_index],axis=-1)
        self.readERA5(inPath=inPath,step=step,cut_levels=cut_levels)
        self.pam.p['obs_height'] = np.zeros((self.pam._shape2D[0],self.pam._shape2D[1],self.pam.p['noutlevels']))
        self.pam.p['obs_height'][:,:,:] = self.output_heights

        # changed for new obs height levels -20230511-
        #self.pam.p['obs_height'][:,:,:] = [833000., 12000., 11900., 11800., 11700., 
        #    11600., 11500., 11400., 11300., 11200.,
        #   11100., 11000., 10900., 10800., 10700., 10600., 10500., 10400.,
        #   10300., 10200., 10100., 10000.,  9900.,  9800.,  9700.,  9600.,
        #    9500.,  9400.,  9300.,  9200.,  9100.,  9000., 8900., 8800., 8700., 
        #    8600., 8500.,5200., 5100., 5000., 4900., 4800., 4700., 4600., 4500., 
        #    4400., 4300., 4200., 4100., 4000., 3900., 3800., 3700., 3600.,
        #    3500., 3400., 3300., 3200., 3100., 3000., 2900., 2800., 2700.,
        #    2600., 2500., 2400., 2300., 2200., 2100., 2000., 0.]
        if applyFilter:
            filter = np.empty(self.pam._shape2D,dtype=bool)
            filter[:,:] = False
            filter[self.pam.p['sfc_type'] == 0] = True
            self.pam.filterProfiles(filter)
        addIntegratedValues()
        self.era5_ds = xr.Dataset(
            {"unixtime": (("x", "y"), self.pam.p['unixtime'][...]),
             "lat": (("x", "y"), self.pam.p['lat'][...]),
             "lon": (("x", "y"), self.pam.p['lon'][...]),
             "obs_height": (("nout"), self.pam.p['obs_height'][0,0,:]),
             "sfc_slf": (("x", "y"), self.pam.p['sfc_slf'][...]),
             "sfc_sif": (("x", "y"), self.pam.p['sfc_sif'][...]),
             "groundtemp": (("x", "y"), self.pam.p['groundtemp'][...]),
             "hgt": (("x", "y", "z"), self.pam.p['hgt'][...]),
             "t": (("x", "y", "z"), self.pam.p['temp'][...]),
             "rh":(("x", "y", "z"), self.pam.p['relhum'][...]),
             "p":(("x", "y", "z"), self.pam.p['press'][...]),
             "iwv":(("x", "y", "nout"), self.pam.p['iwv'][...]),
             "lwp":(("x", "y", "nout"), self.pam.p['hydro_wp'][...,0]),
             "iwp":(("x", "y", "nout"), self.pam.p['hydro_wp'][...,1]),
             "rwp":(("x", "y", "nout"), self.pam.p['hydro_wp'][...,2]),
             "swp":(("x", "y", "nout"), self.pam.p['hydro_wp'][...,3]),},)
        
        self.era5_ds['unixtime'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        self.era5_ds['obs_height'].attrs['units'] = 'm'
        self.era5_ds['groundtemp'].attrs['units'] = 'K'
        self.era5_ds['hgt'].attrs['units'] = 'm'
        self.era5_ds['t'].attrs['units'] = 'K'
        self.era5_ds['rh'].attrs['units'] = '%'
        self.era5_ds['p'].attrs['units'] = 'Pa'
        self.era5_ds['iwv'].attrs['units'] = 'kg/m^2'
        self.era5_ds['lwp'].attrs['units'] = 'kg/m^2'
        self.era5_ds['iwp'].attrs['units'] = 'kg/m^2'
        self.era5_ds['rwp'].attrs['units'] = 'kg/m^2'
        self.era5_ds['swp'].attrs['units'] = 'kg/m^2'
        outFile='era5_'+self.era5_datetime+"_atmos.nc"
        self.era5_ds.to_netcdf(self.outPath + outFile)
        print("ERA Files saved as:",self.outPath+outFile)
        return self.pam
        
    def reducePAMTRAResults(self,instrument='mirac-a'):#,outPath='/scratch/b/b380702/'):
        """
        only usable for entire freq calculations
        """
        def read_attributes():
            """
            Read variable definitions catalog
    
            Returns
            -------
            cat

            """
            import yaml
            fname='updated_instrument_settings.yaml'
            with open(fname,'r') as f:#/work/bb1320/scripts/instrument_settings.yaml', 'r') as f:

                cat = yaml.safe_load(f)

            return cat
        
        cat = read_attributes()
        out_slice = slice(cat[instrument]['obs_heights'][0],cat[instrument]['obs_heights'][1]+1)
        ang_slice = slice(cat[instrument]['angles'][0],cat[instrument]['angles'][1]+1)
        freq_slice = slice(cat[instrument]['frequencies'][0],cat[instrument]['frequencies'][1]+1)
        tb = np.zeros((self.pam.p['ngridx'],self.pam.p['ngridy'],cat[instrument]['nout'],
                       cat[instrument]['nang'],cat[instrument]['nfreq'],cat[instrument]['npol']))
        tb[:,:,:,:,:,:] = self.pam.r['tb'][:,:,out_slice,ang_slice,freq_slice,:]
        #import pdb;pdb.set_trace()
        self.pam_ds = xr.Dataset(
            {"unixtime": (("x", "y"), self.pam.p['unixtime'][...]),
             "lat": (("x", "y"), self.pam.p['lat'][...]),
             "lon": (("x", "y"), self.pam.p['lon'][...]),
             "obs_height": (['nout'], self.pam.p['obs_height'][0,0,
                            slice(cat[instrument]['obs_heights'][0],cat[instrument]['obs_heights'][1]+1)]),
             "ang": (['nang'], np.absolute(self.pam.r['angles_deg'][ang_slice]-180.)),
             "freq": (['nfreq'], self.pam.set['freqs'][freq_slice]),
             "pol": (['npol'],['H','V']),
             "tb": (("x", "y","nout","nang","nfreq","npol"), tb),
            },)
        self.pam_ds['tb'].attrs['units'] = 'K'
        self.pam_ds.attrs['description'] = 'Reduced simulated brightness temperatures for %s based on ERA5 ouput.' % (instrument)
        self.pam_ds.attrs['models'] =  'ERA5 + PAMTRA'
        self.pam_ds.attrs['date'] =  datetime.datetime.utcfromtimestamp(self.pam.p['unixtime'][0,0]).strftime('%Y%m%d %H:%M')
    
        outFile = 'pamtra_' + instrument + '_' +\
            datetime.datetime.utcfromtimestamp(self.pam.p['unixtime'][0,0]).strftime('%Y%m%d_%H') + '.nc'
        
        self.pam_ds.to_netcdf(self.outPath + outFile)
        print("PAMTRA saved as:", self.outPath + outFile) 
