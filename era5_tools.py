import numpy as np
from cdo import *
import xarray as xr
import pyPamtra
import datetime


R_d = 287.0597
R_v = 461.5250
R_g = 9.80665
R = 6371229

def runCDO(area=[-30,50,65,89],yyyy=2019,mm=4,dd=1,timestep=1,threads="32",outPath='/tmp/'):
    """
    cut area from ERA5 model output, selects timestep, transforms grid, and stores as netcdf files.
    
    area: array of integer giving the area of choice [minlon,maxlon,minlat,maxlat]
    yyyy: integer year
    mm:   integer month
    dd:   integer day of month
    timestep: integer starting with 1 = 0 UTC, 13 = 12 UTC
    threads: integer giving threads to be used for the cdo process
    outPath: string giving the path where to store the output
    """
    
    cdo = Cdo()
    yyyy = "%4d" % (yyyy,)
    mm = "%02d" % (mm,)
    dd = "%02d" % (dd,)
    outtime = "%02d" % (timestep-1,)
    area = ','.join([str(coord) for coord in area])
    era5_datetime = yyyy+mm+dd+"_"+outtime

    for var in ['130', '152']:
        cdo_string = "-sp2gpl -seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        
    for var in ['075', '076', '133', '246', '247']:
        cdo_string = "-seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
    
    for var in ['031','034','129','134','137','151','165','166','172','235']:
        cdo_string = "-seltimestep," + str(timestep) +" -selday,"+dd+" -setgridtype,regular "+"/pool/data/ERA5/E5/sf/an/1H/"+var+"/E5sf00_1H_"+yyyy+"-"+mm+"_"+var+".grb"
        cdo.sellonlatbox(area,input=cdo_string, output=outPath+"reduced_sf_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)

    return era5_datetime


def getColumnfromERA5(point=[10.,78.],yyyy=2019,mm=4,dd=1,hour=0,min=0,
                      threads="32",campaign=None,outFile=None,outPath='/tmp/'):
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
    
    cdo = Cdo()
    yyyy = "%4d" % (yyyy,)
    mm = "%02d" % (mm,)
    dd = "%02d" % (dd,)
    outhour = "%02d" % (hour,)
    outmin = "%02d" % (min,)
    point = "lon="+'_lat='.join([str(coord) for coord in point])
    era5_datetime = yyyy+mm+dd+"_"+outhour+outmin
    timestep = hour
    output='/work/bb1320/col_%s.nc' % (outFile)
    
    infiles = list()
    
    for var in ['130', '152']:
        cdo_string = " -sp2gpl -seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.remapnn(point,input=cdo_string, output=outPath+"col_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append("/scratch/b/b380702/"+campaign+"/col_ml_"+era5_datetime+"_"+var+".nc")
        
    for var in ['075', '076', '133', '246', '247']:
        cdo_string = "-seltimestep," + str(timestep) +" -setgridtype,regular "+"/pool/data/ERA5/E5/ml/an/1H/"+var+"/E5ml00_1H_"+yyyy+"-"+mm+"-"+dd+"_"+var+".grb"
        cdo.remapnn(point,input=cdo_string, output=outPath+"col_ml_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append("/scratch/b/b380702/"+campaign+"/col_ml_"+era5_datetime+"_"+var+".nc")
    
    for var in ['031','034','129','134','137','151','165','166','172','235']:
        cdo_string = "-seltimestep," + str(timestep) +" -selday,"+dd+" -setgridtype,regular "+"/pool/data/ERA5/E5/sf/an/1H/"+var+"/E5sf00_1H_"+yyyy+"-"+mm+"_"+var+".grb"
        cdo.remapnn(point,input=cdo_string, output=outPath+"col_sf_"+era5_datetime+"_"+var+".nc", options='-f nc -P ' + threads)
        infiles.append("/scratch/b/b380702/"+campaign+"/col_sf_"+era5_datetime+"_"+var+".nc")

    cdo.merge(input=infiles, output=output)
    
    return era5_datetime

def readERA5(era5_datetime,descriptorFile,inPath='/tmp/',debug=False,verbosity=0,step=1,cut_levels=None):
    """
    import ERA5 full model output from netcdf files
    
    netcdf files have been created with cdo by extracting desired timestep and lat/lon region for all variables needed and 
    converting the temperature from spherical harmonics to Gaussian linear grid. 

    2d vars: 031 034 129 134 137 151 165 166 172 235
    cdo -f nc -P $threads -seltimestep,$timestep -selday,$dd -sellonlatbox,$minlon,$maxlon,$minlat,$maxlat -setgridtype,regular gribfile ncfile
    3d vars: 075 076 133 246 247 130
    cdo -f nc -P $threads -sellonlatbox,$minlon,$maxlon,$minlat,$maxlat [-setgridtype,regular|-sp2gpl] -seltimestep,$timestep gribfile ncfile
    3d but 2d is 152 the logarithm of the surface pressure
    
    era5_datetime: yyyymmmdd_hh of the model output
    descriptorfile: ECMWF descriptorfile
    debug: switch on debugging
    verbosity: increase verbosity
    step: reduce ERA5 grid to nth point in lat/lon
    cut_levels: cut atmosphere from top. This is necessary, cause PAMTRA can not calculate gas absorption for pressure below 3hPa. 
                A sufficiently good value for cut_levels is 5.
    """

    if debug: import pdb;pdb.set_trace()
    
    # define and get 2d vars
    vals2D = dict()
    vals2D_params = {'031':'ci','034':'sst',
                     '129':'z','134':'sp',
                     '151':'msl','165':'10u',
                     '166':'10v','172':'lsm',
                     '235':'skt'}

    for key,value in vals2D_params.items():
        tmp_ds = xr.open_dataset(inPath+'reduced_sf_'+era5_datetime+'_'+key+'.nc')
        vals2D[value] = np.squeeze(np.swapaxes(tmp_ds['var'+str(int(key))].values,1,2))[0::step,0::step]

    # get logarithm of surface pressure, a_coef, and b_coef
    tmp_ds = xr.open_dataset(inPath+'reduced_ml_'+era5_datetime+'_152.nc')
    vals2D['lnsp'] = np.squeeze(np.swapaxes(tmp_ds['lnsp'].values,2,3))[0::step,0::step]
    a_coef = tmp_ds['hyai'].values[cut_levels:]
    b_coef = tmp_ds['hybi'].values[cut_levels:]

    # define and get 3d vars
    vals3D = dict()
    vals3D_params = {'075':'crwc','076':'cswc','130':'t','133':'q','246':'clwc','247':'ciwc'}

    for key,value in vals3D_params.items():
        tmp_ds = xr.open_dataset(inPath+'reduced_ml_'+era5_datetime+'_'+key+'.nc')
#        vals3D[value] = np.squeeze(np.swapaxes(tmp_ds[value].values,1,3)[...,-1:cut_levels:-1])[0::step,0::step,:]
        vals3D[value] = np.squeeze(np.swapaxes(tmp_ds[value].values,1,3)[...,cut_levels:])[0::step,0::step,::-1]
    
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

    # height and pressure grid
    pamData['hgt'] = np.empty(shape3D)
    pamData['hgt_lev'] = np.empty(shape3Dplus)
    pamData['press'] = np.empty(shape3D)
    pamData['press_lev'] = np.empty(shape3Dplus)

    q = vals3D['q']

    def compute_p_level(a_coef,b_coef,sp,p_h,p_f):
        '''Return the presure at half- (p_h) and full-levels (p_f)'''
        p_h[...,0] = a_coef[0] + (b_coef[0] * sp)
        for i in range(1,len(a_coef)):
            p_h[...,i] = a_coef[i] + (b_coef[i] * sp)
            p_f[...,i-1] = (p_h[...,i-1] + p_h[...,i])*0.5

        return p_h, p_f

    pamData['press_lev'][:,:,-1] = np.exp(vals2D['lnsp'][:,:])
    pamData['press_lev'],pamData['press'] = compute_p_level(a_coef,b_coef,np.exp(vals2D['lnsp']),pamData['press_lev'],pamData['press'])

    def compute_z_level(t, q, z_h, p_lev,p_levpo, lev):
        '''Compute z at half- and full-level for the given level, based on t/q/p'''

        # compute moist temperature
        t_v = t * (1. + 0.609133 * q)

        if lev == 0:
            dlog_p = np.log(p_levpo / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(p_levpo / p_lev)
            alpha = 1. - ((p_lev / (p_levpo - p_lev)) * dlog_p)

        t_v = t_v * R_d

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        z_f = z_h + (t_v * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (t_v * dlog_p)

        return z_h, z_f


    tmp_t = pamData['temp'][...,::-1]
    tmp_q = vals3D['q'][...,::-1]
    # surface geopotential
    z_h = vals2D['z'][:,:]
    pamData['hgt_lev'][...,-1] = z_h/R_g
    for i in sorted(range(Nz),reverse=True):
        z_h, z_f = compute_z_level(tmp_t[...,i],tmp_q[...,i],
                                   z_h,pamData['press_lev'][0,0,i],
                                   pamData['press_lev'][0,0,i+1],i)
        pamData['hgt'][...,i] = z_f/R_g
        pamData['hgt_lev'][...,i] = z_h/R_g

    # reverse levels of pressure and height variables
    for var in ['press', 'press_lev', 'hgt', 'hgt_lev']:
        pamData[var] = pamData[var][...,::-1]
        
    pamData['hydro_q'] = np.zeros(shape4D) + np.nan
    pamData['hydro_q'][...,0] = vals3D['clwc']
    pamData['hydro_q'][...,1] = vals3D['ciwc']
    pamData['hydro_q'][...,2] = vals3D['crwc']
    pamData['hydro_q'][...,3] = vals3D['cswc']

    qh = np.zeros(shape3D)
    qh = np.sum(pamData['hydro_q'],axis=3)

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

    return pam

def collectERA5(era5_datetime,inPath='/scratch/b/b380702/',outPath='/scratch/b/b380702/',step=4,cut_levels=5, applyFilter=False):
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
            pam._helperP = dict()
            pam._helperP['dz'] = pam.p['hgt_lev'][...,1::]-pam.p['hgt_lev'][...,0:-1]
            pam._helperP['vapor'] = rh2q(pam.p['relhum']/100.,pam.p['temp'],pam.p['press'])
            pam._helperP['sum_hydro_q'] = np.nansum(pam.p['hydro_q'],axis=-1)
            pam._helperP['rho_moist'] =            moist_rho_rh(pam.p['press'],pam.p['temp'],pam.p['relhum']/100.,pam._helperP['sum_hydro_q'])

            return

        pam._shape4Dret = (pam.p["ngridx"],pam.p["ngridy"],pam.p['noutlevels'],pam.df.nhydro)
        pam._shape3Dret = (pam.p["ngridx"],pam.p["ngridy"],pam.p['noutlevels'])
        pam.p['hydro_wp'] = np.zeros(pam._shape4Dret)
        pam._calcMoistRho() # provies as well dz, sum_hydro_q, and q within dict() self._helperP
        pam.p['iwv'] = np.zeros(pam._shape3Dret)
        for h in range(pam.p['noutlevels']):
            target_height = pam.p['obs_height'][0,0,h]
            for ix in range(pam.p["ngridx"]):
                zgrid = pam.p['hgt_lev'][ix,0,:]
                z_diffs = np.absolute(zgrid - target_height)
                z_index = np.argmin(z_diffs)+1
                #print(z_index)
                #print(target_height, pam.p['hgt_lev'][ix,0,z_index-1], pam.p['hgt_lev'][ix,0,z_index], pam.p['hgt_lev'][ix,0,z_index+1])
                pam.p['iwv'][...,h] = np.nansum(pam._helperP['vapor'][...,0:z_index]*pam._helperP["rho_moist"][...,0:z_index]*pam._helperP["dz"][...,0:z_index],axis=-1)
                #nothing to do without hydrometeors:
                if np.all(pam.p['hydro_q']==0):
                    self.p['hydro_wp'] = np.zeros(pam._shape4Dret)
                else:
                    for i in range(pam.df.nhydro):
                        pam.p['hydro_wp'][...,h,i] = np.nansum(pam.p['hydro_q'][...,0:z_index,i]*pam._helperP["rho_moist"][...,0:z_index]*pam._helperP["dz"][...,0:z_index],axis=-1)

        return

    pam = readERA5(era5_datetime,'/home/b/b380702/pamtra/descriptorfiles/descriptor_file_ecmwf.txt',inPath=inPath,step=step,cut_levels=cut_levels)
    pam.p['noutlevels'] = 71
    pam.p['obs_height'] = np.zeros((pam._shape2D[0],pam._shape2D[1],pam.p['noutlevels']))
    pam.p['obs_height'][:,:,:] = [833000., 12000., 11900., 11800., 11700., 
        11600., 11500., 11400., 11300., 11200.,
       11100., 11000., 10900., 10800., 10700., 10600., 10500., 10400.,
       10300., 10200., 10100., 10000.,  9900.,  9800.,  9700.,  9600.,
        9500.,  9400.,  9300.,  9200.,  9100.,  9000., 8900., 8800., 8700., 
        8600., 8500., 5200., 5100., 5000., 4900., 4800., 4700., 4600., 4500., 
        4400., 4300., 4200., 4100., 4000., 3900., 3800., 3700., 3600.,
        3500., 3400., 3300., 3200., 3100., 3000., 2900., 2800., 2700.,
        2600., 2500., 2400., 2300., 2200., 2100., 2000., 0.]
    if applyFilter:
        filter = np.empty(pam._shape2D,dtype=bool)
        filter[:,:] = False
        filter[pam.p['sfc_type'] == 0] = True
        pam.filterProfiles(filter)
    
    addIntegratedValues()
    era5_ds = xr.Dataset(
        {"unixtime": (("x", "y"), pam.p['unixtime'][...]),
         "lat": (("x", "y"), pam.p['lat'][...]),
         "lon": (("x", "y"), pam.p['lon'][...]),
         "obs_height": (("nout"), pam.p['obs_height'][0,0,:]),
         "sfc_slf": (("x", "y"), pam.p['sfc_slf'][...]),
         "sfc_sif": (("x", "y"), pam.p['sfc_sif'][...]),
         "groundtemp": (("x", "y"), pam.p['groundtemp'][...]),
         "hgt": (("x", "y", "z"), pam.p['hgt'][...]),
         "t": (("x", "y", "z"), pam.p['temp'][...]),
         "rh":(("x", "y", "z"), pam.p['relhum'][...]),
         "p":(("x", "y", "z"), pam.p['press'][...]),
         "iwv":(("x", "y", "nout"), pam.p['iwv'][...]),
         "cwp":(("x", "y", "nout"), pam.p['hydro_wp'][...,0]),
         "iwp":(("x", "y", "nout"), pam.p['hydro_wp'][...,1]),
         "rwp":(("x", "y", "nout"), pam.p['hydro_wp'][...,2]),
         "swp":(("x", "y", "nout"), pam.p['hydro_wp'][...,3]),},)

    era5_ds['unixtime'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    era5_ds['obs_height'].attrs['units'] = 'm'
    era5_ds['groundtemp'].attrs['units'] = 'K'
    era5_ds['hgt'].attrs['units'] = 'm'
    era5_ds['t'].attrs['units'] = 'K'
    era5_ds['rh'].attrs['units'] = '%'
    era5_ds['p'].attrs['units'] = 'Pa'
    era5_ds['iwv'].attrs['units'] = 'kg/m^2'
    era5_ds['cwp'].attrs['units'] = 'kg/m^2'
    era5_ds['iwp'].attrs['units'] = 'kg/m^2'
    era5_ds['rwp'].attrs['units'] = 'kg/m^2'
    era5_ds['swp'].attrs['units'] = 'kg/m^2'

    era5_ds.to_netcdf(outPath + 'era5_'+era5_datetime+"_atmos.nc")
    
    return pam

def reducePamtraResults(pam,instrument='mirac-a',outPath='/scratch/b/b380702/'):
    def read_attributes():
        """
        Read variable definitions catalog

        Returns
        -------
        cat

        """
        import yaml
        with open('/work/bb1320/scripts/instrument_settings.yaml', 'r') as f:

            cat = yaml.safe_load(f)

        return cat
    cat = read_attributes()
    out_slice = slice(cat[instrument]['obs_heights'][0],cat[instrument]['obs_heights'][1]+1)
    ang_slice = slice(cat[instrument]['angles'][0],cat[instrument]['angles'][1]+1)
    freq_slice = slice(cat[instrument]['frequencies'][0],cat[instrument]['frequencies'][1]+1)
    tb = np.zeros((pam.p['ngridx'],pam.p['ngridy'],cat[instrument]['nout'],cat[instrument]['nang'],cat[instrument]['nfreq'],cat[instrument]['npol']))
    tb[:,:,:,:,:,:] = pam.r['tb'][:,:,out_slice,ang_slice,freq_slice,:]
    #import pdb;pdb.set_trace()
    pam_ds = xr.Dataset(
        {"unixtime": (("x", "y"), pam.p['unixtime'][...]),
         "lat": (("x", "y"), pam.p['lat'][...]),
         "lon": (("x", "y"), pam.p['lon'][...]),
         "obs_height": (['nout'], pam.p['obs_height'][0,0,slice(cat[instrument]['obs_heights'][0],cat[instrument]['obs_heights'][1]+1)]),
         "ang": (['nang'], np.absolute(pam.r['angles_deg'][ang_slice]-180.)),
         "freq": (['nfreq'], pam.set['freqs'][freq_slice]),
         "pol": (['npol'],['H','V']),
         "tb": (("x", "y","nout","nang","nfreq","npol"), tb),
        },)
    pam_ds['tb'].attrs['units'] = 'K'

    pam_ds.attrs['description'] = 'Reduced simulated brightness temperatures for %s based on ERA5 ouput.' % (instrument)
    pam_ds.attrs['models'] =  'ERA5 + PAMTRA'
    pam_ds.attrs['date'] =  datetime.datetime.utcfromtimestamp(pam.p['unixtime'][0,0]).strftime('%Y%m%d %H:%M')
    
    outFile = 'pamtra_' + instrument + '_' + datetime.datetime.utcfromtimestamp(pam.p['unixtime'][0,0]).strftime('%Y%m%d_%H') + '.nc'
    
    pam_ds.to_netcdf(outPath + outFile)

    return pam_ds
