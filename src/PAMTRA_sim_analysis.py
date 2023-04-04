# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:12:09 2023

@author: u300737
"""
import numpy as np
import pandas as pd
import xarray as xr

import scipy.interpolate as scint

class PAMTRASIM_analysis():
    def __init__(self,date="20110317",data_path="/scratch/u/u300737/",
                 hour="12",instrument="hamp",height_idx=1):
        
        self.data_path=data_path
        self.date_str=date
        self.hour=hour
        self.instrument=instrument
        self.height_idx=height_idx
    
    def updt(self,total, progress):
        """
        Displays or updates a console progress bar.
        
        Original source: https://stackoverflow.com/a/15860757/1391441
        """
        import sys
        
        barLength, status = 20, ""
        progress = float(progress) / float(total)
        if progress >= 1.:
            progress, status = 1, "\r\n"
        block = int(round(barLength * progress))
        text = "\r[{}] {:.0f}% {}".format("#" * block + "-" * (barLength - block), 
                                          round(progress * 100, 0), status)
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def open_era5_fields(self):
        self.era5_fname="era5_"+self.date_str+"_"+self.hour+"_atmos.nc"
        self.era5_ds=xr.open_dataset(self.data_path+self.era5_fname)
    def open_era5_tbs(self):
        self.pamtra_fname="pamtra_"+self.instrument+"_"+self.date_str+"_"+self.hour+".nc"
        self.pamtra_ds=xr.open_dataset(self.data_path+self.pamtra_fname)
    
    def get_era5_open_ocean_mask(self,):
        self.land_sea_mask=pd.Series(
                                data=self.era5_ds["sfc_slf"].values.flatten(),
                                index=range(self.era5_ds["sfc_slf"].shape[0]))
        self.sea_ice_mask=pd.Series(
                                data=self.era5_ds["sfc_sif"].values.flatten(),
                                index=range(self.era5_ds["sfc_sif"].shape[0]))
        open_ocean=pd.Series(data=np.zeros(self.era5_ds["sfc_slf"].shape[0]),
                             index=self.sea_ice_mask.index)
        
        open_ocean.loc[(self.land_sea_mask==0) & (self.sea_ice_mask==0)]=1
        self.open_ocean=open_ocean
        #iwv_ocean=iwv[open_ocean==1]
    
    def open_era5_ocean_fields(self,):
        self.open_era5_fields()
        self.get_era5_open_ocean_mask()
        
    def choose_relevant_height(self,):
        self.relevant_height=self.era5_ds["obs_height"][self.height_idx]
    
    def get_era5_hmps(self,only_ocean=False):
        self.open_era5_ocean_fields()
        
        iwv=pd.Series(data=self.era5_ds["iwv"][:,0,self.height_idx],
                      index=self.sea_ice_mask.index)
        try:
            lwp=pd.Series(data=self.era5_ds["lwp"][:,0,self.height_idx],
                          index=self.sea_ice_mask.index)
        except:
            lwp=pd.Series(data=self.era5_ds["cwp"][:,0,self.height_idx],index=self.sea_ice_mask.index)
        iwp=pd.Series(data=self.era5_ds["iwp"][:,0,self.height_idx],index=self.sea_ice_mask.index)
        rwp=pd.Series(data=self.era5_ds["rwp"][:,0,self.height_idx],index=self.sea_ice_mask.index)
        swp=pd.Series(data=self.era5_ds["swp"][:,0,self.height_idx],index=self.sea_ice_mask.index)
        
        if only_ocean:
            lwp=lwp[self.open_ocean==1]*1000
            iwp=iwp[self.open_ocean==1]*1000
            rwp=rwp[self.open_ocean==1]*1000
            swp=swp[self.open_ocean==1]*1000
        
        self.era5_hmp=pd.DataFrame(data=np.nan,columns=["Date","IWV","LWP","IWP","SWP","RWP"],index=self.sea_ice_mask.index)
        self.era5_hmp["Date"] = self.date_str
        self.era5_hmp["IWV"]  = iwv
        self.era5_hmp["LWP"]  = lwp
        self.era5_hmp["IWP"]  = iwp
        self.era5_hmp["SWP"]  = swp
        self.era5_hmp["RWP"]  = rwp
        
        self.era5_hmp.index=int(self.date_str)*1000+self.sea_ice_mask.index

    def interp_column_era5_point(self,idx,height_idx,unified_heights,var):
        var_series=pd.Series(self.era5_ds[var][idx,0,:],
                             index=self.era5_ds["hgt"][idx,0,:].values)
        # Interpolation function
        var_fct = scint.interp1d(var_series.index, var_series,fill_value="extrapolate")
        # Get interpolated data
        var_series_interp=pd.Series(var_fct(unified_heights),
                                    index=unified_heights)
        return var_series_interp

    def create_regridded_era5(self,upper_height=16000,res=30,vars=["t","p","rh"]):
        """
        This regrids the era5 along vertical axis onto unified grid for retrieval products
        """
        
        height_series=pd.Series(data=self.era5_ds["hgt"].mean(axis=0)[0,:])
        height_idx=height_series[height_series<upper_height]
        height_max=height_idx.iloc[-1]//res*res
        height_steps=int(height_max/res+1)
    
        unified_heights=np.linspace(0,height_max,height_steps)
        self.regridded_era5=xr.Dataset(coords={"idx":range(self.era5_ds["hgt"].shape[0]),
                                  "z":unified_heights})
        self.regridded_era5["lat"]=xr.DataArray(data=self.era5_ds["lat"].values[:,0],dims=["idx"])
        self.regridded_era5["lon"]=xr.DataArray(data=self.era5_ds["lon"].values[:,0],dims=["idx"])
        
        for var in vars:
            print("Regrid ",var)
            self.regridded_era5[var]=xr.DataArray(np.zeros((self.era5_ds["hgt"].shape[0],
                                                       len(unified_heights))),
                                             dims=["idx","z"])#
            loop_range=self.regridded_era5[var].shape[0]
            for grid_idx in range(loop_range):
                self.regridded_era5[var][grid_idx,:]=self.interp_column_era5_point(grid_idx,height_idx,unified_heights,var)
                self.updt(loop_range,grid_idx)
        
    def open_pamtra_tbs_ocean(self):
        self.open_era5_tbs()
        tb_ds=self.pamtra_ds.isel({#"x":":",
                              "y":0,
                              "nout":self.height_idx-1,
                              "nang":0})
        tb_da=tb_ds["tb"].mean(axis=2)
        tb_da=tb_da.assign_coords(freq=tb_ds.freq)
        tb_da=tb_da.isel(nfreq=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,
                        20,21,22,23,33,34,35,36,37,38,39])
        self.tb_da=tb_da
        #self.cut_era5_to_open_ocean()
    
    @staticmethod
    def calc_specific_humidity_from_relative_humidity(ds):
        """
        moisture data on grid levels is only given as rel. humidity,
        for moisture budget, specific humidity is required

        Returns
        -------
        None.
        
        """
        import metpy.calc as mpcalc
        from metpy.units import units
        pres=ds["p"].data.astype(np.float32)/100
        pres=pres * units.hPa
        rh=ds["rh"].data/100
        rh=rh.astype(np.float32)
        temp=ds["t"].data.astype(np.float32) * units.K
        mixing_ratio=mpcalc.mixing_ratio_from_relative_humidity(
                                        pres,temp,rh)
        print("mixing_ratio calculated")
        specific_humidity=xr.DataArray(np.array(
                                    mpcalc.specific_humidity_from_mixing_ratio(
                                        mixing_ratio)),
                                   dims=["idx","z"])
        print("specific humidity calculated")
        ds=ds.assign({"q":specific_humidity})
        return ds
    
    def list_all_simulated_days(data_path="/scratch/u/u300737/",instrument="hamp",hour="12",
                               check_for_specific_campaigns=True):
        import glob
        # File structure of ERA5-PAMTRA HAMP TBs
        file_structure="pamtra_"+instrument+"_"+"*"+"_"+hour+".nc"
        files=data_path+file_structure
        # List files
        file_list=glob.glob(files)
        print(len(file_list)," days are already simulated")
        date_list=[file.split("/")[-1].split("_")[-2] for file in file_list] 
        date_list=sorted(date_list)
        print(date_list)
        
        ### Check for specific campaigns
        # Synthetic ARs
        synth_ar_days=["20110317","20110423","20150314","20160311",
                       "20180224","20180225","20190319","20200416",
                       "20200419"]
        # HALO-AC3 research flights (HALO)
        halo_ac3_days=["20220311","20220312","20220313","20220314",
                       "20220315","20220316","20220320","20220321",
                       "20220328","20220329","20220330","20220401",
                       "20220404","20220407","20220408","20220410",
                       "20220411","20220412"]
        return date_list 
