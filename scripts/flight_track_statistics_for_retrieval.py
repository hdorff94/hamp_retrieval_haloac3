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


version_to_use="v0.2"
mask_version="v0.6"    
campaign="HALO_AC3"
nc_path=airborne_data_importer_path+"/Flight_Data/"+campaign+"/all_nc/"
rf_no=0
for rf in [*Flight_Dates[campaign].values()][1:3]:
    print(rf)
    bahamas_file_list=glob.glob(nc_path+"bahamas*"+rf+\
                                "*"+version_to_use+".nc")#)
    radiometer_file_list=glob.glob(nc_path+"radiometer*"+rf+\
                                   "*"+version_to_use+".nc")#)
    radar_file_list=glob.glob(nc_path+"radar*"+rf+"*"+mask_version+".nc")
    if not len(bahamas_file_list)>1:
        if len(bahamas_file_list)==0:
            raise FileNotFoundError("Your ",campaign,
                                    " bahamas data is not present under",
                                    nc_path)
        else:
            bahamas_ds=xr.open_dataset(bahamas_file_list[0])
            #bahamas_ds["alt"].plot()
            radiometer_ds=xr.open_dataset(radiometer_file_list[0])
            radar_ds=xr.open_dataset(radar_file_list[0])
            surface_cond=pd.Series(np.array(radar_ds["radar_flag"][:,0]),
                                   index=pd.DatetimeIndex(
                                       np.array(radar_ds.time[:])))
            
            surface_cond=surface_cond.replace(to_replace=-888,value=np.nan)
            surface_cond=surface_cond.interpolate(method="nearest",limit=2)
            surface_cond[surface_cond==-888]=-1
            #sys.exit()
            temp_radiometer_df=pd.DataFrame(data=np.array(radiometer_ds["TB"]),
                                            columns=np.array(radiometer_ds["freq"]),
                                            index=pd.DatetimeIndex(
                                                np.array(radiometer_ds.time[:])))
            temp_bahamas_df=pd.DataFrame(data=np.array(bahamas_ds["alt"]),
                                        columns=["alt"],index=pd.DatetimeIndex(
                                            np.array(bahamas_ds.time[:])))
            temp_bahamas_df["roll"]=pd.Series(data=np.array(bahamas_ds["roll"]),
                                              index=pd.DatetimeIndex(
                                                 np.array(bahamas_ds.time[:])))
            temp_bahamas_df["grad_alt"]=temp_bahamas_df["alt"].diff()
            temp_bahamas_df["cruising"]=abs(temp_bahamas_df["grad_alt"])<1
            temp_bahamas_df["TB_avail_frac"]=(temp_radiometer_df.shape[1]-\
                                              temp_radiometer_df.isna().sum(axis=1))/\
                                            temp_radiometer_df.shape[1]
            temp_bahamas_df["sfc"]=surface_cond
            if rf_no==0:
                bahamas_df=temp_bahamas_df
                radiometer_df=temp_radiometer_df
            else:
                bahamas_df=bahamas_df.append(temp_bahamas_df)
                radiometer_df=radiometer_df.append(temp_radiometer_df)
    rf_no+=1


#%% Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({"font.size":20})

###############################################################################    
# Height hist
height_fig=plt.figure(figsize=(12,9))
ax1=height_fig.add_subplot(111)
height_data=bahamas_df[bahamas_df["cruising"]==True]
height_data=height_data[bahamas_df["TB_avail_frac"]>0.9]
ax1.hist(height_data["alt"]/1000,
         bins=np.linspace(4,12.5,38))
ax1.set_ylabel("Number of Timesteps")
ax1.set_xlabel("Height (km)")
sns.despine(offset=10)
for axis in ["left","bottom"]:
    ax1.spines[axis].set_linewidth(3)
    ax1.tick_params(length=10,width=3)
plot_path=airborne_plotting_module_path+"/plots/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
fig_name="Fig01_Height_Cruising.pdf"
height_fig.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
print("Figure saved as:",plot_path+fig_name)

###############################################################################
# Sea ice hist
surface_hist=plt.figure(figsize=(12,9))
ax1=surface_hist.add_subplot(111)
surface_data=bahamas_df["sfc"][bahamas_df["sfc"]>=-1]
surface_bins=pd.cut(surface_data,bins=[-1,-0.05,0.05,0.5,1.0],
                    include_lowest=True).value_counts()
surface_bins=surface_bins.sort_index()
ax1.bar([1],surface_bins[0]/surface_data.shape[0],
        width=0.75,facecolor="brown",edgecolor="k",lw=3)
ax1.bar([2],surface_bins[1]/surface_data.shape[0],
        width=0.75,facecolor="darkblue",edgecolor="k",lw=3)
ax1.bar([3],surface_bins[2]/surface_data.shape[0],
        width=0.75,facecolor="lightblue",edgecolor="k",lw=3)
ax1.bar([4],surface_bins[3]/surface_data.shape[0],
        width=0.75,facecolor="white",edgecolor="k",lw=3)
ax1.set_xticks([1,2,3,4])

ax1.set_xticklabels(["Ground","Open \n ocean",
                     "sea-ice \n cover < 0.5",
                     "sea-ice \n cover > 0.5"])
sns.despine(offset=10)
for axis in ["left","bottom"]:
    ax1.spines[axis].set_linewidth(3)
    ax1.tick_params(length=10,width=3)
fig_name="Fig02_Sea_Surface_Distribution.pdf"
height_fig.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
print("Figure saved as:",plot_path+fig_name)
ax1.set_ylim([0.0,0.6])
ax1.set_yticks([0,.2,.4,.6]) 
ax1.set_yticklabels(["0%","20%","40%","60%"])
#ax1.hist(,bins=np.linspace(0,1,4))
###############################################################################
tb_channel_hist,axs=plt.subplots(5,5,figsize=(30,25),sharex=True,sharey=True)
f=0
tb_bins=[217.5,222.5,227.5,232.5,237.5,242.5,247.5,
         252.5,262.5,267.5,272.5,277.5,282.5]
for i in range(2):
    for j in range(2):
        freq=radiometer_df.columns[f]
        binned_tbs=pd.cut(radiometer_df.iloc[:,f],bins=tb_bins).value_counts().sort_index()
                            
        axs[i,j].bar([220,225,230,235,240,245,250,255,260,265,270,275,280],
                     binned_tbs.values)
        axs[i,j].text(270,0.9,text=str(freq)+" GHz",s=10)
        f+=1
        axs[i,j].set_xlim([200,300])
sns.despine(offset=10)
            