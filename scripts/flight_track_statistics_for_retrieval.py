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
import Quicklook_Dicts
import measurement_instruments_ql    
try:
    import Flight_Campaign as Campaign
except:
        print("Module Flight Campaign is not listed in the path",
              "Flights need to be defined manually.")

#%% POLAR Aircraft surface mask
def make_POLARLandMask(resolution,polar_dict={},cfg_dict={},
                       polar_aircraft="P5",
                       #outfile,
                       add_sea_ice_mask=True):
    """
    

    Parameters
    ----------
    flight_dates : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    cfg_dict : dict
        Dictionary containing the configuration arguments
    add_sea_ice_mask : boolean, Default: False
        Specifier if sea_ice_mask should be created and added or not. Might not
        be useful for lower latitude flight campaigns.
    use_radiometer : boolean, Default : False
        In general this routine is supposed to be applied for the radar data
        however as also the radiometer can require a surface mask this 
        additional command should set to True and other "radar_ds" ist used.
    Returns
    -------
    None.

    """
    flight_dates=[*polar_dict.keys()]
    # Set file paths
    ls_path = cfg_dict["campaign_path"]+'../Auxiliary/lsmask-world8-var.dist5.5.nc'
    if not os.path.exists(ls_path):
        raise FileNotFoundError('Land mask file not found. ',
                    'Please download the file',
                    'lsmask-world8-var.dist5.5.nc',
                    'from https://www.ghrsst.org/ghrsst-data-services/tools/')
    else:
        pass
    resolution=resolution
    
    print(polar_aircraft," sea ice land mask creation")
    polar_sea_ice_dict={}
    if add_sea_ice_mask:
        #% Loop flights
        for flight in flight_dates:
            print("Land Sea ice mask for ",flight)
            #import measurement_instruments_ql
            # this creates a new surface mask that includes the sea ice cover
            if polar_aircraft.lower()=="p5": 
                polar_ds=polar_dict[flight]
            else:
                polar_ds=polar_dict[flight]
            #%% Land mask first #%%%%%%%%%%%%%%
            import Performance
            performance=Performance.performance()
            print("Add surface mask")
            # Get sea_ice mask    
            sea_ice_cls=measurement_instruments_ql.SEA_ICE(cfg_dict)
            sea_ice_cls.open_sea_ice_ds()
            sea_ice_ds=sea_ice_cls.ds 
            try:
                time_values=polar_ds.TIME[:]
            except:
                time_values=polar_ds.time[:]
            polar_df=pd.DataFrame(data=np.nan,columns=["LAT","LON"],
                             index=pd.DatetimeIndex(np.array(time_values)))
            polar_df["LAT"]=polar_ds.lat.values
            polar_df["LON"]=polar_ds.lon.values
            
            polar_df=polar_df.resample(resolution,convention="start").mean()
            sea_ice_mask=pd.Series(data=np.nan,
                                   index=pd.DatetimeIndex(polar_df.index))
    
            lat_2d=np.array(sea_ice_ds.lat)
            lon_2d=np.array(sea_ice_ds.lon)
            lat_1d=lat_2d.flatten()
            lon_1d=lon_2d.flatten()
            print("check for sea ice concentration")
            for t in range(polar_df.shape[0]):
                if not (np.isnan(polar_df["LAT"].iloc[t])) \
                    and not (np.isnan(polar_df["LON"].iloc[t])):
                    #  Calculate differences of aircraft position to 
                    #  land sea mask grid
                    distances=measurement_instruments_ql.HALO_Devices.\
                                vectorized_harvesine_distance(
                                                polar_df["LAT"].iloc[t],
                                                polar_df["LON"].iloc[t],
                                                lat_1d,lon_1d)
            
                    min_geoloc=np.unravel_index(np.argmin(distances,axis=None),
                                        lat_2d.shape)
                    sea_ice_mask.iloc[t]=sea_ice_ds["seaice"][min_geoloc[0],
                                                              min_geoloc[1]]
                    if polar_df["LAT"].iloc[t]>87:
                        # from upon a specific latitude, the sea ice mask is
                        # not yet provided anymore, but we can certainly 
                        # assume full sea-ice cover.
                        
                        sea_ice_mask.iloc[t]=100
                    else:
                        pass
                performance.updt(polar_df.shape[0],t)                                                             
            sea_ice_mask=sea_ice_mask/100
            sea_ice_mask=sea_ice_mask.fillna(0)
            print("check for land surface")
            # Now add land mask
            from ac3airborne.tools import is_land as il
            t=0
            for x, y in zip(polar_df["LON"], polar_df["LAT"]):
                if il.is_land(x,y):
                    sea_ice_mask.iloc[t]=-0.1*int(il.is_land(x,y))                
                t+=1
                performance.updt(polar_df.shape[0],t)
            polar_df["sea_ice"]=sea_ice_mask.values
            polar_sea_ice_dict[flight]=polar_df
    return polar_sea_ice_dict

def plot_surface_hist_airborne_plattforms(bahamas_df,polar_resolution,
                                          p5_df=pd.DataFrame(),
                                          p6_df=pd.DataFrame(),
                                          add_polar_aircraft=False,
                                          list_flight_hours=False):
    halo_bin_width=0.35
    polar_width=0.2
    halo_label="HALO"
    p5_label="P5"
    p6_label="P6"
    if list_flight_hours:
        halo_hours=str(np.round(bahamas_df.shape[0]/3600,1))
        halo_label+=" "+halo_hours+" h"
        if add_polar_aircraft:
            p5_hours=str(np.round(p5_df.shape[0]*int(polar_resolution[:-1])\
                                  /3600,1))
            p6_hours=str(np.round(p6_df.shape[0]*int(polar_resolution[:-1])\
                                  /3600,1))
            p5_label+=" "+p5_hours+ "h"
            p6_label+=" "+p6_hours+"h"
    import matplotlib.patches as mpatches
    
    halo_patch = mpatches.Patch(facecolor='lightgrey',edgecolor="k",
                                hatch="",label=halo_label,linewidth=2)
    p5_patch   = mpatches.Patch(facecolor='lightgrey',hatch="/",
                                edgecolor="k",label=p5_label,linewidth=2)
    p6_patch   = mpatches.Patch(facecolor='lightgrey',hatch="\\",
                                edgecolor="k",label=p6_label,linewidth=2)

    surface_hist=plt.figure(figsize=(12,9))
    ax1=surface_hist.add_subplot(111)
    surface_data=bahamas_df["sfc"][bahamas_df["sfc"]>=-1]
    surface_bins=pd.cut(surface_data,bins=[-1.05,-0.05,0.15,0.8,1.0],
                        include_lowest=False).value_counts()
    surface_bins=surface_bins.sort_index()
    ax1.bar([1],surface_bins[0]/surface_data.shape[0],
            width=halo_bin_width,facecolor="brown",edgecolor="k",lw=3)
    ax1.bar([2],surface_bins[1]/surface_data.shape[0],
            width=halo_bin_width,facecolor="royalblue",edgecolor="k",lw=3)
    ax1.bar([3],surface_bins[2]/surface_data.shape[0],
            width=halo_bin_width,facecolor="lightblue",edgecolor="k",lw=3)
    ax1.bar([4],surface_bins[3]/surface_data.shape[0],
            width=halo_bin_width,facecolor="white",edgecolor="k",lw=3,
            label=halo_label)
    
    ax1.set_xticks([1,2,3,4])
    
    ax1.set_xticklabels(["Land","Open \nocean",
                         "Marginal \nice zone",
                         "Sea ice"])
    ax1.set_ylim([0.0,0.6])
    ax1.set_yticks([0,.2,.4,.6]) 
    
    if add_polar_aircraft:
        p5_surface_data=p5_df["sea_ice"][p5_df["sea_ice"]>=-1]
        p6_surface_data=p6_df["sea_ice"][p6_df["sea_ice"]>=-1]
        p5_surface_bins=pd.cut(p5_surface_data,bins=[-1.05,-0.05,0.15,0.8,1.0],
                        include_lowest=False).value_counts()
        p5_surface_bins=p5_surface_bins.sort_index()
        p6_surface_bins=pd.cut(p5_surface_data,bins=[-1.05,-0.05,0.15,0.8,1.0],
                        include_lowest=False).value_counts()
        p6_surface_bins=p5_surface_bins.sort_index()
        
        ax1.bar([0.7],p5_surface_bins[0]/p5_surface_data.shape[0],
            width=polar_width,facecolor="brown",edgecolor="k",lw=3,hatch="/")
        ax1.bar([1.7],p5_surface_bins[1]/p5_surface_data.shape[0],
                width=polar_width,facecolor="royalblue",edgecolor="k",
                hatch="/",lw=3)
        ax1.bar([2.7],p5_surface_bins[2]/p5_surface_data.shape[0],
            width=polar_width,facecolor="lightblue",edgecolor="k",
            lw=3,hatch="/")
        ax1.bar([3.7],p5_surface_bins[3]/p5_surface_data.shape[0],
            width=polar_width,facecolor="white",edgecolor="k",lw=3,hatch="/",
            label=p5_label)
        
        ax1.bar([1.3],p6_surface_bins[0]/p6_surface_data.shape[0],
            width=polar_width,facecolor="brown",edgecolor="k",lw=3,hatch="\\")
        ax1.bar([2.3],p6_surface_bins[1]/p6_surface_data.shape[0],
                width=polar_width,facecolor="royalblue",edgecolor="k",
                hatch="\\",lw=3)
        ax1.bar([3.3],p6_surface_bins[2]/p6_surface_data.shape[0],
            width=polar_width,facecolor="lightblue",edgecolor="k",lw=3,hatch="\\")
        ax1.bar([4.3],p6_surface_bins[3]/p6_surface_data.shape[0],
            width=polar_width,facecolor="white",edgecolor="k",lw=3,hatch="\\",
            label=p6_label)
    
    
    ax1.set_yticklabels(["0","20","40","60"])
    ax1.legend(handles=[halo_patch,p5_patch,p6_patch],loc="upper right")
    ax1.set_ylabel("Relative Flight Duration / % ")
    sns.despine(offset=10)
    for axis in ["left","bottom"]:
        ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
    fig_name="Fig02_Sea_Surface_Distribution.pdf"
    surface_hist.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
    print("Figure saved as:",plot_path+fig_name)

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


version_to_use="v1.6"
mask_version="v1.6"    
campaign="HALO_AC3"

plot_height_statistics=False
plot_surface_hist=True
plot_tb_hist=False
plot_joy_hist=False


nc_path=airborne_data_importer_path+"/Flight_Data/"+campaign+"/all_nc/"
rf_no=0
joy_radiometer_df=pd.DataFrame()
desired_path_str="\\Work\\GIT_Repository\\hamp_processing_py\\hamp_processing_python\\"#\\Flight_Data\\HALO_AC3\\sea_ice\\"
airborne_importer_path=actual_working_path+"/../../../"+desired_path_str
campaign_path=airborne_importer_path+"/Flight_Data/"+campaign+"/"

prcs_cfg_dict=Quicklook_Dicts.get_prcs_cfg_dict("RF01", "20220311", campaign,
                                                    campaign_path,
                                                    additional_entries_dict={})
#if flight_day!=None:
#    prcs_cfg_dict["FD"]=flight_day
# Data Handling 
datasets_dict, data_reader_dict=Quicklook_Dicts.get_data_handling_attr_dicts(
        entries_to_change={})

# Get Plotting Handling
plot_handler_dict, plot_cls_args_dict,plot_fct_args_dict=\
                                    Quicklook_Dicts.get_plotting_handling_attrs_dict(
                                        entries_to_change={})
POLAR_Devices_cls=measurement_instruments_ql.POLAR_Devices(prcs_cfg_dict)
POLAR_GPS_INS_cls=measurement_instruments_ql.GPS_INS(POLAR_Devices_cls)
POLAR_GPS_INS_cls.major_data_path=campaign_path

#%% Open Polar5,6 positions
POLAR_GPS_INS_cls.open_aircraft_gps_position(used_polar_aircraft="P5")
# mask resolution
resolution="15s"
p5=POLAR_GPS_INS_cls.P5_GPS
p5_sea_ice=make_POLARLandMask(resolution,polar_dict=p5,
                              cfg_dict=prcs_cfg_dict,polar_aircraft="P5",
                              add_sea_ice_mask=True)
p5_df=pd.concat([*p5_sea_ice.values()])

POLAR_GPS_INS_cls.open_aircraft_gps_position(used_polar_aircraft="P6")
p6=POLAR_GPS_INS_cls.P6_GPS
p6_sea_ice=make_POLARLandMask(resolution,polar_dict=p6,
                              cfg_dict=prcs_cfg_dict,polar_aircraft="P6",
                              add_sea_ice_mask=True)
p6_df=pd.concat([*p6_sea_ice.values()])
for f,rf in enumerate([*Flight_Dates[campaign].values()][1:]):
    print(f,rf)
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
            temp_radiometer_df=temp_radiometer_df.reindex(
                                    sorted(temp_radiometer_df.columns),axis=1)
            col_no=0
            k_band_ch=1
            v_band_ch=1
            w_band_ch=1
            f_band_ch=1
            g_band_ch=1
                
            for col in temp_radiometer_df.columns[:]:
                print("Frequency GHz:",col)
                test_df=pd.DataFrame(data=temp_radiometer_df[col].values,
                                     columns=["TB"],index=range(temp_radiometer_df.shape[0]))
                
                test_df["freq"]=float(col)
                if float(col)<50:
                    
                    test_df["band"]="K-Band"
                    test_df["Ch "+str(k_band_ch)]=test_df["TB"].values
                    k_band_ch+=1
                elif 50<float(col)<80:
                    test_df["band"]="V-Band"
                    test_df["Ch "+str(v_band_ch)]=test_df["TB"].values
                    v_band_ch+=1
                elif 80<float(col)<100:
                    test_df["band"]="W-Band"
                    test_df["Ch "+str(w_band_ch)]=test_df["TB"].values
                    w_band_ch+=1
                elif 115<float(col)<130:
                    test_df["band"]="F-Band"
                    test_df["Ch "+str(f_band_ch)]=test_df["TB"].values
                    f_band_ch+=1
                else:
                    test_df["band"]="G-Band"
                    test_df["Ch "+str(g_band_ch)]=test_df["TB"].values
                    g_band_ch+=1
                joy_radiometer_df=joy_radiometer_df.append(test_df)
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

#sys.exit()
#%% Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams.update({"font.size":24})
plot_path=airborne_plotting_module_path+"/plots/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
        
    
if plot_height_statistics:
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
    fig_name="Fig01_Height_Cruising.pdf"
    height_fig.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
    print("Figure saved as:",plot_path+fig_name)

###############################################################################
# Sea ice hist
if plot_surface_hist:
    plot_surface_hist_airborne_plattforms(bahamas_df,resolution,p5_df=p5_df,
                                          p6_df=p6_df,
                                          add_polar_aircraft=True,
                                          list_flight_hours=True)
    #ax1.hist(,bins=np.linspace(0,1,4))
###############################################################################
radiometer_df=radiometer_df.sort_index(axis=1)

color_series=pd.Series(data=np.nan,index=radiometer_df.columns)

# first band
color_series.loc[22.24]="k"
color_series.loc[23.04]="darkslateblue"
color_series.loc[23.84]="darkmagenta"
color_series.loc[25.44]="violet"
color_series.loc[26.24]="thistle"
color_series.loc[27.84]="gray"
color_series.loc[31.40]="silver"
# second band
color_series[50.3]="maroon"
color_series[51.76]="red"
color_series[52.8]="tomato"
color_series[53.75]="indianred"
color_series[54.94]="salmon"
color_series[56.66]="rosybrown"
color_series[58.0]="grey"
# third band
color_series[90.0]="darkgreen"
color_series[120.15]="green"
color_series[121.05]="forestgreen"
color_series[122.95]="mediumseagreen"
color_series[127.25]="darkseagreen"
# fourth band
color_series[183.91]="k"
color_series[184.81]="midnightblue"
color_series[185.81]="blue"
color_series[186.81]="royalblue"
color_series[188.31]="steelblue"
color_series[190.81]="skyblue"
color_series[195.81]="grey"
        
### -----> fill them
# fourth band

        
        
#sys.exit()
#sys.exit()
if plot_tb_hist:
    tb_channel_hist,axs=plt.subplots(5,5,figsize=(30,25),sharex=True,sharey=True)
    f=0
    tb_bins=[177.5,182.5,187.5,192.5,197.5,202.5,207.5,
             212.5,217.5,222.5,227.5,232.5,237.5,242.5,247.5,
             252.5,257.5,262.5,267.5,272.5,277.5,282.5]
    bin_mids=[180,185,190,195,200,205,210,215,
              220,225,230,235,240,245,
              250,255,260,265,270,275,280]
    for i in range(5):
        for j in range(5):
            freq=radiometer_df.columns[f]
            binned_tbs=pd.cut(radiometer_df.iloc[:,f],bins=tb_bins).value_counts().sort_index()
            binned_tbs=binned_tbs/sum(binned_tbs)                
            axs[i,j].bar(bin_mids,
                         binned_tbs.values,facecolor=color_series.loc[freq],
                         width=2,edgecolor="k",lw=1)
            axs[i,j].text(0.7,0.9,text=str(freq)+" GHz",s=10,
                          transform=axs[i,j].transAxes)
            f+=1
            axs[i,j].set_xlim([180,280])
            axs[i,j].set_ylim([0,0.6])
            if j==0:
                axs[i,j].set_yticks([0,0.2,0.4,0.6])
                axs[i,j].set_yticklabels([0, 20, 40 ,60 ])
    sns.despine(offset=10)
    fig_name="Fig03_HAMP_Tb_Distribution.pdf"
    tb_channel_hist.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
    print("Figure saved as:",plot_path+fig_name)
    
k_band_colors=[matplotlib.colors.to_rgb(colour) for colour in \
               ["k","darkslateblue","darkmagenta","violet",
                "thistle","gray","silver"]]

v_band_colors=[matplotlib.colors.to_rgb(colour) \
               for colour in ["maroon","red",
                              "tomato","indianred",
                              "salmon","rosybrown","grey"]]

w_band_colors=[matplotlib.colors.to_rgb(colour) \
               for colour in ["darkgreen","white","white","white",
                              "white","white","white"]]
f_band_colors=[matplotlib.colors.to_rgb(colour) \
               for colour in ["green","forestgreen","mediumseagreen",
                              "darkseagreen","white","white","white"]]
g_band_colors=[matplotlib.colors.to_rgb(colour) \
               for colour in ["k","midnightblue","blue","royalblue",
                              "steelblue","skyblue","white"]]                    

from matplotlib.colors import LinearSegmentedColormap
channel_1_cm=LinearSegmentedColormap.from_list("channel_1",[f_band_colors[0],
                                                            g_band_colors[0],
                                                            k_band_colors[0],
                                                            v_band_colors[0],
                                                            w_band_colors[0]],N=7)       
channel_2_cm=LinearSegmentedColormap.from_list("channel_2",[f_band_colors[1],
                                                            g_band_colors[1],
                                                            k_band_colors[1],
                                                            v_band_colors[1],
                                                            w_band_colors[1]],N=7)       
channel_3_cm=LinearSegmentedColormap.from_list("channel_3",[f_band_colors[2],
                                                            g_band_colors[2],
                                                            k_band_colors[2],
                                                            v_band_colors[2],
                                                            w_band_colors[2]],N=7)       
channel_4_cm=LinearSegmentedColormap.from_list("channel_4",[f_band_colors[3],
                                                            g_band_colors[3],
                                                            k_band_colors[3],
                                                            v_band_colors[3],
                                                            w_band_colors[3]],N=7)       
channel_5_cm=LinearSegmentedColormap.from_list("channel_5",[f_band_colors[4],
                                                            g_band_colors[4],
                                                            k_band_colors[4],
                                                            v_band_colors[4],
                                                            w_band_colors[4]],N=7)
channel_6_cm=LinearSegmentedColormap.from_list("channel_6",[f_band_colors[5],
                                                            g_band_colors[5],
                                                            k_band_colors[5],
                                                            v_band_colors[5],
                                                            w_band_colors[5]],N=7)       
channel_7_cm=LinearSegmentedColormap.from_list("channel_7",[f_band_colors[6],
                                                            g_band_colors[6],
                                                            k_band_colors[6],
                                                            v_band_colors[6],
                                                            w_band_colors[6]],N=7)
if plot_joy_hist:
    import matplotlib.cm as cm
    import joypy
    # Import Data
    #mpg = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # Draw Plot
    #plt.figure(, dpi= 300)
    fig,axes=joypy.joyplot(joy_radiometer_df,
                           by="band",alpha=0.5,
                           column=["Ch 1","Ch 2","Ch 3","Ch 4",
                                   "Ch 5","Ch 6","Ch 7"],ylim="own",
                           colormap=[channel_1_cm,channel_2_cm,channel_3_cm,
                                     channel_4_cm,channel_5_cm,channel_6_cm,
                                     channel_7_cm],overlap=0.1,figsize=(12,9))
    plt.xlim([160,240])
    plt.xticks([160,200,240])
    
    plt.tick_params(axis="x",width=3,length=8)
    #plt.setp(axes.spines.values(), linewidth=3)


    plt.xlabel("Brightness Temperature Tb (K)")
    fig_name="Fig04_HAMP_TB_Joyhist.png"
    fig.savefig(plot_path+fig_name,dpi=200,bbox_inches="tight")
    print("Figure saved as:",plot_path+fig_name)
    
    ## Decoration
    #plt.title('Joy Plot of City and Highway Mileage by Class', fontsize=22)
    #plt.show()            