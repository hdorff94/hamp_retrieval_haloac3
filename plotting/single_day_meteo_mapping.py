# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:46:19 2022

@author: u300737
"""

import os
import sys

take_randomn_days=False
take_synth_ar_days=False
take_halo_ac3=True
plotting_path=os.getcwd()+"/plots/"
if not os.path.exists(plotting_path):
    os.makedirs(plotting_path)
if take_randomn_days:
    final_plotting_path=plotting_path+"rand_days/"
elif take_synth_ar_days:
    final_plotting_path=plotting_path+"synth_ar_days/"
elif take_halo_ac3:
    final_plotting_path=plotting_path+"halo_ac3/"
if not os.path.exists(final_plotting_path):
    os.makedirs(final_plotting_path)
    
reanalysis_path=os.getcwd()+"/../../Synthetic_Airborne_Arctic_ARs/src/"
config_path=os.getcwd()+"/../../Synthetic_Airborne_Arctic_ARs/config/"

if take_randomn_days:
    era_data_path=os.getcwd()+"/../../../"+\
        "Work/GIT_Repository/PAMTRA_Retrieval/data/ERA-5/"
if take_halo_ac3:
    era_data_path=os.getcwd()+"/../../../"+\
        "Work/GIT_Repository/HALO_AC3/data/ERA-5/"

sys.path.insert(1,reanalysis_path)
sys.path.insert(2,config_path)
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

try:
    from typhon.plots import styles
except:
    print("Typhon module cannot be imported")

#import data_config
import matplotlib
import cartopy.crs as ccrs
#from cmcrameri import cm
#from era5_on_halo_backup import ERA5
from reanalysis import ERA5
dates=["20220317","20220318","20220319","20220320","20220321"]
#,"20220316"]#['19820320','19900409', '19910428', 
#'19950306', '19970322','20050418',
#'20170425','20210331']
    
#dates=[#"20110317","20110423","20150314",
       #"20160311","20180224","20180225",
       #"20190319","20200416",
#       "20200419"
#       ]
for date in dates:
    print(date)
    if take_synth_ar_days:
        if date in ["20180224","20190319","20200416","20200419"]:
            campaign="NA_February_Run"
        else:
            campaign="Second_Synthetic_Study"
        era_data_path=os.getcwd()+"/../../../"+\
        "Work/GIT_Repository/"+campaign+"/data/ERA-5/"
    yyyy=date[0:4]
    mm=date[4:6]
    dd=date[6:8]

    #era_path=
    hmp_file=era_data_path+"total_columns_"+yyyy+"_"+mm+"_"+dd+".nc"
    theta_file=era_data_path+"temp850hPa_"+date+".nc"        

    hmp_ds=xr.open_dataset(hmp_file)
    iwv=hmp_ds["tcwv"][12,:,:]
    lwp=hmp_ds["tclw"][12,:,:]
    theta_ds=xr.open_dataset(theta_file)
    theta_e=theta_ds["theta_e"][12,:,:]
    
    # Define the plot specifications for the given variables
    met_var_dict={}
    met_var_dict["ERA_name"]    = {"IWV":"tcwv","Theta850":"Theta","LWP":"tclw"}
    met_var_dict["colormap"]    = {"IWV":"ocean_r","Theta850":"Spectral_r",
                                   "LWP":"BuPu"}
    met_var_dict["levels"]      = {"IWV":np.linspace(5,25,31),
                                   "Theta850":np.linspace(250,290,51),
                                   "LWP":np.linspace(0,300,31)}
    met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                   "Theta850":"K",
                                   "LWP":"(g$\mathrm{m}^{-2}$)"}

    thermo_fig=plt.figure(figsize=(18,9))
    ax1=thermo_fig.add_subplot(131,projection=ccrs.AzimuthalEquidistant(
                central_longitude=10.0,central_latitude=70))
    ax2=thermo_fig.add_subplot(132,projection=ccrs.AzimuthalEquidistant(
                central_longitude=10.0,central_latitude=70))
    ax3=thermo_fig.add_subplot(133,projection=ccrs.AzimuthalEquidistant(
                central_longitude=10.0,central_latitude=70))

    ax1.set_extent([-30,50,65,90])
    ax2.set_extent([-30,50,65,90])
    ax3.set_extent([-30,50,65,90])
                    
    ax1.coastlines(resolution="50m")
    ax2.coastlines(resolution="50m")        
    ax3.coastlines(resolution="50m")        

    ax1.gridlines()
    ax2.gridlines()
    ax3.gridlines()

    set_font=16
    matplotlib.rcParams.update({'font.size':set_font})
    plt.rcParams.update({'hatch.color': 'k'})  
    plt.rcParams.update({'hatch.linewidth':1.5})
        
                    
    C1=ax1.contourf(hmp_ds["longitude"],hmp_ds["latitude"],
                iwv.values,levels=met_var_dict["levels"]["IWV"],
                extend="max",transform=ccrs.PlateCarree(),
                cmap=met_var_dict["colormap"]["IWV"],alpha=0.95)
    cb=thermo_fig.colorbar(C1,ax=ax1,shrink=0.6,
                           orientation="horizontal",pad=0.02)
    cb.set_label("IWV"+" "+met_var_dict["units"]["IWV"])
    cb.set_ticks([5,10,15,20,25,30])

    C2=ax2.contourf(hmp_ds["longitude"],hmp_ds["latitude"],
                theta_e.values,levels=met_var_dict["levels"]["Theta850"],
                extend="both",transform=ccrs.PlateCarree(),
                cmap=met_var_dict["colormap"]["Theta850"],alpha=0.95)
    cb=thermo_fig.colorbar(C2,ax=ax2,shrink=0.6,orientation="horizontal",pad=0.02)
    cb.set_label("Theta_E"+" "+met_var_dict["units"]["Theta850"])
    cb.set_ticks([260,280,300])

    C3=ax3.contourf(hmp_ds["longitude"],hmp_ds["latitude"],
                lwp.values*1000,levels=met_var_dict["levels"]["LWP"],
                extend="max",transform=ccrs.PlateCarree(),
                cmap=met_var_dict["colormap"]["LWP"],alpha=0.95)
    cb=thermo_fig.colorbar(C3,ax=ax3,shrink=0.6,orientation="horizontal",pad=0.02)
    cb.set_label("LWP"+" "+met_var_dict["units"]["LWP"])
    cb.set_ticks([0,100,200,300])
    ###########################################################################
    # Mean surface level pressure
    #if meteo_var.startswith("IVT"):
    #pressure_color="royalblue"
    #sea_ice_colors=["mediumslateblue", "indigo"]
    pressure_color="purple"##"royalblue"
    sea_ice_colors=["darkorange","saddlebrown"]#["mediumslateblue", "indigo"]
    pressure_color="green"
    sea_ice_colors=["peru","sienna"]
    C_p1=ax1.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                     hmp_ds["msl"][12,:,:]/100,levels=np.linspace(950,1050,11),
                     linestyles="-.",linewidths=1.5,colors="grey",
                     transform=ccrs.PlateCarree())
    C_p2=ax2.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                     hmp_ds["msl"][12,:,:]/100,levels=np.linspace(950,1050,11),
                     linestyles="-.",linewidths=1.5,colors="grey",
                     transform=ccrs.PlateCarree())
    C_p3=ax3.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                     hmp_ds["msl"][12,:,:]/100,levels=np.linspace(950,1050,11),
                     linestyles="-.",linewidths=1.5,colors="grey",
                     transform=ccrs.PlateCarree())
    
    plt.clabel(C_p1, inline=1, fmt='%03d hPa',fontsize=12)
    plt.clabel(C_p2, inline=1, fmt='%03d hPa',fontsize=12)
    plt.clabel(C_p3, inline=1, fmt='%03d hPa',fontsize=12)
        
    # mean sea ice cover
    C_i1=ax1.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                    hmp_ds["siconc"][12,:,:]*100,levels=[15,85],
                    linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                    transform=ccrs.PlateCarree())
    C_i2=ax2.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                    hmp_ds["siconc"][12,:,:]*100,levels=[15,85],
                    linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                    transform=ccrs.PlateCarree())
    C_i3=ax3.contour(hmp_ds["longitude"],hmp_ds["latitude"],
                    hmp_ds["siconc"][12,:,:]*100,levels=[15,85],
                    linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                    transform=ccrs.PlateCarree())
    
    plt.clabel(C_i1, inline=1, fmt='%02d %%',fontsize=10)
    plt.subplots_adjust(wspace=0.1)
    plt.suptitle(date,y=0.55)
    figure_fname=final_plotting_path+"Atm_Therm_"+date+".png"
    thermo_fig.savefig(figure_fname,dpi=300,bbox_inches="tight")
    print("Figure saved as:", figure_fname)
    #-----------------------------------------------------------------#
    # Quiver-Plot
    #step=15
    #quiver_lon=np.array(ds["longitude"][::step])
    #quiver_lat=np.array(ds["latitude"][::step])
    #u=ds["IVT_u"][i,::step,::step]
    #v=ds["IVT_v"][i,::step,::step]
    #v=v.where(v>200)
    #v=np.array(v)
    #u=np.array(u)
    #quiver=plt.quiver(quiver_lon,quiver_lat,
    #                  u,v,color="lightgrey",edgecolor="k",lw=1,
    #                  scale=800,scale_units="inches",
    #                  pivot="mid",width=0.008,
    #                  transform=ccrs.PlateCarree())
    #plt.rcParams.update({'hatch.color': 'lightgrey'})
    #-----------------------------------------------------------------#
    # Show Guan & Waliser 2020 Quiver-Plot if available (up to 2019)
    #if int(flight_date[0:4])>=2020:
    #   show_AR_detection=False
    """
    if show_AR_detection:    
        import atmospheric_rivers as AR
        AR=AR.Atmospheric_Rivers("ERA",use_era5=use_era5_ARs)
        AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019,
                                               year=campaign_cls.year,
                                               month=campaign_cls.flight_month[flight])
        AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

        if not use_era5_ARs:
            
            if i<6:
                hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start,
                                                 0,:,:],
                                 hatches=['//'],cmap="bone_r",
                                 alpha=0.8,transform=ccrs.PlateCarree())
                for c,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("green")
            elif 6<=i<12:
                plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                             AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,
                                                 0,:,:],
                             hatches=[ '//'],cmap='bone',alpha=0.2,
                             transform=ccrs.PlateCarree())
            elif 12<=i<18:
                plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                             AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,
                                                 0,:,:],
                             hatches=['//'],cmap='bone_r',alpha=0.2,
                             transform=ccrs.PlateCarree())
            else:
                hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,
                                                 0,:,:],
                        hatches=['//'],cmap='bone_r',
                        alpha=0.1,transform=ccrs.PlateCarree())
                for c,collection in enumerate(hatches.collections):
                    collection.set_edgecolor("k")
    """
             
#                else:
#                    hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
#                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+i,
#                                                 0,:,:],
#                                 hatches=["//"],cmap="bone_r",
#                                 alpha=0.2,transform=ccrs.PlateCarree())
#                    for c,collection in enumerate(hatches.collections):
#                            collection.set_edgecolor("k")
#            #-----------------------------------------------------------------#
#            # Flight Track (either real or synthetic)
#            if self.analysing_campaign or self.synthetic_campaign:
#                 # Load aircraft data
#                 plot_halo_df=halo_df[halo_df.index.hour<i]
#                 if flight=="RF10":
#                     plot_real_halo_df=real_halo_df[real_halo_df.index.hour<i]
                 
#                 if i<= pd.DatetimeIndex(halo_df.index).hour[0]:
#                      ax.scatter(halo_df["longitude"].iloc[0],
#                                 halo_df["latitude"].iloc[0],
#                                s=30,marker='x',color="red",
#                                transform=ccrs.PlateCarree())
                 
#                 elif i>pd.DatetimeIndex(halo_df.index).hour[-1]+1:
#                     ax.scatter(halo_df["longitude"].iloc[-1],
#                                 halo_df["latitude"].iloc[-1],
#                                s=30,marker='x',color="red",
#                                transform=ccrs.PlateCarree())
#                 else:
#                      if flight=="RF10":
#                          ax.plot(plot_real_halo_df["longitude"],
#                                  plot_real_halo_df["latitude"],
#                                  lw=2,ls="-.",color="grey",
#                                  transform=ccrs.PlateCarree())
#                      ax.plot(plot_halo_df["longitude"],plot_halo_df["latitude"],
#                          linewidth=3.0,color="red",transform=ccrs.PlateCarree(),
#                          alpha=0.8)
                 #------------------------------------------------------------#
    ###########################################################################
    
    
            
    """
def plot_flight_map_era(self,campaign_cls,coords_station,
                            flight,meteo_var,show_AR_detection=True,
                            show_supersites=True,use_era5_ARs=False):
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"ocean_r",
                                       "IVT_v":"speed",
                                       "IVT_u":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                       "IVT":np.linspace(50,500,101),
                                       "IVT_v":np.linspace(0,500,101),
                                       "IVT_u":np.linspace(0,500,101)}
        met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        flight_date=campaign_cls.year+"-"+campaign_cls.flight_month[flight]
        flight_date=flight_date+"-"+campaign_cls.flight_day[flight]
        
        
        era5=ERA5(for_flight_campaign=True,campaign=campaign_cls.name,
                  research_flights=flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        plot_path=campaign_cls.campaign_path+"/plots/"+flight+"/"
        hydrometeor_lvls_path=campaign_cls.campaign_path+"/data/ERA-5/"
    
        file_name="total_columns_"+campaign_cls.year+"_"+\
                                    campaign_cls.flight_month[flight]+"_"+\
                                    campaign_cls.flight_day[flight]+".nc"    
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #if meteo_var.startswith("IVT"):
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        # Load Flight Track
        halo_dict={}
        if not self.analysing_campaign:
            if not (campaign_cls.campaign_name=="NAWDEX") and \
                not (campaign_cls.campaign_name=="HALO_AC3"):
                    halo_dict=campaign_cls.get_aircraft_position(self.ar_of_day)
            if campaign_cls.campaign_name=="NAWDEX":
                from flight_track_creator import Flighttracker
                Tracker=Flighttracker(campaign_cls,flight,self.ar_of_day,
                          shifted_lat=0,
                          shifted_lon=0,
                          track_type="internal")   
                halo_dict,cmpgn_path=Tracker.run_flight_track_creator()
    
            if campaign_cls.campaign_name=="HALO_AC3":
                campaign_cls.load_AC3_bahamas_ds(flight)
                halo_dict=campaign_cls.bahamas_ds
        print(halo_dict)
        if isinstance(halo_dict,pd.DataFrame):
            halo_df=halo_dict.copy() 
        elif isinstance(halo_dict,xr.Dataset):
            halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                                index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
            halo_df["Lon"]=halo_dict["IRS_LON"].data
            halo_df["Lat"]=halo_dict["IRS_LAT"].data
        else:
            if len(halo_dict.keys())==1:
                halo_df=halo_dict.values()[0]
            else:   
                    halo_df=pd.concat([halo_dict["inflow"],halo_dict["internal"],
                                       halo_dict["outflow"]])
                    halo_df.index=pd.DatetimeIndex(halo_df.index)
            if campaign_cls.name=="NAWDEX" and flight=="RF10":
                real_halo_df,_=campaign_cls.load_aircraft_position()
                real_halo_df.index=pd.DatetimeIndex(real_halo_df.index)
                real_halo_df["Hour"]=real_halo_df.index.hour
                real_halo_df=real_halo_df.rename(columns={"Lon":"longitude",
                                    "Lat":"latitude"})
                
        halo_df["Hour"]=halo_df.index.hour
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                    "Lat":"latitude"})
        
        for i in range(24):
            print("Hour of the day:",i)
            calc_time=era5.hours[i]
            map_fig=plt.figure(figsize=(12,9))
            ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=-5.0,central_latitude=70))
            if flight=="SRF06":
                ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=30.0,central_latitude=70))
            if flight=="SRF07":
                ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=40.0,central_latitude=70))
            
            ax.coastlines(resolution="50m")
            ax.gridlines()
            if campaign_cls.name=="NAWDEX":
                if flight=="RF01" or flight=="RF13":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF02":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF03":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF04":
                    ax.set_extent([-45,5,35,70])
                elif flight=="RF05" or flight=="RF06":
                    ax.set_extent([-45,5,35,70])
                elif (flight=="RF08") or (flight=="RF09") or (flight=="RF11"):
                    ax.set_extent([-40, 0, 40, 70])
                elif flight=="RF10":
                    ax.set_extent([-30,5,40,85])
                elif (flight=="RF07") or flight=="RF12":
                    ax.set_extent([-75,-20,45,70])
                else:
                    pass
            elif campaign_cls.name=="HALO_AC3_Dry_Run":
                if flight=="SRF04":
                    ax.set_extent([-10,40,60,90])
                elif flight=="SRF01":
                    ax.set_extent([-20,70,60,90])
               
                else:
                    raise Exception("Other flights are not yet provided")
            elif campaign_cls.name=="HALO_AC3":
                if (flight=="RF01") or (flight=="RF02") or (flight=="RF03") or \
                   (flight=="RF04") or (flight=="RF05") or\
                   (flight=="RF06") or (flight=="RF07") or\
                   (flight=="RF08") or (flight=="RF16"):
                   ax.set_extent([-40,30,55,90]) 
            elif campaign_cls.name=="NA_February_Run":
                ax.set_extent([-30,5,40,90])
                if flight=="SRF04":
                    ax.set_extent([-25,10,55,90])
                if flight=="SRF06" :
                    ax.set_extent([20,90,50,90])
                if flight=="SRF07":
                    ax.set_extent([10,70,50,90])
            elif campaign_cls.name=="Second_Synthetic_Study":
                ax.set_extent([-25,30,55,90])
            #-----------------------------------------------------------------#
            # Meteorological Data plotting
            # Plot Water Vapour Quantity    
            C1=plt.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"][meteo_var]][i,:,:],
                            levels=met_var_dict["levels"][meteo_var],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"][meteo_var],alpha=0.95)
            
            cb=map_fig.colorbar(C1,ax=ax)
            cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
            if meteo_var=="IWV":
                cb.set_ticks([10,15,20,25,30])
            elif meteo_var=="IVT":
                cb.set_ticks([50,100,200,300,400,500])
            else:
                pass
            # Mean surface level pressure
            if meteo_var.startswith("IVT"):
                #pressure_color="royalblue"
                #sea_ice_colors=["mediumslateblue", "indigo"]
                pressure_color="purple"##"royalblue"
                sea_ice_colors=["darkorange","saddlebrown"]#["mediumslateblue", "indigo"]
   
            else:
                pressure_color="green"
                sea_ice_colors=["peru","sienna"]
            C_p=plt.contour(ds["longitude"],ds["latitude"],
                            ds["msl"][i,:,:]/100,levels=np.linspace(950,1050,11),
                            linestyles="-.",linewidths=1.5,colors=pressure_color,
                            transform=ccrs.PlateCarree())
            plt.clabel(C_p, inline=1, fmt='%03d hPa',fontsize=12)
            # mean sea ice cover
            C_i=plt.contour(ds["longitude"],ds["latitude"],
                            ds["siconc"][i,:,:]*100,levels=[15,85],
                            linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                            transform=ccrs.PlateCarree())
            plt.clabel(C_i, inline=1, fmt='%02d %%',fontsize=10)
            
            #-----------------------------------------------------------------#
            # Quiver-Plot
            step=15
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=ds["IVT_u"][i,::step,::step]
            v=ds["IVT_v"][i,::step,::step]
            v=v.where(v>200)
            v=np.array(v)
            u=np.array(u)
            quiver=plt.quiver(quiver_lon,quiver_lat,
                                  u,v,color="lightgrey",edgecolor="k",lw=1,
                                  scale=800,scale_units="inches",
                                  pivot="mid",width=0.008,
                                  transform=ccrs.PlateCarree())
            plt.rcParams.update({'hatch.color': 'lightgrey'})
            #-----------------------------------------------------------------#
            # Show Guan & Waliser 2020 Quiver-Plot if available (up to 2019)
                #if int(flight_date[0:4])>=2020:
                #   show_AR_detection=False
            if show_AR_detection:    
                import atmospheric_rivers as AR
                AR=AR.Atmospheric_Rivers("ERA",use_era5=use_era5_ARs)
                AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019,
                                               year=campaign_cls.year,
                                               month=campaign_cls.flight_month[flight])
                AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

                if not use_era5_ARs:
            
                    if i<6:
                        hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start,
                                                 0,:,:],
                                 hatches=['//'],cmap="bone_r",
                                 alpha=0.8,transform=ccrs.PlateCarree())
                        for c,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("green")
                    elif 6<=i<12:
                        plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,
                                                 0,:,:],
                                 hatches=[ '//'],cmap='bone',alpha=0.2,
                                 transform=ccrs.PlateCarree())
                    elif 12<=i<18:
                        plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,
                                                 0,:,:],
                                 hatches=['//'],cmap='bone_r',alpha=0.2,
                                 transform=ccrs.PlateCarree())
                    else:
                        hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,
                                                 0,:,:],
                                 hatches=['//'],cmap='bone_r',
                                 alpha=0.1,
                                 transform=ccrs.PlateCarree())
                        for c,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("k")
                   
                else:
                    hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+i,
                                                 0,:,:],
                                 hatches=["//"],cmap="bone_r",
                                 alpha=0.2,transform=ccrs.PlateCarree())
                    for c,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("k")
            #-----------------------------------------------------------------#
            # Flight Track (either real or synthetic)
            if self.analysing_campaign or self.synthetic_campaign:
                 # Load aircraft data
                 plot_halo_df=halo_df[halo_df.index.hour<i]
                 if flight=="RF10":
                     plot_real_halo_df=real_halo_df[real_halo_df.index.hour<i]
                 
                 if i<= pd.DatetimeIndex(halo_df.index).hour[0]:
                      ax.scatter(halo_df["longitude"].iloc[0],
                                 halo_df["latitude"].iloc[0],
                                s=30,marker='x',color="red",
                                transform=ccrs.PlateCarree())
                 
                 elif i>pd.DatetimeIndex(halo_df.index).hour[-1]+1:
                     ax.scatter(halo_df["longitude"].iloc[-1],
                                 halo_df["latitude"].iloc[-1],
                                s=30,marker='x',color="red",
                                transform=ccrs.PlateCarree())
                 else:
                      if flight=="RF10":
                          ax.plot(plot_real_halo_df["longitude"],
                                  plot_real_halo_df["latitude"],
                                  lw=2,ls="-.",color="grey",
                                  transform=ccrs.PlateCarree())
                      ax.plot(plot_halo_df["longitude"],plot_halo_df["latitude"],
                          linewidth=3.0,color="red",transform=ccrs.PlateCarree(),
                          alpha=0.8)
                 #------------------------------------------------------------#
                 # plot Cloudnet Locations
                 if show_supersites:
                    
                    if meteo_var=="IWV":
                        station_marker_color="green"
                    else:
                        station_marker_color="red"
                    for station in coords_station.keys():
                        try:
                            if station=="Mace-Head":
                                ax.scatter(coords_station[station]["Lon"],
                                            coords_station[station]["Lat"]+3.6,
                                            s=100,marker="s",color=station_marker_color,
                                            edgecolors="black",
                                            transform=ccrs.PlateCarree())
                            else:
                                ax.scatter(coords_station[station]["Lon"],
                                            coords_station[station]["Lat"],
                                            s=100,marker="s",color=station_marker_color,
                                            edgecolors="black",
                                            transform=ccrs.PlateCarree())        
                        except:
                            pass
            #-----------------------------------------------------------------#
                 #plot Dropsonde releases
                 date=campaign_cls.year+campaign_cls.flight_month[flight]
                 date=date+campaign_cls.flight_day[flight]
                 if self.analysing_campaign:
                     if not flight=="RF06":                           
                         Dropsondes=campaign_cls.load_dropsonde_data(
                                                        date,print_arg="yes",
                                                        dt="all",plotting="no")
                         print("Dropsondes loaded")
                         # in some cases the Dropsondes variable can be a dataframe or
                         # just a series, if only one sonde has been released
                         if isinstance(Dropsondes["Lat"],pd.DataFrame):
                             dropsonde_releases=pd.DataFrame(index=\
                                    pd.DatetimeIndex(Dropsondes["LTS"].index))
                             dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[\
                                                            :,"6000.0"].values
                             dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[\
                                                            :,"6000.0"].values
        
                         else:
                             index_var=Dropsondes["Time"].loc["6000.0"]
                             dropsonde_releases=pd.Series()
                             dropsonde_releases["Lat"]=np.array(\
                                            Dropsondes["Lat"].loc["6000.0"])
                             dropsonde_releases["Lon"]=np.array(\
                                            Dropsondes["Lon"].loc["6000.0"])
                             dropsonde_releases["Time"]=index_var
        
        
                         try:
                             plotting_dropsondes=dropsonde_releases.loc[\
                                            dropsonde_releases.index.hour<i]
                             ax.scatter(plotting_dropsondes["Lon"],
                                        plotting_dropsondes["Lat"],
                                        s=100,marker="v",color="orange",
                                        edgecolors="black",
                                        transform=ccrs.PlateCarree())
                         except:
                            pass
                 
                     if flight=="RF08":
                        if i>=12:
                            ax.scatter(dropsonde_releases["Lon"],
                                   dropsonde_releases["Lat"],
                                   s=100,marker="v",color="orange",
                                   edgecolors="black",
                                   transform=ccrs.PlateCarree()) 
            #-----------------------------------------------------------------#
            ax.set_title(campaign_cls.name+" "+flight+": "+campaign_cls.year+\
                         "-"+campaign_cls.flight_month[flight]+\
                         "-"+campaign_cls.flight_day[flight]+" "+calc_time)
            
            #Save figure
            fig_name=campaign_cls.name+"_"+flight+'_'+era5.hours_time[i][0:2]+\
                "H"+era5.hours_time[i][3:6]+"_"+str(meteo_var)+".png"
            if not show_AR_detection:
                fig_name="no_AR_"+fig_name
            if show_supersites:
                fig_name="supersites_"+fig_name
            fig_path=plot_path+meteo_var+"/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=150)
            print("Figure saved as:",fig_path+fig_name)
            plt.close()
            
        return None
"""  
