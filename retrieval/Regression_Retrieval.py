import numpy as np
import pandas as pd
import xarray as xr

import os
class Regression_Retrieval():
    def __init__(self,state_vector,state_vector_name,brightness_Ts,order,
                 training_dates,save_path,added_noise=True,
                 obs_height=12000,hour="12"):
        self.x=state_vector
        self.x_name=state_vector_name
        self.TBs=brightness_Ts
        
        self.y               = brightness_Ts.values[:,:-1]
        self.freqs           = brightness_Ts.iloc[:,:-1].columns
        self.order           = order
        self.hour            = hour
        self.training_dates  = training_dates
        self.save_path       = save_path
        self.obs_height      = obs_height
        self.added_noise     = added_noise
    @staticmethod
    def updt(total, progress):
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
    
    def build_K_reg(self):
        """
        Constructs the observation matrix typically used for regression retrievals where the
        rows indicate the samples (i.e., time series). The first column usually contains "1"
        only and the remaining columns contain observations in first and higher order.

        Inherited Parameters:
        -----------
        self.y : array of floats
            Observation vector. Must be a numpy array with M observations and N samples. The
            shape must be N x M. (Also if M == 1, y must be a 2D array.)
        self.order : int
            Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:1
        """

        n_obs = self.y.shape[1]                # == M
        n_samples = self.y.shape[0]        # == N

        assert self.y.shape == (n_samples,n_obs)

        # generate regression matrix K out of obs vector:
        self.K_reg = np.ones((n_samples, self.order*n_obs+1))
        self.K_reg[:,1:n_obs+1] = self.y #----> linear terms

        if self.order > 1: # e.g. quadratic or cubic or whatever
            for kk in range(self.order-1):
                jj = kk + 1
                self.K_reg[:,jj*n_obs+1:(jj+1)*n_obs+1] = self.y**(jj+1)
    
    def pure_regression(self):
        # compute m_est
        self.K_reg_T = self.K_reg.T
        
        try:
            inverse_matrix=np.linalg.inv(self.K_reg_T.dot(self.K_reg))
        except:
            print("This is a singular matrix so use linalg.pinv, which leverages SVD to approximate initial matrix.")
            inverse_matrix=np.linalg.pinv(self.K_reg_T.dot(self.K_reg))
        
        self.m_est = inverse_matrix.dot(self.K_reg_T).dot(self.x)
        
    def entire_regression(self):
        """
        Computes regression coefficients m_est to map observations y (i.e., brightness temperatures)
        to state variable x (i.e., temperature profile at one height level, or IWV). The regression
        order can also be specified.
        Parameters:
        -----------
        x : array of floats
            State variable vector. Must be a numpy array with N samples (N = training data size).
        y : array of floats
            Observation vector. Must be a numpy array with M observations (i.e., M frequencies)
            and N samples. The shape must be N x M. (Also if M == 1, y must be a 2D array.)
            order : int
                Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
                1
        """
        # Generate matrix from observations:
        self.build_K_reg()
        self.pure_regression()
    
    def m_est_as_csv(self,column_integrated=True):
        """
        This saves the coefficients derived from the regression so that they can be intercompared later on
        """
        file_end=".csv"
            
        if column_integrated:
            reg_coeffs=pd.DataFrame(data=np.nan, columns=self.freqs, index=["offset","a**1","b**2"])
            for f, freq in enumerate(reg_coeffs.columns):
                reg_coeffs.iloc[0,f]=self.m_est[0]
                if f==0:
                    reg_coeffs.iloc[1,f]=self.m_est[1]
                    reg_coeffs.iloc[2,f]=self.m_est[2]
                else:
                    reg_coeffs.iloc[1,f]=self.m_est[2*(f+1)-1]
                    reg_coeffs.iloc[2,f]=self.m_est[2*(f+1)]
            self.reg_coeffs=reg_coeffs
            reg_coeffs.name=str(self.training_dates)
            # Path for individual variables will be created
            self.save_path+="/"+self.x_name+"/"
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_name=self.save_path+self.x_name+"_Retrieval_coeffs"
            if not self.added_noise:
                file_name=file_name+"_no_noise"
            if hasattr(self,"obs_height"):
                file_name+="_"+str(int(self.obs_height))
            file_name+="_"+self.hour+"UTC"
            reg_coeffs.to_csv(file_name+file_end)
            print("Retrieval coeffs saved as:",file_name+file_end)
        else:
            #Vertical profiles of m_est have different shape 
            if len(self.TBs["Date"].unique())==1:
                # only one date was considered
                # so date is considered in file_name
                file_name=self.x_name+"_Retrieval_coeffs_"+str(self.TBs["Date"].unique()[0])
            else:
                file_name=self.x_name+"_Retrieval_coeffs_several_dates"
            if not self.added_noise:
                file_name=file_name+"_no_noise"
            if hasattr(self,"obs_height"):
                file_name+="_"+str(int(self.obs_height))
            file_name+="_"+self.hour+"UTC"
            file_name+=file_end
            
            # Path for individual variables will be created
            self.save_path+="/"+self.x_name+"/"
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.height_m_est.to_csv(self.save_path+file_name)
            print("Retrieval coeffs saved as:", self.save_path+file_name)
    
    def get_vertical_m_est(self,x_all_heights,var_to_retrieve="",brightness_Tbs=pd.DataFrame(),
                           take_regridded=True,save_coeffs=True):
        #print(x_all_heights)
        if not var_to_retrieve=="":
            # This means that we use another variable as defined when initiating the class
            self.x_name=var_to_retrieve
        self.x_all_heights=x_all_heights[self.x_name]
        
        if not hasattr(self,"y"):
            if not brightness_Tbs.shape[0]==0:
                if brightness_Tbs.columns[-1]=="Date":
                    brightness_Tbs=brightness_Tbs.iloc[:,:-1]
                self.y=brightness_Tbs
            else:
                raise Exception("you have to reallocate your brightness temperatures")
        print("Retrieve ",self.x_name," via Regression")
        
        columns_list=["error"]
        a_list=[str(freq)+"_a*1" for freq in self.freqs]
        b_list=[str(freq)+"_b**2" for freq in self.freqs]
        #for i in range(len(a_list)):
        #    columns_list.append(a_list[i])
        #    columns_list.append(b_list[i])
        columns_list=columns_list+a_list+b_list
        # Create height specific dataframe
        self.height_m_est=pd.DataFrame(data=np.nan,index=x_all_heights["z"].values,
                                       columns=columns_list)
        print("Height Regression")
        for height in range(self.x_all_heights["z"].shape[0]):
            self.x=self.x_all_heights[:,height]
            self.entire_regression()
            self.height_m_est.iloc[height,:]=self.m_est.T
            self.updt(self.x_all_heights["z"].shape[0],height)
        if save_coeffs:
            self.m_est_as_csv(column_integrated=False)
    @staticmethod
    def get_relevant_retrieval_height(retrieval_coeff_heights,halo_height=5000):
        idx_height=np.argmin(abs(np.array(retrieval_coeff_heights)-halo_height))
        relevant_height=retrieval_coeff_heights[idx_height]
        return idx_height,relevant_height

    def open_height_relevant_retrieval(retrieval_coeff_heights,
        var_to_retrieve,act_halo_height,
            coeff_path="C:\\Users\\u300737\\Desktop\\Desktop_alter_Rechner\\"+\
            "PhD_UHH_WIMI\\Work\\GIT_Repository\\hamp_processing_py\\"+\
                "hamp_processing_python\\Flight_Data\\HALO_AC3\\retrieval\\"):
    
        height=act_halo_height.copy()
        idx_height,relevant_height=\
            Regression_Retrieval.get_relevant_retrieval_height(
                                    retrieval_coeff_heights,halo_height=height)
        m_est_height=pd.read_csv(coeff_path+var_to_retrieve+\
                                 "_Retrieval_coeffs_several_dates_"+\
                                 str(relevant_height)+".csv",
                                 index_col="Unnamed: 0")
        return m_est_height