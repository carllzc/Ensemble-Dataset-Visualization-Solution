# import necessary modules
# reading data
import os
import netCDF4 as nc
# operation data
import numpy as np
import pandas as pd
# machine learing
from sklearn.decomposition import PCA


def get_pca_data():
    #get the file path for loading, data file is under the same dir with the notebook
    filename="20121015_00_ecmwf_ensemble_forecast.PRESSURE_LEVELS.EUR_LL10.120.pl.nc"
    foldername="ECWMF Datasets"
    filepath=os.path.join(os.path.dirname(os.getcwd()),foldername,filename)

    # read the raw data and extract the needed data
    # exrtact the value of Geopotential under the pressure of 500 hPA in the certain
    Pressure_Levels_data = nc.Dataset(filepath,"r")
    g = 9.80655
    # get all the dimension value
    nd_1,nd_2,nd_3,nd_4,nd_5 = Pressure_Levels_data.variables['Geopotential_isobaric'][:].shape
    # get the necessary raw data
    Geopotential_Isobaric_500 = Pressure_Levels_data.variables['Geopotential_isobaric'][0,:,7,:,:]/g
    # reshape the dataset into form of (51,41*101)
    Geopotential_Isobaric_500_reshaped = np.reshape(Geopotential_Isobaric_500,(nd_2, nd_4 * nd_5))


    # use PCA to reduce dimensions under the condition of reaching 80% of all the member infomation
    exp_var = 0
    n_pc = 0
    while exp_var < 0.8:
        n_pc = n_pc + 1
        pca = PCA(n_components = n_pc)
        pca.fit(Geopotential_Isobaric_500_reshaped)
        exp_var = sum(pca.explained_variance_ratio_)

    # get the transformed raw data in the dimension-reduced space    
    pca_transformed_data = pca.transform(Geopotential_Isobaric_500_reshaped)

    return pca_transformed_data