import os
import re
import pandas as pd


def load_data(y, path):
    """ load the folder paths and get the simulations 
    
    Parameters: 
    y(list): list of strings representing time series readings
    sim(strin): simulation file
    fault_type(string): folder where the sim are saved
    
    Returns:
    df(pd.dataframe): dataframe representing the time series
    """
    
    # read data and transform to pandas dataframe
    df = pd.read_parquet(path=path, engine='pyarrow')
    
    # Extract headers and remove units
    headers_with_units = list(df)
    headers = [
        item.split(" [")[0] if "[" in item and "]" in item else item
        for item in headers_with_units
    ]
    
    # Use unit-less headers
    df.columns = headers
    df = df[["Time", *y]]
    
    return df
    
def load_all(y, fault_path):

    # create dataframe for all sims in a fault directory
    df = pd.concat(
        [
            load_data(y, os.path.join(fault_path, sim)).assign(Simulation=sim)
            for sim in os.listdir(fault_path)
        ],
        ignore_index=True,
    )
    return df
    
    