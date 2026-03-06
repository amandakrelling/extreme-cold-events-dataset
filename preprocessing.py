# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:41:03 2025

@author: Amanda
"""


# libraries
import pandas as pd
import numpy as np
from psychrolib import SetUnitSystem, GetTWetBulbFromRelHum, SI
import os
import re
import metpy.calc as mpcalc
from metpy.calc import heat_index
from metpy.calc import wet_bulb_temperature ## <<<<<<<<<<<<<<<<<<<<<<<<< here
from metpy.calc import dewpoint_from_relative_humidity  ## <<<<<<<<<<<<<<<<<<<<<<<<< here
from metpy.units import units
import json



# comb = combinations[18]
def merge_multiple_years (comb, dataset_folder_name, path_main):
    
    city = comb[0]
    prd = comb[1]
    print(prd)
    
    # list all folders and files in directory
    csv_files = os.listdir(f'{path_main}/{dataset_folder_name}/{city}/{prd}')
    
    # read, extract and concatenate only necessary columns
    long_path_main = f"\\\\?\\{path_main}" # handle long file paths by prefixing with '\\?\'
    
    # list of csv files
    # r''	Raw string notation – Ensures that backslashes (\) are treated literally in Python.
    # ^	Start of the string – Ensures the match begins exactly at the start of the string.
    # POWER	Matches "POWER" literally – The string must begin with "POWER".
    # .*	Matches any characters (zero or more) – Allows for any text (letters, numbers, underscores, etc.) between "POWER" and ".csv".
    # \.	Escapes the dot (.) to match a literal period (.) – Otherwise, . would match any character.
    # csv	Matches "csv" literally – Ensures that the string ends with "csv".
    # $	End of the string – Ensures the match ends exactly at ".csv".
    
    # ----------------------------
    # Small US DATASET
    # ----------------------------
    if dataset_folder_name == 'Data':
        if prd == 'historical':
            csvs = [file for file in csv_files if re.match(r'^POWER.*\.csv$', file)]
        else:
            csvs = [file for file in csv_files if re.match(r'^.*\.csv$', file)]
        
        # define columns
        columns_to_extract = [0, 1, 2, 3, 6, 7, 8, 9, 13, 21] # 'year', 'month', 'day', 'hour', 'dbt_c', dpt_c, 'rh_perc', 'atmpressure_pa, 'radglobhor_whpersqm', 'windspeed_mpers'
        
        df_list = [pd.read_csv(os.path.join(long_path_main, dataset_folder_name, city, prd, file),
                               usecols=columns_to_extract,  # Use indices directly
                               skiprows=8, 
                               header=None) 
                   for file in csvs]
        
        # merge all dataframes into one
        mdf = pd.concat(df_list, ignore_index=True)
        
        # rename all columns
        mdf.columns = ['year', 'month', 'day', 'hour', 'dbt_c', 'dpt_c', 'rh_perc', 'atmpressure_pa', 'radglobhor_whpersqm', 'windspeed_mpers']
    
    # ----------------------------
    # WORLDWIDE DATA SET - FROM ANNEX 80
    # ----------------------------
    if dataset_folder_name == 'Data - Worldwide A80':
        csvs = [file for file in csv_files if re.match(r'^.*\.csv$', file)]
        csvs = [c for c in csvs if 'multiyear' not in c]
        print(csvs)
              
        df_list = [pd.read_csv(os.path.join(long_path_main, dataset_folder_name, city, prd, file),
                               skiprows=0) 
                   for file in csvs]
        
        # merge all dataframes into one
        mdf = pd.concat(df_list, ignore_index=True)
        
        ## Convert 'time_lst' to datetime format - but first check data format!
        mdf['time_lst'] = mdf['time_lst'].astype(str).str.replace('-', '/', regex=False)   # date_parts = [p.replace('-','/') for p in date_parts] # replace "-" with "/"
        date_parts = mdf['time_lst'] # get only data column
        date_parts = [p.split(' ')[0] for p in date_parts] # remove hour portion, which is separated by a space
       
        ## separate each part of the date
        date_parts_0 = [int(p.split('/')[0]) for p in date_parts]
        date_parts_1 = [int(p.split('/')[1]) for p in date_parts]
        date_parts_2 = [int(p.split('/')[2]) for p in date_parts]
        
        ## check most common formats
        if (max(date_parts_0) > 1000) and (max(date_parts_1) <= 12): # year / month / day
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%Y/%m/%d %H:%M')
        if (max(date_parts_0) > 1000) and (max(date_parts_2) <= 12): # year / day / month
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%Y/%d/%m %H:%M')
        if (max(date_parts_1) <= 12) and (max(date_parts_2) > 1000): # day / month / year
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%d/%m/%Y %H:%M')
        if (max(date_parts_0) <= 12) and (max(date_parts_2) > 1000): # month / day / year
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%m/%d/%Y %H:%M')    
        if (max(date_parts_0) <= 31) and (max(date_parts_1) <= 12): # day / month / year
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%d/%m/%y %H:%M')
        if (max(date_parts_0) <= 12) and (max(date_parts_1) <= 31): # month / day / year
            mdf['time_lst'] = pd.to_datetime(mdf['time_lst'], format='%m/%d/%y %H:%M')
       
        
        # Create new columns
        mdf['year'] = mdf['time_lst'].dt.year
        mdf['month'] = mdf['time_lst'].dt.month
        mdf['day'] = mdf['time_lst'].dt.day
        mdf['hour'] = mdf['time_lst'].dt.hour
        
        # Drop the original column time_lst, drop huss and clt (not used)
        mdf.drop(columns='time_lst', inplace=True)
        if 'huss' in mdf.columns:
            mdf.drop(columns='huss', inplace=True)
        if 'clt' in mdf.columns:
            mdf.drop(columns='clt', inplace=True)
        # Rename wind speed column (problem with Rome)
        if 'wsp' in mdf.columns:
            mdf.rename(columns={'wsp': 'sfcWind'}, inplace=True)
        # Add pressure if missing
        if 'ps' not in mdf.columns:
            print("Pressure missing!")
            mdf['ps'] = 101325 # sea level
        
        # make sure order of columns is correct
        mdf = mdf[['year', 'month', 'day', 'hour', 'tas', 'hurs', 'rsds', 'sfcWind', 'ps']]
        
        # rename all columns
        mdf.columns = ['year', 'month', 'day', 'hour', 'dbt_c', 'rh_perc', 'radglobhor_whpersqm', 'windspeed_mpers', 'atmpressure_pa']
        
        
    # ----------------------------
    # FINAL US DATASET
    # ----------------------------
    
    if dataset_folder_name == 'Data - US Dataset EPWs':
        
        csvs = [file for file in csv_files if re.match(r'^.*\.epw$', file)]
        
        # define columns
        columns_to_extract = [0, 1, 2, 3, 6, 7, 8, 9, 13, 21] # 'year', 'month', 'day', 'hour', 'dbt_c', dpt_c, 'rh_perc', 'atmpressure_pa, 'radglobhor_whpersqm', 'windspeed_mpers'
        
        df_list = [pd.read_csv(os.path.join(long_path_main, dataset_folder_name, city, prd, file),
                               usecols=columns_to_extract,  # Use indices directly
                               skiprows=8, 
                               header=None) 
                   for file in csvs]
        
        # merge all dataframes into one
        mdf = pd.concat(df_list, ignore_index=True)
        
        # rename all columns
        mdf.columns = ['year', 'month', 'day', 'hour', 'dbt_c', 'dpt_c', 'rh_perc', 'atmpressure_pa', 'radglobhor_whpersqm', 'windspeed_mpers']
        
        
    return mdf




#%% wbgt

def calculate_wbgt (mdf):
    
    # set psychrolib to SI units
    SetUnitSystem(SI)
    
    # Extract relevant meteorological variables from EPW
    Tdb = mdf['dbt_c'].to_numpy()  # Dry Bulb Temperature (C)
    RH = mdf['rh_perc'].to_numpy() / 100  # Convert RH from % to fraction
    WindSpeed = mdf['windspeed_mpers'].to_numpy()  # Wind speed (m/s)
    SolarRad = mdf['radglobhor_whpersqm'].to_numpy()  # Global Horizontal Radiation (W/m²)
    Press = mdf['atmpressure_pa'].to_numpy()  # Atmospheric Pressure (Pa) from EPW file
    
    # step 1: Compute Natural Wet Bulb Temperature (Tnw)
    ## this function calculates the wet bulb temperature (Tnw) based on temperature, humidity, and pressure 
    ## it finds the temperature at which air becomes saturated while undergoing evaporative cooling
    ## example: if it's 30°C with 80% humidity, the effective wet bulb temp (Tnw) is around 24.5°C, which is used in WBGT calculations
    # Tnw = np.vectorize(GetTWetBulbFromRelHum)(Tdb, RH, Press)
    Tnw = np.array([GetTWetBulbFromRelHum(T, rh, p) for T, rh, p in zip(Tdb, RH, Press)])
    
    # step 2: Compute Globe Temperature (Tg)
    Tg = Tdb + 0.05 * SolarRad - 0.5 * WindSpeed
    
    # step 3: Compute WBGT Using ISO 7243 Standard
    WBGT = 0.7 * Tnw + 0.2 * Tg + 0.1 * Tdb
    
    # add WBGT results to DataFrame
    mdf["tnw_c"] = Tnw
    mdf["tg_c"] = Tg
    mdf["wbgt_c"] = WBGT
    
    return mdf





#%% heat index
## https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.heat_index.html


def calculate_heatindex (mdf):

    # Convert entire series to MetPy units
    temp_series = mdf['dbt_c'].to_numpy() * units.degC  # Convert temperature
    humidity_series = mdf['rh_perc'].to_numpy() * units.percent  # Convert humidity
    
    # compute heat index in Celsius (fully vectorized)
    mdf['hi_c'] = heat_index(temp_series, humidity_series, mask_undefined=False).to('degC').m
    
    return mdf


#%% wet bulb temperature

def calculate_wet_bulb_temperature(mdf):
    mdf = mdf.copy()  # Ensures no SettingWithCopyWarning
    
    dbt = mdf['dbt_c'].values * units.degC
    rh = mdf['rh_perc'].values * units.percent
    pressure = (mdf['atmpressure_pa'].values / 100.0) * units.hPa
    # pressure = 1013.25 * units.hPa  # Standard atmospheric pressure in hPa
    
    # Calculate the ambient dewpoint given air temperature and relative humidity
    dewpoint = mpcalc.dewpoint_from_relative_humidity(dbt, rh)
    
    # Calculate the wet-bulb temperature using Normand’s rule (based on pressure, temperature and dewpoint)
    wbt = mpcalc.wet_bulb_temperature(pressure, dbt, dewpoint)
    mdf['wbt_c'] = wbt

    return mdf



#%% update dictionaries
def dict_update(d, u):
    """Recursively updates nested dictionaries."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            dict_update(d[k], v)  # Recursively update nested dict
        else:
            d[k] = v  # Otherwise, just update the value
            


#%% run functions

def run_preprocessing (combinations, dataset_folder_name, path_main):
    
    # check if 'combinations' is within a list (this is a problem when trying only one combination)
    if not isinstance(combinations, list):
        combinations = [combinations]
    
    # Keep first occurrence of each (city, horizon) pair
    seen = set()
    unique_city_period_combinations = []
    for a, b, c in combinations:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            unique_city_period_combinations.append((a, b, c))
    
    weather_stats = {}
    
    # comb = unique_city_period_combinations[0]
    for comb in unique_city_period_combinations:
        print(comb)
        
        city = comb[0]
        prd = comb[1]
        # min_days = comb[2]
        # hw_counter = min_num_days.index(min_days) # check if this is the first heat wave type (specific duration) being identified for that specific climate and period
        
        # check is this is the first heat wave we are detecting
        # if hw_counter == 0:
        # if so, we have to preprocess the weather data
        mdf = merge_multiple_years (comb, dataset_folder_name, path_main)
        
        # quality check
        mdf.loc[mdf['rh_perc'] > 100, 'rh_perc'] = 100  # relative humidity cant be higher than 100!
        
        # calculate other variables
        mdf = calculate_wbgt (mdf)
        mdf = calculate_heatindex (mdf)
        # mdf = calculate_wet_bulb_temperature (mdf) # not calculating WBT at this point - it takes super long and i'm not planning to use it
        
        # compute daily averages by grouping by 'year', 'month', 'day'
        daily_avg_df = mdf.groupby(['year', 'month', 'day'], as_index=False).mean()
        
        # compute daily minimum hi_c
        daily_min_df = mdf.groupby(['year', 'month', 'day'], as_index=False).min()
                
        # calculate a few parameters
        w_stats = {city: {prd:{
            
            'dbt_max_hourly': max(mdf['dbt_c']),
            'dbt_min_hourly': min(mdf['dbt_c']),
            'dbt_max_daily': max(daily_avg_df['dbt_c']),
            'dbt_min_daily': min(daily_avg_df['dbt_c']),
            'wbgt_max_hourly': max(mdf['wbgt_c']),
            'wbgt_max_daily': max(daily_avg_df['wbgt_c']),
            'hi_max_hourly': max(mdf['hi_c']),
            'hi_max_daily': max(daily_avg_df['hi_c']),
            'dbt_spic_daily': daily_avg_df['dbt_c'].quantile(0.995),
            'dbt_sdeb_daily': daily_avg_df['dbt_c'].quantile(0.975),
            'dbt_sint_daily': daily_avg_df['dbt_c'].quantile(0.95),
            'dbt_op_spic_daily': daily_avg_df['dbt_c'].quantile(0.005),
            'dbt_op_sdeb_daily': daily_avg_df['dbt_c'].quantile(0.025),
            'dbt_op_sint_daily': daily_avg_df['dbt_c'].quantile(0.05),
            'wbgt_spic_daily': daily_avg_df['wbgt_c'].quantile(0.995),
            'wbgt_sdeb_daily': daily_avg_df['wbgt_c'].quantile(0.975),
            'wbgt_sint_daily': daily_avg_df['wbgt_c'].quantile(0.95),
            'hi_spic_daily': daily_avg_df['hi_c'].quantile(0.995),
            'hi_sdeb_daily': daily_avg_df['hi_c'].quantile(0.975),
            'hi_sint_daily': daily_avg_df['hi_c'].quantile(0.95),
            'dbt_spic_hourly': mdf['dbt_c'].quantile(0.995),
            'dbt_sdeb_hourly': mdf['dbt_c'].quantile(0.975),
            'dbt_sint_hourly': mdf['dbt_c'].quantile(0.95),
            'dbt_op_spic_hourly': mdf['dbt_c'].quantile(0.005),
            'dbt_op_sdeb_hourly': mdf['dbt_c'].quantile(0.025),
            'dbt_op_sint_hourly': mdf['dbt_c'].quantile(0.05),
            'wbgt_spic_hourly': mdf['wbgt_c'].quantile(0.995),
            'wbgt_sdeb_hourly': mdf['wbgt_c'].quantile(0.975),
            'wbgt_sint_hourly': mdf['wbgt_c'].quantile(0.95),
            'hi_spic_hourly': mdf['hi_c'].quantile(0.995),
            'hi_sdeb_hourly': mdf['hi_c'].quantile(0.975),
            'hi_sint_hourly': mdf['hi_c'].quantile(0.95),
            'hi_85_hourly': mdf['hi_c'].quantile(0.85),
            'hi_85_daily': daily_avg_df['hi_c'].quantile(0.85) #,
            # 'wbt_max_hourly': max(mdf['wbt_c'])
            
            
            }}}
        

        # Save results to a new CSV file
        mdf.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_hourly_weatherdata.csv"), index=False)
        daily_avg_df.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_daily_weatherdata.csv"), index=False)
        daily_min_df.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_dailymin_weatherdata.csv"), index=False)
        
        
        # update dictionary: weather statistics
        dict_update(weather_stats, w_stats)
    
    # save results to JSON files
    output_file = os.path.join(path_main, dataset_folder_name, "weather_stats.json")
    with open(output_file, "w") as json_file:
        json.dump(weather_stats, json_file, indent=4)
        
    return (weather_stats)











