# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:41:03 2025

@author: Amanda
"""


# libraries
import pandas as pd
import os
import re
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
        
        # check is this is the first heat wave we are detecting
        # if hw_counter == 0:
        # if so, we have to preprocess the weather data
        mdf = merge_multiple_years (comb, dataset_folder_name, path_main)
        
        # quality check
        mdf.loc[mdf['rh_perc'] > 100, 'rh_perc'] = 100  # relative humidity cant be higher than 100!
        
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
            'dbt_spic_daily': daily_avg_df['dbt_c'].quantile(0.995),
            'dbt_sdeb_daily': daily_avg_df['dbt_c'].quantile(0.975),
            'dbt_sint_daily': daily_avg_df['dbt_c'].quantile(0.95),
            'dbt_op_spic_daily': daily_avg_df['dbt_c'].quantile(0.005),
            'dbt_op_sdeb_daily': daily_avg_df['dbt_c'].quantile(0.025),
            'dbt_op_sint_daily': daily_avg_df['dbt_c'].quantile(0.05),
            'dbt_spic_hourly': mdf['dbt_c'].quantile(0.995),
            'dbt_sdeb_hourly': mdf['dbt_c'].quantile(0.975),
            'dbt_sint_hourly': mdf['dbt_c'].quantile(0.95),
            'dbt_op_spic_hourly': mdf['dbt_c'].quantile(0.005),
            'dbt_op_sdeb_hourly': mdf['dbt_c'].quantile(0.025),
            'dbt_op_sint_hourly': mdf['dbt_c'].quantile(0.05)
            
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











