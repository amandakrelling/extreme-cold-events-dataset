# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:42:47 2025

@author: amanda
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
import time
from pathlib import Path
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r


# directories
path_main = r'...'
path_codes = r'...'
profile_path = r'...'



# set directory
os.chdir(path_codes)
from preprocessing import merge_multiple_years, calculate_wbgt, calculate_heatindex, calculate_wet_bulb_temperature, dict_update, run_preprocessing
from find_heatwaves_coldsnaps_code import end_and_start_date, find_heatwaves, find_coldsnaps
from characterize_heatwaves_code import characterize_heatwaves, calculate_survivability, define_environmental_variables, define_personal_profile, metrics_from_hourly_hi, max_consecutive_heat_days, hw_category
from characterize_coldsnaps_code import  characterize_coldsnaps
# from detect_heatwave import find_heatwaves, cooling_degree_days, end_and_start_date, return_period_weibull, get_consecutive_hours, categorize_heatwave, define_heatwave_categories, characterize_heatwaves
from HEATLim_ak import *


# Start time
start_time = time.time()


# pre-process weather data?
run_dataset_preprocessing = 'no'
method_name = 'm8'
run_event_detection = 'yes'


# Dataset folder name
dataset_folder_name = 'Data - US Dataset EPWs'
# dataset_folder_name = 'Data'
# dataset_folder_name = 'Data - Worldwide A80'


# Ensure long path handling
long_path_main = f"\\\\?\\{path_main}"


# Define threshold and minimum required consecutive days
if run_event_detection == 'yes':
    # heat waves
    threshold_parameter = 'hi_c'
    threshold_method = 'variable' # or 'absolute' 'variable'
    threshold_abs_value = 19
    threshold_perc = 'hi_sdeb_daily' #'hi_sdeb_daily'
    # cold snaps
    threshold_parameter = 'dbt_c'
    threshold_method = 'variable' # or 'absolute' 'variable'
    threshold_abs_value = 5
    threshold_perc = 'dbt_op_sdeb_daily' #'dbt_op_sdeb_daily'

min_num_days = [2] #, 5  # Change this to set how many consecutive days are required



periods = ['historical'] #, 'midterm', 'longterm'

profile_filename = '65_over_livability_test.txt'

exposure_time = 6

# list all folders and files in directory
files_in_dir = os.listdir(f'{path_main}/{dataset_folder_name}')


# get lits of cities
if dataset_folder_name == 'Data - US Dataset EPWs':
    # ^ → start of string
    # [A-Z]{2} → exactly two uppercase letters (state code)
    # _ → literal underscore
    # [A-Za-z\s-]+ → one or more letters, spaces, or hyphens (city name)
    # $ → end of string
    pattern = r'^[A-Z]{2}_[A-Za-z\s-]+$'
    
    # list of cities
    cities = [c for c in files_in_dir if re.match(pattern, c)] # cities = files_in_dir
else:
    # ^ → Start of the string.
    # \d+ → One or more digits (numbers).
    # [A-Za-z]? → Optional single letter after the number.
    # _ → Underscore separator.
    # [A-Za-z]+ → City name (assuming it contains only letters).
    pattern = r'^\d+[A-Za-z]?_[A-Za-z]+'
    
    # list of cities
    cities = [c for c in files_in_dir if re.match(pattern, c)] # cities = files_in_dir



#%%
combinations = []
for c in cities:
    for p in periods:
        for d in min_num_days:
            combinations.append((c, p, d))
    
# combinations = combinations[15:]
            
#%% run pre-processing?
if run_dataset_preprocessing == 'yes':
    weather_stats = run_preprocessing (combinations, dataset_folder_name, path_main)
else:
    with open(f'{path_main}/{dataset_folder_name}/weather_stats.json', 'r') as file:
        weather_stats = json.load(file)


# # End time
# end_time = time.time()


# # with open(f'{path_main}/{dataset_folder_name}/weather_stats_historical.json', 'r') as file:
# #     weather_stats_historical = json.load(file)


# merged = {
#     city: {
#         **weather_stats.get(city, {}),
#         **weather_stats_historical.get(city, {})
#     }
#     for city in set(weather_stats) | set(weather_stats_historical)
# }


# # save results to JSON files
# output_file = os.path.join(path_main, dataset_folder_name, "weather_stats2.json")
# with open(output_file, "w") as json_file:
#     json.dump(merged, json_file, indent=4)


#%%
all_events = {}

# comb = combinations[-2]
for comb in combinations:
    print(comb)
    
    city = comb[0]
    prd = comb[1]
    min_days = comb[2]
    
    # define threshold
    threshold_param = threshold_parameter
    
    if threshold_method  == 'absolute':
        threshold = threshold_abs_value 
    if threshold_method == 'variable':
        threshold = weather_stats[city][prd][threshold_perc]
    
    # read pre-processed files
    mdf = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_hourly_weatherdata.csv"))
    daily_avg_df = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_daily_weatherdata.csv"))
    daily_min_df = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "multiyear_dailymin_weatherdata.csv"))
    
    
    # find heat waves?
    if run_event_detection == 'yes':
        # run function to detect heat waves
        heatwaves_start_end_dates = find_heatwaves (daily_avg_df, threshold_param, threshold, min_days)
        # write start and end dates as csv file
        heatwaves_start_end_dates.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "heatwaves_start_end_dates.csv"), index=False)
        
        # run function to detect cold snaps
        coldsnaps_start_end_dates = find_coldsnaps (daily_avg_df, threshold_param, threshold, min_days)
        # write start and end dates as csv file
        coldsnaps_start_end_dates.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "coldsnaps_start_end_dates.csv"), index=False)
        
    else:
        # just read csv file with start and end dates of heat waves
        heatwaves_start_end_dates = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "heatwaves_start_end_dates.csv"))
        
        # just read csv file with start and end dates of cold snaps
        coldsnaps_start_end_dates = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "coldsnaps_start_end_dates.csv"))
    
    
    # number of events
    num_heatwaves = len(heatwaves_start_end_dates)
    num_coldsnaps = len(coldsnaps_start_end_dates)
    
    # characterize heat waves and cold snaps
    if num_heatwaves > 0:
        hw_metrics = characterize_heatwaves(mdf, heatwaves_start_end_dates, profile_path, profile_filename, exposure_time)
    else:
        hw_metrics = []
    if num_coldsnaps > 0:
        cs_metrics = characterize_coldsnaps(mdf, coldsnaps_start_end_dates, city, prd)
    else:
        cs_metrics = []
    
    
    # events
    events = {city: {'heatwaves': {prd:{min_days: {'number_events': num_heatwaves,
                                                   'hw_metrics': hw_metrics}}},
                     'coldsnaps': {prd:{min_days: {'number_events': num_coldsnaps,
                                                   'cs_metrics': cs_metrics}}}
                     }}
              
    # events = {city:{prd:{min_days: {'number_events': num_events,
    #                                 'hw_metrics': hw_metrics} }}}
    
    # update dictionaries: events and weather statistics
    dict_update(all_events, events)
    


# output_file = os.path.join(path_main, "events_heatwaves.json")
# with open(output_file, "w") as json_file:
#     json.dump(all_events, json_file, indent=4)


# Save to JSON file
output_file = os.path.join(path_main, dataset_folder_name, "events_heatwaves_coldsnaps.json")
with open(output_file, 'w') as f:
    json.dump(all_events, f, indent=4, default=str)  # default=str is to handle datetime objects



# save JSON file in csv format (separate heat waves from cold snaps)
data = all_events

def build_df(data: dict, event_type: str) -> pd.DataFrame:
    """
    Flatten nested structure:
      data[city][event_type][period][min_days] -> {'number_events': N, '<metrics_key>': [ {...}, {...}, ... ]}
    event_type: 'heatwaves' or 'coldsnaps'
    Returns a tidy DataFrame with one row per event and metric keys as columns.
    """
    metrics_key = "hw_metrics" if event_type == "heatwaves" else "cs_metrics"
    rows = []

    for city, city_block in (data or {}).items():
        et_block = (city_block or {}).get(event_type)
        if not et_block:
            continue

        # periods like 'historical' / 'midterm' / 'longterm'
        for period, period_block in (et_block or {}).items():
            # min_days level, e.g. {"2": {"number_events": ..., "hw_metrics": [...]}}
            for min_days_key, group in (period_block or {}).items():
                # min_days might be string; make it int when possible
                try:
                    min_days = int(min_days_key)
                except Exception:
                    min_days = min_days_key

                num_events = (group or {}).get("number_events")
                metrics_list = (group or {}).get(metrics_key, [])

                # one row per event/metric dict
                for idx, metric in enumerate(metrics_list, start=1):
                    row = {
                        "city": city,
                        "period": period,
                        "min_days": min_days,
                        "number_events_in_group": num_events,
                        "event_idx": idx,  # ordinal within (city,period,min_days)
                    }
                    if isinstance(metric, dict):
                        row.update(metric)  # each metric becomes a column
                    rows.append(row)

    df = pd.DataFrame(rows)

    # Nice column order (put common fields first; keep any extra metric keys at the end)
    preferred = [
        "city", "period", "min_days", "number_events_in_group", "event_idx",
        "start", "end", "duration", "max_flag",
        "tv_hi", "cdh_hi", "overheatingdeg_hi", "maxhourly_hi", "minhourly_hi",
        "tv_dbt", "hdh_dbt", "overcoolingdeg_dbt", "maxhourly_dbt",
        "minhourly_dbt", "avghourly_dbt",
    ]
    if not df.empty:
        ordered = [c for c in preferred if c in df.columns]
        rest = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]

    return df

# Build both DataFrames
heatwaves_df = build_df(data, "heatwaves")
coldsnaps_df = build_df(data, "coldsnaps")

# Save results
out_dir = Path(".")
heatwaves_df.to_csv(os.path.join(path_main, dataset_folder_name, "heatwaves_events.csv"), index=False)
coldsnaps_df.to_csv(os.path.join(path_main, dataset_folder_name, "coldsnaps_events.csv"), index=False)




# End time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")