# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:42:47 2025

@author: amanda
"""



# libraries
import pandas as pd
import os
import re
import json
import time
from pathlib import Path



#%% CHANGE HERE

# Set this path to the root directory containing the EPW dataset folder
# The dataset folder name is defined below (dataset_folder_name)
#
# Expected folder structure:
# path_main/
# └── Data - US Dataset EPWs    ← dataset_folder_name
#     └── STATE_city/
#         └── historical/
#             ├── STATE_city_YYYY.epw
#             ├── STATE_city_YYYY.epw
#             └── ...
#
# Example:
# path_main/
# └── Data - US Dataset EPWs/
#     └── AK_Anchorage/
#         └── historical/
#             ├── AK_Anchorage_2017.epw
#             ├── AK_Anchorage_2018.epw
#
# Replace '...' below with the path to your root directory.
path_main = r'...'


# Name of the folder containing the EPW dataset
# This folder must be located inside the main directory (path_main)
dataset_folder_name = 'Data - US Dataset EPWs'


# Set this path to the directory where all Python scripts used in this workflow are located
# This folder should contain any auxiliary or supporting code modules that are imported
# or referenced by the main script
#
# Replace '...' below with the path to your code directory
path_codes = r'...'


# Control which parts of the workflow to run
# Set to 'yes' or 'no' depending on the desired execution
#
# - run_dataset_preprocessing:
#     Performs preprocessing of the raw weather data (e.g., cleaning, formatting)
#     This step typically only needs to be run once, as it generates intermediate
#     data that can be reused in subsequent runs
#
# - run_event_detection:
#     Runs the cold snap detection and characterization using the preprocessed data
#     This step can be run multiple times (e.g., when adjusting detection parameters)
#
# Example:
# If preprocessing has already been completed, you can skip it by setting:
# run_dataset_preprocessing = 'no'
run_dataset_preprocessing = 'yes'
run_event_detection = 'yes'



#%% JUST RUN


# Get supporting code modules
os.chdir(path_codes)
from preprocessing import merge_multiple_years, dict_update, run_preprocessing
from find_coldsnaps_code import end_and_start_date, find_coldsnaps
from characterize_coldsnaps_code import  characterize_coldsnaps


# Start time
start_time = time.time()


# Ensure long path handling
long_path_main = f"\\\\?\\{path_main}"


# Define threshold and minimum required consecutive days
if run_event_detection == 'yes':
    # cold snaps
    threshold_parameter = 'dbt_c'
    threshold_method = 'variable' # or 'absolute' 'variable'
    threshold_abs_value = 5
    threshold_perc = 'dbt_op_sdeb_daily' #'dbt_op_sdeb_daily'

min_num_days = [2] # Change this to set how many consecutive days are required


periods = ['historical'] #, 'midterm', 'longterm'


# ===============================================================================================

# list all folders and files in directory
files_in_dir = os.listdir(f'{path_main}/{dataset_folder_name}')

# get lits of cities, patter:
## ^ → start of string
## [A-Z]{2} → exactly two uppercase letters (state code)
## _ → literal underscore
## [A-Za-z\s-]+ → one or more letters, spaces, or hyphens (city name)
## $ → end of string
pattern = r'^[A-Z]{2}_[A-Za-z\s-]+$'

# list of cities
cities = [c for c in files_in_dir if re.match(pattern, c)] # cities = files_in_dir




# ===============================================================================================
combinations = []
for c in cities:
    for p in periods:
        for d in min_num_days:
            combinations.append((c, p, d))
    
# combinations = combinations[15:]
          
  
# ===============================================================================================
# run pre-processing?
if run_dataset_preprocessing == 'yes':
    weather_stats = run_preprocessing (combinations, dataset_folder_name, path_main)
else:
    with open(f'{path_main}/{dataset_folder_name}/weather_stats.json', 'r') as file:
        weather_stats = json.load(file)

# ===============================================================================================

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
    
    
    # find cold snap?
    if run_event_detection == 'yes':
        # run function to detect cold snaps
        coldsnaps_start_end_dates = find_coldsnaps (daily_avg_df, threshold_param, threshold, min_days)
        # write start and end dates as csv file
        coldsnaps_start_end_dates.to_csv(os.path.join(path_main, dataset_folder_name, city, prd, "coldsnaps_start_end_dates.csv"), index=False)
        
    else:
        # just read csv file with start and end dates of cold snaps
        coldsnaps_start_end_dates = pd.read_csv(os.path.join(path_main, dataset_folder_name, city, prd, "coldsnaps_start_end_dates.csv"))
    
    
    # number of events
    num_coldsnaps = len(coldsnaps_start_end_dates)
    
    # characterize cold snaps
    if num_coldsnaps > 0:
        cs_metrics = characterize_coldsnaps(mdf, coldsnaps_start_end_dates, city, prd)
    else:
        cs_metrics = []
    
    
    # events
    events = {city: {'coldsnaps': {prd:{min_days: {'number_events': num_coldsnaps,
                                                   'cs_metrics': cs_metrics}}}
                     }}
              
    # update dictionaries: events and weather statistics
    dict_update(all_events, events)
    


# Save to JSON file
output_file = os.path.join(path_main, dataset_folder_name, "events_coldsnaps.json")
with open(output_file, 'w') as f:
    json.dump(all_events, f, indent=4, default=str)  # default=str is to handle datetime objects



# save JSON file in csv format
data = all_events

def build_df(data: dict, event_type: str) -> pd.DataFrame:
    """
    Flatten nested structure:
      data[city][event_type][period][min_days] -> {'number_events': N, '<metrics_key>': [ {...}, {...}, ... ]}
    event_type: 'coldsnaps'
    Returns a tidy DataFrame with one row per event and metric keys as columns.
    """
    metrics_key = "cs_metrics"
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

# Build DataFrame
coldsnaps_df = build_df(data, "coldsnaps")

# Save results
out_dir = Path(".")
coldsnaps_df.to_csv(os.path.join(path_main, dataset_folder_name, "coldsnaps_events.csv"), index=False)



# End time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")