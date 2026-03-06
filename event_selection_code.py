# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 17:03:31 2025

@author: Amanda
"""

import os
import pandas as pd


# directories
path_main = r'...'


# Dataset folder name
dataset_folder_name = 'Data - US Dataset'

# read dataset of detected events
cs = pd.read_csv(os.path.join(path_main, dataset_folder_name, "coldsnaps_events.csv"))




def pick_best_idx(d, target, group, used=None):
    dist = (d - target).abs()

    # Optionally exclude already-used indices
    if used:
        dist = dist.loc[~dist.index.isin(used)]

    # If nothing left to pick, signal to caller
    if dist.empty:
        return None

    # Step 1: closest duration
    min_dist = dist.min()
    tied_idx = dist[dist == min_dist].index
    tied = group.loc[tied_idx]

    # Treat NaNs as "worse" so they don't win ties
    tied_avg = tied['avghourly_dbt'].fillna(float('inf'))
    min_avg = tied_avg.min()
    tied = tied.loc[tied_avg == min_avg]

    tied_min = tied['minhourly_dbt'].fillna(float('inf'))
    min_min = tied_min.min()
    tied = tied.loc[tied_min == min_min]

    # Stable fallback
    return tied.index[0]



def select_events_based_on_duration(group):
    if len(group) == 1:
        return group.assign(which='single', target_duration=group['duration'].iloc[0])

    d = group['duration']

    p_low  = d.quantile(0.025)
    p_med  = d.quantile(0.5)
    p_high = d.quantile(0.975)

    rows = []
    used = set()

    for label, target in [('p2.5', p_low), ('median', p_med), ('p97.5', p_high)]:
        idx = pick_best_idx(d, target, group, used=used)

        # If we can't find any unused event left, stop trying to add more
        if idx is None:
            continue

        rows.append(group.loc[[idx]].assign(which=label, target_duration=target))
        used.add(idx)

    return pd.concat(rows) if rows else group.iloc[0:0]  # empty df with same columns




selected = (
    cs
    .groupby(['city', 'category'], group_keys=False)
    .apply(select_events_based_on_duration)
)

# write as csv file
selected.to_csv(os.path.join(path_main, dataset_folder_name, "coldsnaps_events_selected.csv"), index=False)



# formatting final dataset
selected_formatted = selected[['city', 'start', 'end', 'duration', 'category', 
                               'avghourly_dbt', 'maxhourly_dbt', 'minhourly_dbt',
                               'hdh_dbt_0c', 'overcoolingdeg_dbt_0c']]

selected_formatted.to_csv(os.path.join(path_main, dataset_folder_name, "coldsnaps_events_selected_final.csv"), index=False)



