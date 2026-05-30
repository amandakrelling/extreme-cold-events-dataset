# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 13:57:19 2025

@author: amanda
"""

# libraries
import pandas as pd




def end_and_start_date(group):
    start = pd.Timestamp(year=group['year'].iloc[0], month=group['month'].iloc[0], day=group['day'].iloc[0])
    end = pd.Timestamp(year=group['year'].iloc[-1], month=group['month'].iloc[-1], day=group['day'].iloc[-1], hour=23)
    return pd.Series({'start': start, 'end': end})




def find_coldsnaps (daily_avg_df, threshold_param, threshold, min_days):
    
    # avoid too high thresholds
    if threshold >= 0:
        threshold = 0
    
    # Ensure data is sorted correctly
    daily_avg_df = daily_avg_df.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)
    
    # Step 1: Create a boolean column that flags days where DBT <= threshold
    daily_avg_df['value_below_threshold'] = daily_avg_df[threshold_param] <= threshold
    
    # Step 2: Identify groups of consecutive days where value_below_threshold is True
    daily_avg_df['group'] = (daily_avg_df['value_below_threshold'] != daily_avg_df['value_below_threshold'].shift()).cumsum()
    
    # Step 3: Count streak lengths within each group
    streak_counts = daily_avg_df.groupby('group')['value_below_threshold'].transform('sum')
    
    # Step 4: Flag groups that meet the `min_days` requirement
    daily_avg_df['consecutive'] = (daily_avg_df['value_below_threshold']) & (streak_counts >= min_days)
    
    # Step 5: Filter only relevant days
    consecutive_cs_days = daily_avg_df[daily_avg_df['consecutive']]
    
    # Step 6: Group clusters of consecutive days
    grouped = consecutive_cs_days.groupby('group')
    start_end = grouped.apply(end_and_start_date)
    start_end = start_end.reset_index(drop=True)
    
    
    # Step 7: Guarantee independence of events, that is, number of days between different events needs to be >= min_days, or else they are one single event
    # if events were detected:
    if len(start_end) > 0:
        
        # A) work on a copy
        df = start_end.copy()
        
        # B) make sure datetime
        df["start"] = pd.to_datetime(df["start"])
        df["end"]   = pd.to_datetime(df["end"])
        
        # C) compute gaps to the next event
        ## df["start"].shift(-1) → takes the “start” column and shifts it up one row. So, in row 0, you now have the start of row 1
        ## Subtracting df["end"] from that gives you: start_of_next_event - end_of_this_event
        ## Then .dt.days turns the timedelta into an integer number of days
        ## Result: every row tells you how many days until the next event starts
        df["gap_days"] = (df["start"].shift(-1) - df["end"]).dt.days
        
        # D) merge consecutive rows where gap_days < min_days
        merged_periods = []
        i = 0         # position (row index) while we walk
        n = len(df)   # total number of rows
        
        while i < n:
            # start of the current merged block
            cur_start = df.loc[i, "start"]
            cur_end   = df.loc[i, "end"]
        
            # keep moving forward while the gap to the next event is < min_days
            while i < n - 1 and df.loc[i, "gap_days"] < min_days:
                i += 1
                # extend the end to the end of the next event
                cur_end = df.loc[i, "end"]
        
            # store the merged period
            merged_periods.append({"start": cur_start, "end": cur_end})
        
            # move to the next unprocessed row
            i += 1
        
        # E) final dataframe with merged periods
        start_end = pd.DataFrame(merged_periods)
    

    return start_end
