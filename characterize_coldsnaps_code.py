# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:34:47 2025

@author: amanda
"""

# libraries
import pandas as pd


        
def metrics_from_hourly_dbt (mdf, mask):
    filtered_dbt = mdf.loc[mask, 'dbt_c']
    
    # Calculate most extreme and average conditions - maximum and minimum dbt, average dbt
    maxhourly_dbt = filtered_dbt.max()
    minhourly_dbt = filtered_dbt.min()
    avghourly_dbt = filtered_dbt.mean()
    
    # Base temperatures
    bases = {
        "4c": 4.44,
        "0c": 0.0,
        "m15c": -15.0,
        "m26c": -26.0,
        "m32c": -32.0,
    }
    
    # Heating degree hours (HDH) and overcoolingdegree
    hdh = {}
    for label, base in bases.items():
        # Calculate positive differences only
        differences = base - filtered_dbt #threshold
        positive_diff = differences[differences > 0]
        
        # Sum positive hours - heating degree hours
        hdh_dbt = positive_diff.sum()
        
        # Normalize by number of hours - overheating degree
        overcoolingdeg_dbt = hdh_dbt / len(filtered_dbt) if len(filtered_dbt) > 0 else 0

        hdh[f"hdh_dbt_{label}"] = round(hdh_dbt, 1)
        hdh[f"overcoolingdeg_dbt_{label}"] = round(overcoolingdeg_dbt, 1)
    
    
    # Round numbers
    maxhourly_dbt = round(maxhourly_dbt, 1)
    minhourly_dbt = round(minhourly_dbt, 1)
    avghourly_dbt = round(avghourly_dbt, 1)
    
    return hdh, maxhourly_dbt, minhourly_dbt, avghourly_dbt



def cs_category (avghourly_dbt):
    
    if avghourly_dbt <= 0 and avghourly_dbt > -15:
        cs_cat = 1
    elif avghourly_dbt <= -15 and avghourly_dbt > -26:
        cs_cat = 2
    elif avghourly_dbt <= -26 and avghourly_dbt > -32:
        cs_cat = 3
    elif avghourly_dbt <= -32:
        cs_cat = 4
    else:
        # This case may occur due to differences between daily-average-based
        # detection (used to define cold snaps) and hourly-average-based
        # characterization. These events are still valid cold snaps, so they
        # are assigned to Category 1 (lowest severity)
        cs_cat = 1
    
    return cs_cat



def characterize_coldsnaps(mdf, coldsnaps_start_end_dates, city, prd):
      
    
    # Temporary datetime series creation (only if you plan to use it later)
    temp_datetime = pd.to_datetime(mdf[['year', 'month', 'day', 'hour']])
    coldsnaps_start_end_dates[['start', 'end']] = coldsnaps_start_end_dates[['start', 'end']].apply(pd.to_datetime)

    
    cs_metrics = []
    # start = coldsnaps_start_end_dates['start'][10]
    # end = coldsnaps_start_end_dates['end'][10]
    for start, end in zip(coldsnaps_start_end_dates['start'], coldsnaps_start_end_dates['end']):
        mask = (temp_datetime >= start) & (temp_datetime <= end)
        
        # calculate metrics based on hourly dbt
        hdh, maxhourly_dbt, minhourly_dbt, avghourly_dbt = metrics_from_hourly_dbt (mdf, mask)
        
        # category of cold snap
        cs_cat = cs_category (avghourly_dbt)
        
        # duration of cold snap
        duration = (end.normalize() - start.normalize()).days + 1
        
        
        # Store in dictionary
        cs_metrics.append({
            'start__month_day_year': start,
            'end__month_day_year': end,
            'duration__days': duration,
            'category': cs_cat,
            'dbt_mean__c': avghourly_dbt,
            'dbt_max__c': maxhourly_dbt,
            'dbt_min__c': minhourly_dbt,
            'hdh__c': hdh.get("hdh_dbt_0c"),
            'overcoolingdeg_c': hdh.get("overcoolingdeg_dbt_0c")
        })
    
    return cs_metrics
        
        


    












