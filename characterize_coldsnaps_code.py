# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:34:47 2025

@author: amanda
"""

# libraries
import pandas as pd
import os
import re
import time
import json
import numpy as np
from scipy.stats import genpareto, kstest       
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r


        
def metrics_from_hourly_dbt (mdf, mask):
    filtered_dbt = mdf.loc[mask, 'dbt_c']
    
    # Calculate most extreme and average conditions - maximum and minimum dbt, average dbt
    maxhourly_dbt = filtered_dbt.max()
    minhourly_dbt = filtered_dbt.min()
    avghourly_dbt = filtered_dbt.mean()
    
    # Calculate daily max, min, and average temperatures
    daily_max = [max(filtered_dbt[i:i+24]) for i in range(0, len(filtered_dbt), 24)]
    daily_min = [min(filtered_dbt[i:i+24]) for i in range(0, len(filtered_dbt), 24)]
    daily_mean = [(mx + mn) / 2 for mx, mn in zip(daily_max, daily_min)]
    
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
    
    # Heating degree days (HDD)
    hdd = {}
    for label, base in bases.items():
        diff = [base - t for t in daily_mean]
        total = sum(d for d in diff if d > 0)
        hdd[f"hdd_dbt_{label}"] = round(total, 1)
    
    # Round numbers
    maxhourly_dbt = round(maxhourly_dbt, 1)
    minhourly_dbt = round(minhourly_dbt, 1)
    avghourly_dbt = round(avghourly_dbt, 1)
    
    return hdh, hdd, maxhourly_dbt, minhourly_dbt, avghourly_dbt



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
        cs_cat = 0   # or "None", "No cold snap"
    
    return cs_cat


def cs_duration_category (duration):
    
    if duration == 2:
        category_duration = 'short'
    else:
        category_duration = 'long'
    
    return category_duration


def return_period_weibull (overcoolingdeg_dbt):
    num_years = 10
    
    # Rank by severity (largest to smallest)
    ranked_indices = sorted(range(len(overcoolingdeg_dbt)), key=lambda i: overcoolingdeg_dbt[i], reverse=True)
    sorted_degrees = [overcoolingdeg_dbt[i] for i in ranked_indices]

    # Calculate empirical return period
    return_periods = [round((num_years + 1) / (rank + 1), 1) for rank in range(len(overcoolingdeg_dbt))]

    # Restore return period list to match original order
    return_period_original_order = [None] * len(overcoolingdeg_dbt)
    for sorted_idx, original_idx in enumerate(ranked_indices):
        return_period_original_order[original_idx] = return_periods[sorted_idx]

    return return_period_original_order
 

def empirical_return_periods(vals, n_years):
    """
    vals: 1D array of event severities (higher = more severe)
    n_years: number of years of data
    returns: list of RPs in same order
    """
    import numpy as np
    vals = np.asarray(vals, float)
    valid = np.isfinite(vals)
    vals_valid = vals[valid]

    rps = []
    for v in vals:
        if not np.isfinite(v):
            rps.append(np.nan)
            continue
        # count how many events are at least this severe
        count = np.sum(vals_valid >= v)
        lam = count / float(n_years)  # events per year
        rp = np.inf if lam == 0 else 1.0 / lam
        rps.append(rp)
    return rps




def cs_return_period(cs_metrics,
                     n_years=20,                          # how many years of data these events represent (default 20)
                     tail_quantile=0.9,                   # which upper quantile defines the “extreme” tail (default 0.9 = top 10%)
                     metric_key="hdh_dbt_0c",             # which key inside each event holds the value you want to use for severity
                     rp_key="return_period_years"):       # the name of the new key you want to add to each event to store the return period
    """
    For a list of event dicts (cs_metrics), compute a return period for EACH event.
    - Tail (>= threshold): use POT–GPD
    - Non-tail (< threshold): use empirical frequency
    Results are rounded to 0 digits (whole years).

    Parameters
    ----------
    cs_metrics : list[dict]
        Each item is a dict with at least `metric_key`.
    n_years : int
        Number of years of data represented by these events.
    tail_quantile : float
        Upper quantile to define the tail (e.g. 0.9 = top 10% used for GPD).
    metric_key : str
        Key in each event that holds the HDH value.
    rp_key : str
        Key to write the return period (years).

    Returns
    -------
    list[dict]
        Same list, now with rp_key added to each event.
    """
    
    # Extract the hdh values
    vals = np.array([evt.get(metric_key, np.nan) for evt in cs_metrics], dtype=float)
    # Keep only valid numbers (for fitting)
    valid_mask = np.isfinite(vals)
    vals_valid = vals[valid_mask]
    
    # ALWAYS compute empirical RP first
    empirical_rps = empirical_return_periods(vals, n_years)
    for evt, erp in zip(cs_metrics, empirical_rps):
        # round to whole years, like before
        evt["empirical_rp_years"] = np.round(erp, 0) if np.isfinite(erp) else erp
    
    # if not enough valid data, stop here (we already saved empirical)
    if vals_valid.size < 5:
        print("[cs_return_period] not enough valid values → keeping empirical only.")
        for evt in cs_metrics:
            # since we couldn't fit GPD, make main RP just the empirical one
            evt[rp_key] = evt["empirical_rp_years"]
            evt["rp_source"] = "empirical"
        return cs_metrics
    
    
    # # Make sure we have enough data to fit
    # ## If you have fewer than 5 valid events, fitting a tail distribution is meaningless
    # if vals_valid.size < 5:
    #     print("...not enough data, using empirical RPs only...")
    #     rps = empirical_return_periods(vals, n_years)
    #     for evt, rp in zip(cs_metrics, rps):
    #         evt["return_period_years"] = np.round(rp, 0) if np.isfinite(rp) else rp
    #         evt["rp_source"] = "empirical"
    #     return cs_metrics

    # Tail threshold
    u = np.quantile(vals_valid, tail_quantile)

    # Separate tail events and build exceedances
    tail_vals = vals_valid[vals_valid >= u]   # Separate tail events
    exceedances = tail_vals - u
    
    # Edge case: no tail points
    ## If for some weird reason there are no values >= u, we can't fit a GPD
    ## So we fall back to a simple empirical return period (for each event, count how many events are at least that big, turn that into an annual rate, invert to get years)
    if exceedances.size == 0:
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                rp = np.nan
            else:
                count = np.sum(vals_valid >= v)
                lam = count / float(n_years)
                rp = np.inf if lam == 0 else 1.0 / lam
            cs_metrics[i][rp_key] = np.round(rp, 0) if np.isfinite(rp) else rp
        return cs_metrics
    
    # Fit the GPD to the tail
    ## We fit a Generalized Pareto Distribution to the tail exceedances
    ## floc=0 tells it “the threshold is already subtracted, so start at 0”
    ## We get the GPD parameters: shape (ξ), location (0), and scale (σ)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    
    # Estimate how often tail events happen
    ## tail_vals.size = how many tail events we saw in total
    ## Divide by number of years → average tail events per year
    ## We need this to turn “tail probability” into annual probability
    lambda_tail = tail_vals.size / float(n_years)
    
    # Define a helper to get RP (return period) for one value
    def rp_for_one(x):
        # This inner function does the logic for one event
        ## If x is NaN → return NaN
        if not np.isfinite(x):
            return np.nan
        ##  If x >= u (so it’s in the tail) -> use GPD
        if x >= u:
            y = x - u    # Convert to exceedance y
            Fy = genpareto.cdf(y, shape, loc=0, scale=scale)  # Get GPD CDF at y
            tail_prob = 1.0 - Fy  # 1 - Fy = probability that a tail event is even more extreme than this one
            annual_prob = lambda_tail * tail_prob  # Multiply by lambda_tail to get annual probability of such an event
            if annual_prob <= 0:
                return np.inf
            return 1.0 / annual_prob # Invert → return period in years
        ## If x < u (non-tail): empirical
        count = np.sum(vals_valid >= x)  # Count how many events were at least this big
        lam_emp = count / float(n_years) # Turn into “events per year”
        if lam_emp <= 0:
            return np.inf
        return 1.0 / lam_emp  # Invert → return period
    
    # # Apply to every event and round
    # ## Loop over the original list order (so you don't mess up alignment)
    # ## Compute the return period for that event's metric
    # ## Round to 0 digits → whole years
    # ## Store it back into that event’s dict under "return_period_years" (or whatever you passed)
    # for i, v in enumerate(vals):
    #     rp = rp_for_one(v)
    #     # 🔹 round to 0 digits (whole years)
    #     cs_metrics[i][rp_key] = np.round(rp, 0) if np.isfinite(rp) else rp
    #     cs_metrics[i]["rp_source"] = "gpd"
    
    # Apply GPD-based RP, but keep empirical too
    # Apply to every event and round
    ## Loop over the original list order (so you don't mess up alignment)
    ## Compute the return period for that event's metric
    ## Round to 0 digits → whole years
    ## Store it back into that event’s dict under "return_period_years" (or whatever you passed)
    for i, v in enumerate(vals):
        gpd_rp = rp_for_one(v)
        if np.isfinite(gpd_rp):
            cs_metrics[i][rp_key] = np.round(gpd_rp, 0)
            cs_metrics[i]["rp_source"] = "gpd"
        else:
            # if for some reason it's not finite, fall back to empirical
            cs_metrics[i][rp_key] = cs_metrics[i]["empirical_rp_years"]
            cs_metrics[i]["rp_source"] = "empirical"

    return cs_metrics






# keep the diagnostics helper here so this function is self-contained
def gpd_diagnostics(exceedances,
                    u,
                    shape,
                    scale,
                    lambda_tail,
                    n_years,
                    return_periods=(10, 25, 50, 100),
                    n_boot=300,
                    random_state=42):
    exc = np.asarray(exceedances, dtype=float)
    n = exc.size

    # KS test
    ks_stat, ks_pval = kstest(exc, "genpareto", args=(shape, 0, scale))

    # log-likelihood, AIC, BIC
    logpdf = genpareto.logpdf(exc, shape, loc=0, scale=scale)
    loglik = logpdf.sum()
    k_param = 2  # shape, scale
    AIC = 2 * k_param - 2 * loglik
    BIC = k_param * np.log(n) - 2 * loglik

    def gpd_return_level(T, u, xi, sig, lam):
        if lam * T <= 1:
            return np.nan
        if np.isclose(xi, 0.0):
            y = sig * np.log(lam * T)
        else:
            y = sig / xi * ((lam * T) ** xi - 1)
        return u + y

    rl = {}
    for T in return_periods:
        rl[T] = gpd_return_level(T, u, shape, scale, lambda_tail)

    # bootstrap CIs
    rng = np.random.default_rng(random_state)
    rl_ci = {T: [] for T in return_periods}
    for _ in range(n_boot):
        sample = rng.choice(exc, size=n, replace=True)
        try:
            b_shape, b_loc, b_scale = genpareto.fit(sample, floc=0)
        except Exception:
            continue
        for T in return_periods:
            zT = gpd_return_level(T, u, b_shape, b_scale, lambda_tail)
            rl_ci[T].append(zT)

    rl_ci_final = {}
    for T, vals in rl_ci.items():
        if len(vals) == 0:
            rl_ci_final[T] = (np.nan, np.nan)
        else:
            low, high = np.percentile(vals, [2.5, 97.5])
            rl_ci_final[T] = (low, high)

    return {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pval,
        "loglik": loglik,
        "AIC": AIC,
        "BIC": BIC,
        "return_levels": rl,
        "return_levels_ci": rl_ci_final,
    }


def cs_extreme_plots(cs_metrics,
                     n_years=20,
                     tail_quantile=0.9,
                     metric_key="hdh_dbt_0c",
                     location_name="location",
                     period_label="period",
                     save_dir=None,
                     show_plots=True):
    """
    Generate plots for one city AND save a diagnostics CSV next to the plots.
    """
    # ---------- prep data ----------
    vals = np.array([evt.get(metric_key, np.nan) for evt in cs_metrics], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        print(f"[{location_name}] Not enough valid values to make plots/diagnostics.")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    base_name = f"{location_name}_{period_label}".replace(" ", "_")

    # ---------- 1) MRL plot ----------
    thresholds = np.quantile(vals, np.linspace(0.6, 0.95, 20))
    mrl = []
    for u0 in thresholds:
        exc = vals[vals > u0] - u0
        mrl.append(exc.mean() if exc.size > 0 else np.nan)

    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, mrl, marker="o")
    plt.xlabel("Threshold u")
    plt.ylabel("Mean exceedance above u")
    plt.title(f"MRL - {location_name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{base_name}_MRL.png"), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ---------- fit GPD on chosen tail ----------
    u = np.quantile(vals, tail_quantile)
    tail_vals = vals[vals >= u]
    exceedances = tail_vals - u

    if exceedances.size < 3:
        print(f"[{location_name}] Too few tail events for GPD diagnostics.")
        return

    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    lambda_tail = tail_vals.size / float(n_years)

    # ---------- 2) Return level plot ----------
    plt.figure(figsize=(5, 4))
    vals_sorted = np.sort(vals)[::-1]
    ranks = np.arange(1, len(vals_sorted) + 1)
    emp_rp = n_years / ranks

    T_model = np.logspace(0, 3, 200)  # 1–1000 years
    z_model = []
    for T in T_model:
        p_tail = 1.0 - 1.0 / (lambda_tail * T)
        if p_tail <= 0 or p_tail >= 1:
            z_model.append(np.nan)
            continue
        yq = genpareto.ppf(p_tail, shape, loc=0, scale=scale)
        z_model.append(u + yq)
    z_model = np.array(z_model)

    plt.scatter(emp_rp, vals_sorted, label="Observed", s=20)
    plt.plot(T_model, z_model, color="r", label="Fitted GPD")
    plt.xscale("log")
    plt.xlabel("Return period (years)")
    plt.ylabel(metric_key)
    plt.title(f"Return level plot - {location_name}")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{base_name}_ReturnLevel.png"), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ---------- 3) GPD QQ plot ----------
    plt.figure(figsize=(5, 4))
    exc_sorted = np.sort(exceedances)
    probs = (np.arange(1, exc_sorted.size + 1) - 0.5) / exc_sorted.size
    theo_q = genpareto.ppf(probs, shape, loc=0, scale=scale)
    max_q = max(exc_sorted.max(), theo_q.max())
    plt.scatter(theo_q, exc_sorted, s=20)
    plt.plot([0, max_q], [0, max_q], "r--")
    plt.xlabel("Theoretical GPD quantiles")
    plt.ylabel("Empirical exceedances")
    plt.title(f"GPD QQ plot - {location_name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{base_name}_QQ.png"), dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # ---------- 4) diagnostics + save CSV ----------
    diag = gpd_diagnostics(
        exceedances=exceedances,
        u=u,
        shape=shape,
        scale=scale,
        lambda_tail=lambda_tail,
        n_years=n_years,
        return_periods=(10, 25, 50, 100),
        n_boot=300,
    )

    # flatten return levels for CSV
    row = {
        "city": location_name,
        "period": period_label,
        "threshold": u,
        "shape": shape,
        "scale": scale,
        "tail_points": exceedances.size,
        "lambda_tail": lambda_tail,
        "ks_stat": diag["ks_stat"],
        "ks_pvalue": diag["ks_pvalue"],
        "AIC": diag["AIC"],
        "BIC": diag["BIC"],
    }
    for T, zT in diag["return_levels"].items():
        row[f"RL_{T}"] = zT
        low, high = diag["return_levels_ci"][T]
        row[f"RL_{T}_low"] = low
        row[f"RL_{T}_high"] = high

    df = pd.DataFrame([row])

    if save_dir:
        csv_path = os.path.join(save_dir, f"{base_name}_diagnostics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[{location_name}] diagnostics saved to {csv_path}")



def cs_return_period_cdf(cs_metrics,
                         n_years=20,
                         metric_key="hdh_dbt_0c",
                         rp_key="return_period_cdf_years"):
    """
    Fit a smooth CDF (Gumbel) to the event intensities and compute a
    distribution-based return period for EACH event.

    RP = 1 / (lambda_per_year * P(X >= x))

    - always writes a new key in each event: rp_key
    - does NOT replace your existing 'return_period_years'
    """
    # extract values
    vals = np.array([evt.get(metric_key, np.nan) for evt in cs_metrics], dtype=float)
    vals_valid = vals[np.isfinite(vals)]

    if vals_valid.size < 3:
        # not enough to fit a distribution; just fill NaN and return
        for evt in cs_metrics:
            evt[rp_key] = np.nan
        return cs_metrics

    # events per year (how often do we see a cold snap at all)
    lambda_events = len(cs_metrics) / float(n_years)

    # fit a Gumbel distribution to the severities
    # gumbel is common for extremes and simple
    loc, scale = gumbel_r.fit(vals_valid)

    for i, v in enumerate(vals):
        if not np.isfinite(v):
            cs_metrics[i][rp_key] = np.nan
            continue

        # CDF = P(X <= v)
        Fv = gumbel_r.cdf(v, loc=loc, scale=scale)
        # exceedance prob = P(X >= v)
        p_exceed = 1.0 - Fv

        # annual probability = how often we get an event at least this strong
        annual_prob = lambda_events * p_exceed

        if annual_prob <= 0:
            rp = np.inf
        else:
            rp = 1.0 / annual_prob

        # round to whole years like the rest
        cs_metrics[i][rp_key] = np.round(rp, 0) if np.isfinite(rp) else rp

    return cs_metrics






def characterize_coldsnaps(mdf, coldsnaps_start_end_dates, city, prd):
      
    
    # Temporary datetime series creation (only if you plan to use it later)
    temp_datetime = pd.to_datetime(mdf[['year', 'month', 'day', 'hour']])
    coldsnaps_start_end_dates[['start', 'end']] = coldsnaps_start_end_dates[['start', 'end']].apply(pd.to_datetime)

    
    cs_metrics = []
    # start = coldsnaps_start_end_dates['start'][10]
    # end = coldsnaps_start_end_dates['end'][10]
    for start, end in zip(coldsnaps_start_end_dates['start'], coldsnaps_start_end_dates['end']):
        mask = (temp_datetime >= start) & (temp_datetime <= end)
        
        # calculate metrics based on hourly hi
        hdh, hdd, maxhourly_dbt, minhourly_dbt, avghourly_dbt = metrics_from_hourly_dbt (mdf, mask)
        
        # category of cold snap
        cs_cat = cs_category (avghourly_dbt)
        
        # duration of cold snap
        duration = (end.normalize() - start.normalize()).days + 1
        
        category_duration = cs_duration_category (duration)
        
        # Store in dictionary
        cs_metrics.append({
            'start': start,
            'end': end,
            'duration': duration,
            # 'survive': survive,
            # 'max_flag': max_flag,
            # 'tv_dbt': tv_dbt,
            'maxhourly_dbt': maxhourly_dbt,
            'minhourly_dbt': minhourly_dbt,
            'avghourly_dbt': avghourly_dbt,
            'category': cs_cat,
            'category_duration': category_duration,
            **hdh,
            **hdd
            # Add more if needed
        })
    
    
    cs_metrics = cs_return_period(cs_metrics, n_years=20, tail_quantile=0.9)
    
    # NEW: smooth / CDF-based RP
    cs_metrics = cs_return_period_cdf(cs_metrics,
                                      n_years=20,
                                      metric_key="hdh_dbt_0c",
                                      rp_key="return_period_cdf_years")
    
    # 2) generate plots (this will save 3 pngs in save_dir)
    # filename will contain city + period
    cs_extreme_plots(cs_metrics,
                     n_years= 20,
                     tail_quantile=0.9,
                     metric_key="hdh_dbt_0c",
                     location_name=city,
                     period_label=prd,
                     save_dir=r'C:\Users\amanda\Documents\AmandaKrelling\Dataset_ColdSnaps\Graphs',
                     show_plots=False)
    
    # overcooling_list = pd.DataFrame(cs_metrics)['overcoolingdeg_dbt'].dropna().tolist()
    # returnp = return_period_weibull (overcooling_list)
    
    return cs_metrics
        
        


    












