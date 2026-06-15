#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:26:21 2024

@author: nicolaspinzonmatapi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

#READ AND LOAD DATA

from pathlib import Path

BASE_DIR = Path(__file__).parent

dates = BASE_DIR / "Data" / "2_sigma_mean_w.txt"



# ----------------------------------
# LOAD DATA and Definir Parametros
# ----------------------------------
dat = np.loadtxt(dates) 

# ----------------------------
# PARAMETERS
# ----------------------------
input_file = dates   # your file
reference_year = 2024             # you used 2024 as zero reference for BP
num_MC_attempts = 1000            # number of random catalogues
#seed = 42                         

#np.random.seed(seed)

# ----------------------------
# LOAD & PREPROCESS DATA
# ----------------------------
# Expecting a 2 x N matrix: row0 = max (older), row1 = min (younger)
dat = np.loadtxt(input_file)  

# If your file is oriented differently (N x 2), adapt accordingly.
# Here we apply the same conversion you used before: convert BC/AD to BP with 2024 as zero.
# Values < 0 (BC) -> -val + 2024 ; Values >=0 (AD) -> 2024 - val
data_bp = np.where(dat < 0, -1 * dat + reference_year, reference_year - dat)

# Ensure float
data_bp = np.array(data_bp, dtype=float)

# Check shape
if data_bp.ndim != 2 or data_bp.shape[0] != 2:
    raise ValueError(f"Expected a 2xN matrix after loading; got shape {data_bp.shape}. Please check your file orientation.")

age_max = data_bp[0, :]   # older bound (e.g. 95% upper)
age_min = data_bp[1, :]   # younger bound (e.g. 5% lower)
N_events = age_min.size

print("Loaded data shape:", data_bp.shape)
print("Number of events:", N_events)

# ----------------------------
# FUNCTIONS
# ----------------------------
def memory_coefficient(tau):
    
    #Compute memory coefficient M following Goh & Barabási (2008).
    #tau: 1D array of interevent times (positive)
    #Returns M (float) or np.nan if undefined (zero variance).
    
    tau = np.asarray(tau)
    if tau.size < 2:
        return np.nan
    tau_i = tau[:-1]
    tau_ip1 = tau[1:]
    mu1 = np.mean(tau_i)
    mu2 = np.mean(tau_ip1)
    s1 = np.std(tau_i, ddof=0)
    s2 = np.std(tau_ip1, ddof=0)
    denom = s1 * s2
    if denom == 0:
        return np.nan
    return np.mean((tau_i - mu1) * (tau_ip1 - mu2)) / denom

# ----------------------------
# MONTE CARLO TO NOT CONSIDER ONLY THE MEAN AGE
# ----------------------------
memory_values = []
cov_values = []
burstiness_values = []   # NEW
catalogs = []            # optional

for it in range(num_MC_attempts):

    # 1) pick a random age within each event's allowed range
    random_ages = np.random.uniform(age_min, age_max)

    # 2) sort them to ensure temporal order
    random_ages_sorted = np.sort(random_ages)

    # 3) compute interevent times
    interevent_times = np.diff(random_ages_sorted)

    # Guard against zero or negative intervals (should not happen with sort)
    if np.any(interevent_times <= 0):
        continue

    # ---- 4) MEMORY COEFFICIENT ----
    M = memory_coefficient(interevent_times)
    memory_values.append(M)

    # ---- 5) COV = std / mean ----
    mean_it = np.mean(interevent_times)
    std_it = np.std(interevent_times, ddof=0)
    cov = std_it / mean_it if mean_it != 0 else np.nan
    cov_values.append(cov)

    # ---- 6) BURSTINESS ----
    if (std_it + mean_it) != 0:
        B = (std_it - mean_it) / (std_it + mean_it)
    else:
        B = np.nan
    burstiness_values.append(B)

    # Optional: store a few catalogs
    # if it < 50:
    #     catalogs.append(interevent_times)

# Convert to arrays
memory_values = np.array(memory_values)
cov_values = np.array(cov_values)
burstiness_values = np.array(burstiness_values)





kde = gaussian_kde(memory_values, bw_method='silverman')  # Adjust bandwidth as needed

# Generate x values for evaluating the KDE
x_values = np.linspace(min(memory_values), max(memory_values), 10000)

# Evaluate the KDE
y_values = kde(x_values)
dx = x_values[1] - x_values[0]
print(dx)
area_total= np.sum(y_values) * dx

M_plus = np.sum(y_values[x_values>0]*dx)
M_minus = np.sum(y_values[x_values<0]*dx)


print("La probabilidad de Memory coefficient (P|M>0)", M_plus)
print("La probabilidad de Memory coefficient (P|M<0)", M_minus)
    


kde = gaussian_kde(burstiness_values, bw_method='silverman')  # Adjust bandwidth as needed

# Generate x values for evaluating the KDE
x_values1 = np.linspace(min(burstiness_values), max(burstiness_values), 10000)

# Evaluate the KDE
y_values1 = kde(x_values1)
dx = x_values1[1] - x_values1[0]
area_total= np.sum(y_values1) * dx

M_plus = np.sum(y_values1[x_values1 > -0.33]*dx)



print("La probabilidad de Memory coefficient (P|0 > B > -0.33)", M_plus)



# ----------------------------
# SUMMARY & SAVE
# ----------------------------
print("\n=== Summary ===")
print("Simulations kept:", memory_values.size)

print("\nMemory (M):")
print("  mean = ", np.nanmean(memory_values))
print("  std  = ", np.nanstd(memory_values))
print("  5th, 50th, 95th pct:", np.nanpercentile(memory_values, [5,50,95]))

print("\nCOV:")
print("  mean = ", np.nanmean(cov_values))
print("  std  = ", np.nanstd(cov_values))
print("  5th, 50th, 95th pct:", np.nanpercentile(cov_values, [5,50,95]))

print("\nBurstiness (B):")
print("  mean = ", np.nanmean(burstiness_values))
print("  std  = ", np.nanstd(burstiness_values))
print("  5th, 50th, 95th pct:", np.nanpercentile(burstiness_values, [5,50,95]))

# Save arrays
#np.savetxt('memory_values.txt', memory_values, header='Memory coefficient (M) per simulation')
#np.savetxt('cov_values.txt', cov_values, header='COV per simulation')
#np.savetxt('burstiness_values.txt', burstiness_values, header='Burstiness (B) per simulation')

# ----------------------------
# PLOTS
# ----------------------------
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.hist(memory_values[~np.isnan(memory_values)], bins=30, color="lightgrey",edgecolor='black', alpha=0.7)
#plt.axvline(x=np.nanmean(memory_values), color='r', linestyle='--')
plt.xlabel('Memory coefficient (M)')
plt.ylabel('Frequency')
plt.ylim(0,140)
plt.title('Distribution of Memory Coefficient')

plt.subplot(1,3,2)
plt.hist(burstiness_values[~np.isnan(burstiness_values)], bins=30, color="lightgrey",edgecolor='black', alpha=0.7)
plt.xlabel('Burstiness B')
plt.ylabel('Frequency')
plt.title('Distribution of Burstiness')

plt.subplot(1,3,3)
plt.hist(cov_values[~np.isnan(cov_values)], bins=30, color="lightgrey",edgecolor='black', alpha=0.7)
plt.xlabel('COV (std/mean)')
plt.ylabel('Frequency')
plt.title('Distribution of COV')



plt.subplot(2,3,1)
plt.plot(x_values, y_values, label='KDE of Mean Interevent Times', color='red')

plt.tight_layout()
output = BASE_DIR / "Outputs"
plt.savefig(output / "M_COV_B_distributions.pdf", bbox_inches="tight")
plt.show()




"""
y = memory_values
x = burstiness_values


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.02
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    
    ax_histx.hist(x, bins=bins,
                  color='steelblue', alpha=0.7,
                  label='Memory')

    ax_histy.hist(y, bins=bins,
                  orientation='horizontal',
                  color='tomato', alpha=0.7,
                  label='Burstiness')
    
    

fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')

scatter_hist(x, y, axs['scatter'], axs['histx'], axs['histy'])
"""

