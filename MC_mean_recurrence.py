#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:26:21 2024

@author: nicolaspinzonmatapi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



#READ AND LOAD DATA

from pathlib import Path

BASE_DIR = Path(__file__).parent

dates = BASE_DIR / "Data" / "2_sigma_mean_w.txt"



# ----------------------------------
# LOAD DATA and Definir Parametros
# ----------------------------------
dat = np.loadtxt(dates)  # define the file name as needed
#dat = np.loadtxt('mongolia_ages.txt')
data = np.where(dat<0, -1*dat + 2024, 2024 -  dat) #convert BC and AD to BP I used 2024 as cero reference
#print(data1)
data=np.int_(data)
print(data[1,0])
#np.savetxt('mean_BP.txt',data2, fmt='%i', delimiter='\t' )


# Step size for time intervals
step_size = 5.0

# Number of samples to take per attempt
sampling = 100

# Number   of attempts to compute mean interevent times
num_attempts = 1000

# Create a list to store mean interevent times for each pair of columns
mean_interevent_times_lists = []


# Loop through consecutive columns to compute mean interevent times
for col_index in range(data.shape[1] - 1):
    # Extract the min and max from each column to create the time range
    time_range_1 = np.arange(data[1, col_index], data[0, col_index], step_size)
    time_range_2 = np.arange(data[1, col_index + 1], data[0, col_index + 1], step_size)

    # List to store the mean interevent times for this specific pair of time ranges
    mean_interevent_times = []
  
    # Perform 1000 attempts to compute mean interevent times
    for i in range(num_attempts):
        # Here we impose chronological order for the random samples
        random_samples_1 = []
        random_samples_2 = []
        while(data[1,0]>0):
            random_sample_1 = np.random.choice(time_range_1)
            random_sample_2 = np.random.choice(time_range_2)
            if random_sample_1 > random_sample_2:
                random_samples_1.append(random_sample_1)
                random_samples_2.append(random_sample_2)
                if(np.size(random_samples_1) == sampling and np.size(random_samples_2) == sampling):
                    random_samples_1 = np.array(random_samples_1)
                    random_samples_2 = np.array(random_samples_2)
                    print(random_samples_1.dtype)
                    break

        # Compute the interevent times (PDF2 - PDF1)
        interevent_times = random_samples_1 - random_samples_2
        print(interevent_times)
        
        # Calculate the mean interevent time for this attempt
        mean_interevent_time = np.mean(interevent_times) #since we compute the difference in BP, the mean is negative, so we multi times -1
        #mean_interevent_time = np.mean(np.abs(interevent_times))
        # Save the mean to the list
        mean_interevent_times.append(mean_interevent_time)
    # Store the mean interevent times in the main list
    mean_interevent_times_lists.append(mean_interevent_times)
#print(mean_interevent_times_lists)
# Create subplots for histograms with percentiles
fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # Adjust the grid layout as needed
axs = axs.flatten()  # Flatten the array of axes


######## FIT A GAUSSIAN KDE FUCNTION #######

kde = gaussian_kde(mean_interevent_times, bw_method='silverman')  # Adjust bandwidth as needed

# Generate x values for evaluating the KDE
x_values = np.linspace(min(mean_interevent_times), max(mean_interevent_times), 10000)

# Evaluate the KDE
y_values = kde(x_values)



n_intervals = len(mean_interevent_times_lists)
errors_list = []
#mean_interevent_times_lists = mean_interevent_times_lists[::-1]
# PLOT HISTOGRAMS AND PRINT A TABLE WITH the 5th, 50th, and 95th PERCENTILES
for ax, (index, means) in zip(axs, enumerate(mean_interevent_times_lists)):
    # Plot histogram
    ax.hist(means, bins=30, alpha=0.6, color='deepskyblue', edgecolor='black') #salmon
    plt.plot(x_values, y_values, label='KDE of Mean Interevent Times', color='red')
    # Calculate and plot the 5th, 50th, and 95th percentiles
    percentiles = np.percentile(means, [2.5, 50, 97.5, 25, 75])
    #ax.axvline(percentiles[0], color='red', linestyle='--', label='5th Percentile')
    #ax.axvline(percentiles[1], color='black', linestyle='-', label='50th Percentile')
    #ax.axvline(percentiles[2], color='green', linestyle='--', label='95th Percentile')  
    #print(percentiles)
    mean_interevent_times_df = pd.DataFrame()
    mean_interevent_times_df[f'Interval {index}-{index + 1}'] = percentiles
    th25 = mean_interevent_times_df.iloc[3]
    th75 = mean_interevent_times_df.iloc[4]
    plas = mean_interevent_times_df.iloc[2]  #calcular el limite superior del 2sigma error
    minus=mean_interevent_times_df.iloc[0] #calcular el limite inferior del 2sigma error
    errors=pd.DataFrame({'-2σ':np.int_(minus),'25th percentile':np.int_(th25), 'mean IT':np.int_(percentiles[1]), '75th percentile':np.int_(th75), '+2σ':np.int_(plas)})
    errors_list.append(errors)
    errors_all = pd.concat(errors_list, ignore_index=True) #concatenar el mean interevent time y los errores en una sola tabla
    
    # Set labels and titles
    ax.set_ylabel('Frequency')
    interval_start = n_intervals - index - 1
    interval_end = n_intervals - index
    ax.set_title(f'Interval {interval_start}-{interval_end}')
    #ax.legend()
    output = BASE_DIR / "Outputs"
    plt.savefig(output / 'histograms(w_mean).pdf', bbox_inches='tight')
ax.set_xlabel('Mean Interevent Time')
print(errors_all)    



######## AVERAGE MEAN INTEREVENT TIME #######
average_return_time=[]


average = errors_all.loc[:, 'mean IT'].mean()
plus_average = errors_all.loc[:, '+2σ'].mean()
minus_average = errors_all.loc[:, '-2σ'].mean()
average_return_time.append([average,plus_average,minus_average])
#print(average_return_time)
    
    
# If there are fewer than nine subplots, turn off unused axes

for ax in axs[len(mean_interevent_times_lists):]:
    ax.axis('off')  # Turn off unused axes

# Adjust layout and show the plots
plt.tight_layout() 

#SAVE IN EXCEL FORMAT
df1 = pd.DataFrame(errors_all)


plt.figure(figsize=(10, 4))

# Plot boxplots for mean interevent times with compressed width and without outliers

a=plt.boxplot(
    mean_interevent_times_lists,'red', 'deepskyblue',
    whis=[5, 95],vert=False,
    labels=[f'{i}-{i + 1}' for i in range(len(mean_interevent_times_lists))],
    showfliers=False,  # Don't show outliers
    widths = 0.2  # Compress box width
)
for i in a['boxes']:
    # change outline color
    i.set(color='deepskyblue', linewidth=0.7)
#red

# Set axis labels and title
plt.xlabel('Mean Interevent Time')
plt.ylabel('Interval Index')
#plt.xlim(200, 1800)
plt.xlim(200, 1800)
plt.title('Mean Interevent Times (Whiskers at 5th and 95th Percentiles)')
plt.grid(axis='x', color='0.95')


############
#EXPORT DATA
############

output = BASE_DIR / "Outputs"


df1.to_excel(output / '2_sigma_mean_w.xlsx', index=False)
plt.savefig(output / "boxplot(2σ_mean_w).pdf", bbox_inches="tight")

# Show the plot
plt.show()



