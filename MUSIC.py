import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import simps
from scipy.integrate import simpson
import pandas as pd
import seaborn as sns

#READ AND LOAD DATA

from pathlib import Path

BASE_DIR = Path(__file__).parent
output = BASE_DIR / "Outputs"

trench1 = BASE_DIR / "Data" / "T1_aksay.txt"
trench2 = BASE_DIR / "Data" / "mobaer.txt"


df1 = pd.read_csv(trench1, sep='\t', header=0).fillna(0)
df2 = pd.read_csv(trench2, sep='\t', header=0).fillna(0)

data1 = df1.columns.tolist()
data2 = df2.columns.tolist()

time = df1['time']
dt = 5 #time step resolution from Oxcal
#print(time)

df = pd.DataFrame(np.zeros((len(df2.columns)-1,len(df1.columns)-1)), # stores the overlap from the B coefficient 
                  index=[x for x in data2 if x!= "time"], 
                  columns = [y for y in data1 if y != "time"])

df_x = df.copy() # stores the overlap express only in terms of time based on the B coefficient
df4 = df.copy() # stores the overlap from the minimium values between PDFs

for i in df1.columns:
    if i == 'time':
        continue
    for k in df2.columns:
        if k == 'time':
            continue
        pdf_a = df1[i].values
        pdf_b = df2[k].values
        pdf_a /= np.sum(pdf_a)
        pdf_b /= np.sum(pdf_b)
        bc = simpson(np.sqrt(pdf_a * pdf_b),
                     np.arange(0,np.size(time)), dx=dt)
        
        overlap = np.sum(np.minimum(pdf_a, pdf_b))
        df4.loc[k, i] = overlap
        df.loc[k, i] = bc
        x = np.sqrt(pdf_a * pdf_b)
        vector=[] # este vector en cada iteracion se llena de valores y al terminar una iter y comenzar la otra queda vacia
        for j in x:
            if j > 1e-3:
                vector.append(j)
                dx = np.size(vector)* dt - dt
                df_x.loc[k, i] = dx
print(df_x)



#-------------------------------------------------
#PLOT MATRIX SHOWING OVERLAP + TOTAL BC VALUES
#-------------------------------------------------

#plt.plot(time, pdf1, "k:", label='PDF 1')
plt.show()
plt.figure()
sns.heatmap(df,annot=True, cmap="Oranges", cbar_kws={'label': 'B coefficient'})  
plt.savefig(output / 'T1_annanbaa.pdf', bbox_inches='tight')    
plt.show()


"""
ACA POR SI QUIEREN SABER PROBABILIDADES DEL PRODUCTO
b_c =  pdf1*pdf2   
#b_c2 =  np.sqrt(pdf3*pdf2) 

P_agreement = [index for index, value in enumerate(b_c) if value != 0]
min_idx = np.min(P_agreement)
max_idx = np.max(P_agreement) 



cdf = np.cumsum(b_c)
#cdf = np.cumsum(PDFs_3['w_mean'].values)
cdf /= cdf[-1]

lower_index = np.argmax(cdf >= 0.06)
upper_index = np.argmax(cdf >= 0.921)

# extract the calendar dates corresponding to these indexes
lower_bound = time[lower_index]
upper_bound = time[upper_index]

print("The probabilistic agreement between two site PDFs is:", lower_bound, upper_bound) 
"""

#-------------------------------------------------
#If you want to plot the BC area 
#-------------------------------------------------

#after running the previous loop..if you want to check the area of the B_coeficient between any particular PDF's
pdf1 = df2['An'].values 
pdf2 = df1['E5'].values
pdf3 = df1['E3'].values 

pdf1 /= np.sum(pdf1)
pdf2 /= np.sum(pdf2)
pdf3 /= np.sum(pdf3)


area_1 =  np.sqrt(pdf1*pdf2)  
area_2 =  np.sqrt(pdf1*pdf3) 
print("area:",np.sum(area_1))   


plt.plot(time, pdf1, "k--", label='An"')
plt.plot(time, pdf2,"k", label='E5')
plt.plot(time, pdf3, "k:", label='E3' )
plt.fill_between(time, 0, area_1, alpha=0.5, color='orange', label='overlap E8-E5"')
plt.fill_between(time, 0, area_2, alpha=0.5, color='slategrey', label='overlap E8-E4"')
plt.xlim(-770, 1500)
#plt.xlim(time[min_idx]-10, time[max_idx]+10)
plt.ylim(0, np.max(pdf3) + 0.0005)
plt.legend()
plt.savefig(output / 'An-E5-E3.pdf', bbox_inches='tight')
plt.show()
            



###########################################################
######### COMBINE PROBABILITY DENSITY FUNCTIONS ###########
###########################################################


######### here I call a file that have all the paleoevents documented in ATF #######

All_events = BASE_DIR / "Data" / "ATF_events.txt"

ATF = pd.read_csv(All_events, sep='\t').fillna(0)

####### Which event are you gonna process – Indicate the name of the event ######
event = "Event_9"

#sub_events = ['TH1', 'E6', 'Bn', 'Bs', 'CM4', 'BS2']
sub_events = ['E8', 'E4_T3', 'TH3']
time = ATF['time'].values



#create a first dataframe with normalized sub events 
# create dataframe with normalized PDFs
PDFs = pd.DataFrame()
count = {}

for ev in sub_events:
    pdf = ATF[ev].values.astype(float)
    pdf /= np.sum(pdf)                 # normalize PDF
    PDFs[ev] = pdf
    count[ev] = np.count_nonzero(pdf)  # number of non-zero bins


 
####  Call each group of overlaping PDF's that represent the event (previously defined) ####
results = {}

for candidate in sub_events:

    pdf_fixed = PDFs[candidate].values
    PDFs_temp = pd.DataFrame()

    for other in sub_events:

        pdf_moving = PDFs[other].values

        overlap = np.where(pdf_fixed * pdf_moving > 0, pdf_moving, 0)
        non_overlap = np.where(pdf_fixed * pdf_moving == 0, pdf_moving, 0)

        area = np.sum(overlap)
        area_non = np.sum(non_overlap)

        weight_value = max(area, area_non)

        PDFs_temp[other] = np.where(
            pdf_fixed * pdf_moving > 0,
            pdf_moving * weight_value,
            pdf_moving * (1 - weight_value)
        )

    # Mean rupture combination
    combined = PDFs_temp.mean(axis=1).values
    combined /= combined.sum()

    # Compute weighted std of combined PDF
    mean_comb = np.average(time, weights=combined)
    std_comb = np.sqrt(np.average((time - mean_comb)**2, weights=combined))

    results[candidate] = std_comb

master_event = min(results, key=results.get)
master = PDFs[master_event]

print("The master PDF that minimizes final variance is:", master_event)



# Create a DataFrame to include only secondary PDFs (non-modified yet)
PDFs_2 = pd.DataFrame()
for event_name in PDFs:
    if event_name != master_event:     #master_event
        PDFs_2[event_name] = PDFs[event_name]
        
#print(PDFs_2)   
          
########### WE WILL KEEP THE SHAPE OF THE MASTER PDF #################
### AND WILL WEIGHT THE OTHER(S) BASED ON THE INTERACTION REGION #####
### Support overlap -> The domain of the secondary PDF’s that 
###     parametrically overlaps with the master PDF. 


PDFs_3 = pd.DataFrame() #aca se almacenan las PDF moduladas y ponderadas
for a in PDFs_2:
    # Support overlap -> The domain of the secondary PDF’s that parametrically overlaps with the master PDF. 
    overlap = np.where((master * PDFs_2[a].values) > 0, PDFs_2[a].values, 0.0)
    non_overlap = np.where((master*PDFs_2[a].values)==0, PDFs_2[a].values, 0.0)
    area = np.sum(overlap)
    area_non = np.sum(non_overlap)
    #print(area)
    #print(area_non)
    weight_value = np.where(area > area_non, area, area_non)
    ################ Here starts we start to modulate the secondary(s) PDF ################
    ##### Weight the support overlap and reduce the weight of non-support overlap #########
    PDFs_3[a]= np.where(master * PDFs_2[a].values > 0, PDFs_2[a].values * weight_value, PDFs_2[a].values * (1 - weight_value))
    

    
#print(PDFs_3['E1_T2'].sum())
  
##### COMBINE THE PDFs by using the MEAN RUPTURE METHOD ######
# 1) add the master PDF to the dataframe of modulated PDFs
PDFs_2['master'] = master #original PDFs 
PDFs_3['master'] = master # Modulated PDFs + master

# 2) compute the combination 
PDFs_2['mean']   = PDFs_2.mean(axis=1) # mean rupture
PDFs_3['w_mean'] = PDFs_3.mean(axis=1) # weigthed mean rupture 
PDFs_3['w_mean'] /= np.sum(PDFs_3['w_mean'])
#print("suma_final", np.sum(PDFs_3['w_mean']) )
 
#### create a gaussian bell #####

def gaussian_filter(kernel_size, kernel_std):
    kernel = np.arange(-size//2+1, size//2+1)
    kernel = np.exp(-0.5*(kernel/std)**2)
    kernel /= np.sum(kernel)
    return kernel 

size = 9
std = 2

gaussian = gaussian_filter(size, std)

###### convolve PDF with kernel to smoth ######
smoth_PDF = np.convolve(PDFs_3['w_mean'], gaussian)
FPDF = smoth_PDF[4:-4]

######### OBTAIN THE WEIGHTED MEAN AND 95TH PERCENTILE OF A NON-NORMAL PDF ########
######### For the Weighted and modulated PDF #########

FPDF_mean = np.average(time, weights=PDFs_3['w_mean']) 


# So far the FPDF is modulated (shape) and weigthed
#if you want to normalize to ensure the area is 1, then:

#FPDF /= np.sum(FPDF)   
#print(np.sum(FPDF)) 

# Compute the cumulative distribution function CDF
cdf = np.cumsum(FPDF)
#cdf = np.cumsum(PDFs_3['w_mean'].values)
cdf /= cdf[-1]
#print(np.size(cdf))


# Find the indices corresponding to the lower and upper bounds of the 95.4% confidence interval

lower_index4 = np.argmax(cdf >= 0.025)
upper_index4 = np.argmax(cdf >= 0.977)
lower_index = np.argmax(cdf >= 0.159)
upper_index = np.argmax(cdf >= 0.841)

# extract the calendar dates corresponding to these indexes
lower_bound = time[lower_index]
upper_bound = time[upper_index]
lower_bound4 = time[lower_index4]
upper_bound4 = time[upper_index4]



######### For the usual rupture mean PDF #########

mean_rupture = PDFs_2['mean'].values
mean_r = np.average(time, weights = mean_rupture)


cdf_mean= np.cumsum(mean_rupture)
cdf_mean /= cdf_mean[-1]

lower_index5 = np.argmax(cdf_mean >= 0.025)
upper_index5 = np.argmax(cdf_mean >= 0.977)
lower_index2 = np.argmax(cdf_mean >= 0.159)
upper_index2 = np.argmax(cdf_mean >= 0.841)

lower_bound2 = time[lower_index2]
upper_bound2 = time[upper_index2]
lower_bound5 = time[lower_index5]
upper_bound5 = time[upper_index5]



print("2_SIGMA")    
print("Weighted mean rupture method")
print("the age of", event, " is between", lower_bound4, "and", upper_bound4)
print("the weighted mean age is:", FPDF_mean,"(+", -FPDF_mean + upper_bound4, "/-", -lower_bound4 + FPDF_mean,")")
print("mean rupture method")
print("the age of", event, " is between", lower_bound5, "and", upper_bound5)
print("the weighted mean age is:", mean_r,"(+", -mean_r + upper_bound5, "/-", -lower_bound5 + mean_r,")")
print("----------------------------------------------------------")


print("1_SIGMA")    
print("Weighted mean rupture method")
print("the age of", event, " is between", lower_bound, "and", upper_bound)
print("the weighted mean age is:", FPDF_mean,"(+", -FPDF_mean+upper_bound, "/-", -lower_bound+FPDF_mean,")")
print("mean rupture method")
print("the age of", event, " is between", lower_bound2, "and", upper_bound2)
print("the weighted mean age is:", mean_r,"(+", -mean_r + upper_bound2, "/-", -lower_bound2 + mean_r,")")
print("----------------------------------------------------------")
#check area 
#print('suma E8:', np.sum(PDFs_3['E8'].values))
#print('suma TH3:', np.sum(PDFs_3['TH3'].values))

################################################
######### PLOTEAR THE COMBINED PDF #############
################################################



#styles = ['k--', 'k:', 'k-.', 'k-', 'r-', 'g-.','grey--']
styles = ['k--', 'k:', 'k-.', 'k-', 'g-', 'r-.']
lower_index3 = np.argmax(cdf_mean >= 0.025)
upper_index3 = np.argmax(cdf_mean >= 0.977)
lower_bound3 = time[lower_index3]
upper_bound3 = time[upper_index3]

plt.figure(2, figsize=(9,6))
  
for i, col in enumerate(PDFs_3):
    if col == 'w_mean':
        continue
    else:
        plt.plot(time, PDFs_3[col].values, styles[i], label = col)
        
plt.plot(time, PDFs_3['w_mean'], 'k', label = 'Weighted_mean age ')
plt.fill_between(time, 0, PDFs_3['w_mean'], alpha=0.5, color='lightskyblue', label='modulated, weighted, smoothed')
xerr = [[(-lower_bound+FPDF_mean)], [(-FPDF_mean+upper_bound)]]
plt.errorbar(FPDF_mean, 0.016, xerr=xerr, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
xerr2 = [[(-lower_bound4+FPDF_mean)], [(-FPDF_mean+upper_bound4)]]
plt.errorbar(FPDF_mean, 0.0153, xerr=xerr2, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
plt.xlim(lower_bound3 - 300, upper_bound3 + 300)
#plt.xlim(-6000, -3600)
plt.ylim(0)
plt.xlabel('Canlendar Modeled Date')
plt.ylabel('Probability Density')
plt.title(event)
plt.legend()
#plt.savefig(output / f"{event}weighted.pdf", bbox_inches='tight')
#np.savetxt('event_4w.txt', PDFs_3['w_mean'], fmt='%.6f', delimiter='\t')

################################################
######## plot normal mean rupture method #######
################################################

plt.figure(3, figsize=(9,6))

for i, col in enumerate(PDFs_2):
    if col == 'mean':
        continue
    else:
        plt.plot(time, PDFs_2[col].values, styles[i], label = col)
        
if 'mean' in PDFs_2.columns:
    plt.plot(time, PDFs_2['mean'].values, 'k', label = 'Mean_age')
    
plt.fill_between(time, 0, PDFs_2['mean'].values, alpha=0.5, color='tomato', label='Mean rupture method')
xerr = [[(-lower_bound2+mean_r)], [(-mean_r+upper_bound2)]]
plt.errorbar(mean_r, 0.0163, xerr=xerr, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
xerr3 = [[(-lower_bound5+mean_r)], [(-mean_r+upper_bound5)]]
plt.errorbar(mean_r, 0.015, xerr=xerr3, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
plt.xlim(lower_bound3 - 300, upper_bound3 + 300)
plt.ylim(0)
plt.xlabel('Canlendar Modeled Date')
plt.ylabel('Probability Density')
plt.title(event)
plt.legend()
#plt.savefig(output / f"{event}mean.pdf", bbox_inches='tight')
#np.savetxt('event_2.txt', PDFs_2['mean'], fmt='%.6f', delimiter='\t')

################################################
######## plot normal mean rupture method #######
################################################
"""
plt.figure(4)#, figsize=(9,6))

for i, col in enumerate(PDFs_2):
    if col == 'mean' :
        continue
    elif col == 'master' :
        continue
    else:
        plt.plot(time, PDFs_2[col].values, styles[i], label = col)
        
if 'master' in PDFs_2.columns:
    plt.plot(time, PDFs_2['master'].values, 'k', label = 'Master PDF')
    
plt.plot(time, PDFs_2['mean'].values, 'red')    
plt.fill_between(time, 0, PDFs_2['mean'].values, alpha=0.5, color='tomato', label='Mean rupture method')
xerr = [[(-lower_bound2+mean_r)], [(-mean_r+upper_bound2)]]
plt.errorbar(mean_r, 0.016, xerr=xerr, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
xerr3 = [[(-lower_bound5+mean_r)], [(-mean_r+upper_bound5)]]
plt.errorbar(mean_r, 0.0153, xerr=xerr3, fmt='o', capsize=5, ecolor='gray', markerfacecolor='gray', markeredgecolor='gray', alpha=1)
#plt.xlim(lower_bound3 - 300, upper_bound3 + 300)
plt.xlim(-6000, -3600)
#plt.xlim(lower_bound3 - 300, upper_bound3 + 300)
plt.ylim(0)
plt.xlabel('Canlendar Modeled Date')
plt.ylabel('Probability Density')
plt.title(event)
plt.legend()
plt.savefig(output / f"{event}_mean2.pdf", bbox_inches='tight')
plt.show()    
    
"""
