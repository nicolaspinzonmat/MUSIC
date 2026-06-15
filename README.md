# MUSIC: MUlti Site Integrated Chronologies


MUSIC is a probabilistic framework for the correlation assessment and combination of multisite paleoseismic data. The methodology operates on event-age probability density functions (PDFs) to evaluate event-age correlations, combine correlated event ages into unified chronologies, and estimate recurrence intervals through Monte Carlo simulations.

MUSIC accompanies the manuscript:

> Pinzón and Klinger. (2026), *A Probabilistic Framework for Correlation Assessment and Combination of Multisite Paleoseismic Data: Evidence for Accelerated Strain Release Along the Altyn Tagh Fault*.

## Repository contents

### Notebooks

* **correlation.ipynb**
  Computes Bhattacharyya coefficients and evaluates temporal overlap among event-age PDFs.

* **combination.ipynb**
  Implements the weighted-mean method for combining correlated event-age PDFs into a unified earthquake chronology.

### Python scripts

* **MUSIC.py**
  Core functions used by the notebooks.

* **MC_mean_recurrence.py**
  Monte Carlo simulation of earthquake recurrence intervals.

* **memory_coefficient.py**
  Computes Memory Coefficients, Burstiness and Coefficient of Variations on 2sigma range 
  paleoearthquake chronologies.

### Data

The `Data/` directory contains example paleoseismic datasets from the central–eastern Altyn Tagh Fault, including event-age PDFs from each trench site and 2sigma range event-ages used to analyse recurrence of events.

### Output

The `Output/` directory stores generated figures and results.

## Requirements

The code was developed in Python 3 and requires:

* numpy
* pandas
* scipy
* matplotlib
* jupyter
* seaborn

Install dependencies using:

```bash
pip install numpy pandas scipy matplotlib seaborn jupyter
```

## Quick start

1. Clone the repository.

```bash
git clone https://github.com/nicolaspinzonmat/MUSIC.git
```

2. Navigate to the repository.

```bash
cd /path/to/MUSIC
```

3. Launch Jupyter Notebook.

```bash
jupyter lab
```

4. Open and run:

* `correlation.ipynb`
* `combination.ipynb`


## Repository structure

```text
MUSIC/
│
├── README.md
│
├── correlation.ipynb
├── combination.ipynb
├── MUSIC.py
├── MC_mean_recurrence.py
├── memory_coefficient.py
│
├── Data/
│   ├── ATF_events.txt
│   ├── T1_aksay.txt
│   ├── T2_aksay.txt
│   ├── T3_aksay.txt
│   ├── 2_sigma_mean_w.txt
│   ├── 2_sigma_mean.txt
│   └── ...
│
└── Output/
```



## Citation

If you use MUSIC in your research, please cite:

Pinzón and Klinger. (2026), *A Probabilistic Framework for Correlation Assessment and Combination of Multisite Paleoseismic Data: Evidence for Accelerated Strain Release Along the Altyn Tagh Fault*.


