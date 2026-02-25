# HPC usage by LL

This is Léonard Lebrun's personal space for Unige's Baobab HPC facility usage.

## Launch interactive session
To access a compute node interactively, source .bashrc if needed then run command :
`launch_interactive`

This will log you in a cpu node. Maybe later we shall add an option for gpu nodes.

## Launch jupyter server

To launch a jupyter server **from the compute node**, execute the `start_jupyter.sh` script.
You can then run notebooks under `~/notebooks` from e.g. vscode by choosing the appropriate kernel.

Ctrl + I in terminal to open copilot

## Data

### Files
This codebase was developed to work with the Dampe data contained in ```DmlNtup_Data_9years_FullPhotonData_AllSky.root```. Download it in the ```data``` folder.

About MC data : no need to download, paths to the shared drive are in the scripts.

### Filtering
In order to parallelize the processing of heavy MC data, I chose the (not very optimized) option
to split the original root files into numerous smaller files organised per cos(theta) bins.

The script ```scripts/split_and_scale_thetaBin.py``` does this splitting, and adds weights to every
entry ; the MC files show a non-uniform spectrum in log(E), which is required for our IRFs computation.
When weights are accounted for, the spectrum is uniform **in every cos(theta) bin**. You can achieve
this splitting and scaling of all bins in parallel using the ```split_and_scale_all_bins.sh``` slurm
batch script, with ```sbatch scripts/split_and_scale_all_bins.sh``` . You can indicate an exponent to the scaling : this is the exponent of the power law followed in log(E) by the event distribution. Recommended value is 0 for uniformity in log(E), useful for IRFs computation.
/!\ This operation has already been run and data saved under ```/dpnc/beegfs/dampe/tryf/ForKimberly/```.

Once data is splitted per theta bin and weighted, it can be filtered using Jenni's selection algorithm.
To achieve this, you can run ```sbatch scripts/filter_all_bins.sh```, that runs the filtering in parallel for 
every bin using the cript ```scripts/filter_thetaBin.py```.
/!\ This operation has already been run and data saved under ```/dpnc/beegfs/dampe/tryf/ForKimberly/```.

### IRFs
With the MC data in hand, you can compute DAMPE's IRFs. 
- Effective Area : ```sbatch scripts/effArea.sh```
    - runs ```scripts/effArea.py```, works fairly well
- PSF : ```scripts/PSF/PSF.sh```
    - runs ```scripts/PSF/PSF_two_stages.py```, works OK : the PSF is fitted through a two-step fitting process in eevry cos(theta) bins. See Improvements section for more thoughts.
- Energy Distribution Function : was not useful for pulsars study, but ```eDist.py``` exists and works fairly well.

### Observing time
You can compute the heatmap of observing time of a sky position in function of line-of-sight coordinates
in the detector frame, cos(theta) and phi. This is achieved with ```sbatch scripts/obsTime.sh``` or ```sbatch scripts/obsTimeBounded.sh``` that allows to run the same map but for a portion of the total mission duration. Indeed, the mission is still running and you may want to tweak your map to fit the 
acquisition period of the considered flight data.
/!\ These scripts exclude time of observation spent in the South Anomaly region. Notice the way this region is excluded : I used a polyngone found in the DampeSt directory, but with no reference justifying it. While the polygon fits fairly well the excluded region 

### Exposure Maps
The exposure maps were computed per Energy bin with the intention to match the binning in use by Fermi in their Galactic Interstellar Emission model (GIE). This might not be the most useful use case and can be changed by tweaking the energy bins ; note that in this case you must be in possession of appropriate effArea histogram (follow appropriate binning scheme). A more polyvalent script was thus created along : ```scripts/exposure/exposureMap_year.sh``` where you can specify the energy at which you ant your map. The underlying python script ```scripts/exposure/exposureMap.py``` will load the effArea histogram and select the bin edges in which the indicated energy falls into.

In order to create these exposure maps, ```sbatch scripts/exposure/exposureMap_year.sh``` launches up to 366 jobs in parallel, one for each day of the considered year. All days are not necessarily present in the housekeeping data due to e.g. glitches or maintenance, so that the script needs a list of dates to be available prior to running. To create it, use ```scripts/find_all_dates.sh```.

Once per-year per-energy-bin exposure maps are created, you can sum them using ```sbatch scripts/exposure/sum_exposureMap.sh``` and obtain several-year per-energy-bin maps. The script does **not** sum over energy bins.

You can also create exposure maps for a given target in the sky : same process ,use e.g. ```sbatch scripts/exposure/exposureMap_target_year_array.sh```

## Pulsars
To compute light curves from pulsars, we use a csv conversion of the dataset. To generate it use the ```dataSaver.ipynb``` notebook.

We also need ephemerides files; the .par files inside ```data/par/``` describe the orbit of pulsars as computed with PTAs; if you use these, do not forget to acknowledge Paul Ray and Matthew Kerr for the private communication of the ephemerides.

The package used for light curve computation is a fork of PINT available at ```https://github.com/Lgtlebrun/PINT_dampe``` . Since the cern environment in use in LCP group has a non-writeable site_packages directory, you will have to create a new venv or conda environment. This is where having the data saved under .csv comes handy, because it is not necessary to have ROOT installed to read the data.

Once you have cloned the fork and installed it in your environment via e.g. ```pip install .``` inside the root directory of the repo, you can run the ```lightCurve.ipynb``` notebook with this environment. Make sure you have installed all the required packages (pandas...).


## Improvements
I detected a few caveats in the method implemented during this master's thesis.

- Storage : the splitting of MC data into per-bin files is not storage-efficient. I believe this ca nbe greatly improved by either making use of Root in the entire pipeline, or by making a better use of dataframes
- Maps : While the [cos(theta), phi] bins all represent an equal solid angle, the plots using 2D numpy arrays of [theta, phi] bins don't; a better approach would be to use Healpix all the way. Introducing some Healpy techniques would also probably allow a big refactor of the code, especially in the exposure maps computation, that would make it more efficient and reliable.
- PSF : the PSF obtained does not match the one from Duan et al. I believe this might either come from :
    - Incomplete MC dataset : weighting does not prevent from bias in the data in the case of PSF
    - Selection algorithm non-uniformity in line-of-sight directions
    - Bad usage of angular bins : healpy could help here
- Flux computation : I wasn't able to compute flux from pulsars, modt probably due to skill issue on my side