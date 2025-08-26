# Heterogeneous MCOT
Code for the netgcoop2025 paper

The main results of the paper are shown in the `examples.ipynb` notebook. 

## Getting started

Install requirements :
```
pip install -r requirements.txt
```

You can either launch the `examples.ipynb` notebook or the `main.py` file of each experiments.
The experiments are structured as follows:

### Electric Vehicles (EVs)

We optimize the consumption of a fleet of 1000 EVs by using real data from the [Elaad OpenDataset](https://platform.elaad.io/analyses/index.php?url=ElaadNL_opendata.php).

In the folder `src/experiments/RealEVs`, launch the `main.py` file to run the Model Predictive Control(MPC) optimization with the EVs only. This is also the first result shown in `examples.ipynb`.

### Water Heaters (WHs)

We optimize the consumption of a fleet of 1000 WHs by using real data from the [SMACH Platform](https://hal.science/hal-03195500/document).

In the folder `src/experiments/WHs`, launch the `main.py` file to run the Model Predictive Control(MPC) optimization with the WHs only. This is also the second result shown in `examples.ipynb`.

### Combination of both WHs and EVs

In the folder `src/experiments/RealEVsandWHs`, launch the `main.py` file to run the Model Predictive Control(MPC) optimization with both Whs and EVs. This is also the last result shown in `examples.ipynb`.