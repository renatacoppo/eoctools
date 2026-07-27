# eocene_postprocessing

This repository provides a workflow to postprocess eocene simulations.

It consists of two main parts:
- remaping
- data_analysis

# remaping
This part takes the output from the simulations, selects desired variables, computes annual means, remaps the files if needed, and saves them in a new folder. It produces three main files:
atm_annual_mean_{first_year}-{last_year}.nc  
oce_annual_mean_{first_year}-{last_year}.nc
moc_annual_mean_{first_year}-{last_year}.nc

The main entry point is:

```bash
./remaping/run_remap.job
```

