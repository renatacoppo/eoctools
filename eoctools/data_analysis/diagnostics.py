import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from tas import TASDiagnostics
from pr import PRDiagnostics
from toa import TOADiagnostics
from sst import SSTDiagnostics
from sos import SOSDiagnostics
from deepmip import DeepMIPDiagnostics
from utils import load_last_years, load_simulation, count_warm_fix_activations


class Diagnostics():
     """
     Central container for loading climate model data and running diagnostics from multiple experiments.

     This class reads the experiment configuration file, loads the atmospheric, ocean, and overturning ciruclation
     datasets, and initialized the iindividual diagnostics clases.

     Parameters
     config_file : str of pathlib.Path
          Path to the YAML configuration file containing experiment definition file locations, CO2 concentrations, 
          and plotting settings.

     Attributes:
     base_path : pathlib.Path
          Base directory containing the model experiments.

     nyears : int
          Number of final simulation years used for equilibrium diagnostics.

     reference : str
          Name of the reference experiment.

     experiments : dict
          Dictionary containing metadata and file paths for all experiments.

     atm : dict[str, xarray.Dataset]
          Atmospheric datasets containing the final "nyears" of each simulation.

     atm_full : dict[str, xarra.Dataset]
          Atmospheric datasets containing the full simulation after the spin-up period.

     oce : dict[str, xarray.Dataset]
          Ocean datasets containing the final "nyears" of each simulation. 

     moc : dict[str, xarray.Dataset]
          Overturning ciruclation datasets containing the final "nyears" of each simulation.
     
     """
     def __init__(self, config_file):
            """
            Initialize the diagnostics workflow.

            The configuration file is read first, followed y loading all available model datasets.
            Individual diagnostics classes are then initialized and provided with the appropriate 
            datasets and experiment metadata.
            """
            #Read configuration
            #------------------
            with open(config_file, "r") as f:
                  config = yaml.safe_load(f)
            
            # Base directory containing all experiment folders
            self.base_path = Path(config["base_path"])

            # Number of initial simulation years to skip
            self.skip_years = config.get("skip_years", 0)
            
            # Number of final years used for equilibrium diagnostics
            self.nyears = config["nyears"]

            # Reference experiment
            self.reference = config["reference"]

            # Experiment definitions
            self.experiments = config["experiments"]

            # Warm-fix configuration
            # Construct the full path to the warm-fix log for each experiment
            # The YAML stores the log rellative to the experiment directory
            self.warm_fix_logs = {
                  exp: (
                        self.base_path
                        / info["directory"]
                        / info["warm_fix"]
                  )
                  for exp, info in self.experiments.items()
                  if info.get("warm_fix") is not None
            }
            # Count warm-fix activations for each experiment
            self.warm_fix_activations = {
                  exp: count_warm_fix_activations(log_file)
                  for exp, log_file in self.warm_fix_logs.items()
            }

            # Configuration for DeepMIP comparison datasets
            self.deepmip_config = config["deepmip"]

            # Relative directory used to store plots within each experiment folder
            self.plot_directory = config["plot_directory"]

            # Directory used for plots comparing this model with DeepMIP models or comparing 
            # multiple experiments from the same model
            self.comparison_plot_dir = (self.base_path / config["comparison_plot_directory"])

            #Data containers 
            # ---------------     
            self.atm = {}
            self.atm_full = {}
            self.oce = {}
            self.moc = {}

            #Load model data
            #----------------
            self._load_data()

            # Consistent colors for all experiments across diagnostics
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            self.colors = {
                  exp: color_cycle[i % len(color_cycle)]
                  for i, exp in enumerate(self.atm_full.keys())
            }

            #Initialize variable-specific diagnostics 
            #----------------------------------------
            # Surface air temperature diagnostics     
            self.tas = TASDiagnostics(
                 atm=self.atm,
                 reference=self.reference,
                 co2_levels=self.co2_levels,
                 plot_dirs=self.plot_dirs,
            )

            # Precipitation diagnostics
            self.pr = PRDiagnostics(
                 atm=self.atm,
                 reference=self.reference,
                 co2_levels_log2=self.co2_levels_log2,
                 plot_dirs=self.plot_dirs,
            )

            # Top-of-atmosphere radiation diagnostics
            # The full atmospheric simulations are used here because diagnostics
            # such as Gregory plots and time-series plots require the temporal evolution of the simulation.
            # Initial spin-up years can be removed indide TOADiagnostics using skip_years.
            self.toa = TOADiagnostics(
                 atm=self.atm_full,
                 reference=self.reference,
                 plot_dirs=self.plot_dirs,
                 comparison_plot_dir=self.comparison_plot_dir,
                 experiment_labels=self.experiment_labels,
                 colors=self.colors,
                 skip_years=self.skip_years,
                 warm_fix_activations=self.warm_fix_activations,
            )

            # Sea surface temperature diagnostics
            self.tos = SSTDiagnostics(
                 oce=self.oce,
                 reference=self.reference,
                 co2_levels=self.co2_levels,
                 plot_dirs=self.plot_dirs,
            )

            # Sea surface salinity diagnostics
            self.sos = SOSDiagnostics(
                 oce=self.oce,
                 reference=self.reference,
                 plot_dirs=self.plot_dirs,
            )
            
            # DeepMIP model comparison diagnostics
            self.deepmip = DeepMIPDiagnostics(
                 base_path=self.base_path,
                 reference=self.reference,
                 deepmip_models=self.deepmip_config,
                 plot_dir=self.comparison_plot_dir,
                 experiment_dirs=self.experiment_dirs
            )

            #self.zos = {}
            #self.so = {}
            #self.thetao = {}
            #self.moc = {}


                  
                  
     # Data loading  
     # -------------      
     def _load_data(self):
             """"
             Load all datasets defined in the experiment configuration.
             For each experiment, the configuration may specify atmospheric, ocean,
             and/or meridional overturning circulation datasets.

             Atmospheric data are loaded twice:

             - The final "nyears" are stored in "self.atm" for equilibium diagnostics.
             - The simulaltion after the initial 10-year spin-up is store in "self.atm_full"
             for time-dependent diagnostics.

             Ocean and MOC datasets are loaded only for the final "nyears".

             An optional "end_year" can be specified for an experiment to exclude incomplete or 
             unwanted years at the end of a simulation.
             """
             # Loop through all experiments defined in the configuration file
             for exp, info in self.experiments.items():
                  directory = self.base_path / info["directory"]

                  #Optional explicit end year
                  end_year = info.get("end_year", None)

                # Atmospheric data
                  if "atm" in info:
                       
                       file_path=directory / info["atm"]

                       # Load only the last N years: used by diagnostics representing the equilibrated state
                       self.atm[exp] = load_last_years(directory / info["atm"], nyears=self.nyears, end_year=end_year)
                       
                       # Load the complete simulation after removing the initial 10-year spin-up period.
                       # It can be used for diagnostics that depend on the temporal evolution of the simulation (e.g., TOA/Gregory plots)
                       self.atm_full[exp] = load_simulation(
                            file_path,
                            end_year=end_year
                       )
                
                # Ocean data
                  if "oce" in info:
                       
                       # Load only the last N years: used by diagnostics representing the equilibrated state
                       self.oce[exp] = load_last_years(directory / info["oce"], nyears=self.nyears, end_year=end_year)
                
                # MOC data
                  if "moc" in info:
                       
                       # Load only the last N years: used by diagnostics representing the equilibrated state
                       self.moc[exp] = load_last_years(directory / info["moc"], nyears=self.nyears, end_year=end_year)

     # Experiment metadata
     #--------------------
     @property
     def co2_levels(self):
            """
            Return the CO2 concentration multiplier for each experiment.
            The values are read directly from the experiment configuration.
            """
            return {
                 exp: info["co2"]
                 for exp, info in self.experiments.items()
                 } 
        
     @property
     def co2_levels_log2(self): 
            """
            Return log2-transformed CO2 concentration levels.
            This representation is useful for diagnostics that assume an approximately logarithmic
            relationship between climate response and atmospheric CO2 concentration.
            """
            return {
               exp: np.log2(co2)
               for exp, co2 in self.co2_levels.items()
            }
        
     @property
     def experiment_names(self):
            """
            Return the names of all configured experiments.
            e.g. : pi, x1, x1.1, x3
            """
            return list(self.experiments.keys())
     
     @property
     def experiment_labels(self):
            """
            Map internal experiment names to their original simulation directory names.

            These labels are intended for figure legends and other user-facing output,
            while internal experiment keys are retained for accessing diagnostics.
            e.g. : pix1, oix1, pex1, new3
            """
            return {
               exp: info["directory"]
               for exp, info in self.experiments.items()
            }
        
     @property
     def experiment_dirs(self):
             """
             Return the relative directory name for each experiment.
             These directories are defined in the experiment configuration and can be used when 
             constructing paths for comparison diagnostics.
             """
             return [
               info["directory"]
               for info in self.experiments.values()
          ]

     @property
     def perturbation_experiments(self):
            """
            Return all experients except the reference experiment.
            This is useful when calculating anomalies or climate responses relative to the reference state.
            """

            return [
                exp
                for exp in self.experiments
                if exp != self.reference
            ]
        
     @property
     def plot_dirs(self):
            """
            Construct the plot output directory for each experiment.
            Each experiment has its own plot directory located inside its experiment folder.
            """
            return {
               exp: self.base_path / info["directory"] / self.plot_directory
               for exp, info in self.experiments.items()
         }
        
        


                
                


