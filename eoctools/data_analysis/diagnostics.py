import numpy as np
import xarray as xr
import yaml
from pathlib import Path
from tas import TASDiagnostics
from pr import PRDiagnostics
from toa import TOADiagnostics
from sst import SSTDiagnostics
from sos import SOSDiagnostics
from deepmip import DeepMIPDiagnostics
from utils import load_last_years, load_simulation


class Diagnostics():
        """
    Container for climate diagnostics from multiple experiments.
    """
        def __init__(self, config_file):
            
            with open(config_file, "r") as f:
                  config = yaml.safe_load(f)
            self.base_path = Path(config["base_path"])
            self.nyears = config["nyears"]
            self.reference = config["reference"]
            self.experiments = config["experiments"]
            self.deepmip_config = config["deepmip"]
            self.plot_directory = config["plot_directory"]
            self.comparison_plot_dir = (self.base_path / config["comparison_plot_directory"])

            #Data containers      
            self.atm = {}
            self.atm_full = {}
            self.oce = {}
            self.moc = {}

            #Load data
            self._load_data()

            #Variable diagnostics      
            self.tas = TASDiagnostics(
                 atm=self.atm,
                 reference=self.reference,
                 co2_levels=self.co2_levels,
                 plot_dirs=self.plot_dirs,
            )

            self.pr = PRDiagnostics(
                 atm=self.atm,
                 reference=self.reference,
                 co2_levels_log2=self.co2_levels_log2,
                 plot_dirs=self.plot_dirs,
            )

            self.toa = TOADiagnostics(
                 atm=self.atm_full,
                 reference=self.reference,
                 plot_dirs=self.plot_dirs,
                 comparison_plot_dir=self.comparison_plot_dir
            )

            self.tos = SSTDiagnostics(
                 oce=self.oce,
                 reference=self.reference,
                 co2_levels=self.co2_levels,
                 plot_dirs=self.plot_dirs,
            )

            self.sos = SOSDiagnostics(
                 oce=self.oce,
                 reference=self.reference,
                 plot_dirs=self.plot_dirs,
            )
            
            self.deepmip = DeepMIPDiagnostics(
                 base_path=self.base_path,
                 reference=self.reference,
                 deepmip_models=self.deepmip_config,
                 plot_dir=self.base_path / self.comparison_plot_dir,
                 experiment_dirs=self.experiment_dirs
            )

            #self.tos = {}
            #self.sos = {}
            #self.zos = {}
            #self.so = {}
            #self.thetao = {}
            #self.moc = {}
                  
                  
        # Data loading        
        def _load_data(self):
             for exp, info in self.experiments.items():
                  directory = self.base_path / info["directory"]

                  #Optional explicit end year
                  end_year = info.get("end_year", None)

                # Atmosphere
                  if "atm" in info:
                       
                       file_path=directory / info["atm"]
                       #Last N years: used by the normal diagnostics
                       self.atm[exp] = load_last_years(directory / info["atm"], nyears=self.nyears, end_year=end_year)
                       
                       #Complete simulation: can be used by TOA/Gregory
                       self.atm_full[exp] = load_simulation(
                            file_path,
                            skip_years=10,
                            end_year=end_year
                       )
                
                # Ocean
                  if "oce" in info:
                       self.oce[exp] = load_last_years(directory / info["oce"], nyears=self.nyears, end_year=end_year)
                # MOC
                  if "moc" in info:
                       self.moc[exp] = load_last_years(directory / info["moc"], nyears=self.nyears, end_year=end_year)

        # General properties
        @property
        def co2_levels(self):
            return {
                 exp: info["co2"]
                 for exp, info in self.experiments.items()
                 } 
        
        @property
        def co2_levels_log2(self): 
            return {
               k: np.log2(v)
               for k, v in self.co2_levels.items()
            }
        
        @property
        def experiment_names(self):

            return list(self.experiments.keys())
        
        @property
        def experiment_dirs(self):
             return [
               info["directory"]
               for info in self.experiments.values()
          ]

        @property
        def perturbation_experiments(self):

            return [
                exp
                for exp in self.experiments
                if exp != self.reference
            ]
        
        @property
        def plot_dirs(self):
            return {
               exp: self.base_path / info["directory"] / self.plot_directory
               for exp, info in self.experiments.items()
         }
        
        


                
                


