# Wind-Induced Changes to Surface Gravity Wave Shape in Shallow Water
The repository contains the source code and LaTeX manuscript
for the
"Wind-Induced Changes to Surface Gravity Wave Shape in Shallow Water"
paper by Thomas Zdyrski and Falk Feddersen,
[doi:10.1017/jfm.2021.15](https://doi.org/10.1017/jfm.2021.15).

## Layout
- `ShallowWaterBiphaseManuscript.tex` LaTeX source for the paper
- `PhysicalScales.tex` LaTeX source for brief discussion of the model's
  physical scales
- `Plots.tex` LaTeX document with plots for visualization purposes
- `Figures` Additional figures, including both figures used in the paper
and figures not used
- `Data/README.txt` Description of the simulation output data format
- `Code/intermediate_solver.py` Python script for numerical wave solver
- `Code/postprocessor.py` Python script for post-processing numerical
  results from `intermediate_solver.py`, such as calculating shape
  statistics
- `Code/plotter.py` Python script for plotting post-processed results
- `Code/strong_profile.py` Attempt to calculate analytic solution
  for wave evolution under different wind-forcing types
- `Code/RMS.txt.py` Log of root-mean-square difference
  between unforced wave profile at beginning and end of simulation
  to check convergence of numerical solver.
  Saved from runs of `plotter.py`
- `Code/useful_functions.py` Python script with utility functions
  like derivative calculations and significant-figure rounding
- `Code/pi_formatter.py` Python script for formatting angles in radians
  as fractions of pi
- `Code/data_csv.py` Python script for reading/writing to CSV files
- `Code/texplot.py` Python script plotting generating matplotlib PGF
  outputs with styling consistent to latex document

## License
The code located in the `Code` directory is licensed under the terms of the GPL v3.0 or later license.
The manuscript located in the `Reports` directory is licensed under the terms of the Creative Commons Attribution 4.0 International Public License.
