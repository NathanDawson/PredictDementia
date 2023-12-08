# Exploring Dementia and Alzheimer's Disease Using Patient Records From Longitudinal MRI Data

This repository leverages longitudinal MRI data to classify healthy 
patients records from cases of dementia, including Alzheimer’s
disease. It aims to refine our understanding of the key features 
that are most predictive of dementia, facilitating early and 
accurate detection. The codebase encompasses the entire analysis 
pipeline, from exploratory analysis and data pre-processing to the 
deployment of a robust predictive model, highlighting the critical
attributes for dementia diagnosis.

### Script Execution

Each Python script in the repository is structured to operate 
independently, allowing for stand-alone execution. This provides 
flexibility in analysing specific aspects of the data in isolation.
However, to achieve a comprehensive analysis and the best results, 
it is recommended to run the scripts in their intended order.
For convenience, a `CSC3432_Report2.py` file is provided to 
execute all scripts in the correct order with a single command,
streamlining the process from data pre-processing to the final 
predictive modeling.

### Visualisations

To enhance the usability and interpretability of the analysis, all 
graphs generated by the scripts are automatically saved in the
`Graphs` folder. This centralised folder of visualisations 
allows users to gain insights without the need to navigate through 
the codebase. Each graph is named and organised corresponding to 
the script that generated it.


### Additional Information

#### Handling Matplotlib Errors

An issue has been observed with the `matplotlib` package that may interrupt script execution. 
Despite attempts to diagnose and address this problem, it has proven difficult due to its 
non-reproducible nature. If any unexpected errors with `matplotlib` are encountered, it is 
recommended to uninstall and then reinstall the package, as this has been found to resolve
the issue:

```bash
pip uninstall matplotlib
pip install matplotlib
