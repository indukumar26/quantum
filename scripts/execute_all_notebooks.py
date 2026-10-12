"""
Pre-executes all Jupyter notebooks in the notebooks/ directory so they have rich rendered outputs.
"""
import os
import nbformat
from nbclient import NotebookClient

notebook_files = [
    "notebooks/01_biomedical_feature_extraction.ipynb",
    "notebooks/02_vqc_training_and_baselines.ipynb",
    "notebooks/03_processor_benchmarking_and_phase2_processor_c.ipynb",
    "notebooks/PS_Q1_Biomedical_Signal_Classifier_Full_Pipeline.ipynb"
]

for nb_path in notebook_files:
    print(f"Executing {nb_path}...")
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"  Successfully executed and saved: {nb_path}")

print("All notebooks pre-rendered with rich visual outputs!")
