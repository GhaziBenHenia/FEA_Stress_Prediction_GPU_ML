# FEA_Stress_Prediction_GPU_ML

GPU-accelerated machine learning surrogate model for predicting Von Mises stress fields from finite element analysis (FEA) data.  
Includes scripts for data preprocessing, validation, visualization, and a Jupyter notebook for model training and evaluation.

## Project Structure
```
project_root/
├── data/
│   ├── process_data.py               # Convert raw .vtu/.pvtu/.pvd → .npz
│   ├── validate_processed_data.py    # Integrity checks on processed data
│
├── notebooks/
│   ├── data_visualisation.ipynb      # Visualize raw FEA data
│   └── ml_surrogate_fea_stress.ipynb # Model training & evaluation
```

## Dataset
This project uses the dataset:  

Kerfriden, Pierre. (2022). *Deep learning dataset: finite element stress analysis of biaxial specimen with random elastic properties - 1000 samples* (V0.9) [Data set]. Zenodo.  
[https://doi.org/10.5281/zenodo.6819576](https://doi.org/10.5281/zenodo.6819576)

## Requirements
- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- PyTorch
- PyVista
- scikit-learn
