# Human Activity Recognition Using Hidden Markov Models (HMM)

**Formative 2 — HMM Project**

This repository implements a **Hidden Markov Model from scratch using NumPy** to infer hidden human activity states (standing, walking, jumping, still) from smartphone accelerometer and gyroscope sensor data.

## Repository Structure

```
Formative 2 HMM/
├── data/
│   ├── train/          # Labelled training CSV files (real recordings)
│   │   ├── standing/
│   │   ├── walking/
│   │   ├── jumping/
│   │   └── still/
│   └── test/           # Unseen test CSV files (real recordings)
│       ├── standing/
│       ├── walking/
│       ├── jumping/
│       └── still/
├── src/
│   ├── convert_real_data.py 
│   ├── features.py         
│   ├── hmm.py              
│   ├── visualization.py    
│   ├── run_pipeline.py     
│  
├── notebooks/
│   └── HMM_Activity_Recognition.ipynb   # Main analysis notebook
├── figures/             
├── report/
│   └── report_template.md  # Report template (4–5 pages)
└── videos/              
```

## Requirements

```bash
pip install numpy matplotlib seaborn pandas jupyter
```

## Quick Start

### 1. Prepare the dataset

If you have your own sensor-logger session folders in `data/` (e.g. `Walking_1-.../`, `Standing_1-.../`), run the conversion script to reorganize them into `data/train/<activity>/` and `data/test/<activity>/`:

```bash
cd src
python convert_real_data.py
```

This reads each session's `TotalAcceleration.csv`, auto-detects the sampling rate, and writes pipeline-ready CSVs. By default, the first 3 sessions per activity go to training and the last 2 to testing.

*(The `data_generator.py` script generates synthetic data and is kept for reference only; it is not run when using real recordings.)*

### 2. Run the full pipeline
```bash
cd src
python run_pipeline.py
```
This performs:
- Feature extraction (SMA, mean, variance, dominant frequency, spectral energy)
- Z-score normalization
- Observation discretisation (20 symbols)
- Baum-Welch HMM training (with log-likelihood convergence check)
- Viterbi decoding on unseen test data
- Evaluation metrics (sensitivity, specificity, accuracy)
- All visualizations saved to `figures/`

### 3. Open the notebook
```bash
jupyter notebook notebooks/HMM_Activity_Recognition.ipynb
```

## Key Implementation Details

| Component | Details |
|-----------|---------|
| Hidden states | 4 (standing, walking, jumping, still) |
| Observations | 20 discretised symbols from 5 features |
| Features | SMA, mean accel, var accel (time); dominant freq, spectral energy (FFT) |
| Normalization | Z-score (fit on train, applied to test) |
| Training algorithm | Baum-Welch EM, convergence at Δlog-likelihood < 1e-4 |
| Decoding algorithm | Viterbi |
| Window size | 2 seconds (~125 samples at ~62.5 Hz) |
| Data | Real sensor-logger recordings (~8–11 s per file), ~62.5 Hz |

