# PM2.5 Forecasting – Colombo (Kirulapone)

This repository contains code and resources for **forecasting hourly PM2.5 concentrations** in Colombo, Kirulapone using **deep learning models**. The project explores **LSTM and N-BEATS** for time-series prediction and is part of the **final research project for the Higher National Diploma in Data Science (HNDDS) at NIBM**.

## Project Structure


## Features

- Data Preprocessing: clipping, log transformation, scaling  
- Sequence Creation: prepare time-series data for model input  
- Deep Learning Models: LSTM and N-BEATS  
- Evaluation Metrics: RMSE, MAE, R², MAPE  
- Academic research project: part of **HNDDS final research at NIBM**

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt

Run the notebook:

jupyter notebook notebooks/pm25_lstm_nbeats.ipynb

Trained models (optional):

LSTM: models/lstm_pm25_model.h5

N-BEATS: models/nbeats_pm25_model.pth

Note: Trained models are not included. Run the notebook to train locally.

Customize preprocessing or model parameters in src/ scripts.

License

This project is free for academic or research use.
For commercial use, contact the author (Kal) for licensing.

References

TensorFlow: https://www.tensorflow.org/

PyTorch: https://pytorch.org/

N-BEATS PyTorch: https://github.com/philipperemy/n-beats


---

✅ This is **everything you need** to push your repo to GitHub:  

- `src/` with clean scripts  
- `notebooks/` for workflow  
- `data/` folder for your CSVs  
- `requirements.txt` for dependencies  
- `.gitignore` to keep repo clean  
- README highlighting **NIBM HNDDS final research**  

---

If you want, I can **also draft a ready-to-copy `LICENSE` (dual academic/commercial) for this repo** so you can just add it and push.  

Do you want me to do that?
