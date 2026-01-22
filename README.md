PM2.5 Air Quality Forecasting for Kirulapone (Colombo)
Predicting hourly PM2.5 pollution levels using time‑series deep learning models (LSTM and N‑BEATS) with key evaluation metrics. This project was completed as the final research project for the Higher National Diploma in Data Science (HNDDS) at NIBM.

## Dataset

- **Source:** Local air quality monitoring station, Kirulapone, Colombo  
- **Time period:** Hourly readings from **9th January 2022 to 5th December 2025**  
- **Number of records:** 24,864  
- **Features:** 
  - **Target variable:** `pm2_5` (fine particulate matter)  
  - **Other environmental variables:** `pm10`, `carbon_monoxide`, `carbon_dioxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`, `uv_index_clear_sky`, `ammonia`, `methane`  
- **Data quality:** No missing values in the `pm2_5` column
- 

## Project Structure


## Features

- Data Preprocessing: clipping, log transformation, scaling  
- Sequence Creation: prepare time-series data for model input  
- Deep Learning Models: LSTM and N-BEATS  
- Evaluation Metrics: RMSE, MAE, R², MAPE  
- Academic research project: part of **HNDDS final research at NIBM**

## Model Results - PM2.5 Forecasting

The models were evaluated on standard metrics to predict hourly PM2.5 levels in Kirulapone.

| Metric | LSTM | N-BEATS |
|--------|------|---------|
| MSE    | 4.5972 | 2.7522 |
| RMSE   | 2.1441 | 1.6590 |
| MAE    | 1.5150 | 1.1541 |
| R²     | 0.9056 | 0.9435 |
| MAPE (%) | 10.6882 | 8.4915 |

**🏆 Winner:** N-BEATS (best overall performance on all metrics)

## Visualizations - LSTM Model/N-BEATS Model Training vs Validation Curve

<img width="720" height="407" alt="Picture2" src="https://github.com/user-attachments/assets/f4620e95-ea76-44e7-9836-73772d7e82da" />
<img width="672" height="380" alt="Picture1" src="https://github.com/user-attachments/assets/c06bc621-083d-44f4-9c6d-89e34c42627c" />

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

This is **everything you need** to push your repo to GitHub:  

- `src/` with clean scripts  
- `notebooks/` for workflow  
- `data/` folder for your CSVs  
- `requirements.txt` for dependencies  
- `.gitignore` to keep repo clean  
- README highlighting **NIBM HNDDS final research**  

---

If you want, I can **also draft a ready-to-copy `LICENSE` (dual academic/commercial) for this repo** so you can just add it and push.  

Do you want me to do that?
