# PM2.5 Forecasting – Kirulapone, Colombo

This project forecasts hourly PM2.5 air pollution levels in Kirulapone, Colombo using time‑series deep learning models (LSTM and N‑BEATS). Models are evaluated using RMSE, MAE, R², and MAPE. The goal is to provide a data‑driven forecasting solution for air quality monitoring as part of my Higher National Diploma in Data Science (HNDDS) research.


## Dataset

- **Source:** Local air quality monitoring station, Kirulapone, Colombo  
- **Time period:** Hourly readings from **9th January 2022 to 5th December 2025**  
- **Number of records:** 24,864  
- **Features:** 
  - **Target variable:** `pm2_5` (fine particulate matter)  
  - **Other variables:** `pm10`, `carbon_monoxide`, `carbon_dioxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`, `uv_index_clear_sky`, `ammonia`, `methane`  
- **Data quality:** No missing values in the `pm2_5` column


## Model Results - PM2.5 Forecasting

| Metric     | LSTM   | N‑BEATS |
|------------|--------|---------|
| MSE        | 4.5972 | 2.7522  |
| RMSE       | 2.1441 | 1.6590  |
| MAE        | 1.5150 | 1.1541  |
| R²         | 0.9056 | 0.9435  |
| MAPE (%)   | 10.6882 | 8.4915 |

**🏆 Winner:** N‑BEATS (best overall performance)


## Visualizations - LSTM Model/N-BEATS Model Training vs Validation Curve

<img width="642" height="377" alt="image" src="https://github.com/user-attachments/assets/eff9b3ba-b9ae-465c-a43e-36584bd887dd" />
<img width="640" height="377" alt="image" src="https://github.com/user-attachments/assets/30c63374-91ae-4346-b22e-ee611c3b40cd" />


## How to Run

1. Clone the repository  
   `git clone https://github.com/Kalingu/pm25-forecast-kirulapone.git`
2. Install dependencies  
   `pip install -r requirements.txt`
3. Open and run the notebook  
   `jupyter notebook notebooks/pm25_lstm_nbeats.ipynb`
4. View results and plots


## Tech Stack
- Python, Pandas, NumPy, Matplotlib
- PyTorch (or TensorFlow)
- Jupyter Notebook
- Git/GitHub


## License

This project is licensed under the MIT License.
