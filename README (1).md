
# Rainfall Forecasting for Adaptive Infrastructure Planning

This repository contains simplified Python scripts and sample data to demonstrate AI-based rainfall forecasting for Rajshahi, Bangladesh. It includes two modeling approaches:

-  Prophet: A time-series forecasting model developed by Facebook.
-  LSTM (Long Short-Term Memory): A deep learning model using TensorFlow/Keras.

##  Files Included

- `sample_rainfall.csv` — Synthetic annual rainfall data for Rajshahi and Ishwardi (1980–2024).
- `prophet_forecast.py` — Univariate rainfall forecast for Rajshahi using Prophet.
- `lstm_forecast.py` — LSTM model implementation for rainfall forecasting.
- `prophet_forecast.png` — Output plot of Prophet model prediction.
- `lstm_forecast.png` — Output plot of LSTM model prediction.

##  How to Run

1. Install required Python packages:
   ```bash
   pip install pandas matplotlib scikit-learn tensorflow prophet
   ```

2. Run Prophet forecast:
   ```bash
   python prophet_forecast.py
   ```

3. Run LSTM forecast:
   ```bash
   python lstm_forecast.py
   ```

Each script will generate a PNG plot of the forecast.

## Dataset Notes

- Data is synthetic and mimics rainfall trends for Rajshahi and Ishwardi.
- Original source: Bangladesh Meteorological Department (BMD) — simulated for academic use.

##  License

This code is distributed under the MIT License. You may use and modify it for research or academic purposes.

##  Contact

For any issues or academic collaboration, please contact: hasan.alif@researchmail.com
