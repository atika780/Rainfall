
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

df = pd.read_csv('sample_rainfall.csv')
data = df['Rajshahi_mm'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(data, time_step=3):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

pred_input = scaled_data[-3:].reshape(1, 3, 1)
predictions = []
for _ in range(15):
    pred = model.predict(pred_input)[0]
    predictions.append(pred)
    pred_input = np.append(pred_input[:,1:,:], [[pred]], axis=1)
forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
future_years = np.arange(2025, 2040)
plt.plot(df['Year'], df['Rajshahi_mm'], label='Historical')
plt.plot(future_years, forecast, label='LSTM Forecast', linestyle='--')
plt.title('LSTM Forecast for Rajshahi Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_forecast.png')
plt.show()
