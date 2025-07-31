import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('sample_rainfall.csv')
df = df[['Year', 'Rajshahi_mm']].rename(columns={'Year': 'ds', 'Rajshahi_mm': 'y'})
df['ds'] = pd.to_datetime(df['ds'], format='%Y')
model = Prophet(yearly_seasonality=True)
model.fit(df)
future = model.make_future_dataframe(periods=15, freq='Y')
forecast = model.predict(future)
model.plot(forecast)
plt.title('Prophet Forecast for Rajshahi Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.tight_layout()
plt.savefig('prophet_forecast.png')
plt.show()
