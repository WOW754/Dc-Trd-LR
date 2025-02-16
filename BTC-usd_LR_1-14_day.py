import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
import io

# Get user input
crypto2 = 'BTC-USD'
test_size2 = 0.2
random_state2 = 42

# Define the day interval
days = [1, 3, 5, 7, 10, 14]

days_webhook = {1:'https://discord.com/api/webhooks/1340004132291088504/vOp8Gt1MQJxKyip90CkoUdzqc5gXFqecBTwPiatoRcrLU_TKpVTgdWHOEhNuLueHaJp1',
                3:'https://discord.com/api/webhooks/1340008729281691751/Rp3sWsqlT_2sbwhB-ONHiMVzDWuNTgcst3R5jCeHRU-BkFKFAvTZ_9FcOWUP0eWPolrx',
                5:'https://discord.com/api/webhooks/1340008850392088587/TFOh5N2d4z55naWKHuz1gRaS9ncTIMMpkmxHlzI_zYTLfMzHuFxsVpbyvBlo8sWTnGcz',
                7:'https://discord.com/api/webhooks/1340008966121324677/AUyBOySnCwRQIKKqhK7VWY3AVfVjlyDPzc53TNgT-lwL2RHE6sKUcIXjEQhFQSb4o4MN',
                10:'https://discord.com/api/webhooks/1340696984624955393/VZOv-lJ4hpmchw-HW3hK5Z9wS0cRxwQ-nG3_05jhKT_DYDs3w-uL0DDaPDHQR1n5mxUu',
                14:'https://discord.com/api/webhooks/1340697209464684625/MpxxOIp5RvbNFaijlo6NiOT_GSyY84si2DIO8VxlbK7CwNnz3OnqqZTpCJ2n_7GsUglm'}


intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']


for day in days:

    end_date = datetime.now()
    start_date = end_date - timedelta(days=day)
    
    for i in intervals:
        try:

            # Fetch historical price data
            data = yf.download(crypto2, start=start_date, end=end_date, interval=i)

            # Check if data is empty
            if data.empty:
                print("No data found for the given input.")
            else:
                # Prepare the data
                data['Date'] = pd.to_datetime(data.index)
                data['Date'] = data['Date'].map(mdates.date2num)  # Convert dates to numeric values

                # Create a dataframe with 'Date' and 'Close' columns
                df = data[['Date', 'Close']].copy()

                # Split the data into training and testing sets
                X = df[['Date']]
                y = df['Close']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size2, random_state=random_state2)

                # Train the linear regression model
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)

                # Make predictions
                y_pred_train = regressor.predict(X_train)
                y_pred_test = regressor.predict(X_test)

                # Plot the actual and predicted prices
                plt.figure(figsize=(12, 6))
                plt.scatter(X_train, y_train, color='blue', label='Actual (Train)')
                plt.scatter(X_test, y_test, color='green', label='Actual (Test)')
                plt.plot(X_train, y_pred_train, color='red', linestyle=':', label='Predicted (Train)')
                plt.plot(X_test, y_pred_test, color='orange', linestyle=':', label='Predicted (Test)')
                plt.xlabel('Date')
                plt.ylabel(f'{crypto2} Price (USD)')
                plt.title(f'{crypto2} Price Prediction ({day}-Day {i}-Interval)')
                plt.legend()
                plt.grid(True)

                # Adaptive label formatting based on the number of days
                if day <= 1:  # 1-Day Chart (Shorter Intervals)
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format as HH:MM

                elif day <= 7:  # 3-7 Days Chart (Daily Labels)
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format as 'Feb 14'

                else:  # 10+ Days Chart (Label Every 2 Days)
                    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # 'Feb 14'


                # Convert the plot to an in-memory file-like object
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)  # Move cursor to the start of the file-like object

                # Send the plot directly to the Discord webhook
                webhook_url = days_webhook[day]
                files = {'file': ('plot.png', buf, 'image/png')}
                response = requests.post(webhook_url, files=files)

                if response.status_code == 200:
                    print("Plot sent successfully to Discord!")
                else:
                    print("Failed to send the plot to Discord.")

        except Exception as e:
            print(f"Error fetching data from Yahoo Finance for {crypto2} ({day}-Day, {i}-Interval): {e}")

    # Send a simple message instead of the plot
    webhook_url = days_webhook[day]
    message_data = {"content": "**-+-+-+-+-+-COMPLETED-+-+-+-+-+-**"}  # Message to be sent

    response = requests.post(webhook_url, json=message_data)

    if response.status_code == 200:
        print("Message sent successfully to Discord!")
    else:
        print(f"Failed to send the message to Discord. HTTP Status: {response.status_code}, Response: {response.text}")
