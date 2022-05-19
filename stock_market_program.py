# Luca Shapiro final project - stock charts plus simple prediction
# uses yfinance for stock data
# uses pandas to create the dataframe to receive the finance data
# using plotly for plots
# uses streamlit for web interface
# uses datetime for date features
# uses prophet for stock prediction using time series forecasting with seasonal and holiday 

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import pylab
import matplotlib.pyplot as plt

import numpy as np
import yfinance as yf
import datetime as dt
import streamlit as st
import plotly.express as px
from datetime import date
from plotly import graph_objs as go
from pandas_datareader import data as pdr
from PIL import Image

image = Image.open('crystal-ball-magic-icon-hands.jpg')

end = dt.datetime.now()
start = end - dt.timedelta(days=5000)
#print (start, end)
# List of symbols for technical indicators
#INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'BASP', 'PIVOT_FIB', 'VORTEX']
#yf.pdr_override()

col_header1, col_header2 = st.columns(2)
header = st.container()
topofscreen = st.container()
bottomofscreen = st.container()


with col_header1:
    st.image(image, width = 150)
with col_header2:
    st.title("Luca's stock prediction app")
    
stock = st.text_input("Enter Stock Ticker:", value = "AMZN")
    
#stock=input("Enter a stock ticker symbol: ")

days = int(st.text_input("Enter number of days in the future for price prediction:", value = "365"))
#days_str =input("How many days into the future do you want to predict?")
print ("days = ", days)



train_size = 252*3                     # Use 3 years of data as train set
val_size = 252                         # Use 1 year of data as validation set
train_val_size = train_size + val_size # Size of train+validation set
i = train_val_size                     # Day to forecast
H = 21                                 # Forecast horizon

#close_data.reset_index(level=0, incplace=True)

# this procedure takes the symbol and makes the prediction - note that I graphed it inside the procedure
# in order to use the stucture with the yfinance data which is called inside it

#@st.cache(suppress_st_warning=True) # cache the function to improve the performance of the interactive graph
def predict(ticker,days):
    #print("Ticker = ", ticker)
    df = pdr.get_data_yahoo(stock, start, end)
    df.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)

    close_data = df.close
    yfin = yf.Ticker(ticker)
    hist = yfin.history(period="max")
    hist = hist[['Close']]
    
    hist.reset_index(level=0, inplace=True)
    hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
    #hist.info

    string_logo = '<img src=%s>' % yfin.info['logo_url']
    string_name = yfin.info['longName']
    string_summary = yfin.info['longBusinessSummary']
    current_price = yfin.info['currentPrice']
    percent_shorted = yfin.info['shortPercentOfFloat']
    open_price = yfin.info['open']
    days_volume = yfin.info['volume24Hr']
    market_cap = yfin.info['marketCap']
    fifty_two_week_high = yfin.info['fiftyTwoWeekHigh']
    fifty_two_week_low = yfin.info['fiftyTwoWeekLow']
    pe_ratio = yfin.info['forwardPE']
    change = current_price - open_price
    analyst_info = yfin.recommendations.tail(25)
    


    

    first_col, second_col = st.columns(2)
    
    with st.container():
        first_col.markdown(string_logo, unsafe_allow_html=True)
        first_col.header('**%s**' % string_name)
        #second_col.header(' %s*' % current_price_str)
        second_col.code("Price: %.2f         PE ratio: %.2f" % (current_price, pe_ratio))
        second_col.code("Day Change: %.2f" % change)
        second_col.code("52 Week High/low: %.2f / %.2f" % (fifty_two_week_high, fifty_two_week_low))
        #second_col.code("52 Week Low: %.2f" % fifty_two_week_low)
        #second_col.header("Days volume: %.2f" % days_volume)
        #second_col.code("Market Cap: %.2f" % market_cap)
        second_col.code("Percent shorted: %.2f" % (percent_shorted*100))

    with st.container():
        st.info(string_summary)
        
    
    m=Prophet(daily_seasonality=True)
    m.fit(hist)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    #print("Predicted Data")
    #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    #print(forecast[['ds', 'yhat']].tail())
    #print ("forcase index", forecast.index)

    #plot forecast in streamlit app


    st.subheader("Forecast plot for " + stock)
    st.write(f'Forcasting for {days} days')
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    forecast_list = forecast.values.tolist()
    #print (forecast_list[1])
    #print (forecast_list[2])

    #show the last 25 analyst recommendationss
    st.subheader("Analyst recommendations for " + stock)
    st.write(analyst_info)

    #use plotly to plot the data
    fig = px.line(close_data, title="Luca's stock prediction for " + stock)

    fig.update_xaxes(tickformat="%d %m %Y")
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
    buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")
    ])
    )
    )
    #fig.show()

    #st.write (f'Forecast plot for 1 year')
    #fig1 = plot_plotly(m, forecast)
    #st.plotly_chart(fig1)

    #st.write("Forecast components")
    
    
    #fig2 = m.plot(forecast)
    #fig2.show()

    
  
  

predict(stock,days)







