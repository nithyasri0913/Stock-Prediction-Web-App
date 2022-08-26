import streamlit as st
import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START='2015-01-01'
TODAY=datetime.date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")
stocks=('AAPL','GOOG','GOOGL','MSFT','GME','AMZN', 'UNH','TSM','META','NVDA','V','JNJ','XOM','WMT','PG','JPM','MA','CVX','HD','LLY','TSLA','BAC','KO','PFE','BABA','NVO')
selected_stock=st.selectbox('Select dataset for prediction',stocks)
n_years=st.slider("Years of prediction",1,4)
period=n_years*365
name=""

@st.cache
def load_data(ticker):
    data=yf.download(ticker,START, TODAY)
    data.reset_index(inplace=True)
    return data

def get_name(ticker):
    n=yf.Ticker(ticker)
    name=n.info['longName']
    return name



data_load_state=st.text("Load data...")
data=load_data(selected_stock)
name=get_name(selected_stock)
data_load_state.text("Loading state...done!")

st.header(name)
st.subheader('Recent data')
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#forecasting
df_train=data[['Date', 'Close']]
df_train=df_train.rename(columns={'Date':'ds','Close':'y'})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)
