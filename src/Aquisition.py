import quandl
import yfinance as yf
import pandas as pd

from tqdm import tqdm
import pandas_datareader.data as web
from alpaca_trade_api import REST

START_DATE = "2017-04-01"
END_DATE = "2022-06-01"


def get_stock_data():
    """
    Get stock data on Apple Inc.

    Returns:
        APPL_stock_data: the dataframe contains the APPL stock data
    """
    # Acquire the data from yahoo finance
    try:
        APPL_stock_data = yf.download('AAPL', start=START_DATE, end=END_DATE)
        print("Successfully acuired Apple stock data!")
    except Exception as e:
        print('[Error] Please check the following details: ', e)

    return APPL_stock_data


def get_auxiliary_data():
    """
    Get all the auxiliary data, include news, inflation and SP500

    Returns:
        auxiliary_data: a dict that contain all the auxiliary data
    """
    # a28757a83b124399bc4bd68308266987
    # The required KEYs
    API_KEY = 'PK066CXSHJPQ0EFGAVBP'
    API_SECRET = 'z38smetLDRnGYc3wdxVdGHT1hDZjNTHWc1fxFSEu'
    auxiliary_data = {"news": None, "SP500": None, "inflation": None}

    # News data
    rest_client = REST(API_KEY, API_SECRET)
    news = rest_client.get_news(
        "AAPL", START_DATE, END_DATE, limit=7500, exclude_contentless=True)
    print("Successfully acuired Apple news data!")

    news_benziga = pd.DataFrame()
    with tqdm(total=len(news), desc="Geting News Data") as pbar:
        for i in range(len(news)):
            dataframe = pd.DataFrame.from_dict(news[i]._raw, orient='index')
            news_benziga = pd.concat([news_benziga, dataframe], axis=1)
            pbar.update(1)

    news_benziga.columns = [str(x) for x in range(len(news))]
    news_benziga_df = news_benziga.transpose()
    new_created_at = pd.to_datetime(
        news_benziga_df['created_at']).dt.strftime('%Y-%m-%d')
    news_benziga_df['created_at'] = new_created_at
    auxiliary_data["news"] = news_benziga_df

    # S&P 500 index data
    sp500 = web.DataReader('SP500', 'fred', START_DATE, END_DATE)
    # Drop the 3% of Nan data
    auxiliary_data["SP500"] = sp500.dropna(subset=['SP500'])
    print("Successfully acuired SP500 index stock data!")

    # Inflation rate data
    quandl.ApiConfig.api_key = 'oQSBUMZLG-1w2sqinQNU'
    inflation_df = quandl.get("RATEINF/INFLATION_USA", START_DATE='2017-03-31',
                              END_DATE='2022-05-31', authtoken="oQSBUMZLG-1w2sqinQNU")
    auxiliary_data["inflation"] = inflation_df
    print("Successfully acuired US inflation rate data!")

    return auxiliary_data
