import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def Data_integrity_check(dataframe, key):
    """
    Check any missing value and interpolate with if there is any

    Args:
        dataframe: The dataframe need to be checked
        key: Indicate which database need to be checked

    Returns:
        dataframe: A dataframe with out any NaN data
    """

    report = dataframe.isnull().sum()
    if (report != 0).any():

        # Actually only SP500 data need to be dealing with missing value
        print("There is a null element in the " + str(key) + " Dataframe.")

        # interpolate the NaN data
        sp500_original = dataframe.copy()

        # Interpolate the missing values
        dataframe.interpolate(method='linear', inplace=True)

        # Plot the interpolated data
        fig, axes = plt.subplots(figsize=(16, 9))

        axes.plot(dataframe, color='red', label='Interpolated Data')

        axes.plot(sp500_original, color='blue', label='Original Data')
        # Add labels and legend
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
        plt.grid()
        plt.ylabel('Stock Price')
        plt.legend()
        fig.savefig('./fig/Interpolated_SP500.png')
        plt.close()

    else:
        print("This " + str(key) + " Dataframe is full of data!")

    return dataframe


def data_transformation(raw_df, key):
    """
    Transform the dataframe to the same format

    Args:
        raw_df: the original dataframe
        key: indicate which dataframe need to be transformed

    Returns:
        new_df: A transformed dataframe
    """

    if key == 'stock':
        new_df = raw_df.copy()
        new_df.index = pd.to_datetime(new_df.index)
        print("APPL stocl date is already well structured!")

    elif key == 'news':
        nltk.download('vader_lexicon')
        nltk.download('punkt')

        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Calculate the sentiment score
        raw_df['headline_sentiment'] = raw_df['headline'].apply(
            lambda x: sia.polarity_scores(x)['compound'])

        new_df = {'Date': raw_df['created_at'],
                  'Value': raw_df['headline_sentiment']}
        new_df = pd.DataFrame(new_df).set_index('Date')
        new_df.index = pd.to_datetime(new_df.index)

        # Drop the value that out of range
        new_df = new_df[(new_df.index >= "2017-04-01")
                        & (new_df.index <= "2022-05-31")]
        new_df = new_df.groupby(new_df.index).sum()
        # new_df.sort_index()

        print("News data is now well structured!")

    elif key == 'inflation':
        new_df = raw_df.copy()

        new_df.index = new_df.index.astype('datetime64[ns]')

        date_range = pd.date_range(
            start='2017-03-31', end='2022-05-31', freq='D')

        new_df = pd.DataFrame(new_df['Value'], index=date_range)
        new_df.index = new_df.index.rename('Date')

        # Interpolate the missing values
        new_df.interpolate(method='linear', inplace=True)
        new_df = new_df.iloc[1:]

        # Plot the interpolated data
        _, axes = plt.subplots(figsize=(16, 9))

        axes.plot(new_df, color='red', label='Interpolated Data')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
        plt.grid()
        plt.ylabel('Stock Price')
        plt.savefig('./fig/interpolate_inflation.png')

        print("Inflation rate data is now well structured!")

    elif key == 'SP500':
        new_df = raw_df.copy()
        new_df.rename(columns={'SP500': 'Value'}, inplace=True)
        new_df.index.name = 'Date'
        new_df.index = new_df.index.astype('datetime64[ns]')
        print("SP500 data is already well structured!")
    return new_df


class data_visualisation():
    """
        A class to plot all the figures
    """

    def scatter_plot(dataframe, datatype, key):
        """
        Plot the scatter of the data given

        Args:
            dataframe: the whole dataframe
            datatype: indicate the datatpe
            key: which dataset is going to be ploted
        """
        # Calculate the IQR
        Q1 = np.quantile(dataframe[datatype], 0.25)
        Q3 = np.quantile(dataframe[datatype], 0.75)
        IQR = Q3-Q1
        Minimum = Q1-1.5*IQR
        Maximum = Q3+1.5*IQR

        out_Q1 = dataframe.query(datatype + ' < @Minimum')
        out_Q3 = dataframe.query(datatype + ' > @Maximum')

        # plot the scatter
        if key == 'stock':
            ylabel = 'Number of Shares'
        elif key == 'inflation':
            ylabel = 'Inflation Rate'
        elif key == 'news':
            ylabel = 'Sentiment Score'
        plt.subplots(figsize=(16, 9))
        plt.title('Scatter Plot of ' + key)
        plt.scatter(dataframe.index, dataframe[datatype])
        plt.scatter(out_Q1.index, out_Q1[datatype], c='r')
        plt.scatter(out_Q3.index, out_Q3[datatype], c='r')
        plt.xlabel('Years')
        plt.ylabel(ylabel)
        plt.grid()

        plt.savefig("./fig/scatter_" + key + ".png")
        plt.close()

    def box_plot(dataframe, key):
        """
        Plot the boxplot of the data given

        Args:
            dataframe: the whole dataframe
            key: which dataset is going to be ploted
        """
        if key == 'stock':
            plt.figure(num=1, figsize=(16, 9))
            sns.boxplot(data=dataframe[['Open', 'High', 'Low', 'Close', 'Adj Close']],
                        orient="v").set_title('Box Plot for stock data without Volume')
            plt.savefig("./fig/boxplot_" + key + "_without_Volume.png")
            plt.figure(num=2, figsize=(16, 9))
            sns.boxplot(data=dataframe[['Volume']], orient="v", width=0.3).set_title(
                'Box Plot for stock data - Volume')
            plt.savefig("./fig/boxplot_" + key + "_Volume.png")
            plt.close()

            data_visualisation.scatter_plot(dataframe, 'Volume', key)
        else:
            plt.figure(figsize=(16, 9))
            sns.boxplot(data=dataframe[['Value']], orient="v", width=0.3).set_title(
                'Box Plot for ' + key)
            plt.savefig("./fig/boxplot_" + key + ".png")
            plt.close()
            if key != 'SP500':
                data_visualisation.scatter_plot(dataframe, 'Value', key)

    def histogram(dataframe, key):
        """
        Plot the histogram of the data given

        Args:
            dataframe: the whole dataframe
            key: which dataset is going to be ploted
        """
        if key == 'news':
            # Create a histogram of the sentiment scores
            positive_count = (dataframe['Value'] > 0).sum()
            zero_count = (dataframe['Value'] == 0).sum()
            negative_count = (dataframe['Value'] < 0).sum()

            print("Total counts of positive sentiment score: ", positive_count)
            print("Total counts of neutral sentiment score: ", zero_count)
            print("Total counts of negative sentiment score: ", negative_count)

            # Plot the histograms
            plt.figure(figsize=(16, 9))
            plt.hist(dataframe['Value'].where(dataframe['Value'] > 0), bins=10,
                     color='green', alpha=0.5, label='Positive', align='mid')
            plt.hist(dataframe['Value'].where(dataframe['Value'] == 0), bins=11,
                     color='blue', alpha=0.5, label='Zero', align='mid')
            plt.hist(dataframe['Value'].where(dataframe['Value'] < 0), bins=10,
                     color='red', alpha=0.5, label='Negative', align='mid')

            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Sentiment Score')
            plt.grid(True)
            plt.legend()
            plt.savefig("./fig/histograms_news_sentiment_score.png")
            plt.close()
        else:
            # Create a bar chart of the volumn
            plt.figure(figsize=(16, 9))
            plt.bar(dataframe.index, dataframe['Volume'])
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.title('Distribution of Sentiment Score')
            plt.grid(True)
            plt.savefig("./fig/barchart_volume.png")
            plt.close()

    def graph_plot(dataframe, key):
        """
        Plot the visulisation of the dataframe

        Args:
            dataframe: the whole dataframe
            key: which dataset is going to be ploted
        """
        if key == 'stock':
            fig, axes = plt.subplots(figsize=(16, 9))
            for column in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                axes.plot(dataframe.index, dataframe[column], label=column)
                axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
            plt.title("Stock data")
            plt.grid(True)
            plt.legend()
            plt.xlabel("Data point index")
            plt.xticks(rotation=45)
            plt.ylabel("Stock price")
            fig.savefig("./fig/stock_data.png")
            plt.close()

            data_visualisation.histogram(dataframe, key)
        else:
            if key == 'news':
                ylabel = 'Sentiment Score'
                title = 'Sentiment Analysis of News Headlines'

                data_visualisation.histogram(dataframe, key)
            elif key == 'inflation':
                ylabel = 'Inflation rate'
                title = 'Inflation rate Data Trand'

            elif key == 'SP500':
                ylabel = 'Stock Price'
                title = 'Stock Data for SP500 Index'

            fig, axes = plt.subplots(figsize=(16, 9))
            axes.plot(dataframe.index, dataframe["Value"])
            axes.xaxis.set_major_locator(ticker.MultipleLocator(100))
            plt.grid(True)
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig("./fig/" + key + ".png")
            plt.close()
