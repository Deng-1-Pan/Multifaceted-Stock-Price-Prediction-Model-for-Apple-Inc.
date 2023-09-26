import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix


def EDA_hypo(data_dic):
    """
    Doing the hypothesis tesing with
    H0: there is no relationship between the closing data and auxiliary data
    H1: there is relationship between the closing data and auxiliary data
    And plot the scatter-matrix and heatmap bewteen all variables

    Args:
        data_dic: A dict contains all the dataframes
    """
    # create a new column with the same New Date on both dataframe
    data_dic["SP500"]['New Date'] = data_dic["SP500"].index
    data_dic["stock"]['New Date'] = data_dic["stock"].index
    data_dic["news"]['New Date'] = data_dic["news"].index
    data_dic["inflation"]['New Date'] = data_dic["inflation"].index

    # merge the dataframe on New Date column
    merged_df = pd.concat(
        [data_dic["SP500"], data_dic["stock"]], axis=1).dropna(subset=['Close', 'Adj Close'])
    merged_df.rename(columns={'Value': 'SP500'}, inplace=True)
    merged_df = pd.concat([merged_df, data_dic["news"]], axis=1).dropna(
        subset=['Close', 'Adj Close'])
    merged_df.rename(columns={'Value': 'Sentiment Score'}, inplace=True)
    merged_df = pd.concat([merged_df, data_dic["inflation"]], axis=1).dropna(
        subset=['Close', 'Adj Close'])
    merged_df.rename(columns={'Value': 'Inflation Rate'}, inplace=True)

    # drop the New Date column
    merged_df.drop(columns=['New Date'], inplace=True)

    merged_df['Sentiment Score'] = merged_df['Sentiment Score'].fillna(0)

    corr = merged_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True)
    plt.savefig('./fig/heatmap.png')
    plt.close()

    # Plot the Scattered-Plot Matrix
    scatter_matrix(merged_df, alpha=0.2, figsize=(20, 10), diagonal='hist')
    plt.savefig('./fig/scatter_matrix.png')
    plt.close()

    # Hypothesis Part
    for datatype in ['SP500', 'Sentiment Score', 'Inflation Rate', 'Adj Close', 'Volume']:
        x = merged_df['Close']
        y = merged_df[datatype]
        corr, p_value = pearsonr(x, y)
        print("correlation", corr)
        print("p-value: ", p_value)

        alpha = 0.05  # significance level
        if p_value < alpha:
            print("Reject the null hypothesis, there is a statistically significant correlation between the stock prices of APPLE and the " + datatype)
        else:
            print("Fail to reject the null hypothesis, there is no statistically significant correlation between the stock prices of APPLE and the " + datatype)

        plt.figure(figsize=(16, 9))
        plt.scatter(x, y)
        plt.grid(True)
        plt.title('Correlation between APPL - Close and ' + datatype)
        plt.ylabel(datatype)
        plt.xlabel('APPL - Close')
        plt.savefig('./fig/Correlation_Close_' + datatype + '.png')
        plt.close()

