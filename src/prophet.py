import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from fbprophet import Prophet
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_data(data, target_feature):
    """
    prepare the data for ingestion by fbprophet: 
    see: https://facebook.github.io/prophet/docs/quick_start.html
    """
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    new_data = new_data.rename(
        {'Date': 'ds', '{}'.format(target_feature): 'y'}, axis=1)

    return new_data


def train_test_split(data):
    """
    Function to split the data into train dataset and test dataset
    """
    train = data.set_index('ds').loc[:'2022-05-01', :].reset_index()
    test = data.set_index('ds').loc['2022-05-01':, :].reset_index()

    return train, test


def make_predictions_df(forecast, data_train, data_test):
    """
    Function to convert the output Prophet dataframe to a datetime
    index and append the actual target values at the end
    """
    forecast.index = pd.to_datetime(forecast.ds)
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:, 'y'] = data.loc[:, 'y']

    return forecast


def plot_predictions(forecast, start_date):
    """
    Function to plot the predictions 
    """
    figure, ax = plt.subplots(figsize=(16, 9))

    train = forecast.loc[start_date:'2022-04-30', :]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    ax.fill_between(train.index, train.yhat_lower,
                    train.yhat_upper, color='steelblue', alpha=0.3)

    test = forecast.loc['2022-05-01':, :]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower,
                    test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc['2022-05-02', 'ds'], color='k', ls='--', alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    ax.grid(ls=':', lw=0.5)

    return figure, ax


def df_prepare(stock_df, data_dic, datatype):
    """
    prepare the dataframe for ingestion by fbprophet: 
    """
    if datatype == 'Adj Close' or datatype == 'Volume':
        return stock_df
    # Create sample DataFrames
    df1 = stock_df
    df2 = data_dic[datatype]

    # # Concatenate the DataFrames using outer join (union of indices)
    new_df = pd.concat([df1, df2], axis=1, join='outer')
    new_df = new_df.dropna()
    new_df = new_df.rename(columns={'Value': datatype})

    return new_df


def training(model, dataframe, datatype, mode):
    """
    A function taht doing the train test split, model fit and predict

    Args:
        model: the model used for training and predicting
        dataframe: the dataframe feed in
        datatype: that kind of data it needs
        mode: single data predict or multi-variable data predic

    Returns:
        train_df: the traning dataframe
        test_df: the testing datafram
        test_predictions: the predict value
    """
    train_df, test_df = train_test_split(data=dataframe)
    if mode == 'single':
        # Only use Close data to train
        model.fit(train_df[['ds', 'y']])

        test_predictions = model.predict(dataframe.drop(columns='y'))

    elif mode == 'multi':
        # Predic using other auxiliary data
        model.add_regressor(datatype)
        model.fit(train_df[['ds', 'y', datatype]])

        test_predictions = model.predict(dataframe.drop(columns='y'))

    elif mode == 'all':
        # Predic using all auxiliary data
        data_list = ['ds', 'y']
        for name in datatype:
            model.add_regressor(name, mode='multiplicative')
            data_list.append(name)

        model.fit(train_df[data_list])

        test_predictions = model.predict(dataframe.drop(columns='y'))

    return train_df, test_df, test_predictions


def create_joint_plot(forecast, x='yhat', y='y', color='b', title=None):
    """
    Function to create joint plot between two variables

    Args:
        forecast: the dataframe contain predict data and true data
        x, y, color: string
        title: title of the plot

    Returns:
        graph, ax: the plot variable
    """
    plt.figure(figsize=(16, 9))
    graph = sns.jointplot(x=x, y=y, data=forecast, kind="reg", color=color)
    graph.fig.set_figwidth(8)
    graph.fig.set_figheight(8)

    ax = graph.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    ax = graph.fig.axes[0]
    ax.add_artist(AnchoredText(
        "R = {:+4.2f}".format(forecast.loc[:, [y, x]].corr().iloc[0, 1]), loc=2))
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Observations', fontsize=15)
    ax.grid(ls=':')
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    ax.grid(ls=':')

    return graph, ax


def error_check(test_predictions):
    """
    Check the MAE and RMSE

    Args:
        test_predictions: the predicted value and true value
    """
    # Calculate the MAE and RMSE for the train set
    date = '2022-05-01'
    train_mae = mean_absolute_error(
        test_predictions.loc[:date, 'y'].dropna(),
        test_predictions.loc[:date, 'yhat'].drop(
            test_predictions.loc[:date, 'yhat'][test_predictions.loc[:date, 'yhat'] == 0].index)
    )
    train_rmse = mean_squared_error(
        test_predictions.loc[:date, 'y'].dropna(),
        test_predictions.loc[:date, 'yhat'].drop(
            test_predictions.loc[:date, 'yhat'][test_predictions.loc[:date, 'yhat'] == 0].index)
    )**0.5

    # Calculate the MAE and RMSE for the test set
    test_mae = mean_absolute_error(
        test_predictions.loc[date:, 'y'].dropna(), test_predictions.loc[date:, 'yhat'].drop(
            test_predictions.loc[date:, 'yhat'][test_predictions.loc[date:, 'yhat'] == 0].index)
    )
    test_rmse = mean_squared_error(
        test_predictions.loc[date:, 'y'].dropna(), test_predictions.loc[date:, 'yhat'].drop(
            test_predictions.loc[date:, 'yhat'][test_predictions.loc[date:, 'yhat'] == 0].index)
    )**0.5

    # Print the evaluation metrics'
    print(f'Train MAE: {train_mae:.2f}')
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test MAE: {test_mae:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')


def analysis(model, test_predictions, train_df, test_df, datatype='close'):
    """
    Analysis the data by ploting the prediction trend, joint-plot and 
    evaluating the mean, median and skewness of the residual distribution

    Args:
        model: the model use to train and predict
        test_predictions: dataframe contains y and yhat
        train_df: trainning dataframe
        test_df: testing dataframe
        datatype: the type of data

    Returns:
        (List[Dict]): the results for all pages of the requests,
        concatenated in one list
    """
    figure = model.plot_components(test_predictions, figsize=(12, 16))
    plt.savefig('./fig/predic_close_' + datatype + '.png')
    plt.close()

    result = make_predictions_df(test_predictions, train_df, test_df)
    result.loc[:, 'yhat'] = result.yhat.clip(lower=0)
    result.loc[:, 'yhat_lower'] = result.yhat_lower.clip(lower=0)
    result.loc[:, 'yhat_upper'] = result.yhat_upper.clip(lower=0)

    figure, ax = plot_predictions(result, '2017-04-03')
    figure.savefig('./fig/predictions_Close_' + datatype + '.png')
    plt.close()

    graph, ax = create_joint_plot(
        result.loc[:'2022-05-01', :], title='Train set', color='b')
    graph.savefig('./fig/jointplot_Close_' + datatype + '_train.png')
    plt.close()
    graph, ax = create_joint_plot(
        result.loc['2022-05-01':, :], title='Test set', color='r')
    graph.savefig('./fig/jointplot_Close_' + datatype + '_test.png')
    plt.close()

    error_check(test_predictions)

    #  mean, median and skewness of the residual distribution
    residuals = (result.loc['2022-05-01':,
                            'y'] - result.loc['2022-05-01':, 'yhat']).dropna()

    skewness = skew(residuals)
    print('The mean of residuals distribution for' +
          datatype + 'is: ', np.mean(residuals))
    print('The median of residuals distribution for' +
          datatype + 'is: ', np.median(residuals))
    print('The skewness of residuals distribution for' +
          datatype + 'is: ', skewness)

    sns.histplot(residuals, bins=20, kde=True)
    plt.title('The Residuals Distribution for ' + datatype)
    plt.xlabel("Residuals")
    plt.savefig('./fig/distribution_residuals_for_' + datatype + '.png')
    plt.close()


def inference(data_dic, datatype, mode=None):
    """
    The part that use all the funtion above, include prepare the data,
    set the model, train the model, predict and analysis.

    Args:
        data_dic: a dict contains all the dataframe
        datatype: the type of data
        modeN: single variable predicting or multi-variables predicting
    """
    if mode == 'single':
        dataframe = prepare_data(data_dic['stock'], target_feature='Close')

        model = Prophet(seasonality_mode='multiplicative',
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False)

        # Predict use only Close data
        train_df, test_df, test_predictions = training(
            model, dataframe, datatype, 'single')

        analysis(model, test_predictions, train_df, test_df)

    else:
        modified_df = df_prepare(data_dic['stock'], data_dic, datatype)

        dataframe = prepare_data(data=modified_df, target_feature='Close')

        model = Prophet(seasonality_mode='multiplicative',
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False)

        # Predict use one of Auxiliary data
        train_df, test_df, test_predictions = training(
            model, dataframe, datatype, 'multi')
        analysis(model, test_predictions, train_df, test_df, datatype)
