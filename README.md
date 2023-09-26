[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9510471&assignment_repo_type=AssignmentRepo)

# Data Acquisition and Processing Systems (DAPS) (ELEC0136)

## Abstrct

This report discusses the use of machine learning models and data related to the stock market, but from different sources, to predict stock prices, with a focus on predicting the stock price movement of Apple Inc. in May 2022. Data from various sources, including stock market data, news data, S&P500 data and US inflation data, are used to analyse their impact on stock market movements. The report describes the procedures for collecting and storing the data, the pre-processing of the data obtained, and the mining methods for the processed data. The models used to learn and predict stock prices are evaluated using the predicting results and associated image and numerical evaluation methods. The best performing auxiliary data is the S&P500 data, but the possibility of overfitting the model cannot be ruled out due to the questionable overperformance of the model. The report concludes with a comprehensive analysis of the performance of the models and their limitations.

## Requirement

- python=3.7.0
- Cython>=0.22
- pystan>=2.14
- numpy>=1.10.0
- pandas>=0.23.4
- matplotlib>=2.0.0
- LunarCalendar>=0.0.9
- convertdate>=2.1.2
- holidays>=0.9.5
- setuptools-git>=1.2
- python-dateutil>=2.8.0
- tqdm>=4.36.1
- nltk
- pymongo
- scipy
- seaborn
- quandl
- yfinance == 0.1.87
- pandas_datareader
- alpaca_trade_api
- scikit-learn
- fbprophet

**Note** : If fbprophet cannot be installed by pip, try `conda install -c conda-forge fbprophet`

**Note** : UserWarning: FixedFormatter should only be used together with FixedLocator ax.set_yticklabels(yticklabels) This Warning does not affect the training
