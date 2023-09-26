import src.Aquisition as aq
import src.Storage as st
import src.Preprocessing as pp
import src.EDA as EDA
import src.prophet as ph


def main():
    """
        This is the main block of my code
    """
    # Data Aquisition
    print('================Start Data Aquisition================')
    stock_data = aq.get_stock_data()
    auxiliary_data = aq.get_auxiliary_data()

    # Data Storage
    print('================Start Data Storage================')
    st.store_data(stock_data, 'stock')
    st.store_data(auxiliary_data["news"], 'news')
    st.store_data(auxiliary_data["SP500"], 'SP500')
    st.store_data(auxiliary_data["inflation"], 'inflation')

    # Store all the data into dictionary
    data_dic = {"stock": None, "news": None,
                "SP500": None, "inflation": None}
    data_dic["stock"] = stock_data
    data_dic["news"] = auxiliary_data["news"]
    data_dic["SP500"] = auxiliary_data["SP500"]
    data_dic["inflation"] = auxiliary_data["inflation"]

    # Data Preprocessing
    print('================Start Data Preprocessing================')
    for key, dataframe in data_dic.items():
        dataframe = pp.data_transformation(dataframe, key)
        dataframe = pp.Data_integrity_check(dataframe, key)
        data_dic[key] = dataframe
        # Outlier Check
        pp.data_visualisation.box_plot(dataframe, key)
        pp.data_visualisation.graph_plot(dataframe, key)

    # Data Exploration
    print('================Start Data Exploration================')
    EDA.EDA_hypo(data_dic)

    # Date Inference
    print('================Start Data Inference================')
    ph.inference(data_dic, datatype='Close', mode='single')
    for datatype in ['Adj Close', 'Volume', 'news', 'SP500', 'inflation']:
        ph.inference(data_dic, datatype=datatype, mode='None')


if __name__ == "__main__":
    main()
