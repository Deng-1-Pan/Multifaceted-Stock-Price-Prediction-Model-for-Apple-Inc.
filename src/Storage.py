import os
import pymongo


DATABASE_NAME = "daps_final"
STOCK_COLLECTION_NAME = "STOCK"
NEWS_COLLECTION_NAME = "NEWS"
SP500_COLLECTION_NAME = "SP500"
INFLATION_COLLECTION_NAME = "INFLATION"


def Data_presence_check_MongoDB(MongoDB_server, collection_name):
    """
    Check if the data exists in the MongoDB cloud

    Args:
        MongoDB_server: Server information for MongoDB
        collection_name: Name of the collection data to be stored 

    Returns:
        Case 1: database not exsit
        Case 2: collection not exist
        Case 3: Data not exist
    """
    # check if database exists
    databases = MongoDB_server.list_database_names()
    if DATABASE_NAME not in databases:
        return "Case 1"

    # check if collection exists
    collections = MongoDB_server[DATABASE_NAME].list_collection_names()
    if collection_name not in collections:
        return "Case 2"

    # check if collection contains elements
    collection = MongoDB_server[DATABASE_NAME][collection_name]
    if collection.count_documents({}) <= 0:
        return "Case 3"

    return True

def store_data(datasets, datatype):
    """
    Store the data locally and to the cloud

    Args:
        datasets: the datasets to be stored
        datatype: the datatype of data
    """
    if datatype == 'stock':
        collection_name = STOCK_COLLECTION_NAME
    elif datatype == 'news':
        collection_name = NEWS_COLLECTION_NAME
    elif datatype == 'SP500':
        collection_name = SP500_COLLECTION_NAME
    elif datatype == 'inflation':
        collection_name = INFLATION_COLLECTION_NAME

    # Save the data locally
    Dataset_folder = os.getcwd() + "\\Dataset\\"
    Figure_folder = os.getcwd() + "\\fig\\"
    # Crete the folder if not exist
    if not os.path.exists(Dataset_folder):
        os.makedirs(Dataset_folder)
        print("Dataset Folder Creted!")
    if not os.path.exists(Figure_folder):
        os.makedirs(Figure_folder)
        print("Dataset Folder Creted!")
    # It will rewrite if it already exist
    datasets.to_csv('./Dataset/' + collection_name +
                    '.csv', index=True, mode='w+')
    print('Data' + datatype + ' has stored to the local folder!')

    # Connect to MongoDB
    try:
        MongoDB_server = pymongo.MongoClient(
            "mongodb+srv://admin:Assignment@cluster0.dqr3am3.mongodb.net/?retryWrites=true&w=majority")
        print("Connected successfully!")

    except Exception as e:
        print('[Error] Please check the following details: ', e)

    # Check if there is a existing Database and Collection
    if Data_presence_check_MongoDB(MongoDB_server, collection_name) == "Case 1":
        # Create Database, Collection and upload the datasets
        database = MongoDB_server[DATABASE_NAME]
        collection = database[collection_name]
        collection.insert_many(datasets.to_dict('records'))
        print("Data stored to MongoDB server!")

    elif Data_presence_check_MongoDB(MongoDB_server, collection_name) == "Case 2":
        # Create Collection and upload the datasets
        collection = MongoDB_server[DATABASE_NAME][collection_name]
        collection.insert_many(datasets.to_dict('records'))
        print("Collection created and data uploaded! ")

    elif Data_presence_check_MongoDB(MongoDB_server, collection_name) == "Case 3":
        # Upload the datasets
        MongoDB_server[DATABASE_NAME][collection_name].insert_many(
            datasets.to_dict('records'))
        print("Data uploaded!")

    else:
        print("The data and the database are already there!")
