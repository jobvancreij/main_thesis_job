#to start a new experiment the settings have to be added to firestore
import sys
from LJT_database.firestore_codes import  add_update
from LJT_database.make_new_dataset import create_dataset
from LJT_database.make_database import make_new_table
from LJT_helper_functions.helpers import send_message_telegram
from LJT_database.merge_dataset import check_for_updates_needed
from datetime import datetime
from google.cloud import bigquery
from hashlib import sha256
from deep_learning_models.feature_selection import make_components_pca
from LJT_database.merge_dataset import retrieve_data_predictors
from LJT_database.feature_prep import feature_preperation





dataset_schema = [
    bigquery.SchemaField("optimizer",'STRING'),
    bigquery.SchemaField("neurons",'INTEGER'),
    bigquery.SchemaField("dropout",'FLOAT'),
    bigquery.SchemaField("momentum",'FLOAT'),
    bigquery.SchemaField("activation_function","STRING"),
    bigquery.SchemaField("number_layers",'INTEGER'),
    bigquery.SchemaField("batch_size",'INTEGER'),
    bigquery.SchemaField("epochs",'INTEGER'),
    bigquery.SchemaField("learning_rate",'FLOAT'),
    bigquery.SchemaField("window_size",'INTEGER'),
    bigquery.SchemaField("bidrectional",'BOOL'),
    bigquery.SchemaField("time_ahead_prediction",'INTEGER'),
    bigquery.SchemaField("loss_func",'STRING'),
    bigquery.SchemaField("loss", 'FLOAT'),
    bigquery.SchemaField("status",'STRING',mode="REQUIRED"),
    bigquery.SchemaField("sensitivity_val",'FLOAT'),
    bigquery.SchemaField("specificity_val",'FLOAT'),
    bigquery.SchemaField("precision_val",'FLOAT'),
    bigquery.SchemaField("accuracy_val",'FLOAT'),
    bigquery.SchemaField("highest_train_ac",'FLOAT'),
    bigquery.SchemaField("f_1_val",'FLOAT'),
    bigquery.SchemaField("experiment",'STRING'),
    bigquery.SchemaField("experiment_date",'STRING'),

    ]

def init_settings(coin,algorithm):
    """

    :param coin: the coin that is going to be analyzed
    :param algorithm: the algorithm that is used to analyze the coin
    :return: a list with settings for the experiment
    """

#1588554000000 accuracy of : 55 percent
    settings = {
        "first_timestamp": 1581616800000,#1581012000000, #start year
        #full dataset timestamp = 1593028800000
        "last_timestamp": 1590105600000,#1590105600000,#1581616800000 + 60000*60*24*7,#1588554000000,#1583431200000, 1588122840000 #22-04-2020 1588122846000 #1588122840000 <-- use
        "coin": coin,
        "training_size":0.8,
        "minutes_lookback":30,
        'bins': True,
        'algorithm': algorithm,
        "database_name": "prediction_datasets",
        'minimum_sensitivity': 0.4,
        'minimum_specifity': 0.4,
        'minimum_precision': 0.4,
        'minimum_f1_val': 0.4,
        "include_transaction_cost": False,
        "columns":["*"], #for init we need all the columns
        "dataset_name" : f"{coin}_dataset",
        "experiment_date":f"{datetime.now().month}_{datetime.now().day}",
        "prediction_dataset_name": "prediction_scores",
        "prediction_database_name": f"{coin}_{algorithm}_scores",
    }
    return settings



if __name__ == "__main__":
    if sha256(input('password is jaikwil, Enter pw if you want to continue: ').encode(
            'utf-8')).hexdigest() == "964de380401cf82374b24a8a4dabc0a564b852e7a0d99e573b92ec0554886d96":
        if len(sys.argv) - 1 !=2:
            raise TypeError("Please give arguments. Arg 1 = coin, Arg2 = algorithm")
        coin = sys.argv[1].upper()
        algorithm = sys.argv[2].upper()
        settings = init_settings(coin,algorithm)
        dataset_prepared = feature_preperation(retrieve_data_predictors(settings))
        df,columns = make_components_pca(dataset_prepared,
                                         settings,
                                         percentage_variance=0.99,
                                         type_scaler='min_max_scaler',
                                         min_cor=0.4)
        columns = [col for col in columns if "price_change_lag_" not in col] #otherwise it queries the change cols
        if 'last_start_time' not in columns: #add last_start_time since always needed
            columns.append('last_start_time')
        if f"{coin}__ticker_info__close_price" not in columns: #add close price since always needed
            columns.append(f"{coin}__ticker_info__close_price")
        settings['columns'] = columns
        create_dataset(settings["prediction_dataset_name"])
        make_new_table(settings["prediction_dataset_name"],
                       settings["prediction_database_name"],
                       schema=dataset_schema)
        print("Check if data needs to be updated")
        check_for_updates_needed(settings) # if the dataset is not updated yet do this
        #add settings to firestore
        add_update(dataset=f"{coin}_{algorithm}_experiments",updates=settings,document="experiment_general_settings")
        print(f"Experiment succesfully initialized for {coin} {algorithm}. Now run the file to initialize experiment 1")
        send_message_telegram("init_experiment_general", f"{coin}_{algorithm} is initialized. Start initialize experiment 1")
    else:
        print("Not correct password, finish")


