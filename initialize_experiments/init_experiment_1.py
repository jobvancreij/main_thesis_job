import sys
import os
from LJT_database.firestore_codes import  add_update, retrieve_updates
from LJT_database.merge_dataset import retrieve_data_predictors
from LJT_database.feature_prep import feature_preperation
from statsmodels.tsa.stattools import acf
import numpy as np
import matplotlib.pyplot as plt
from LJT_helper_functions.helpers import send_message_telegram
from LJT_helper_functions.dataset_prep import percentage_fee
from itertools import product
from hashlib import sha256

prediction = [1] #to to predict ahead
windows_extend = [1,2,4, 6, 11, 35, 50, 60, 70, 100, 150, 200]

space = {
    'optimizer': 'adam',
    'neurons': 75,
    'dropout': 0.1,
    'momentum': 0.1,
    'loss_func': 'binary_crossentropy',
    'activation_function': 'sigmoid',
    'number_layers': 1,
    'batch_size': 1000,
    'epochs': 500,
    'learning_rate': 0.001,
    "bidrectional": False,
}



def correlation_tests(df,settings_general):
    '''
    This function calculates which windows are expected to give good results. Besides that, it extends the
    tested windows with values to make sure that unpopular choices are also tested

    :param df: the dataframe from the specific coin
    :param settings_general: the settings file contains the information about the experiment
    :return: the windows that are being tested in the experiment
    '''
    dataset_prepared = feature_preperation(df)
    target_column = f"{settings_general['coin']}__ticker_info__close_price"
    dataset_prepared[f"{target_column}_next"] = dataset_prepared[target_column].shift(-1).fillna(0)
    if settings_general["include_transaction_cost"]:
        ## you make rev if next price is greater than buy_price + buy_price * transction cost
        dataset_prepared['difference'] = dataset_prepared[f"{target_column}_next"] - (dataset_prepared[target_column] + dataset_prepared[target_column] * (
                percentage_fee / 100))  # make float from perc
    else:
        dataset_prepared['difference'] = dataset_prepared[f"{target_column}_next"] - dataset_prepared[target_column]

    dataset_prepared['bins'] = dataset_prepared['difference'].apply(
                                    lambda x: 1 if x > 0 else 0)  # value 1 if raise 0 if downwards
    cor = acf(dataset_prepared['bins'], nlags=200) #calculate the lagged correlation
    sort_index = np.argsort(-cor) #sorted by highest lagged
    plt.plot(cor[1:]) #make a plot that contains the laggged corr
    plt.savefig(f"{settings_general['coin']}_{settings_general['algorithm']}/lagged_correlation_{settings_general['experiment_date']}.png")
    windows = sorted([i + 1 for i in sort_index[:10]]) #take top ten highest corr windows
    windows.extend(windows_extend) #fill in some standard valeus
    final_windows = sorted([win for win in set(window for window in windows)]) #make sure they are unique
    return final_windows,dataset_prepared

def prepare_experiments(windows,prediction_ahead):
    '''
    Determine the experiments that have to be done in experiment_1
    :param windows: the windows to be tested
    :param prediction_ahead: the prediction ahead to be tested
    :return: dictionary that contains the experiments. Where they value is (window, prediction ahead)
    '''
    number_experiments = len(windows) * len(prediction_ahead)
    print(f"Number of experiments = {number_experiments}")
    #make integers because firestore does not accept np integers
    final_experiment = {}
    for i,experiment in enumerate((product(windows, prediction_ahead))):
        final_experiment[f"experiment_{i}"] = {**space, **{'window_size':int(experiment[0]),'time_ahead_prediction':int(experiment[1])}}
    return final_experiment



if __name__ == "__main__":
    if sha256(input('password is jaikwil, Enter pw if you want to continue: ').encode(
            'utf-8')).hexdigest() == "964de380401cf82374b24a8a4dabc0a564b852e7a0d99e573b92ec0554886d96":
        if len(sys.argv) - 1 !=2:
            raise TypeError("Please give arguments. Arg 1 = coin, Arg2 = algorithm")
        coin = sys.argv[1].upper()
        algorithm = sys.argv[2].upper()
        os.makedirs(f"{coin}_{algorithm}", exist_ok=True) #make folder if not yet exists
        settings_general = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                            document="experiment_general_settings")
        df = retrieve_data_predictors(settings_general)
        final_windows,dataset_prepared = correlation_tests(df,settings_general)
        prediction_ahead = prediction #change this upper side of the file
        experiment_1 = prepare_experiments(final_windows,prediction_ahead)
        add_update(dataset=f"{coin}_{algorithm}_experiments",updates=experiment_1,document="experiment1_settings")
        final_text = f"""
                        Experiment 1 succesfully initialized for {coin} {algorithm}. 
                        windows are: {final_windows} \npredictions are: {prediction_ahead}  
                        Now run experiment 1
                      """
        print(final_text)
        send_message_telegram("init_experiment_1",final_text)
    else:
        print("Not correct password, finish")
