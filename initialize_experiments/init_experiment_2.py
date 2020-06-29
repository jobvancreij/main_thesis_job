import sys
from LJT_database.firestore_codes import  add_update, retrieve_updates
from LJT_database.make_query import extract_dataset_predictions
from LJT_helper_functions.helpers import send_message_telegram
from hashlib import sha256

values_for_hyperopt = {
                    'optimizer': ['rmsprop', 'adam'],
                    'neurons': [75],
                    'dropout': {'lowest_poss': 0, 'highest_poss': 0.5},
                    'loss_func': ['binary_crossentropy'],
                    'activation_function': ['tanh','sigmoid','relu','softmax'],
                    'number_layers': [1],
                    'batch_size': {'lowest_poss': 800, 'highest_poss': 3500, 'increments': 20},
                    'epochs': {'lowest_poss': 30, 'highest_poss': 1500, 'increments': 50},
                    'learning_rate': {'lowest_poss': -20, 'highest_poss': 1},
                    'bidirectional': [False],
                    'number_experiments':200,
}

def prepare_text_for_message(final_model):
    """
    to show in the text message which values are important they are highlighted in the dataframe
    """
    final_model["window_size"] = f"<b>-->{final_model['window_size']}</b>"
    final_model["time_ahead_prediction"] = f"<b>-->{final_model['time_ahead_prediction']}</b>"
    return str(final_model)

def remove_columns(df):
    """
    Removes the columns for which specifity, precision, recall, f1 is to low
    :return: df without rows that have columns that are to low
    """
    print(f"len options = {len(df)}")
    # remove values that do not have values high enough for precision,recall,specifity,f1
    df_cleaned = df[df["precision_val"] >= settings["minimum_precision"]]
    df_cleaned = df_cleaned[df_cleaned["specificity_val"] >= settings["minimum_specifity"]]
    df_cleaned = df_cleaned[df_cleaned["f_1_val"] >= settings["minimum_f1_val"]]
    df_cleaned = df_cleaned[df_cleaned["sensitivity_val"] >= settings["minimum_sensitivity"]]
    print(f"len options after cleaning = {len(df_cleaned)}")
    return df_cleaned

if __name__ == "__main__":
    if sha256(input('password is jaikwil, Enter pw if you want to continue: ').encode(
            'utf-8')).hexdigest() == "964de380401cf82374b24a8a4dabc0a564b852e7a0d99e573b92ec0554886d96":
        if len(sys.argv) - 1 !=2:
            raise TypeError("Please give arguments. Arg 1 = coin, Arg2 = algorithm")
        coin = sys.argv[1].upper()
        algorithm = sys.argv[2].upper()
        settings = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                            document="experiment_general_settings")

        df = extract_dataset_predictions(coinname=settings['prediction_dataset_name'],
                                         type_table=settings['prediction_database_name'],
                                         experiment="experiment_1",
                                         experiment_date=settings['experiment_date'])

        # take the best model
        df_cleaned = remove_columns(df)
        final_model = df_cleaned.sort_values('accuracy_val', ascending=False).iloc[0]
        # take the window and prediction
        if input("Manually select window and size?? Type --> YES  ").lower() == "yes":
            window = int(input("Type window value"))
            prediction = int(input("Type prediction ahead value"))
        else:
            window, prediction = final_model[['window_size', 'time_ahead_prediction']]
        experiment_2_values = {'window': [int(window)],
                               'prediction': [int(prediction)]}
        data_to_store = {**experiment_2_values, **values_for_hyperopt}

        add_update(dataset=f"{coin}_{algorithm}_experiments",
                   updates=data_to_store,
                   document="experiment2_settings")
        try:
            print(prepare_text_for_message(data_to_store))
            send_message_telegram("init_experiment_2", prepare_text_for_message(data_to_store))
        except Exception as e:  # in case the highlight gives are send normal string
            print(e)
            print(str(data_to_store))
            send_message_telegram("init_experiment_2", str(data_to_store))
    else:
        print("Not correct password, finish")

