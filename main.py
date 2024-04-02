# This is a sample Python script.

from generate_list_of_all_root_links import dump_list_of_wine_type
from dump_page import dump_all_page
from parse_all_files import parse_all_files
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import time
import os
from tools import build_dataframe
from tools import load_and_clean_dataframe, preprocess_train, preprocess_test
import collections.abc
collections.Iterable = collections.abc.Iterable
from hts import HTSRegressor
from hts import functions
from autots import AutoTS
from lazypredict import LazyRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



def smape(predicted, actual):
    return 100 * (2 * abs(predicted - actual) / (abs(predicted) + abs(actual))).mean(axis=1)



if __name__ == '__main__':
    print('commence')
    use_hierarchy=False
    # GENERATE DATA (BABA IS THE BEST)
    #dump_list_of_wine_type("bordeaux")
    #dump_list_of_wine_type("bourgogne")
    #dump_list_of_wine_type("rhone")
    #filenames = ['list_of_link_wine_bordeaux.txt',
    #             'list_of_link_wine_bourgogne.txt',
    #             'list_of_link_wine_rhone.txt']
    #with open('list_of_link_wine.txt', 'w') as outfile:
    #    for fname in filenames:
    #        with open(fname) as infile:
    #            for line in infile:
    #                outfile.write(line)
    dump_all_page ("list_of_link_wine.txt")
    parse_all_files()

    # CREATE THE PANDAS DATAFRAME
    #folder_path = 'data_formated'
    #df = build_dataframe(folder_path)
    #print(df[:10].to_string())
    #os.makedirs("dataframes", exist_ok=True)
    #df.to_csv('dataframes/dataframe.csv', index=False)

    df = load_and_clean_dataframe('dataframes/dataframe.csv')
    print(df.nlargest(100, 'Price_2024').to_string())
    df['Price_2024'] = df['Price_2024'].clip(upper=1000)

    #train_df,sum_mat_train,sum_mat_labels_train = preprocess_train(df,use_hierarchy)

    #Fitting the model using AutoTS
    #model = AutoTS(
    #   num_validations=0,
    #   no_negatives=True,
    #   forecast_length=1,
    #   remove_leading_zeroes=True,
    #   ensemble=None,
    #   model_list="superfast",
    #   transformer_list="superfast",
    #   subset=1000,
    #   max_generations=0# more
    #)

    #model = model.fit(train_df)

    # Print the description of the best model
    #print(model)

    #prediction = model.predict()

    # point forecasts dataframe
    #forecasts_df = prediction.forecast

    #if use_hierarchy:
    #    pred_dict = collections.OrderedDict()
    #    for label in sum_mat_labels_train:
    #        pred_dict[label] = pd.DataFrame(data=forecasts_df[label].values, columns=['yhat'])
    #    revised = functions.optimal_combination(pred_dict, sum_mat_train, method='OLS', mse={})
    #    revised_forecasts = pd.DataFrame(data=revised[0:, 0:],
    #                                     index=forecasts_df.index,
    #                                     columns=sum_mat_labels_train)

    #forecasts_naive = train_df.loc[['2023-01-01']]
    #forecasts_naive.index = forecasts_naive.index.where(~(forecasts_naive.index == '2023-01-01'), '2024-01-01')

    # Calculate SMAPE for each column
    #test_df = preprocess_test(df)
    #print(test_df.columns)
    #print(test_df.sum(axis=1))

    #forecasts_df = forecasts_df.reindex(columns=test_df.columns)
    #forecasts_naive = forecasts_naive.reindex(columns=test_df.columns)
    #print(forecasts_df.sum(axis=1))
    #print(forecasts_naive.sum(axis=1))

    #smape_ts = smape(forecasts_df,test_df)
    #smape_naive = smape(forecasts_naive,test_df)
    #print(smape_ts)
    #print(smape_naive)

    #if use_hierarchy:
    #    revised_forecasts = revised_forecasts.reindex(columns=test_df.columns)
    #    print(revised_forecasts.sum(axis=1))
    #    smape_revised = smape(revised_forecasts, test_df)
    #    print(smape_revised)

    ##PRICE QUALITY
    ## Step 1: Calculate the average price per note
    #average_price_per_note = df.groupby('note')['Price_2024'].mean().reset_index(name='avg_price')

    ##Factor in stocking fees, 1,44 euro per year per bottle for an average of 10 years
    #stocking_fees = 14.4

    ## Step 2: Merge the average price back with the original DataFrame
    #df_merged = df.merge(average_price_per_note, on='note', how='left')

    ## Step 3: Calculate the score as Price / Avg Price for the note
    #df_merged['score'] = (df_merged['Price_2024']+stocking_fees) / df_merged['avg_price']

    ## Step 4: Select the 100 wines with the best (lowest) scores
    #best_wines = df_merged.nsmallest(100, 'score')
    #best_wines_absolute = df_merged.nlargest(100, 'note')

    ## Output the result
    #print(best_wines.to_string())
    #print(best_wines_absolute.to_string())

    #Removing older wines than 1950 as they mess up predictions (note=NAN becomes a good thing). Better altnerative
    # would be to put constraints on monotonicity, plus NAN cannot be a good thing

    def predict_best_wines(df,note_column_decorators):
        # Splitting the data into features and target
        note_columns = ['note']
        for decorator in note_column_decorators:
            note_columns = note_columns + [col for col in df.columns if col.startswith(decorator)]
        X = df[note_columns]  # Your features (all note columns)
        y = df['Price_2024']  # Your target variable

        # Randomize train/test split and tak average predictions
        nb_random_trials=5
        df['Predicted_Price'] = 0
        for r in range(nb_random_trials):
            rand_seed = np.random.randint(100)
            # Splitting the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            ## Initialize LazyRegressor
            #reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

            ## Fit and compare models
            #models, predictions = reg.fit(X_train, X_test, y_train, y_test)
            #print (X_train.shape)
            #print (X_test.shape)
            #print (predictions.shape)
            #print (predictions)

            ## Display the performance of all models
            #print(models)

            #best_model_name = models.sort_values(by='Adjusted R-Squared',ascending=False).index[0]
            #print(f"The best model based on Adjusted R-Squared is: {best_model_name}")

            monotone_constraints = [1] * len(note_columns)
            regressor = LGBMRegressor(monotone_constraints=monotone_constraints,verbose=-1,random_state=rand_seed,
                                      subsample=0.8,feature_fraction=0.8)

            regressor.fit(X,y)

            # Make predictions for the entire dataset
            df['Predicted_Price'] = regressor.predict(X)+df['Predicted_Price']
        df['Predicted_Price']=df['Predicted_Price']/nb_random_trials

        #print(df[:10].to_string())

        # Calculate the quality-price ratio or any other evaluation metric
        penalization = 15*12*0.12
        df['Quality_Price_Ratio'] = df['Predicted_Price'] / (df['Price_2024']+penalization)  # Adjust this calculation as needed
        df['Max_Buying_Price'] = np.minimum(df['Predicted_Price']/1.5,df['Price_2024']*2.0)
        #print (df[df['Domaine']=="Château Carbonnieux"].to_string())
        #print(df[df['Domaine'] == "Château Clos Haut Peyraguey"].to_string())
        #print(df[:10].to_string())
        #print (df.shape)
        best_value_wines = df[df['Quality_Price_Ratio']>1.5].dropna(subset=note_columns, how='all')
        print(best_value_wines.to_string())
        print (best_value_wines.shape)
        return best_value_wines

    years_to_cut = [1945]
    for cut in years_to_cut:
        df_cut = df.copy()
        print (cut)
        df_cut = df_cut[pd.to_datetime(df['Vintage']).dt.year > cut]
        print(df_cut.shape)
        #bvw_simple = predict_best_wines(df,['Note_'])

        # Identifiers excluding 'Vintage'
        identifiers = ['Pays/région', 'Couleur', 'Appellation', 'Classement', 'Viticulture', 'Domaine']

        # Note columns
        note_columns = [col for col in df_cut.columns if col.startswith('Note_')]+['note']

        # Initialize an empty DataFrame for storing averages
        avg_notes_df = pd.DataFrame()
        for col in note_columns:
            # Group by identifiers, then calculate the mean for each note column, excluding NaNs
            #print (df.shape)
            grouped = df_cut.groupby(identifiers)[col].mean().reset_index(name=f'avg_{col}')
            #print (df.shape)
            #print ("kookie")
            # If first iteration, avg_notes_df is empty
            if avg_notes_df.empty:
                avg_notes_df = grouped
            else:
                # Merge on identifiers to ensure each avg column aligns with the right group
                avg_notes_df = pd.merge(avg_notes_df, grouped, on=identifiers, how='outer')

        # Assuming 'df' has a unique identifier made from the combination of identifiers and 'Vintage'
        df_cut['unique_id'] = df_cut[identifiers + ['Vintage']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        # Merge the averages back based on the wine identifiers (excluding vintage)
        df_cut = pd.merge(df_cut, avg_notes_df, on=identifiers, how='left')

        # Cleanup if needed
        df_cut.drop(columns=['unique_id'], inplace=True)

        bvw_simple = predict_best_wines(df_cut,['Note_','avg_'])




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
