import pandas as pd
import numpy as np
import os
import json
import collections.abc
collections.Iterable = collections.abc.Iterable
from hts import functions
from collections import defaultdict

def process_file(file_path):
    file_name = os.path.basename(file_path)
    vintage_year = file_name.split('-')[1]
    with open(file_path, 'r') as file:
        data = json.load(file)
    attributes = {key: value for key, value in data.items() if key not in ['degustation', 'cote']}
    #df_cote = pd.DataFrame(list(data['cote'].items()), columns=['Year', 'Price'])
    #df_cote['Price'] = pd.to_numeric(df_cote['Price'])
    df = pd.DataFrame([attributes])
    # Add the vintage year to each row
    df['Vintage'] = vintage_year
    # Extract and add the price for the year 2024 as a new column, if it exists
    price_2024 = data.get('cote', {}).get('2024', np.nan)  # Use np.nan as default if 2024 price is not found
    df['Price_2024'] = price_2024
    # Handling 'degustation' - adding a column for each critic's note
    # Check if 'degustation' exists and is a dictionary
    if 'degustation' in data and isinstance(data['degustation'], dict):
        for key, value in data['degustation'].items():
            critic_name = key.split(' ')[0]  # Assuming the critic's name is before the first space
            df[f'Note_{critic_name}'] = value
    return df


def build_dataframe(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    frames = []

    for file in files:
        frame = process_file(file)
        frames.append(frame)
    # Concatenate all frames into a single DataFrame
    return pd.concat(frames, ignore_index=True)

def load_and_clean_dataframe(file_path):
    df = pd.read_csv(file_path)
    columns_to_keep = [
        'Pays/région', 'Couleur', 'Appellation', 'Vintage',
        'Classement', 'Viticulture', 'Domaine', 'note','Price_2024'
    ]

    # Dynamically find all columns that start with 'Note_'
    note_columns = [col for col in df.columns if col.startswith('Note_') and col not in
                    ("Note_JR","Note_RVF","Note_B+D")]

    # Combine the specified columns with the 'Note_' columns
    final_columns_to_keep = columns_to_keep + note_columns

    # Simplify the DataFrame
    df = df[final_columns_to_keep]

    # Fonction personnalisée pour convertir les notes
    def convert_note_str(note):
        if pd.isna(note):
            return np.nan  # Retourne NaN si la note est NaN
        else:
            return int(note.split('/')[0])  # Convertit la partie avant '/' en entier

    def convert_note_float(note):
        if pd.isna(note):
            return np.nan  # Retourne NaN si la note est NaN
        else:
            return int(note)

    # Application de la fonction à la colonne 'note'
    df['note'] = df['note'].apply(convert_note_str)
    for notes in note_columns:
        df[notes] = df[notes].apply(convert_note_float)

    #df = df.groupby(final_columns_to_keep,as_index=False).mean()

    df=df[df['Vintage'].isna() == False]
    df['Vintage']=pd.to_datetime(df['Vintage'].astype(int).astype(str) + '-01-01')
    #df['Year'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
    df['Classement'] = df['Classement'].astype(str)
    df['Viticulture'] = df['Viticulture'].astype(str)
    df['Domaine'] = df['Domaine'].astype(str)

    return df

def preprocess_train(df,use_hierarchy=True):
    train_df = df[df['Vintage'] <= pd.Timestamp(year=2019, month=1, day=1)]

    level_names = [
    'Pays/région', 'Couleur', 'Appellation',
    'Classement', 'Viticulture','Domaine']
    if use_hierarchy:
        hierarchy = [
        ['Pays/région'],
        ['Pays/région','Couleur'],
        ['Pays/région','Couleur','Appellation']
        ]
    else:
        hierarchy=[]

    train_df, sum_mat_train, sum_mat_labels_train = functions.get_hierarchichal_df(train_df,
                                                                       level_names=level_names,
                                                                       hierarchy=hierarchy,
                                                                       date_colname='Vintage',
                                                                       val_colname='note')
    if not use_hierarchy:
        train_df = train_df[train_df.columns.difference(['total'])]
    return train_df,sum_mat_train,sum_mat_labels_train

def preprocess_test(df):
    test_df = df[df['Vintage'] >= pd.Timestamp(year=2020, month=1, day=1)]

    level_names = [
    'Pays/région', 'Couleur', 'Appellation',
    'Classement', 'Viticulture','Domaine']

    hierarchy = []

    test_df, sum_mat_test, sum_mat_labels_test = functions.get_hierarchichal_df(test_df,
                                                                       level_names=level_names,
                                                                       hierarchy=hierarchy,
                                                                       date_colname='Vintage',
                                                                       val_colname='note')
    test_df = test_df[test_df.columns.difference(['total'])]
    return test_df