import pandas as pd
import numpy as np

def split_cabin(data):
    deck = []
    num = []
    side = []
    for _, row in data.iterrows():
        if pd.isna(row['Cabin']):
            deck.append(None)
            num.append(None)
            side.append(None)
            continue
        info = row['Cabin'].split('/')
        deck.append(info[0])
        num.append(int(info[1]))
        side.append(info[2])
    data['Deck'] = deck
    data['Num'] = num
    data['Side'] = side
    return data

def sum_spendings(data):
    neces = []
    luxuries = []
    for _, row in data.iterrows():
        if pd.isna(row['RoomService']):
            neces.append(0)
        else:
            neces.append(row['RoomService'])

        if pd.isna(row['FoodCourt']):
            neces[-1] += 0
        else:
            neces[-1] += row['FoodCourt']


        if pd.isna(row['VRDeck']):
            luxuries.append(0)
        else:
            luxuries.append(row['VRDeck'])

        if pd.isna(row['ShoppingMall']):
            luxuries[-1] += 0
        else:
            luxuries[-1] += row['ShoppingMall']

        if pd.isna(row['Spa']):
            luxuries[-1] += 0
        else:
            luxuries[-1] += row['Spa']

    data['Nec'] = neces
    data['Lux'] = luxuries
    return data

def load_data(filename):
    if '.csv' in filename:
        return pd.read_csv(filename)
    return np.fromfile(filename)

def standardize(data, sample, column_name):
    mean = data[column_name].mean()
    std = data[column_name].std()
    data[column_name] = (data[column_name] - mean) / std
    sample[column_name] = (sample[column_name] - mean) / std
    return data, sample

def normalize(data, sample, column_name):
    maximum = data[column_name].max()
    minimum = data[column_name].min()
    data[column_name] = (data[column_name] - minimum) / (maximum - minimum)
    sample[column_name] = (sample[column_name] - minimum) / (maximum - minimum)
    return data, sample

if __name__ == "__main__":
    data = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/train.csv')
    sample = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/test.csv')
    new_data = split_cabin(data)
    new_sample = split_cabin(sample)
    new_data = sum_spendings(new_data)
    new_sample = sum_spendings(new_sample)
    new_data = new_data.set_index('PassengerId')
    new_sample = new_sample.set_index('PassengerId')
    new_data.to_csv('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_train.csv')
    new_sample.to_csv('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_test.csv')