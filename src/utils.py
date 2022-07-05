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

def load_data(filename):
    if '.csv' in filename:
        return pd.read_csv(filename)
    return np.fromfile(filename)

if __name__ == "__main__":
    data = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/train.csv')
    sample = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/test.csv')
    new_data = split_cabin(data)
    new_sample = split_cabin(sample)
    new_data = new_data.set_index('PassengerId')
    new_sample = new_sample.set_index('PassengerId')
    new_data.to_csv('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_train.csv')
    new_sample.to_csv('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_test.csv')