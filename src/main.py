from econml.grf import RegressionForest
from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *


def main(configurations):
    number_of_cross = 20
    data = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_train.csv')
    sample = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_test.csv')
    processed_data, processed_sample = preprocessing(data, sample)
    for config in configurations:
        sample_predictions = None
        current_score = 0
        all_scores = []
        best_score = load_score()
        for _ in tqdm(range(number_of_cross)):
            x_train, x_test, y_train, y_test = split_data(processed_data)
            model = train_model(x_train, y_train, config)
            predictions = predict(model, x_test)
            score = compute_score(y_test, predictions)
            all_scores.append(score)
            if score > current_score:
                current_score = score
                sample_predictions = predict(model, processed_sample)
        all_scores = np.array(all_scores)
        score = all_scores.mean()
        std = all_scores.std()
        inside_string = "\033[1;32;40m" + str(score) + '\033[0m' if score > 0.8 else str(score)
        print(f'{config} | Score: {inside_string} | Std: {std}')
        if best_score - score < 0.005 * best_score:
            save_predictions(sample, sample_predictions, score)


def load_score():
    best = 0
    for file in os.listdir('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/'):
        if 'submission' in file:
            score = float(file.split('_')[-1][:-4])
            if score > best:
                best = score
    return best


def load_data(filename):
    if '.csv' in filename:
        return pd.read_csv(filename)
    return np.fromfile(filename)


def preprocessing(data, sample):
    # Remove useless columns
    data.drop(['Name', 'Cabin'], axis=1, inplace=True)
    sample.drop(['Name', 'Cabin'], axis=1, inplace=True)
    # Replace discrete values with padnas dummies
    data = pd.get_dummies(data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
    sample = pd.get_dummies(sample, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
    # Remove all null columns
    data.fillna(method='ffill', inplace=True)
    sample.fillna(method='ffill', inplace=True)
    # # Standardize
    # data, sample = standardize(data, sample, 'Age')
    # data, sample = standardize(data, sample, 'RoomService')
    # data, sample = standardize(data, sample, 'FoodCourt')
    # data, sample = standardize(data, sample, 'ShoppingMall')
    # data, sample = standardize(data, sample, 'Spa')
    # data, sample = standardize(data, sample, 'VRDeck')
    # data, sample = standardize(data, sample, 'Nec')
    # data, sample = standardize(data, sample, 'Lux')

    # Normalize
    data, sample = normalize(data, sample, 'Age')
    data, sample = normalize(data, sample, 'RoomService')
    data, sample = normalize(data, sample, 'FoodCourt')
    data, sample = normalize(data, sample, 'ShoppingMall')
    data, sample = normalize(data, sample, 'Spa')
    data, sample = normalize(data, sample, 'VRDeck')
    data, sample = normalize(data, sample, 'Nec')
    data, sample = normalize(data, sample, 'Lux')
    return data, sample


def split_data(data, rate=0.3, state=70):
    return train_test_split(data.drop('Transported', axis=1), data['Transported'], test_size=rate)


def train_model(feature, label, config):
    model = RegressionForest(n_estimators=config['n_of_trees'], min_samples_leaf=config['min_samples_leaf'],
                             max_depth=config['max_depth'], max_samples=config['max_samples'],
                             max_features=config['max_features'], subforest_size=config['subforest_size'])
    model.fit(feature, label)
    return model


def predict(model, data):
    predictions = model.predict(data)
    return [p[0] >= 0.5 for p in predictions]


def compute_score(truth, predictions):
    iterator = truth.iteritems()
    score = 0
    for prediction in predictions:
        if prediction == next(iterator)[1]:
            score += 1
    return score / len(truth)


def save_predictions(sample, predictions, score):
    df = pd.DataFrame(predictions, index=sample['PassengerId'], columns=['Transported'])
    df.to_csv(f'C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/submission_{score}.csv')


if __name__ == '__main__':
    configurations = []
    for n_of_trees in [5000]:
        for min_sample_leaf in [5]:
            for max_features in [10]:
                for subforest_size in [4]:
                    config = {'n_of_trees': n_of_trees*subforest_size, 'min_samples_leaf': min_sample_leaf,
                              'max_depth': None, 'max_samples': 0.45,
                              'max_features': max_features, 'subforest_size': subforest_size}
                    configurations.append(config)
    main(configurations)
