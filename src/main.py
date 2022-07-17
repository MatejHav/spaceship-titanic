from econml.grf import RegressionForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
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
        all_predictions = []
        best_score = load_score()
        for _ in tqdm(range(number_of_cross)):
            x_train, x_test, y_train, y_test = split_data(processed_data)
            model = train_model(x_train, y_train, config)
            predictions = predict(model, x_test)
            score = compute_score(y_test, predictions)
            all_scores.append(score)
            all_predictions.append((score, predict(model, processed_sample)))
            if score > current_score:
                current_score = score
                sample_predictions = predict(model, processed_sample)
        all_scores = np.array(all_scores)
        score = all_scores.mean()
        std = all_scores.std()
        inside_string = "\033[1;32;40m" + str(score) + '\033[0m' if score > 0.8 else str(score)
        print(f'{config} | Score: {inside_string} | Std: {std}')
        if best_score + std - score < 0.02 * best_score:
            for computed_score, pred in all_predictions:
                if abs(computed_score - score - std) < abs(current_score - score - std):
                    current_score = computed_score
                    sample_predictions = pred
            save_predictions(sample, sample_predictions, current_score)


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
    # model = RegressionForest(n_estimators=config['n_of_trees'], min_samples_leaf=config['min_samples_leaf'],
    #                          max_depth=config['max_depth'], max_samples=config['max_samples'],
    #                          max_features=config['max_features'], subforest_size=config['subforest_size'])
    forests = []
    for i in range(config['n_of_estimators']):
        forests.append((f'grf_{i}', GradientBoostingClassifier(n_estimators=config['n_trees'], min_samples_leaf=config['min_samples_leaf'])))
    model = VotingClassifier(estimators=forests).fit(feature, label)
    return model


def predict(model, data):
    predictions = model.predict(data)
    return predictions


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
    for n_trees in [250]:
        for min_samples in [5]:
            config = {'n_trees': n_trees, 'min_samples_leaf': min_samples, 'n_of_estimators': 20}
            configurations.append(config)
    main(configurations)
