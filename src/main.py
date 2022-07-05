from econml.grf import RegressionForest
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    best_score = load_score()
    data = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_train.csv')
    sample = load_data('C:/Users/Matej/Desktop/Interest/spaceship-titanic/data/new_test.csv')
    processed_data, processed_sample = preprocessing(data, sample)
    x_train, x_test, y_train, y_test = split_data(processed_data)
    model = train_model(x_train, y_train)
    predictions = predict(model, x_test)
    score = compute_score(y_test, predictions)
    print(f'Score: {score}')
    if score > best_score:
        sample_predictions = predict(model, processed_sample)
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
    print('PREPROCESSING...')
    # Remove useless columns
    data.drop(['Name', 'Cabin'], axis=1, inplace=True)
    sample.drop(['Name', 'Cabin'], axis=1, inplace=True)
    # Replace discrete values with padnas dummies
    data = pd.get_dummies(data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
    sample = pd.get_dummies(sample, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
    # Replace booleans with 0/1

    # Remove all null columns
    data.fillna(method='ffill', inplace=True)
    sample.fillna(method='ffill', inplace=True)
    return data, sample


def split_data(data, rate=0.25):
    return train_test_split(data.drop('Transported', axis=1), data['Transported'], test_size=rate)


def train_model(feature, label):
    print("TRAINING...")
    model = RegressionForest(n_estimators=200, min_samples_leaf=16, max_depth=10, max_samples=0.45)
    model.fit(feature, label)
    return model


def predict(model, data):
    predictions = model.predict(data)
    mean = np.mean(predictions)
    print(f'Mean prediction: {mean}')
    return [p[0] >= mean for p in predictions]


def compute_score(truth, predictions):
    print('SCORING...')
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
    main()
