import pickle
import argparse
import yaml
import numpy as np
import cnn_model
import cv2
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

class Digit(object):

    def __init__(self):
        # Configuration
        self.base_config = None
        self.model_config = None
        # Model
        self.model = None

    def parse_args(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('-m', '--mode', choices=['train', 'predict'])
        argparser.add_argument('-p', '--path')
        return argparser.parse_args()

    def parse_configuration(self, file_name):
        with open(file_name) as file:
            config = yaml.load(file)
            self.base_config = config['base']
            self.model_config = config[self.base_config['model_type']]

    def load_samples(self, file_name):
        with open(file_name, 'rb') as file:
            raw_data = pickle.load(file)
        x = raw_data['data']
        # Check if label is None
        if raw_data['label'] is not None:
            y = np.array(raw_data['label'])
        else:
            y = None
        return x, y

    def preprocess(self, x, y, mode):
        if mode == 'train':
            # One hot encode
            if self.base_config['y_do_one_hot_encode'] is True:
                self.onehot_encoder = OneHotEncoder(sparse=False)
                y = y.reshape(-1,1)
                y = self.onehot_encoder.fit_transform(y)
        # Normalize
        x = x / 255
        return x, y

    def create_model(self):
        # Create model by model type and pass model config
        model_type = self.base_config['model_type']
        if model_type == 'cnn_model':
            self.model = cnn_model.CNNModel(self.model_config)

    def load_trained_model(self):
        model_type = self.base_config['model_type']
        if model_type == 'cnn_model':
            self.model = cnn_model.CNNModel(self.model_config)
            self.model.load_trained_model()

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)
        print('--> Finished training!')

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict

    def run(self):
        # Parse args.
        args = self.parse_args()
        # Pars configuration
        self.parse_configuration(args.path)
        # Mode: Train
        if args.mode == 'train':
            x_train, y_train = self.load_samples(self.base_config['dataset_path'] + 'train.pkl')
            # Preprocess
            x_train, y_train = self.preprocess(x_train, y_train, 'train')
            # Build model
            self.create_model()
            # Train
            self.train(x_train, y_train)
        # Mode: Test
        elif args.mode == 'predict':
            x_test, y_test = self.load_samples(self.base_config['dataset_path'] + 'test.pkl')
            # Preprocess
            x_test, y_test = self.preprocess(x_test, y_test, 'test')
            #
            if self.model is not None:
                y_predict = self.predict(x_test)
            elif self.model is None:
                self.load_trained_model()
                y_predict = self.predict(x_test)
            if y_test is not None:
                input(type(y_test))
                print(self.base_config['label_names'])
                print(classification_report(y_test, y_predict, target_names=self.base_config['label_names']))
                print(confusion_matrix(y_test, y_predict, labels=range(10)))
            # Make a submission.
            if self.base_config['make_submission'] is True:
                submission = pd.DataFrame({'ImageId': range(1, len(y_predict) + 1), 'Label': y_predict}, columns=['ImageId', 'Label'], index=None)
                with open(self.base_config['submission_path'], 'w') as file:
                    submission.to_csv(file, index=False)


if __name__ == '__main__':
    digit = Digit()
    digit.run()
