import pandas
import argparse
import yaml
import numpy as np


class Digit(object):

    def __init__(self):
        pass

    def parse_args(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('-m', '--mode', choices=['train', 'test'])
        argparser.add_argument('-p', '--path')
        return argparser.parse_args()

    def parse_configuration(self):
        with open(file_name) as file:
            config = yaml.load(file)

    def load_samples(self, file_name):
        with open(file_name) as file:
            raw_data = pd.read.csv()
        y = np.array(raw_data.ix[:, 0])
        x = raw_data.ix[:, 1:].values
        return x, y

    def preprocess(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def run(self):
        # Parse args.
        args = self.parse_args()
        # Pars configuration
        configuration = self.parse_configuration(args.path)
        if args.mode == 'train':
            pass
        elif args.mode == 'test':
            pass
