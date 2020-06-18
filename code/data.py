# -*- coding: utf-8 -*-
import os
import os.path as osp
import pandas as pd

import numpy as np


class Dataset(object):
    def __init__(self, name, train_ratio, timesteps, features, steps_ahead, interval=1, overwrite=False):
        self.name = name
        self.train_ratio = train_ratio
        self.timesteps = timesteps
        self.features = features
        self.steps_ahead = steps_ahead
        self.interval = interval
        self.train_path = './datasets/{}_train.txt'.format(name)
        self.test_path = './datasets/{}_test.txt'.format(name)
        self.raw_path = './datasets/raw/{}.csv'.format(name)
        if osp.exists(self.train_path) and osp.exists(self.test_path) and not overwrite:
            self.train = self._load_file(self.train_path)
            self.test = self._load_file(self.test_path)
        else:
            self.train, self.test = self._preprocess()
            self._save()

    def prepare_train_test(self):
        steps_ahead = self.steps_ahead
        train = np.asarray(self.train)
        test = np.asarray(self.test)
        return train[:, :self.timesteps], train[:, self.timesteps:self.timesteps + steps_ahead], \
            test[:, :self.timesteps], test[:, self.timesteps:self.timesteps + steps_ahead]

    def _load_file(self, file_path):
        ext = osp.splitext(file_path)[-1]
        if ext == '.txt':
            with open(file_path, 'r') as f:
                data = f.readlines()
                self.min, self.max = list(map(lambda x: float(x), data[0].split(' ')))
                data = list(map(lambda l: l.strip().split(' '), data[1:]))
                data = np.asarray(data, np.float)
        else:
            raw_data = pd.read_csv(file_path)['Close']
            self.min = raw_data.min()
            self.max = raw_data.max()
            data = raw_data.apply(lambda x: (x - raw_data.min()) / (raw_data.max() - raw_data.min()))
        return data

    def _preprocess(self):
        raw_data = self._load_file(self.raw_path).tolist()
        num_train = int(len(raw_data) * self.train_ratio)
        print('total: {}, train: {}, test: {}.'.format(len(raw_data), num_train, len(raw_data) - num_train))
        train, test = [], []
        for i in range(0, len(raw_data) - self.timesteps - self.steps_ahead, self.interval):
            sample = raw_data[i: i + self.timesteps + self.steps_ahead]
            if i + 1 < num_train:
                train.append(sample)
            else:
                test.append(sample)
        return train, test

    def _save(self):
        content = [self.train, self.test]
        for i, path in enumerate([self.train_path, self.test_path]):
            with open(path, 'w') as f:
                f.write('{:.6f} {:.6f}\n'.format(self.min, self.max))
                for sample in content[i]:
                    f.write(' '.join(map(lambda x: '{:.6f}'.format(x), sample)) + '\n')


if __name__ == '__main__':
    for d in os.listdir('./datasets/raw'):
        if osp.isfile(osp.join('./datasets/raw', d)):
            dataset_name = osp.splitext(d)[0]
            dataset = Dataset(dataset_name, 0.8, 5, 1, 5, 2, overwrite=True)
