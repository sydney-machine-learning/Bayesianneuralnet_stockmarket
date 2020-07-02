# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import os
import os.path as osp
import shutil


def reset_folder(path):
    if not osp.exists(temp_dir):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def plot(path, ys, labels, title, xlabel, ylabel):
    x = np.linspace(0, ys[0].shape[0], num=ys[0].shape[0])
    for i, y in enumerate(ys):
        plt.plot(x, y, label=labels[i])
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=200)
    plt.clf()


def save(path, data):
    data = np.asarray(data)
    # print('data:', data)
    if data.shape[0] == 2:
        data = data.T
    np.savetxt(path, data, fmt='%1.5f')


class TestMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.test_metrics = {
            'mean_absolute_error': [],
            'mean_absolute_percentage_error': [],
            'indi_rmse_test': []
        }

    def on_epoch_end(self, batch, logs={}):
        fx_test = self.model.predict(test_x)
        #fx_train = self.model.predict(train_x)
        individual_rmse = [0] * steps_ahead
        individual_mae = [0] * steps_ahead
        individual_mape = [0] * steps_ahead

        for i in range(0, steps_ahead):
            individual_rmse[i] = np.sqrt(np.square(fx_test[:, i] - test_y[:, i]).mean())
            individual_mae[i]= np.abs(fx_test[:, i] - test_y[:, i]).mean()
            individual_mape[i]=np.abs((fx_test[:, i] - test_y[:, i]) / (test_y[:, i] + 1e-6)).mean()*100
        self.test_metrics['indi_rmse_test'].append(individual_rmse)
        self.test_metrics['mean_absolute_error'].append(individual_mae)
        self.test_metrics['mean_absolute_percentage_error'].append(individual_mape)


for d in os.listdir('./datasets/raw'):
    if osp.isfile(osp.join('datasets/raw', d)):
        name = osp.splitext(d)[0]
    data_path_base = 'datasets/' + name

    timesteps = 5
    steps_ahead = 5

    data_min = 0
    data_max = 1

    with open(data_path_base + '_train.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        train = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        train = np.asarray(train, np.float)

    with open(data_path_base + '_test.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        test = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        test = np.asarray(test, np.float)

    if steps_ahead == 1:
        result_dir = 'one_step_results/' + name
        temp_dir = 'one_step_problemfolder/' + name + '/nns_temp'
    else:
        result_dir = 'results/' + name
        temp_dir = 'problemfolder/' + name + '/nns_temp'

    reset_folder(temp_dir)

    assert timesteps + steps_ahead <= train.shape[-1]
    train_x = train[:, :timesteps]
    train_y = train[:, timesteps: timesteps + steps_ahead]

    test_x = test[:, :timesteps]
    test_y = test[:, timesteps: timesteps + steps_ahead]

    # FNN_adam
    FNN_adam = keras.Sequential()
    FNN_adam.add(layers.Dense(5, input_dim=timesteps, activation='relu'))

    FNN_adam.add(layers.Dense(steps_ahead, activation='sigmoid'))

    FNN_adam.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=[keras.metrics.RootMeanSquaredError(),
                 keras.metrics.mae,
                 keras.metrics.mape])
    fa_test_his = TestMetrics()
    history = FNN_adam.fit(train_x, train_y, epochs=500, batch_size=32)#callbacks=[fa_test_his])

    """path = temp_dir + '/{}_indi_mae_test.txt'.format('fa')
    save(path, fa_test_his.test_metrics['mean_absolute_error'])

    path = temp_dir + '/{}_indi_mape_test.txt'.format('fa')
    save(path, fa_test_his.test_metrics['mean_absolute_percentage_error'])

    path = temp_dir + '/{}_indi_rmse_test.txt'.format('fa')
    save(path, fa_test_his.test_metrics['indi_rmse_test'])"""

    fa_pred_test = FNN_adam.predict(test_x)
    n = len(test_y)
    fa_mae = sum(np.abs(test_y - fa_pred_test)) / n
    path = result_dir + '/{}_mae.txt'.format('fa')
    save(path, fa_mae)
    fa_rmse = np.sqrt(sum(np.square(test_y - fa_pred_test)) / n)
    path = result_dir + '/{}_rmse.txt'.format('fa')
    save(path, fa_rmse)
    fa_mape=sum(np.abs((test_y - fa_pred_test) / (test_y + 1e-6))) / n * 100
    path = result_dir + '/{}_mape.txt'.format('fa')
    save(path, fa_mape)
    path = result_dir + '/{}_pred_test.txt'.format('fa')
    fa_pred_test = fa_pred_test * (data_max-data_min) + data_min
    save(path, fa_pred_test)

    # FNN_sgd
    FNN_sgd = keras.Sequential()
    FNN_sgd.add(layers.Dense(256, input_dim=timesteps, activation='relu'))

    FNN_sgd.add(layers.Dense(steps_ahead, activation='sigmoid'))

    FNN_sgd.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='sgd',
        metrics=[keras.metrics.RootMeanSquaredError(),
                 keras.metrics.mae,
                 keras.metrics.mape])
    fs_test_his = TestMetrics()
    sgd_history = FNN_sgd.fit(train_x, train_y, epochs=500, batch_size=32)#callbacks=[fs_test_his],shuffle=True)

    """path = temp_dir + '/{}_indi_mae_test.txt'.format('fs')
    save(path, fs_test_his.test_metrics['mean_absolute_error'])

    path = temp_dir + '/{}_indi_mape_test.txt'.format('fs')
    save(path, fs_test_his.test_metrics['mean_absolute_percentage_error'])

    path = temp_dir + '/{}_indi_rmse_test.txt'.format('fs')
    save(path, fs_test_his.test_metrics['indi_rmse_test'])"""

    fs_pred_test = FNN_sgd.predict(test_x)

    n = len(test_y)
    fs_mae = sum(np.abs(test_y - fs_pred_test)) / n
    path = result_dir + '/{}_mae.txt'.format('fs')
    save(path, fs_mae)
    fs_rmse = np.sqrt(sum(np.square(test_y - fs_pred_test)) / n)
    path = result_dir + '/{}_rmse.txt'.format('fs')
    save(path, fs_rmse)
    fs_mape = sum(np.abs((test_y - fs_pred_test) / (test_y + 1e-6))) / n * 100
    path = result_dir + '/{}_mape.txt'.format('fs')
    save(path, fs_mape)
    path = result_dir + '/{}_pred_test.txt'.format('fs')
    fs_pred_test = fs_pred_test * (data_max - data_min) + data_min
    save(path, fs_pred_test)

    # LSTM
    """train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(timesteps, 1)))
    model.add(keras.layers.LSTM(64, return_sequences=False))
    model.add(keras.layers.Dense(steps_ahead, activation='sigmoid'))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=[keras.metrics.RootMeanSquaredError(),
                 keras.metrics.mae,
                 keras.metrics.mape])
    lstm_test_his = TestMetrics()
    lstm_history = model.fit(train_x, train_y, epochs=1000, batch_size=32, callbacks=[lstm_test_his])

    path = temp_dir + '/{}_indi_mae_test.txt'.format('lstm')
    save(path, lstm_test_his.test_metrics['mean_absolute_error'])

    path = temp_dir + '/{}_indi_mape_test.txt'.format('lstm')
    save(path, lstm_test_his.test_metrics['mean_absolute_percentage_error'])

    path = temp_dir + '/{}_indi_rmse_test.txt'.format('lstm')
    save(path, lstm_test_his.test_metrics['indi_rmse_test'])

    lstm_pred_test = model.predict(test_x)
    lstm_pred_test = lstm_pred_test * (data_max-data_min) + data_min


    path = result_dir + '/{}_pred_test.txt'.format('lstm')
    save(path, lstm_pred_test)"""

    # plot prediction
    """test_y = test_y * (data_max-data_min)+ data_min
    ptmcmc_pred_test = np.loadtxt(result_dir + '/ptmcmc_pred_test.txt')
    if steps_ahead == 1:
        plot(
            result_dir + '/all_pred_test.png',
            [test_y, ptmcmc_pred_test, fa_pred_test, fs_pred_test],
            ['Actual', 'ptmcmc', 'FNN_adam', 'FNN_sgd'],
            '{} one_step_ahead'.format(name),
            'time',
            'close price'
        )
    else:
        for i in [0, 1, 4]:
            plot(
                result_dir + '/all_pred_test_{}.png'.format(i + 1),
                [test_y[:, i], ptmcmc_pred_test[:, i], fa_pred_test[:, i], fs_pred_test[:, i]],
                ['Actual', 'ptmcmc', 'FNN_adam', 'FNN_sgd'],
                '{} five_steps_ahead_{}'.format(name, i + 1),
                'time',
                'close price'
            )

    # plot individual rmse
    if steps_ahead == 1:
        ptmcmc_indirmse_path = 'one_step_results/{}/ptmcmc_indi_rmse.txt'.format(name)
    else:
        ptmcmc_indirmse_path = 'results/{}/ptmcmc_indi_rmse.txt'.format(name)

    ptmcmc_indirmse_test = np.loadtxt(ptmcmc_indirmse_path)
    fa_indirmse_test = fa_rmse
    fs_indirmse_test = fs_rmse
    lstm_indirmse_test = np.asarray(lstm_test_his.test_metrics['indi_rmse_test'])[-1]

    x = np.linspace(1, steps_ahead, num=steps_ahead).astype(np.int)
    width = 0.1
    plt.bar(x, ptmcmc_indirmse_test, label='ptmcmc', width=width)
    plt.bar(x + width, fa_indirmse_test, label='FNN_Adam', width=width)
    plt.bar(x + width * 2, fs_indirmse_test, label='FNN_Sgd', width=width)
    plt.bar(x + width * 3, lstm_indirmse_test, label='LSTM', width=width)
    plt.legend(loc='upper right')
    plt.xlabel('step')
    plt.ylabel('mean RMSE')
    plt.title('{} mean of RMSE for 30 iterations'.format(name))
    plt.xticks(x + width, x)
    plt.savefig(result_dir + '/mean_rmse.png', dpi=200)
    plt.clf()"""
