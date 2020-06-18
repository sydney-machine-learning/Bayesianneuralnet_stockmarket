import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def rmse_value(y_true, y_pred):
    n = len(y_true)
    rmse = np.sqrt(sum(np.square(y_true - y_pred)) / n)
    return rmse

def mape_value(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / (y_true + 1e-6))) / n * 100
    return mape

def inverse_min_max(X, min, max):
    return X * (max - min) + min



def MODEL_FNN_adam(train_x, test_x, train_y, test_y,timesteps,steps_ahead,min,max,name):
    FNN_adam = keras.Sequential()
    FNN_adam.add(layers.Dense(5, input_dim=timesteps, activation='relu'))
    FNN_adam.add(layers.Dense(steps_ahead, activation='sigmoid'))
    FNN_adam.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )
    FNN_adam.fit(train_x, train_y, epochs=1000, batch_size=32)
    yhat11 = FNN_adam.predict(test_x, verbose=0)
    test_y = inverse_min_max(test_y, min, max)
    yhat11 = inverse_min_max(yhat11, min, max)

    plt.plot(test_y, c="g", label='real')
    plt.plot(yhat11, c="r", label='predict')
    plt.legend(loc='upper right')
    plt.savefig(f'{steps_ahead}_{name}_adam_pred.png')
    plt.clf()

    mae = mae_value(test_y, yhat11)
    rmse = rmse_value(test_y, yhat11)
    mape = mape_value(test_y, yhat11)
    return rmse, mae, mape

def MODEL_FNN_sgd(train_x, test_x, train_y, test_y,timesteps,steps_ahead,min,max,name):
    FNN_sgd = keras.Sequential()
    FNN_sgd.add(layers.Dense(5, input_dim=timesteps, activation='relu'))
    FNN_sgd.add(layers.Dense(steps_ahead, activation='sigmoid'))
    FNN_sgd.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='sgd',
        metrics=['accuracy']
    )
    FNN_sgd.fit(train_x, train_y, epochs=1000, batch_size=32)
    yhat11 = FNN_sgd.predict(test_x, verbose=0)
    test_y = inverse_min_max(test_y, min, max)
    yhat11 = inverse_min_max(yhat11, min, max)

    plt.plot(test_y, c="g", label='real')
    plt.plot(yhat11, c="r", label='predict')
    plt.legend(loc='upper right')
    plt.savefig(f'{steps_ahead}_{name}_sgd_pred.png')
    plt.clf()

    mae = mae_value(test_y, yhat11)
    rmse = rmse_value(test_y, yhat11)
    mape = mape_value(test_y, yhat11)
    return rmse, mae, mape

def MODEL_LSTM(train_x, test_x, train_y, test_y,timesteps,steps_ahead,min,max,name):
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(timesteps, 1)))
    model.add(keras.layers.LSTM(100, return_sequences=False))
    model.add(keras.layers.Dense(steps_ahead, activation='sigmoid'))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(train_x, train_y, epochs=1000, batch_size=32)
    yhat11 = model.predict(test_x, verbose=0)
    test_y = inverse_min_max(test_y, min, max)
    yhat11 = inverse_min_max(yhat11, min, max)
    plt.plot(test_y, c="g", label='real')
    plt.plot(yhat11, c="r", label='predict')
    plt.legend(loc='upper right')
    plt.savefig(f'{steps_ahead}_{name}_lstm_pred.png')
    plt.clf()
    mae = mae_value(test_y, yhat11)
    rmse = rmse_value(test_y, yhat11)
    mape = mape_value(test_y, yhat11)
    return rmse, mae, mape

problem=2
if problem==1:
    name = 'ACFINANCE'
    with open('./datasets/ACFINANCE_train.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        train = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        train = np.asarray(train, np.float)
    with open('./datasets/ACFINANCE_test.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        test = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        test = np.asarray(test, np.float)
    with open('./results/ACFINANCE/ptmcmc_pred_test.txt', 'r') as f:
        pred_ptmcmc = f.readlines()
        rmse_ptmcmc = list(map(lambda l: l.strip().split(' '), pred_ptmcmc[:]))
        rmse_ptmcmc = np.asarray(rmse_ptmcmc, np.float)
elif problem==2:
    name='AAL'
    with open('./datasets/AAL_train.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        train = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        train = np.asarray(train, np.float)
    with open('./datasets/AAL_test.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        test = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        test = np.asarray(test, np.float)
    with open('./results/AAL/ptmcmc_pred_test.txt', 'r') as f:
        pred_ptmcmc = f.readlines()
        rmse_ptmcmc = list(map(lambda l: l.strip().split(' '), pred_ptmcmc[:]))
        rmse_ptmcmc = np.asarray(rmse_ptmcmc, np.float)
elif problem==3:
    name='TWTR'
    with open('./datasets/TWTR_train.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        train = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        train = np.asarray(train, np.float)
    with open('./datasets/TWTR_test.txt', 'r') as f:
        raw_data = f.readlines()
        data_min, data_max = list(map(lambda x: float(x), raw_data[0].split(' ')))
        test = list(map(lambda l: l.strip().split(' '), raw_data[1:]))
        test = np.asarray(test, np.float)
    with open('./results/TWTR/ptmcmc_pred_test.txt', 'r') as f:
        pred_ptmcmc = f.readlines()
        rmse_ptmcmc = list(map(lambda l: l.strip().split(' '), pred_ptmcmc[:]))
        rmse_ptmcmc = np.asarray(rmse_ptmcmc, np.float)

timesteps = 5
steps_ahead = 5
train_x = train[:, :timesteps]
train_y = train[:, timesteps: timesteps + steps_ahead]
test_x = test[:, :timesteps]
test_y = test[:, timesteps: timesteps + steps_ahead]

rmse_FNNa, mae_FNNa, mape_FNNa=MODEL_FNN_adam(train_x, test_x, train_y, test_y,timesteps,steps_ahead,data_min,data_max,name)
rmse_FNNs, mae_FNNs, mape_FNNs=MODEL_FNN_sgd(train_x, test_x, train_y, test_y,timesteps,steps_ahead,data_min,data_max,name)
rmse_lstm, mae_lstm, mape_Flstm=MODEL_LSTM(train_x, test_x, train_y, test_y,timesteps,steps_ahead,data_min,data_max,name)


y_true=inverse_min_max(test_y,data_min,data_max)
rmse_ptmcmc=rmse_value(y_true,rmse_ptmcmc)
print(rmse_ptmcmc)
print(rmse_FNNa)
print(rmse_FNNs)
print(rmse_lstm)

barWidth = 0.15
bars1 = rmse_ptmcmc
bars2 = rmse_FNNa
bars3 = rmse_FNNs
bars4 = rmse_lstm
ra1 = np.arange(len(bars1)) + 1
ra2 = [y + barWidth for y in ra1]
ra3 = [y + barWidth for y in ra2]
ra4 = [y + barWidth for y in ra3]
plt.bar(ra1, bars1, width=barWidth, color='red', edgecolor='black',
        capsize=7, label='ptmcmc')
plt.bar(ra2, bars2, width=barWidth, color='blue', edgecolor='black',
        capsize=7, label='FNN_ADAM')
plt.bar(ra3, bars3, width=barWidth, color='yellow', edgecolor='black',
        capsize=7, label='FNN_SDG')
plt.bar(ra4, bars4, width=barWidth, color='green', edgecolor='black',
        capsize=7, label='LSTM')

plt.ylabel('RMSE')
plt.xlabel('Number of steps')
plt.legend(loc='upper right', fontsize=8)
plt.savefig(f'{name}_allrmse.png')
plt.show()



