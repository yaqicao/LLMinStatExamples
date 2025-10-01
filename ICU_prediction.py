import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

time = np.arange(0, 1200, 1)
amplitude = 15
base_value = 85
raw_signal = base_value + amplitude * np.sin(time * 0.1) + 5 * np.sin(time * 0.5)

noise = np.random.normal(0, 3, len(time))
blood_pressure_data = raw_signal + noise
dataset = blood_pressure_data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(dataset)

train_size = int(len(scaled_dataset) * 0.80)
test_size = len(scaled_dataset) - train_size
train_data, test_data = scaled_dataset[0:train_size,:], scaled_dataset[train_size:len(scaled_dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 30
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)


trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=32, verbose=1)

train_predict = model.predict(trainX)
test_predict = model.predict(testX)

train_predict = scaler.inverse_transform(train_predict)
trainY_orig = scaler.inverse_transform([trainY])
test_predict = scaler.inverse_transform(test_predict)
testY_orig = scaler.inverse_transform([testY])

train_predict_plot = np.empty_like(scaled_dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

test_predict_plot = np.empty_like(scaled_dataset)
test_predict_plot[:, :] = np.nan
test_start_index = len(train_predict) + (look_back * 2) + 1
test_predict_plot[test_start_index:len(scaled_dataset)-1, :] = test_predict


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  
plt.figure(figsize=(15,7))
plt.plot(scaler.inverse_transform(scaled_dataset), label='真实值', color='C0', alpha=0.6)
plt.plot(train_predict_plot, label='训练集预测', color='red')
plt.plot(test_predict_plot, label='测试集预测', color='green')
plt.axvline(x=train_size, c='grey', linestyle='--', label='训练/测试分割')
plt.title('模拟的ICU血压的LSTM时间序列预测')
plt.xlabel('时间点')
plt.ylabel('模拟血压值 (mmHg)')
plt.legend()
plt.grid(True)
plt.savefig("icu_prediction_plot.svg", format="svg", bbox_inches='tight')
plt.show()