import tensorflow as tf
import pandas_datareader.data as pdr
import yfinance as yf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

"""
start и end - период, на котором будет проходить обучение.
start_test и end_test - период, на котором будет проходить предсказание.
seq_len - количество предыдущих дневных цен (признаков), на основе которых делается предсказание цены дня.
""" 

start = "2003-01-01"
end = "2019-04-01"
#start_test = "2017-12-15"
#end_test = "2018-02-01"
start_test = "2019-03-19"
end_test = "2019-05-01"
ticker = "BMW.DE"
train_file_all = "stock_prices_all.csv"
test_file_all = "test_all.csv"
scal_file_all = "scal_all.csv"
seq_len = 10
x = []
x_all = []
x_1 = []
x_2 = []
x_3 = []
x_4 = []
x_5 = []
y = []
new_x = []
new_x_all = []
new_x_1 = []
new_x_2 = []
new_x_3 = []
new_x_4 = []
new_x_5 = []
new_y = []
scal_x = []
scal_x_all = []
scal_x_1 = []
scal_x_2 = []
scal_x_3 = []
scal_x_4 = []
scal_x_5 = []
scal_y = []

yf.pdr_override()


def get_stock_data_all(ticker, start_date, end_date, file):
    """
    Получает исторические данные по дневным ценам акций между датами
    :param ticker: компания или компании, чьи данные должны быть извлечены
    :type ticker: string or list of strings
    :param start_date: начальная дата получения цен на акции
    :type start_date: string of date "YYYY-mm-dd"
    :param end_date: конечная дата получения цен на акции
    :type end_date: string of date "YYYY-mm-dd"
    :param file: имя возвращаемого файла с данными
    :return: файл формата csv
    """
    i = 1
    all_data = 0
    while i > 0:
        try:
            all_data = yf.download(ticker, start_date, end_date)
            i = 0
        except ValueError:
            i += 1
            if i < 5:
                print("ValueError, trying again")
                time.sleep(10)
            else:
                print("Tried 5 times, Yahoo error. Trying after 2 minutes")
                i = 1
                time.sleep(120)
    print("download ok!")
    all_data.to_csv(file)

def get_X_Y_all(data, seq_len, list_x, list_x_all, list_x_1, list_x_2, list_x_3, list_x_4, list_x_5, list_y):
    """
    Преобразует данные, разбивая на признаки и ответы.
    :param data: исходный массив данных
    :param seq_len: количество признаков
    :param list_x: список, в который добавляются признаки
    :param list_y: список, в который добавляются ответы
    """
    for i in range(len(data) - seq_len):
        i1 = np.array(data.iloc[i: i + seq_len, 1])
        i2 = np.array([data.iloc[i + seq_len, 1]], np.float64)
        i3 = np.array(data.iloc[i: i + seq_len, 1:])
        i4 = np.array(data.iloc[i: i + seq_len, lambda df: [1, 3, 4, 5, 6]])
        i5 = np.array(data.iloc[i: i + seq_len, lambda df: [1, 2, 4, 5, 6]])
        i6 = np.array(data.iloc[i: i + seq_len, lambda df: [1, 2, 3, 5, 6]])
        i7 = np.array(data.iloc[i: i + seq_len, lambda df: [1, 2, 3, 4, 6]])
        i8 = np.array(data.iloc[i: i + seq_len, lambda df: [1, 2, 3, 4, 5]])
        list_x.append(i1)
        list_x_all.append(i3)
        list_x_1.append(i4)
        list_x_2.append(i5)
        list_x_3.append(i6)
        list_x_4.append(i7)
        list_x_5.append(i8)
        list_y.append(i2)

#Получим три массива данных - тренировочный, для предсказания и для настройки скалера.

get_stock_data_all(ticker, start, end, train_file_all)
get_stock_data_all(ticker, start_test, end_test, test_file_all)
get_stock_data_all(ticker, start, end_test, scal_file_all)

#Разобьем полученные данные на признаки и ответы.

data = pd.read_csv(train_file_all, encoding='utf-8')
get_X_Y_all(data, seq_len, x, x_all, x_1, x_2, x_3, x_4, x_5, y)

new_data = pd.read_csv(test_file_all, encoding='utf-8')
get_X_Y_all(new_data, seq_len, new_x, new_x_all, new_x_1, new_x_2, new_x_3, new_x_4, new_x_5, new_y)

scal_data = pd.read_csv(scal_file_all, encoding='utf-8')
get_X_Y_all(scal_data, seq_len, scal_x, scal_x_all, scal_x_1, scal_x_2, scal_x_3, scal_x_4, scal_x_5, scal_y)

#Преобразуем данные в формат, используемый в нейронной сети.

x = np.array(x)
x_all = np.array(x_all)
x_1 = np.array(x_1)
x_2 = np.array(x_2)
x_3 = np.array(x_3)
x_4 = np.array(x_4)
x_5 = np.array(x_5)
y = np.array(y)
new_x = np.array(new_x)
new_x_all = np.array(new_x_all)
new_x_1 = np.array(new_x_1)
new_x_2 = np.array(new_x_2)
new_x_3 = np.array(new_x_3)
new_x_4 = np.array(new_x_4)
new_x_5 = np.array(new_x_5)
new_y = np.array(new_y)
scal_x = np.array(scal_x)
scal_x_all = np.array(scal_x_all)
scal_x_1 = np.array(scal_x_1)
scal_x_2 = np.array(scal_x_2)
scal_x_3 = np.array(scal_x_3)
scal_x_4 = np.array(scal_x_4)
scal_x_5 = np.array(scal_x_5)
scal_y = np.array(scal_y)



#Произведем скалинг данных

X = [x_all, x_1, x_2, x_3, x_4, x_5]
NEW_X = [new_x_all, new_x_1, new_x_2, new_x_3, new_x_4, new_x_5]

scalers = {}
for i in range(scal_x_all.shape[2]):
    scalers[i] = MinMaxScaler(feature_range = (0, 1))
    scalers[i].fit_transform(scal_x_all[:, :, i])
    scal_x_all[:, :, i] = scalers[i].transform(scal_x_all[:, :, i])
    if i == 0:
        x = scalers[i].transform(x)
        new_x = scalers[i].transform(new_x)
        for t in range(6):
            X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
            NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])
    else:
        for t in range(6):
            if t == 0:
                X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
                NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])
            else:
                if t < i:
                    X[t][:, :, i - 1] = scalers[i].transform(X[t][:, :, i - 1])
                    NEW_X[t][:, :, i - 1] = scalers[i].transform(NEW_X[t][:, :, i - 1])
                if t > i:
                    X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
                    NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])
                    
scaler_y = MinMaxScaler(feature_range = (0, 1))
scaler_y.fit_transform(scal_y)
y = scaler_y.transform(y)


scal_x_all_pca = scal_x_all[:, :, 1:5]
scal_x_all_pca.shape = (scal_x_all_pca.shape[0] * scal_x_all_pca.shape[1], 4)
pca = PCA(n_components = 1)
scal_XPCAreduced = pca.fit_transform(scal_x_all_pca)

x_all_pca = np.zeros((len(x_all),10,3))
for i in range(len(x_all)):
    z = np.zeros((10,3))
    for j in range(len(z)):
        z[j][0] = x_all[i, j, 0]
        z[j][1] = pca.transform(x_all[i, j, 1:5].reshape(1, -1))
        z[j][2] = x_all[i, j, 5]
    x_all_pca[i] = z

x_6 = np.zeros((len(x_all_pca),10,2))
for i in range(len(x_all_pca)):
    for j in range(10):
        x_6[i][j][0] = x_all_pca[i, j, 0]
        x_6[i][j][1] = x_all_pca[i, j, 2]

x_pca = x_all_pca[:, :, :2]

new_x_all_pca = np.zeros((len(new_x_all),10,3))
for i in range(len(new_x_all)):
    z = np.zeros((10,3))
    for j in range(len(z)):
        z[j][0] = new_x_all[i, j, 0]
        z[j][1] = pca.transform(new_x_all[i, j, 1:5].reshape(1, -1))
        z[j][2] = new_x_all[i, j, 5]
    new_x_all_pca[i] = z

new_x_6 = np.zeros((len(new_x_all_pca),10,2))
for i in range(len(new_x_all_pca)):
    for j in range(10):
        new_x_6[i][j][0] = new_x_all_pca[i, j, 0]
        new_x_6[i][j][1] = new_x_all_pca[i, j, 2]

new_x_pca = new_x_all_pca[:, :, :2]

#Разобьем тренировочные данные на обучение и проверку в соотношении 9 к 1 и перемешаем их.

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)
X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(x_all, y, test_size=0.1, random_state=42, shuffle=True)
X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(x_1, y, test_size=0.1, random_state=42, shuffle=True)
X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(x_2, y, test_size=0.1, random_state=42, shuffle=True)
X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(x_3, y, test_size=0.1, random_state=42, shuffle=True)
X_train_4, X_valid_4, y_train_4, y_valid_4 = train_test_split(x_4, y, test_size=0.1, random_state=42, shuffle=True)
X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(x_5, y, test_size=0.1, random_state=42, shuffle=True)
X_train_6, X_valid_6, y_train_6, y_valid_6 = train_test_split(x_6, y, test_size=0.1, random_state=42, shuffle=True)
X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(x_pca, y, test_size=0.1, random_state=42, shuffle=True)
X_train_all_pca, X_valid_all_pca, y_train_all_pca, y_valid_all_pca = train_test_split(x_all_pca, y, test_size=0.1, random_state=42, shuffle=True)

"""
Настроим нейтронную сеть.
Модель Sequential представляет собой линейный стек слоев.
Создадим модель Sequential, передав в конструктор список экземпляров слоя.
Для первых двух слоев используем MLP (multilayer perceptron) - многослойный перцептрон.
Полностью связанные слои определяются с помощью класса Density. Мы можем указать количество нейронов или узлов в слое в качестве первого аргумента, а также указать функцию активации, используя аргумент активации.
Мы будем использовать функцию активации ReLU (rectified linear unit).
Выходной слой также будет состоять из 1 нейрона с функцией активации ReLU. Функция активации определяет выходное значение нейрона. Выходной нейрон также оставим без нелинейности, чтобы иметь возможность прогнозировать любое значение.
Раньше считалось, что Sigmoid и Tanh активационные функции были предпочтительны для всех слоев. В наши дни более высокая производительность достигается с помощью функции активации ReLU.
"""

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

"""
Перед обучением модели необходимо настроить процесс обучения, что делается с помощью метода компиляции. Он получает два аргумента:
1) Оптимизатор. Используем Adam, алгоритм градиентной оптимизации стохастических целевых функций первого порядка, основанный на адаптивных оценках моментов более низкого порядка. Этот метод прост в реализации, вычислительно эффективен, имеет небольшие требования к памяти, инвариантен к диагональному масштабированию градиентов и хорошо подходит для задач, которые являются большими с точки зрения данных и/или параметров. Этот метод также подходит для нестационарных задач и задач с очень шумными и / или разреженными градиентами. Эмпирические результаты показывают, что Адам хорошо работает на практике и выгодно отличается от других методов стохастической оптимизации.
2) Функция потерь. Это та цель, которую модель постарается свести к минимуму. Используем среднеквадратическую ошибку (mean squared error, MSE). Физического смысла MSE не имеет, но чем ближе к нулю, тем модель лучше.
"""

#model.compile(optimizer="adam", loss="mean_squared_error")

"""
Обучим модель, используя 100 эпох.
Эпоха - один проход по всему набору данных, используемый для разделения обучения на отдельные фазы, что полезно для ведения логов и периодической оценки.
"""

#model.fit(X_train, y_train, epochs=100)

#print(model.evaluate(X_valid, y_valid))

#Предскажем цены с помощью обученной модели и произведем обратный скалинг.

#y1 = scaler_y.inverse_transform(model.predict(X_valid))
#y2 = scaler_y.inverse_transform(model.predict(new_x))
#y3 = scaler_y.inverse_transform(y_valid)

#print(y2)

#Запишем данные в файлы, для дальнейшего сравнения и построения графиков в другой части программы.

#np.save('MLP_y_pred', y1)
#np.save('MLP_new_y', y2)
#np.save('MLP_y', y3)

X_train_all = np.array([x.flatten() for x in X_train_all])
X_valid_all = np.array([x.flatten() for x in X_valid_all])
new_x_all = np.array([x.flatten() for x in new_x_all])
X_train_1 = np.array([x.flatten() for x in X_train_1])
X_valid_1 = np.array([x.flatten() for x in X_valid_1])
new_x_1 = np.array([x.flatten() for x in new_x_1])
X_train_2 = np.array([x.flatten() for x in X_train_2])
X_valid_2 = np.array([x.flatten() for x in X_valid_2])
new_x_2 = np.array([x.flatten() for x in new_x_2])
X_train_3 = np.array([x.flatten() for x in X_train_3])
X_valid_3 = np.array([x.flatten() for x in X_valid_3])
new_x_3 = np.array([x.flatten() for x in new_x_3])
X_train_4 = np.array([x.flatten() for x in X_train_4])
X_valid_4 = np.array([x.flatten() for x in X_valid_4])
new_x_4 = np.array([x.flatten() for x in new_x_4])
X_train_5 = np.array([x.flatten() for x in X_train_5])
X_valid_5 = np.array([x.flatten() for x in X_valid_5])
new_x_5 = np.array([x.flatten() for x in new_x_5])
X_train_6 = np.array([x.flatten() for x in X_train_6])
X_valid_6 = np.array([x.flatten() for x in X_valid_6])
new_x_6 = np.array([x.flatten() for x in new_x_6])
X_train_pca = np.array([x.flatten() for x in X_train_pca])
X_valid_pca = np.array([x.flatten() for x in X_valid_pca])
new_x_pca = np.array([x.flatten() for x in new_x_pca])
X_train_all_pca = np.array([x.flatten() for x in X_train_all_pca])
X_valid_all_pca = np.array([x.flatten() for x in X_valid_all_pca])
new_x_all_pca = np.array([x.flatten() for x in new_x_all_pca])


#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
#model.compile(optimizer="adam", loss="mean_squared_error")

#model.fit(X_train_all, y_train_all, epochs=100)
#print(model.evaluate(X_valid_all, y_valid_all))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_all))
#y2 = scaler_y.inverse_transform(model.predict(new_x_all))
#print(y2)
#np.save('MLP_y_pred_all', y1)
#np.save('MLP_new_y_all', y2)

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
#model.compile(optimizer="adam", loss="mean_squared_error")

#model.fit(X_train_1, y_train_1, epochs=100)
#print(model.evaluate(X_valid_1, y_valid_1))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_1))
#y2 = scaler_y.inverse_transform(model.predict(new_x_1))
#print(y2)
#np.save('MLP_y_pred_1', y1)
#np.save('MLP_new_y_1', y2)

#model.fit(X_train_2, y_train_2, epochs=100)
#print(model.evaluate(X_valid_2, y_valid_2))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_2))
#y2 = scaler_y.inverse_transform(model.predict(new_x_2))
#print(y2)
#np.save('MLP_y_pred_2', y1)
#np.save('MLP_new_y_2', y2)

#model.fit(X_train_3, y_train_3, epochs=100)
#print(model.evaluate(X_valid_3, y_valid_3))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_3))
#y2 = scaler_y.inverse_transform(model.predict(new_x_3))
#print(y2)
#np.save('MLP_y_pred_3', y1)
#np.save('MLP_new_y_3', y2)

#model.fit(X_train_4, y_train_4, epochs=100)
#print(model.evaluate(X_valid_4, y_valid_4))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_4))
#y2 = scaler_y.inverse_transform(model.predict(new_x_4))
#print(y2)
#np.save('MLP_y_pred_4', y1)
#np.save('MLP_new_y_4', y2)

#model.fit(X_train_5, y_train_5, epochs=100)
#print(model.evaluate(X_valid_5, y_valid_5))
#y1 = scaler_y.inverse_transform(model.predict(X_valid_5))
#y2 = scaler_y.inverse_transform(model.predict(new_x_5))
#print(y2)
#np.save('MLP_y_pred_5', y1)
#np.save('MLP_new_y_5', y2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train_all_pca, y_train_all_pca, epochs=100)
print(model.evaluate(X_valid_all_pca, y_valid_all_pca))
y1 = scaler_y.inverse_transform(model.predict(X_valid_all_pca))
y2 = scaler_y.inverse_transform(model.predict(new_x_all_pca))
print(y2)
np.save('MLP_y_pred_all_pca', y1)
np.save('MLP_new_y_all_pca', y2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train_pca, y_train_pca, epochs=100)
print(model.evaluate(X_valid_pca, y_valid_pca))
y1 = scaler_y.inverse_transform(model.predict(X_valid_pca))
y2 = scaler_y.inverse_transform(model.predict(new_x_pca))
print(y2)
np.save('MLP_y_pred_pca', y1)
np.save('MLP_new_y_pca', y2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train_6, y_train_6, epochs=100)
print(model.evaluate(X_valid_6, y_valid_6))
y1 = scaler_y.inverse_transform(model.predict(X_valid_6))
y2 = scaler_y.inverse_transform(model.predict(new_x_6))
print(y2)
np.save('MLP_y_pred_6', y1)
np.save('MLP_new_y_6', y2)
