import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Verileri yükleme
data = pd.read_csv('C:/Users/ugurc/Desktop/energy.csv')

# Verileri özellikler ve etiketler olarak ayırma
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Verileri ölçeklendirme
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine verileri ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Derin öğrenme modeli oluşturma
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))

# Modeli derleme
model.compile(loss='mse', optimizer='adam')

# Early stopping ile modeli eğitme
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1, callbacks=[early_stop])

# Model performansını değerlendirme
mse = model.evaluate(X_test, y_test)
print('MSE: %.2f' % (mse))
