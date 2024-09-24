from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('/content/sonar.all-data.csv')

df.head()

df.shape

df.info()

df.describe()

df.isnull()

df.isnull().sum()

df.groupby('Label').size()

import matplotlib.pyplot as plt
df.hist(figsize=(10,10), sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

df.columns

import numpy as np
a = np.mean(df[df['Label'] == 'R'].values[:, :-1], axis = 0)

b = np.mean(df[df['Label'] == 'M'].values[:, :-1], axis = 0)

plt.figure(figsize=(8,5))
plt.plot(a, label='Rock')
plt.plot(b, label='Metal')
plt.legend()
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()


df.corr(numeric_only=True)

import seaborn as sns
sns.heatmap(df.corr(numeric_only=True))
plt.show()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X

y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.3, random_state=42)

print("훈련 데이터 : ", X_train.shape, y_train.shape)
print("테스트 데이터 : ", X_test.shape, y_test.shape)

from sklearn.preprocessing import LabelEncoder

y_train_encode = LabelEncoder().fit_transform(y_train)
y_test_encode = LabelEncoder().fit_transform(y_test)

y_train_encode

from keras import Sequential
from keras.layers import Dense

def build_model():
  model = Sequential([
        Dense(100, input_shape=(60,), activation='relu'),
	    Dense(1, activation='sigmoid'),
      ])

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model = build_model()
model.summary()

history = model.fit(X_train, y_train_encode, epochs = 200)

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()

loss, acc = model.evaluate(X_train, y_train_encode)
print("Train Data loss:", loss)
print("Train Data accuracy:", acc)


loss, acc = model.evaluate(X_test, y_test_encode)
print("Test Data Test loss:", loss)
print("Test Data Test accuracy:", acc)


from keras import Sequential
from keras.layers import Dense

def build_model():
  model = Sequential([
        Dense(256, input_shape=(60,), activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
      ])

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model = build_model()
model.summary()

history = model.fit(X_train, y_train_encode, epochs = 200)

loss, acc = model.evaluate(X_train, y_train_encode)
print("Train Data Test loss:", loss)
print("Train Data Test accuracy:", acc)


loss, acc = model.evaluate(X_test, y_test_encode)
print("Test Data loss:", loss)
print("Test Data accuracy:", acc)

y_test_pred = model.predict(X_test)

print(y_test_pred[:10])


import tensorflow as tf
y_test_pred_encode = tf.greater(y_test_pred, .5)
print(y_test_pred_encode[:10])

from sklearn.metrics import classification_report

print('\n', classification_report(y_test_encode, y_test_pred_encode))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_encode, y_test_pred_encode)
print(cm)

sns.heatmap(cm, linewidths=1, cbar=False, annot=True, fmt='d')
plt.show()
