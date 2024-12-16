# -*- coding: utf-8 -*-
"""Sonar_test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/171Kxzz1Mp2eEKbdkrch6RFatG_Lr9PBj

##**[1] 데이터 불러오기**

1.1 파일 업로드하기
"""

from google.colab import files
uploaded = files.upload()

"""1.2 데이터프레임 생성하기"""

import pandas as pd
df = pd.read_csv('/content/sonar.all-data.csv')

df.head()

"""## **[2] 탐색적 데이터 분석 및 전처리하기**

2.1 전체적인 데이터 살펴보기
"""

df.shape

df.info()

df.describe()

"""2.2 결측치 확인하기"""

df.isnull()

df.isnull().sum()

"""2.3 기뢰와 바위 수 확인하기"""

df.groupby('Label').size()

"""2.4 주파수별 시각화하기"""

import matplotlib.pyplot as plt
df.hist(figsize=(10,10), sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

"""2.5 주파수별 데이터의 수 시각화하기"""

df.columns

import numpy as np
a = np.mean(df[df['Label'] == 'R'].values[:, :-1], axis = 0)

b = np.mean(df[df['Label'] == 'M'].values[:, :-1], axis = 0)

plt.figure(figsize=(8,5))
plt.plot(a, label='Rock')
plt.plot(b, label='Mine')
plt.legend()
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

"""2.6 주파수별 상관관계 파악하기"""

df.corr(numeric_only=True)

import seaborn as sns
sns.heatmap(df.corr(numeric_only=True))
plt.show()

"""2.7 특징과 타깃 선정하기"""

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X

y

"""2.8 훈련 데이터, 테스트 데이터 분할하기"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.3, random_state=42)

print("훈련 데이터 : ", X_train.shape, y_train.shape)
print("테스트 데이터 : ", X_test.shape, y_test.shape)

"""2.9 카테고리형 문자열을 수치형으로 변환하기"""

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encode = label_encoder.fit_transform(y_train)
y_test_encode = label_encoder.transform(y_test)

print(y_train_encode)

"""## **[3] 모델 생성하기**

3.1 신경망 모델 설계하기
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(60,)),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

"""3.2 신경망 모델 학습하기"""

history = model.fit(X_train, y_train_encode, epochs = 200)

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()

loss, acc = model.evaluate(X_train, y_train_encode)
print("Train Data loss:", loss)
print("Train Data accuracy:", acc)

"""## **[4] 모델 평가 및 예측하기**"""

loss, acc = model.evaluate(X_test, y_test_encode)
print("Test Data Test loss:", loss)
print("Test Data Test accuracy:", acc)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model():
    model = Sequential([
        Input(shape=(60,)),
        Dense(256, activation='relu'),
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

