# 파일 업로드
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('/content/Europe Hotel Booking Satisfaction Score.csv')
df.head()

df.info()

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Hotel wifi service', data=df)
plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(df.iloc[:, 1:].corr(numeric_only=True), annot=True, cmap='Blues')
plt.show()

X = df.iloc[:, 6:16]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, max_depth=2, filled=True)
plt.show()

import numpy as np
print("feature importances : ", model.feature_importances_)

n_features = X_train.shape[1]
plt.figure(figsize=(25,5))
plt.bar(np.arange(n_features), model.feature_importances_)
plt.xticks(np.arange(n_features), df.columns[6:16], rotation=15)
plt.show()

print("훈련 데이터를 이용한 모델 분류 정확도 : ", model.score(X_train, y_train))

print("테스트 데이터 성능평가 : ", model.score(X_test, y_test))

prediction = model.predict(X_test)
print(prediction[:5])
print(y_test[:5])

from sklearn.metrics import confusion_matrix

prediction = model.predict(X_test)
conf = confusion_matrix(y_test, prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, cmap='Blues', fmt='d')
plt.title('Hotel satisfaction classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 파일 업로드
from google.colab import files
uploaded = files.upload()

df_new = pd.read_csv('/content/hotel_satisfaction_new.csv')
df_new.head()

print(model.predict(df_new))

# LabelEncoder 클래스를 임포트함
from sklearn.preprocessing import LabelEncoder

# df 데이터 프레임의 마지막 열을 y0 변수에 저장함
y0 = df.iloc[:, -1]

# LabelEncoder 객체를 생성함
encoder = LabelEncoder()

# LabelEncoder 객체를 사용하여 y0 변수의 값을 레이블 인코딩함
y = encoder.fit_transform(y0)

y    # y 변수의 값을 출력함
