# 파일 업로드
from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('/content/penguins_size.csv')
df.head()

df.info()

df.isnull().sum()

df.dropna(inplace=True)
df.isnull().sum()

df['sex'].unique()

df[df['sex']=='.']

# 데이터프레임 336 인덱스 행 삭제하기 (axis=0 : 행, inplace=True : 원본에서 바로 바꾸기)
df.drop(axis=0, inplace=True, index=336)

# 데이터프레임 요약정보 구하기
df.info()

df['species'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='species', data=df)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot( x='species', y='flipper_length_mm', hue='species', data=df)
plt.show()

sns.scatterplot(x='culmen_depth_mm', y='culmen_length_mm', hue='species', data=df)
plt.show()

sns.scatterplot(x='flipper_length_mm', y='culmen_length_mm', hue='species', data=df)
plt.show()

df1 = df[['culmen_depth_mm', 'culmen_length_mm', 'flipper_length_mm', 'species']]
df1.head()

dataset = df1.values
X = dataset[:, :-1]
y = dataset[:, -1]

print("특징 모양 : ", X.shape)
print("타겟 모양 : ", y.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[0]

# 훈련, 테스트 데이터셋 구분하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,
                                                    stratify=y, random_state=0)

print("훈련 데이터 : ", X_train.shape, y_train.shape)
print("테스트 데이터 : ", X_test.shape, y_test.shape)

print(X_train[0], y_train[0])

y_train[:10]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

print("훈련 데이터를 이용한 모델 분류 정확도 : ", knn.score(X_train, y_train))

print("테스트 데이터를 이용한 모델 성능 평가 : ", knn.score(X_test, y_test))

for k in range(2, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print('k: %2d, accuracy: %.2f' % (k, score*100))

predictions = knn.predict(X_test)

print(predictions[:5])
print(y_test[:5])

from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8, 6))

conf = confusion_matrix(y_test, predictions)
sns.heatmap(conf, annot=True, cmap="BuPu")

plt.title("Penguin Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 파일 업로드
from google.colab import files
uploaded = files.upload()

df_new = pd.read_csv('/content/penguin_new.csv')
df_new.head()


dataset_new = df_new.values
new_scaled = scaler.fit_transform(dataset_new)
new_scaled

print(knn.predict(new_scaled))


