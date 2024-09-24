from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('/content/diabetes.csv')
df.head()

df.info()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(4,3))
sns.countplot(x="Outcome", data=df)
plt.show()

df.describe()

print("Number of rows with 0 values for each variable")
for col in df.columns:
  missing_rows=df.loc[df[col]==0].shape[0]
  print(col+":", missing_rows)

sns.scatterplot(x='Age', y='BloodPressure', hue='Outcome', data=df)
plt.show()

sns.scatterplot(x='BloodPressure', y='Insulin', hue='Outcome', data=df)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, orient='h')

Preg_range=(df.Pregnancies>=0)&(df.Pregnancies<=13.5)
Gluc_range=(df.Glucose>=36.75)&(df.Glucose<=202.75)
Bloo_range=(df.BloodPressure>=35)&(df.BloodPressure<=107)
Skin_range=(df.SkinThickness>=1)&(df.SkinThickness<=80)
Insu_range=(df.Insulin>=1)&(df.Insulin<=318.75)
BMI_range=(df.BMI>=13.35)&(df.BMI<=50.55)
diab_range=(df.DiabetesPedigreeFunction>=0)&(df.DiabetesPedigreeFunction<=1.23)
Age_range=(df.Age>=21)&(df.Age<=65.5)
df2=df.loc[Preg_range&Gluc_range&Bloo_range&Skin_range&Insu_range&BMI_range&diab_range&Age_range,:]
df2

X = df2.drop('Outcome', axis = 1) # Outcome을 제외하고 독립변수 X에 저장
y = df2['Outcome']                # 종속변수 Outcome을 y에 저장

print("특징 모양: ", X.shape)
print("타깃 모양: ", y.shape)

#  최소-최대 정규화(Min-Max Normalization)
X_scaled= (X-X.min())/(X.max()-X.min())
X_scaled.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y)

print("훈련 데이터 : ", X_train.shape, y_train.shape)
print("테스트 데이터 : ", X_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression( solver='lbfgs', max_iter=1000, random_state=42)     # Logistic Regression 모델 생성
model.fit(X_train,y_train)

import numpy as np
w=model.coef_          #회귀 계수
b=model.intercept_     #절편
print("w= ", np.round(w,2))
print("b= ", np.round(b,2))  #np.round()를 이용하여 소숫점 2자리까지 출력

print("훈련 데이터로 학습한 모델 분류 정확도 : ", model.score(X_train, y_train))

print("테스트 데이터를 이용한 모델 성능 평가 : ", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print(y_pred[5:10])
print(y_test[5:10])

from sklearn.metrics import confusion_matrix
plt.figure(figsize=(6, 4))
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, cmap="Blues",fmt="g")
plt.title("Diabetes Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y.value_counts()              # y의 빈도수 확인

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X2, y2 = oversample.fit_resample(X,y)   # SMOTE 활용 데이터 개수 맞축기

y2.value_counts()     # 오버샘플링 후 y2의 빈도수 확인

# 최소-최대 정규화(Min-Max Normalization)
X2_scaled= (X2-X2.min())/(X2.max()-X2.min())
X2_scaled.head()

X_train, X_test, y_train, y_test = train_test_split(X2_scaled, y2, test_size=0.3, stratify=y2)

model2 = LogisticRegression( solver='lbfgs', max_iter=1000, random_state=42)
model2.fit(X_train,y_train)

print("개선된 모델 분류 정확도 : ", model2.score(X_train, y_train))

from sklearn import metrics
y_pred2 = model2.predict(X_test)
print("개선된 모델 성능 평가:",metrics.accuracy_score(y_test, y_pred2))      # 실제값과 모델2의 예측값 비교

from sklearn.metrics import confusion_matrix
plt.figure(figsize=(6, 4))

conf = confusion_matrix(y_test,y_pred2)
sns.heatmap(conf, annot=True, cmap="Greens",fmt="g")

plt.title("Diabetes Classification(2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 파일 업로드
from google.colab import files
uploaded = files.upload()

df_new = pd.read_csv('new_diabetes.csv')
df_new.head()

#  최소-최대 정규화(Min-Max Normalization)
df_new_scaled = (df_new-X2.min())/(X2.max()-X2.min())

df_new_scaled

print(model2.predict(df_new_scaled))
