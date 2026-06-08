# 파일업로드
from google.colab import files
uploaded = files.upload( )

import pandas as pd
df=pd.read_csv('/content/자장면소비자물가지수(1975-2022).csv', encoding='cp949')
df.head()

#데이터 속성 확인하기
df.info()

#물가지수 통계 값 확인하기
df.describe()

# 2020년도 자장면 재료 가격
price2020=[3734, 2032, 1356, 5195]   #양파, 돼지고기,밀가루, 자장면 가격

df['양파 가격']=round(df['양파']*price2020[0]/100,2)
df['돼지고기 가격']=round(df['돼지고기']*price2020[1]/100,2)
df['밀가루 가격']=round(df['밀가루']*price2020[2]/100,2)
df['자장면 가격']=round(df['자장면']*price2020[3]/100,2)

# 자장면 재료 가격 확인하기
df[41:48]              # 2020년 기준년도의 소비자 물가지수는 100이다

df2=df.iloc[:,[0,5,6,7,8]]
df2.head()

#속성간 상관관계 출력
df2.corr()

#한글 라이브러리 설치하기
!pip install koreanize-matplotlib

#히트맵으로 상관관계 시각화하기
import koreanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df2.corr(),annot=True, cmap='Greens')   #히트맵 출력

# pairplot으로 시각화하기
sns.pairplot(df2)


X=df2.iloc[:,0:4]   # 자장면 재료 가격 데이터를 X에 저장
y=df2.iloc[:,4]     # 자장면 가격 데이터를 y데이터에 저장
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)

print(model.coef_, model.intercept_)   # coef_ : 회귀계수, intercept_ : 절편

for i in range(4):
  print("w%d = %.3f"%(i+1,model.coef_[i]))
print("b = %.3f"%(model.intercept_))

print('훈련 데이터로 학습한 모델의 성능(R2):', model.score(X_train, y_train))

# 테스트 데이터로 모델 성능 평가
print('테스트 데이터로 모델의 성능(R2) 평가:', model.score(X_test, y_test))

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
y_pred=model.predict(X_test)

print('Mean squared error :', mean_squared_error(y_pred, y_test))
print('Mean absolute error :', mean_absolute_error(y_pred, y_test))
print('R2 score : ', r2_score(y_pred, y_test))

y_pred=model.predict(X_test)
print('실제값:')
print(y_test[:5])
print('예측값:', y_pred[:5])

df_new=pd.read_csv('new_data.csv', encoding='cp949')
df_new

print(model.predict(df_new))
