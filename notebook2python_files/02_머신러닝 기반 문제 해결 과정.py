from sklearn.datasets import load_iris

iris = load_iris()  # iris 데이터셋 불러오기

X, y = iris.data, iris.target  # 특징(X), 타깃(y)으로 데이터 분할

from sklearn.model_selection import train_test_split
# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# 의사결정트리 모델 생성 및 학습
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)  # 테스트 데이터를 사용하여 모델의 예측값을 계산

# 실제값과 예측값을 비교하여 정확도 계산
print("정확도: ", accuracy_score(y_test, y_pred))

import numpy as np

# 붓꽃 테스트 데이터 3개 생성
test_data = np.array([[5.0, 3.6, 1.3, 0.25],
                      [6.7, 3.0, 5.0, 1.7],
                      [5.8, 2.7, 5.1, 1.9]])

# 예측
predictions = model.predict(test_data)
print(predictions)


