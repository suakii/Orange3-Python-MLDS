import pandas as pd

# 데이터 생성
day = [1, 2, 3, 4, 5, 6, 7]
temp = [-4.7, -11.7, -13.2, -10.7, -7, -6.3, -5.1]

# 데이터프레임 생성
df = pd.DataFrame({'Day': day, 'Temp': temp})
df.head()

import numpy as np

mean_temp = np.mean(temp)  # 온도의 평균 계산
std_temp = np.std(temp)    # 온도의 표준편차 계산

print("평균 온도:", mean_temp)
print("온도 표준편차:", std_temp)

import matplotlib.pyplot as plt

# 선 그래프 그리기
plt.plot(day, temp, marker='o', color='blue')

# 그래프 제목 및 축 레이블 설정
plt.title('Temp Variation')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()  # 그래프 출력

import seaborn as sns

# 산점도 그리기
sns.scatterplot(x='Day', y='Temp', data=df, s=100)

# 그래프 제목 및 축 레이블 설정
plt.title('Temp Scatter Plot')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()   # 그래프 출력
