# 1.1 파일 업로드하기
from google.colab import files
uploaded = files.upload()

# 2.2 데이터프레임 생성하기
import pandas as pd
df = pd.read_csv('/content/Groceries_dataset.csv')
df.head()

df.info()

df.isnull().sum()

df['itemDescription'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,8))
df['itemDescription'].value_counts().plot(kind='bar')

products = df['itemDescription'].unique()
products

one_hot=pd.get_dummies(df['itemDescription'])
one_hot

df2 = df.drop('itemDescription', axis=1)      # 데이터 프레임에서 품목 제외하기
df2 = df2.join(one_hot)
df2.head()

transaction= df2.groupby(["Member_number","Date"])[products[:]].apply(sum)
transaction =transaction.reset_index('Member_number')                  #고객 번호 인덱스에서 해제하기
transaction

transaction.describe()

transaction2=transaction.iloc[:,1:]   # 품목 데이터만 추출하여 transaction2에 저장하기
transaction2[transaction2>=1]=1       # 속성값이 1 이상인 것을 1로 대체
transaction2

transaction2.describe()

transaction2.to_csv('groceries_transaction.csv')   # 트랜잭션 데이터 파일로 저장하기

# 연관분석을 위한 라이브러리 추가하기
from mlxtend.frequent_patterns import fpgrowth, association_rules

frequent_itemsets=fpgrowth(transaction2, min_support=0.01, max_len=3, use_colnames=True)
frequent_itemsets.sort_values(by=['support'], ascending=True).head(10)

frequent_itemsets.sort_values(by=['support'], ascending=True).tail(10)

frequent_itemsets.shape

frequent_itemsets=fpgrowth(transaction2, min_support=0.001, max_len=3, use_colnames=True)
frequent_itemsets.sort_values(by=['support'], ascending=True).head(10)

frequent_itemsets.sort_values(by=['support'], ascending=True).tail(10)

frequent_itemsets.shape

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)
rules.sort_values(by=['lift'], ascending=False)

fp_rules=rules.loc[:, ['antecedents','consequents','support','confidence','lift']]
fp_rules.sort_values(by=['lift'], ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=fp_rules, x='support', y='confidence', color='blue', alpha=0.6)

sns.scatterplot(data=fp_rules, x='support', y='lift', color='green', alpha=0.6)
plt.axhline(y=1, color='red')
plt.show()

sns.scatterplot(data=fp_rules, x='confidence', y='lift', color='red', alpha=0.6)

fp_rules[(fp_rules['lift']) >1 ].sort_values(by=['confidence'], ascending=False)

fp_rules[(fp_rules['antecedents']==frozenset({'rolls/buns'}))].sort_values(by = 'lift', ascending = False)
