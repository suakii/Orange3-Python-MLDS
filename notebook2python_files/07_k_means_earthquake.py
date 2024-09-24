from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('/content/database.csv')

df.head()

df.shape

df.info()

df.isnull()

df.isnull().sum()

df['Type'].value_counts()

import matplotlib.pyplot as plt
df['Type'].value_counts().plot(kind='bar')
plt.show()

import folium
m = folium.Map(location=(0, 0), zoom_start=2)
m

for i in range(len(df)):
    folium.Circle(
        location=[df.iloc[i]['Latitude'], df.iloc[i]['Longitude']],
        radius=10,
    ).add_to(m)
m


earthquake = (df.Type=="Earthquake")
X = df.loc[earthquake, ['Depth', 'Magnitude', 'Latitude', 'Longitude']]
print(X)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_km = km.fit_predict(X[['Depth', 'Magnitude']])
print(y_km[:10])



plt.scatter(X[y_km==0]['Depth'], X[y_km==0]['Magnitude'])
plt.scatter(X[y_km==1]['Depth'], X[y_km==1]['Magnitude'])
plt.scatter(X[y_km==2]['Depth'], X[y_km==2]['Magnitude'])
plt.xlabel('Depth')
plt.ylabel('Magnitude')

plt.show()


mm = folium.Map(location=(0, 0), zoom_start=2)
colors = ['red', 'blue', 'green']

for i in range(3):
    latitudeT =  X[y_km==i]['Latitude']
    longitudeT = X[y_km==i]['Longitude']

    for latitude, longitude in zip(latitudeT, longitudeT) :
        folium.Circle(
            location=[latitude, longitude],
            radius=10,
            color = colors[i]
        ).add_to(mm)

mm

from sklearn.metrics import silhouette_score

scores = []

for i in range(2, 8):
    km = KMeans(n_clusters=i)
    km.fit(X[['Depth', 'Magnitude']])
    score = silhouette_score(X[['Depth', 'Magnitude']], km.labels_)
    scores.append(score)

plt.plot(range(2, 8), scores, marker='o')
plt.xlabel('Number of clusters')
plt.show()
