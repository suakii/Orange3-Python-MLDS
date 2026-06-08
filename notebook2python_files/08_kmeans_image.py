from google.colab import files
uploaded_image = files.upload()


import matplotlib.image as image
my_image = image.imread('/content/flower.jpg')
print(my_image.shape)
row = my_image.shape[0]
col = my_image.shape[1]
print("row ", row)
print("col ", col)

import matplotlib.pyplot as plt

plt.imshow(my_image)
plt.show()

import numpy as np
histogram, bin_edges = np.histogram(my_image[:,:,0], bins=256, range=(0,256))
plt.bar(bin_edges[0:-1], histogram, color="red")
plt.show()

histogram, bin_edges = np.histogram(my_image[:,:,1], bins=256, range=(0,256))
plt.bar(bin_edges[0:-1], histogram, color="green")
plt.show()

histogram, bin_edges = np.histogram(my_image[:,:,2], bins=256, range=(0,256))
plt.bar(bin_edges[0:-1], histogram, color="blue")
plt.show()

my_image = my_image.reshape(-1, 3)
print(my_image.shape)

from sklearn.cluster import KMeans

new_images = []
inertias = [] #군집이 얼마나 잘 응집되었는지 보여주는 지표
for i in range(2, 11, 2):
   km = KMeans(
       n_clusters=i, init='random',
       n_init=10, max_iter=300,
       tol=1e-04, random_state=0
   )
   km.fit(my_image)
   inertias.append(km.inertia_)
   new_image = km.cluster_centers_[km.labels_]
   new_image = new_image.astype('uint8')
   new_image = new_image.reshape(row, col, -1)
   new_images.append(new_image)


fig = plt.figure(figsize=(10,5))
graph_rows = 2
graph_cols = 3
ax = fig.add_subplot(graph_rows, graph_cols, 1)
ax.set_title("Original")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(my_image.reshape(row, col, -1))

j = 2
for i, image in enumerate(new_images):
    ax = fig.add_subplot(graph_rows, graph_cols, i+2)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cluster: " + str(i+j))
    j += 1

plt.show()


plt.plot(range(2, 11, 2), inertias, marker='o')
plt.xlabel('Clusters')
plt.ylabel('Inertia value')
plt.title('Clusters vs Inertia value')
plt.show()
