"""The code contains code for task 1,2,3 of task3

ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html"""


import numpy as np
import matplotlib.pyplot as plt
import random

X = np.array([[5.9, 3.2],
	[4.6 ,2.9],
	[6.2, 2.8],
	[4.7, 3.2],
	[5.5, 4.2],
	[5.0, 3.0],
	[4.9, 3.1],
	[6.7, 3.1],
	[5.1, 3.8],
	[6.0, 3.0]])
# plt.scatter(X[:,0], X[:,1], s = 150)
# plt.show()
k = 3
color = ["g","r","b"]
centroids=np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
C = np.array(list(zip(centroids[:,0], centroids[:,1])))
centroids ={0:[6.2,3.2],1:[6.6,3.7],2:[6.5,3.0]}

classifications = {}
for i in range(k):
	centroids[i]=X[i]

classifications = {}
for i in range(k):
	classifications[i]=[]
for featureset in X:
	distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
	classification = distances.index(min(distances))
	classifications[classification].append(featureset)
old_centroids = dict(centroids)
for classification in classifications:
	centroids[classification] = np.average(classifications[classification],axis = 0)
for classification in classifications:
	colors = color[classification]
	for featureset in classifications[classification]:
		plt.scatter(featureset[0],featureset[1],marker="^",color=colors,s = 15)
		plt.savefig('task3_iter1_a.jpg')
plt.show()





print("initial",centroids)

def iter(max_itr):
	for i in range(max_itr):
		classifications = {}
		for i in range(k):
			classifications[i]=[]
		for featureset in X:
			distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
			classification = distances.index(min(distances))
			classifications[classification].append(featureset)
		old_centroids = dict(centroids)
		for classification in classifications:
			centroids[classification] = np.average(classifications[classification],axis = 0)
			return centroids,classifications

def cen_print(a,it):
	for centroid in centroids:
		colors= color[centroid]
		plt.scatter(centroids[centroid][0],centroids[centroid][1],marker="o",color=colors,s = 15)
		plt.savefig("task3_iter"+str(it)+"_"+str(a)+".jpg")
	plt.show()
def feat_print(a,it):
	for classification in classifications:
		colors = color[classification]
		for featureset in classifications[classification]:
			plt.scatter(featureset[0],featureset[1],marker="^",color=colors,s = 15)
			plt.savefig("task3_iter"+str(it)+"_"+str(a)+".jpg")
	plt.show()
e = "b"	
f = "a"
for i in range(k):
	centroids[i]=X[i]
	centroids, classifications = iter(1)
cen_print(e,1)

for i in range(k):
	centroids[i]=X[i]
	centroids, classifications = iter(2)
cen_print(e,2)
feat_print(f,2)


print("final centroid",centroids)
