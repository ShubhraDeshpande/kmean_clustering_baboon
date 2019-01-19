import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


UBIT = 'shubhraj'
np.random.seed(sum([ord(c)for c in UBIT]))
img1 = cv2.imread("baboon.jpg")
img1 = img1/256.0
w, h , c = img1.shape


X = np.zeros([w*h,c])
count = 0
for i in range(w):
	for j in range(h):
		X[i*w+j,:] = img1[i,j,:]
		count += 1
def kmean(K):
	print("running for "+str(K)+" clustures")
	max_itr = 5
	centroids = np.random.rand(K,c)
	print("initial",centroids)
	centroids = X[np.random.choice(np.arange(len(X)), K), :]
	for i in range(max_itr):
		
		# for x_i in X:	
		# 	for y_k in centroids:
		# 		y = np.sqrt(np.dot(x_i-y_k, x_i-y_k))
		# 	z = np.argmin([y])
		# C = np.array([z])
		C = np.array([np.argmin([np.sqrt(np.dot(i-j, i-j)) for j in centroids]) for i in X])#inline function for above code

		# for k in range(K):
		# 	centroids = X[C==k].mean(axis = 0)
		centroids = [X[C == k].mean(axis = 0) for k in range(K)] #inline function for above code

		print("iter: ", i)


	centroids = np.array(centroids)
	C = np.array(C)

	X_recon = np.zeros(img1.shape)
	for i in range((X_recon.shape[0])):
		for j in range((X_recon.shape[1])):
			label = C[i*X_recon.shape[0]+j]
			X_recon[i,j,:] = centroids[label,:]
			#X_recon = int(X_recon * 256)
			#compute loss
	loss = np.sqrt(sum(sum(sum((img1-X_recon)**2))))
	print("loss during "+str(K)+"clusturing is ",loss)
	X_recon = X_recon *256.0
	return X_recon

cv2.imwrite("task3_baboon_3.jpg",kmean(3))
cv2.imwrite("task3_baboon_5.jpg",kmean(5))
cv2.imwrite("task3_baboon_15.jpg",kmean(15))