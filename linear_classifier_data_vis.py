import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset_size = 2000

a = np.random.multivariate_normal(
    mean=[2,0],
    cov=[[1,0.5],[0.5,1]],
    size=dataset_size//2)

b = np.random.multivariate_normal(
    mean=[-2,0],
    cov=[[0.5,1],[1,0.5]],
    size=dataset_size//2)

inputs = np.vstack((a,b)).astype("float32")

labels = np.vstack((np.ones((dataset_size//2,1), dtype="float32"), np.zeros((dataset_size//2,1), dtype="float32")))

plt.scatter(inputs[:,0], inputs[:,1], c=labels[:,0])
plt.show()