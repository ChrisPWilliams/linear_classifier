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

class linear_model:
    def __init__(self, learn_rate=0.1, input_dim=2, output_dim=1):
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim,output_dim)))
        self.b = tf.Variable(initial_value=tf.random.uniform(shape=(output_dim,)))
        self.learn_rate = learn_rate

    def forward_pass(self, inputs):
        outputs = tf.matmul(inputs, self.W) + self.b
        return outputs
    
    def square_loss(targets, predictions):
        per_sample_losses = tf.square(targets-predictions)
        return tf.reduce_mean(per_sample_losses)

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:                                         #open tape to record computation graph
            predictions = self.forward_pass(inputs)
            loss = self.square_loss(targets, predictions)
        loss_wrt_W, loss_wrt_b = tape.gradient(loss, (self.W, self.b))          #retrieve gradients from tape
        self.W.assign_sub(loss_wrt_W * self.learn_rate)
        self.b.assign_sub(loss_wrt_b * self.learn_rate)
        return loss
    
    def get_weights(self):
        return self.W, self.b

