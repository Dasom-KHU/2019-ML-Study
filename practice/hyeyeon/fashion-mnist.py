import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pprint

xy = np.loadtxt('fashion-mnist_train의 복사본.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 1:]
y_data = xy[:, [0]]

nb_classes = 10  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6, shape(?,1)

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot, shape(?,1,7) one-hot function은 한 차원을 더해준다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape(?,7) 따라서 reshape가 필요하다.

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.int32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

'''
TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.
'''
