import numpy as np
import tensorflow as tf
### Goal : 년도를 제외한 날짜에 따른 아보카도의 가격 예측.
# load data set
xy_data = np.loadtxt('avocado.csv', delimiter=',', dtype=np.str)
x_data = xy_data[:15000, 2:4]
y_data = xy_data[:15000, [4]]
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)
print(x_data.shape)
print(y_data.shape)
# test set
x_test = xy_data[15000:, 2:4]
y_test = xy_data[15000:, [4]]
# variable setting
X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# hypothesis
hypothesis = tf.matmul(X, W) + b
# cost
cost = tf.reduce_mean(tf.square(Y - hypothesis))
# error rate
errorRate = tf.reduce_mean(Y - hypothesis)
# train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)
# session set
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100001):
        h, c ,_ = sess.run([hypothesis, cost, train], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print("step: ", step, "\nhypothesis\n", h, "\ncost\n", c)
    p, a = sess.run([hypothesis, errorRate], feed_dict={X:x_test, Y:y_test})
    print("prediction\n", p, "\nanswer\n" , y_test, "error rate : ", a)