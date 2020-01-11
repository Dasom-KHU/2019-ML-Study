import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('Iris.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 3  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6, shape(?,1)

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot, shape(?,1,7) one-hot function은 한 차원을 더해준다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape(?,7) 따라서 reshape가 필요하다.

W = tf.Variable(tf.random_normal([5, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# **달라진 부분 Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
                                        
        if step % 200 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


'''
Step:     0     Cost: 122.562   Acc: 33.33%
Step:   200     Cost: 170.467   Acc: 52.67%
Step:   400     Cost: 118.285   Acc: 46.67%
Step:   600     Cost: 27.001    Acc: 66.67%
Step:   800     Cost: 153.385   Acc: 64.00%
Step:  1000     Cost: 22.289    Acc: 82.00%
Step:  1200     Cost: 21.261    Acc: 78.00%
Step:  1400     Cost: 17.668    Acc: 70.00%
Step:  1600     Cost: 10.473    Acc: 84.00%
Step:  1800     Cost: 20.605    Acc: 70.00%
Step:  2000     Cost: 14.209    Acc: 82.00%
Step:  2200     Cost: 12.055    Acc: 84.67%
Step:  2400     Cost: 12.326    Acc: 84.00%
Step:  2600     Cost: 11.330    Acc: 84.67%
Step:  2800     Cost: 11.440    Acc: 84.67%
Step:  3000     Cost: 18.114    Acc: 70.00%
Step:  3200     Cost: 19.079    Acc: 74.00%
Step:  3400     Cost: 10.886    Acc: 84.67%
Step:  3600     Cost: 10.630    Acc: 84.67%
Step:  3800     Cost: 10.393    Acc: 84.67%
Step:  4000     Cost: 10.197    Acc: 84.67%
Step:  4200     Cost: 10.018    Acc: 84.67%
Step:  4400     Cost: 9.855     Acc: 84.67%
Step:  4600     Cost: 9.697     Acc: 84.00%
Step:  4800     Cost: 9.571     Acc: 84.00%
Step:  5000     Cost: 20.589    Acc: 67.33%
Step:  5200     Cost: 15.482    Acc: 75.33%
Step:  5400     Cost: 8.715     Acc: 86.67%
Step:  5600     Cost: 8.921     Acc: 86.00%
Step:  5800     Cost: 10.184    Acc: 78.67%
Step:  6000     Cost: 7.065     Acc: 88.67%
Step:  6200     Cost: 14.239    Acc: 74.00%
Step:  6400     Cost: 14.162    Acc: 79.33%
Step:  6600     Cost: 8.507     Acc: 86.67%
Step:  6800     Cost: 8.452     Acc: 86.67%
Step:  7000     Cost: 8.459     Acc: 86.00%
Step:  7200     Cost: 5.858     Acc: 88.67%
Step:  7400     Cost: 11.760    Acc: 82.67%
Step:  7600     Cost: 10.445    Acc: 78.67%
Step:  7800     Cost: 5.029     Acc: 88.67%
Step:  8000     Cost: 11.709    Acc: 82.67%
Step:  8200     Cost: 7.023     Acc: 85.33%
Step:  8400     Cost: 4.784     Acc: 88.67%
Step:  8600     Cost: 4.868     Acc: 88.67%
Step:  8800     Cost: 5.121     Acc: 88.67%
Step:  9000     Cost: 5.750     Acc: 86.67%
Step:  9200     Cost: 4.368     Acc: 88.67%
Step:  9400     Cost: 4.700     Acc: 88.67%
Step:  9600     Cost: 4.397     Acc: 88.67%
Step:  9800     Cost: 12.323    Acc: 81.33%
Step: 10000     Cost: 7.028     Acc: 87.33%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[False] Prediction: 0 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[False] Prediction: 2 True Y: 1
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
'''
