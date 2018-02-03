import tensorflow as tf
import numpy as np

#====================================================================================
#====================================================================================
#           A convolutional neural network programmed using tensorflow
#====================================================================================
#====================================================================================

class TFConvNetwork:

    def __init__(self, iWidth, iHeight, outputSize, trainData, trainLabels, testData, testLabels):
        tf.reset_default_graph()
        self.imageWidth = iWidth
        self.imageHeight = iHeight

        # the 1600 is the number of pixels in an image and the 10 is the number of images in a batch
        # ...aka for labels
        self.X = tf.placeholder(tf.float32, shape=[None, iHeight, iWidth, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, outputSize])

        self.trX = np.asarray(trainData).reshape(np.max(trainData.shape), iHeight, iWidth, 1)
        #self.trX = trainData
        self.trY = trainLabels

        self.teX = np.asarray(testData).reshape(np.max(testData.shape), iHeight, iWidth, 1)
        #self.teX = testData
        self.teY = testLabels

        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.oSize = outputSize


        #self.py_x = self.model(self.X, w, w2, w3, w4, w_o, self.p_keep_conv, self.p_keep_hidden)
        self.py_x = self.model()
        
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))

        with tf.name_scope("Train_Op"):
            self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost) #0.001

        with tf.name_scope("Test_Op"):
            self.correct_prediction = tf.equal(tf.argmax(self.py_x, 1), tf.argmax(self.Y, 1))
            #self.predict_op = tf.argmax(self.py_x, 1)

        with tf.name_scope("Loose_Test_Op"):
            self.loose_predict_op = self.py_x

        with tf.name_scope("Accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        tf.summary.scalar('cost', self.cost)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("TensorBoard/1")
        self.writer.add_graph(tf.Session().graph)

    def model(self):
        #def model(self, X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        conv1 = self.conv_layer(self.X, [3, 1, 1, 32], [1, 4, 1, 1], "timeWise_Conv", self.p_keep_conv)
        conv2 = self.conv_layer(conv1, [1, 3, 32, 64], [1, 1, 4, 1], "electrodeWise_Conv", self.p_keep_conv)

        with tf.name_scope("flatten"):
            flatten = tf.reshape(conv2, [-1, 64 * 7 * 10])  # reshape to (?, 2048)
            flatten = tf.nn.dropout(flatten, self.p_keep_conv)

        fc1 = self.fc_layer(flatten, [64 * 7 * 10, 625], "fullyConnected_1", dropout=self.p_keep_hidden)
        fc2 = self.fc_layer(fc1, [625, self.oSize], "fullyConnected_2")
        
        return fc2

    def conv_layer(self, input, shape, stride, name, dropout=1):
        with tf.name_scope(name):
            w = self.init_weights(shape, "w")  # 3x3x1 conv, 32 outputs
            l1a = tf.nn.relu(tf.nn.conv2d(input, w,  # l1a shape=(?, 39, 25, 32)
                                        strides=[1, 1, 1, 1], padding='SAME'))

            l1 = tf.nn.max_pool(l1a, ksize=stride,  # l1 shape=(?, 10, 25, 32)
                                strides=stride, padding='SAME')

            tf.summary.histogram("weights", w)
            tf.summary.histogram("activations", l1a)
            return(tf.nn.dropout(l1, dropout))

    def fc_layer(self, input, shape, name, dropout=1):
        with tf.name_scope(name):
            w = self.init_weights(shape, "w")  # 3x3x1 conv, 32 outputs
            l1 = tf.nn.relu(tf.matmul(input, w))
            return(tf.nn.dropout(l1, dropout))

    def init_weights(self, shape, name):  
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def trainAndClassify(self, iter):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("         --------    Train  |  Test")
            for i in range(iter):
                for batch in range(0, len(self.trX) - 50, 50):
                    if (i % 5 == 0):
                        s = sess.run(self.merged_summary, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50],
                                                      self.p_keep_conv: 1, self.p_keep_hidden: 1})
                        self.writer.add_summary(s,i)

                    sess.run(self.train_op, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50],
                                                      self.p_keep_conv: 0.8, self.p_keep_hidden: 0.5})

                
                [trainVal] = sess.run([self.accuracy], feed_dict={self.X: self.trX, self.Y: self.trY,
                                                                            self.p_keep_conv: 1.0,
                                                                            self.p_keep_hidden: 1.0})
                '''
                [testVal] = sess.run([self.accuracy], feed_dict={self.X: self.teX, self.Y: self.teY,
                                                                            self.p_keep_conv: 1.0,
                                                                            self.p_keep_hidden: 1.0})
                '''
                trainVal = round(trainVal, 4)
                #testVal = round(testVal, 4)

                '''
                trainVal = round(np.mean(np.argmax(self.trY, axis=1) ==
                                  sess.run(self.predict_op, feed_dict={self.X: self.trX,
                                                                            self.p_keep_conv: 1.0,
                                                                            self.p_keep_hidden: 1.0})),4)

                testVal =  round(np.mean(np.argmax(self.teY, axis=1) ==
                                              sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                                   self.p_keep_conv: 1.0,
                                                                                   self.p_keep_hidden: 1.0})),4)
                '''
                print("           ", str(i).ljust(7), str(trainVal).ljust(6), " | ", str(trainVal).ljust(6))
                #print("           ", str(i).ljust(7), str(trainVal).ljust(6), " | ", str(testVal).ljust(6))

                #if testVal > 0.9 and trainVal > 0.9:
                #    break

            value =  np.mean(np.argmax(self.teY, axis=1) ==
                                                  sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                                       self.p_keep_conv: 1.0,
                                                                                       self.p_keep_hidden: 1.0}))

            value2 = sess.run(self.loose_predict_op, feed_dict={self.X: self.teX, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
            return [value2, value]



#HELPER FUNCTIONS
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')