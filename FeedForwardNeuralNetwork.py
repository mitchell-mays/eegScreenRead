import tensorflow as tf
import numpy as np
import time


class FFNN:
    def __init__(self, data, labels, iterations, givenModel=[], learn=0.01, keep_input=0.8, keep_hidden=0.5):
        tf.reset_default_graph()
        self.DATA = data
        self.LABELS = labels
        self.model = []
        self.modelStructure = []
        if (len(givenModel) > 0):
            self.load_model(givenModel)

        self.iSize = len(data[0])
        self.oSize = len(labels[0])
        self.it = iterations
        self.learning_rate = learn
        self.kInput = keep_input
        self.kHidden = keep_hidden

        self.X = tf.placeholder("float", [None, self.iSize])
        self.Y = tf.placeholder("float", [None, self.oSize])

        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")

        #Creates model based on each layer -- with relu's inbetween (see function)
        self.py_x = self.tfmodel(self.X, self.model, self.p_keep_input, self.p_keep_hidden, self.learning_rate)

        #computes cross entropy between output and truth, (performs softmax first)
        if self.oSize == 1:
            self.cost = tf.reduce_sum(tf.square(self.py_x - self.Y))
            self.predict_op = self.py_x
        else:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y)) # compute costs
            self.predict_op = tf.argmax(self.py_x, 1)

        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost) # construct an optimizer
        #self.train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()
    def restoreTF(self, path):
        fName = path.split("/")[-1]
        fPath = path.split("/")[:-1]

        temp = open(path+".model")
        self.modelStructure = eval(temp.readline())
        temp.close()

        #####NEED TO WRITE CODE TO LOAD IN SAVE from here to end of function
        #https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model

        sess = tf.Session()
        saver = tf.train.import_meta_graph(path+".meta")
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        for layer in range(len(self.modelStructure)-1):
            self.model.append(graph.get_tensor_by_name("w_" + str(layer)+":0")) 
            if (layer == (len(self.modelStructure)-2)):
                self.model.append(graph.get_tensor_by_name("b_out:0"))
            else:
                self.model.append(graph.get_tensor_by_name("b_" + str(layer)+":0"))
            

        print("Model Successfully Restored")
        print("Model = " + str(self.modelStructure))


    def load_model(self, givenModel):
        print("Loading Model")
        self.modelStructure = givenModel

        if (givenModel[0] != self.DATA.shape[1]):
            raise Exception('Model input does not match data input....') 
        if (givenModel[-1] != self.LABELS.shape[1]):
            raise Exception('Model output does not match data output....') 

        print("Model = " + str(self.modelStructure))
        self.model = []
        for layer in range(len(self.modelStructure)-1):
            self.model.append(self.init_weights([givenModel[layer], givenModel[layer+1]], "w_" + str(layer))) # create symbolic variables)
            if (layer == (len(self.modelStructure)-2)):
                self.model.append(self.init_weights([givenModel[layer+1]], "b_out"))
            else:
                self.model.append(self.init_weights([givenModel[layer+1]], "b_" + str(layer)))

        print(self.model)
    def init_weights(self, shape, varName):
       return tf.Variable(tf.random_normal(shape, stddev=0.01), name=varName)

    def tfmodel(self, X, model, p_keep_input, p_keep_hidden, alpha):
        X = tf.nn.dropout(X, p_keep_input)
        curr = X
        for layer in range(len(model)-1):
            if (layer%2 == 0):
                curr = tf.matmul(curr, model[layer])
                #Leaky RELU
                curr = tf.maximum(alpha*curr, tf.nn.relu(curr)) + model[layer+1]
                curr = tf.nn.dropout(curr, p_keep_hidden)


        out = tf.maximum(alpha*curr, tf.nn.relu(curr)) + model[len(model)-1]
        return out

    def train(self, crossVal=10, fName="myModel", printFreq=500, save=False):
        if (crossVal > 0):
            folds = 10
            chunk = len(self.DATA)//folds
            for fold in range(crossVal):
                #tf.reset_default_graph()
                self.teX = self.DATA[fold*chunk:(fold+1)*chunk]
                self.teY = self.LABELS[fold*chunk:(fold+1)*chunk]

                if fold > 0:
                    if fold < (crossVal-1):
                        self.trX = np.concatenate((self.DATA[0:(fold*chunk)-1:], self.DATA[((fold+1)*chunk)+1:len(self.DATA)-1]))
                        self.trY = np.concatenate((self.LABELS[0:(fold*chunk)-1:], self.LABELS[((fold+1)*chunk)+1:len(self.LABELS)-1]))
                    else:
                        self.trX = self.DATA[0:(fold*chunk)-1:]
                        self.trY = self.LABELS[0:(fold*chunk)-1:]
                else:
                    self.trX = self.DATA[((fold+1)*chunk)+1:len(self.DATA)-1]
                    self.trY = self.LABELS[((fold+1)*chunk)+1:len(self.LABELS)-1]

        else:
            self.trX = self.DATA
            self.trY = self.LABELS
            self.teX = []
            self.teY = []

        if (save):
            saver = tf.train.Saver()

        with tf.Session() as sess:
            #you need to initialize all variables
            #tf.initialize_all_variables().run()
            sess.run(self.init_op)
            print("         ------------------------------")
            print("         --------    Num Features: " + str(len(self.trX[0])))
            print("         --------    Data Points: " + str(len(self.trX)))
            print("         ------------------------------")
            print("         --------    Train  |  Test")


            for i in range(self.it):
                #run training
                for batch in range(0, (len(self.trX) - 50), 50):
                    sess.run(self.train_op, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50], self.p_keep_input: self.kInput, self.p_keep_hidden: self.kHidden})

                #leftovers
                #print(self.trX[len(self.trX)-(len(self.trX)%50):len(self.trX)])
                #print(self.trY[len(self.trX)-(len(self.trX)%50):len(self.trX)])


                sess.run(self.train_op, feed_dict={self.X: self.trX[len(self.trX)-(len(self.trX)%50):len(self.trX)], self.Y: self.trY[len(self.trX)-(len(self.trX)%50):len(self.trX)], self.p_keep_input: self.kInput, self.p_keep_hidden: self.kHidden})

                #time.sleep(10000)

                if (i % printFreq) == 0:
                    if self.oSize == 1:
                        trainVal = np.mean(abs(self.trY -
                                         sess.run(self.predict_op, feed_dict={self.X: self.trX,
                                                                         self.p_keep_input: 1.0,
                                                                         self.p_keep_hidden: 1.0})))
                        if (crossVal > 0):
                            testVal = np.mean(abs(self.teY -
                                         sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                         self.p_keep_input: 1.0,
                                                                         self.p_keep_hidden: 1.0})))

                    else:
                        trainVal = round(np.mean(np.argmax(self.trY, axis=1) ==
                                         sess.run(self.predict_op, feed_dict={self.X: self.trX,
                                                                         self.p_keep_input: 1.0,
                                                                         self.p_keep_hidden: 1.0})), 4)

                        if (crossVal > 0):
                            testVal = round(np.mean(np.argmax(self.teY, axis=1) ==
                                         sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                         self.p_keep_input: 1.0,
                                                                         self.p_keep_hidden: 1.0})), 4)


                    
                    print("           ", str(i).ljust(7), str(round(trainVal,4)).ljust(6), " | ")
                    
                    if (crossVal > 0):
                        print("           ", str(i).ljust(7), str(trainVal).ljust(6), " | ", str(testVal).ljust(6))


                #if testVal > 0.97 and trainVal > 0.97:
                #    break


            value = sess.run(self.predict_op, feed_dict={self.X: self.trX, self.p_keep_input: 1.0,
                                                                 self.p_keep_hidden: 1.0})

            if (save):
                print("Saving Model")
                saver.save(sess,fName)
                modSave = open(fName+".model",'r')
                modSave.write(str(self.modelStructure))
                modSave.close()
                print("Model Saved as: " + fName)

        if self.oSize != 1:
            correct = 0
            temp = self.teY
            if (crossVal <= 0):
                temp = self.trY

            for item in range(len(value)):
                if temp[item][value[item]] == 1:
                    correct += 1

            return correct/len(temp)

        print("Final: "+str(np.mean(abs(self.trY - value))))

        #return [value2, correct/len(value)]

