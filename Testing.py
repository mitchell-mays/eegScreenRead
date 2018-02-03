import FeedForwardNeuralNetwork as FFNN
import preprocessing

# Test 1
model = [2, 4, 2]
#(put in the preprocessing bit when I get a chance!)
dataFile = pd.read_csv("testingData.csv")
data = dataFile.drop(dataFile.columns[len(dataFile.columns)-(model[-1]):len(dataFile.columns)], axis=1)
labels = dataFile[dataFile.columns[len(dataFile.columns)-(model[-1]):len(dataFile.columns)]]
labelsVals = labels.values
if (len(labels.columns) == 1):
    labelsVals = np.reshape(labels.values, (labels.values.shape[0], 1))


NN = FFNN.FFNN(data.values, labelsVals, 10000, model, learn=0.5, keep_input=1, keep_hidden=1)
print(NN.train(crossVal=0))