###################################################################
# Logistic regression classifier which can load pkl.gz data set
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

import numpy
import theano
import theano.tensor as T
import os
import gzip
import cPickle
rng = numpy.random



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    data = []
    import re

    if re.search('\.csv$', dataset):
        with open(dataset) as f:
            f.readline()
            for line in f:
                if line=='\n':
                    continue
                data.append([float(x.strip()) for x in line.strip().split(',')])

        train_x = [data[x][:-1] for x in range(int(len(dataset)*.7))]
        train_y = [data[x][-1] for x in range(int(len(dataset)*.7))]

        valid_x = [data[x][:-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]
        valid_y = [data[x][-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]

        test_x = [data[x][:-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]
        test_y = [data[x][-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]

        train_set = (numpy.array(train_x), numpy.array(train_y))
        valid_set = (numpy.array(valid_x), numpy.array(valid_y))
        test_set = (numpy.array(test_x), numpy.array(test_y))

    else:
        #############
        # LOAD DATA #
        #############

        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if (not os.path.isfile(dataset)):
            print '%s does NOT exist.' % dataset
            return
        else:
            print '... loading data'

        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

    return (train_set, valid_set, test_set)


# Sample datasets, uncomment one of them to load
# Note: this algorithm can only be used for two-class problems, 
#       so there will be unreasonable results for non-two-class problems
# vehicle_2_classes_small: Two classes: vehicle or non-vehicle
train_set, valid_set, test_set = load_data('../data/vehicle_2_classes_small.pkl.gz')

# imageclipper: four classes: bike, car, motorcycle, pedestrian
#train_set, valid_set, test_set = load_data('../data/imageclipper.pkl.gz')

# MNIST (0~9 digits 10 classes)
#train_set, valid_set, test_set = load_data('../data/mnist.pkl.gz')


N = train_set[0].shape[0]
test_N = test_set[0].shape[0]
feats = train_set[0].shape[1]

# training set of N examples of feature vectors
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2)) 
training_steps = 10000
print "Training set feature vectors: "
print train_set[0]
print "Training set output/target/labels: "
print train_set[1]

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print("Initial model:")
print("w.get_value(): " + str(w.get_value()))
print("b.get_value(): " + str(b.get_value()))

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
          allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=prediction,
          allow_input_downcast=True)

# Train
for i in range(training_steps):
    pred, err = train(train_set[0], train_set[1])

print("pred: " + str(pred))
print("err: " + str(err))

print("Final model:")
print("w.get_value(): " + str(w.get_value()))
print("b.get_value(): " + str(b.get_value()))

print("target values for train_set:")
print(train_set[1])
print("prediction on train_set:")
trainset_prediction = predict(train_set[0])
print(trainset_prediction)
trainset_correct_count = 0
for i in range(0, N-1):
    if trainset_prediction[i] == train_set[1][i]:
        trainset_correct_count =  trainset_correct_count + 1
print("train set classification accuracy:" + str(float(trainset_correct_count)/N))

### test set
print("target values for test_set:")
print(test_set[1])
print("prediction on test_set:")
testset_prediction = predict(test_set[0])
print(testset_prediction)
testset_correct_count = 0
for i in range(0, test_N-1):
    if testset_prediction[i] == test_set[1][i]:
        testset_correct_count =  testset_correct_count + 1
print("test set classification accuracy:" + str(float(testset_correct_count)/test_N))


