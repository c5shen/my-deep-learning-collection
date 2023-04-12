import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle

RATE = 0.001      # learning rate for stochastic G.D.
EPOCH = 100     # X epoches for SGD
BATCH = 64      # batch size
DIM = 28        # dimension of the image (single)
OUT = 10        # output layer number
ROUND = 1       # number of rounds to run different training (initial weights)
SAVE = 1        # by default, graph/weights are not saved
TEST = 1        # by default testing is disabled
MOMENTUM = 0.9  # by default, momentum is 0
BN = 0          # by default, not doing batch normalization
DROP = 0        # by default, no dropout
DROP_rate = 1   # by default, no dropout

# some other constants settings
features = 1    # number of feature maps
imgsize = 32    # image width/height
filtersize = 5 # filter size (i.e. 10x10)
padding = 2     # size of 0 padding around inputs
pooling = 2     # max pooling size (i.e. 2x2)
stride = 1      # size of stride (i.e. 1)
small = 0       # small set
fname = "cnn"   # output file name
trial_n = 1     # trial number for output file


###################### helper functions #########################

# weight randomizer
# return type: np array
def getWeights(bot, top, seed):
    np.random.seed(seed)
    b = np.sqrt(6.0/(bot + top))
    #print b
    #exit()

    # create a bot X top shape weight array
    weights = [[np.random.uniform(-b, b) for x in range(top)] for y in range(bot)]
    weights = np.array(weights, dtype=float)

    #print weights
    return weights

# data reading function
def readData(file_loc):
    data = open(file_loc, 'r')
    lines = data.read().splitlines()
    instances = []  # a vector storing digits and labels
    labels = []

    for line in lines:
        line = line.split(',')
        label = int(line[len(line)-1])  # int label
        labels.append(label)
        del line[len(line)-1]

        digits = [float(x) for x in line]  # float pixel
        instances.append(digits)
        
    return (instances, labels)
    #print len(instances[0][0]),len(instances[0][1])

# convert Nx1 labels to Nx19 labels
def convertLabel(labels):
    new_label = np.zeros(shape=(labels.shape[0], OUT))
    for i in range(labels.shape[0]):
        new_label[i,labels[i]] = 1
    return new_label

# batch normalization
# adapted from https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
def BN_forward(X, gamma, beta):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    new_X = (X - mean)/np.sqrt(var + 1e-8)
    out = gamma * new_X + beta

    cache = (X, new_X, mean, var, gamma, beta)
    return out, cache, mean, var

# batch norm backward
# adapted from https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
def BN_backward(dout, cache):
    X, X_norm, mean, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mean
    std_inv =1.0 / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -0.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta

# function to do the convolutional layer forward-feed
# filters -> keeps original image size. i.e. 32x32
# for 3 channels, sum over after the convolutional operation
# p: padding size, s: stride, f: filters, pl: pooling size
def conv_forward(ori, p, s, f, pl):
    # assuming the ori is a tensor of size [batch_size, img_size, img_size, 3]
    tmp_size1 = ori.shape[1]
    tmp_size2 = ori.shape[2]
    f_size = f.shape[1]
    
    # initialize zero tensor for padded img
    padded = np.zeros((ori.shape[0], tmp_size1 + 2*p,
                        tmp_size2 + 2*p, ori.shape[3]))
    
    # assign middle portion of padded layer as the original
    padded[:, p:p+tmp_size1, p:p+tmp_size2, :] = np.copy(ori)

    #plt.imshow(padded[0])
    #plt.show()
    #exit()
    
    # start to filter out using the filters on padded layer
    # size: batch_size x 32 x 32 x num_filters
    filtered = np.zeros((padded.shape[0], (padded.shape[1]-2*p)//s, 
                            (padded.shape[2]-2*p)//s, f.shape[0]))

    # for loop to do matrix multiplication
    for i in range(0, padded.shape[1] - f_size + 1, s):
        for j in range(0, padded.shape[2] - f_size + 1, s):
            filtered[:,i,j,:] = np.tensordot(padded[:,i:i+f_size,j:j+f_size,:],
                                f, axes=([1,2,3],[1,2,3]))

    # STEP 2: ReLU function (x * (x > 0))
    relu = filtered * (filtered > 0)
    
    # STEP 3: max pooling, also recored indexes of max for backprop
    pooled, pooled_mask = max_pooling(relu, pl)

    # STEP 4: flatten max pool and pass to output layer, include bias term
    biased_pooled = pooled.reshape((pooled.shape[0],
                pooled.shape[1]*pooled.shape[2]*pooled.shape[3]))
    biased_pooled = np.insert(biased_pooled, 0, 1, axis=1)

    return padded, pooled, pooled_mask, biased_pooled

# very similar function for backward convolution
# left: original data, right: gradient passed down
# RETURN: gradient of filters
def conv_backward(ori, prev_grad, s, f_size, batch_size):
    # size: (1,5,5,3) or something like (36,5,5,3)
    f_grad = np.zeros((prev_grad.shape[3], f_size, f_size, ori.shape[3]))
        
    # rotate previous gradient
    rotated_grad = np.rot90(prev_grad, 2, (1,2))
    
    #print f_grad.shape, prev_grad.shape, ori.shape
    # let's just assume we have squared filters
    tmp_size1 = prev_grad.shape[1]
    tmp_size2 = prev_grad.shape[2]
    for i in range(0, f_size):
        for j in range(0, f_size):
            f_grad[:,i,j,:] = np.tensordot(ori[:,i:i+tmp_size1,j:j+tmp_size2,:],
                            rotated_grad,
                            axes=([0,1,2],[0,1,2])).transpose([1,0])
    return f_grad/batch_size

# function for max pooling from relu output
# r: relu tensor (i.e. batch_size x 32 x 32 x num_features), p: pooling size
def max_pooling(r, p):
    indexes = np.zeros((r.shape[0], r.shape[1], r.shape[2], r.shape[3]))
    pooled = np.zeros((r.shape[0], r.shape[1]//p, r.shape[2]//p, r.shape[3]))

    #---------- another way to implement max pool ----------#
    # index indicates the location (flattened) in a single pool
    #for i in range(0, pooled.shape[1]):
    #    for j in range(0, pooled.shape[2]):
    #        pooled[:,i,j,:] = np.amax(r[:,i*p:i*p+p,j*p:j*p+p,:], axis=(1,2))
    
    # shape: (32,32,batch_size, filter_size)
    tmp = r.transpose([1,2,0,3])
    # shape: (32,32,(batch_size x filter size)=omega)
    cpr = tmp.reshape(( tmp.shape[0], tmp.shape[1], tmp.shape[2]*tmp.shape[3]))
    # shape: (16,2,2,16,omega), for later indexes processing
    #indexes = cpr.reshape((cpr.shape[0]/p, p, cpr.shape[1]/p, p, cpr.shape[2])).transpose([0,1,3,2,4])
    # shape: (16,16,omega), after max pooling
    cpr = cpr.reshape(cpr.shape[0]//p, p, cpr.shape[1]//p, p, cpr.shape[2]).max(axis=(1,3))
    # shape: (16,16,64,1) -> (64,16,16,1), max pooled
    cpr = cpr.reshape((cpr.shape[0],cpr.shape[1],cpr.shape[2]//r.shape[3], cpr.shape[2]//r.shape[0]))
    pooled = cpr.transpose([2,0,1,3])
    #print np.array_equal(pooled, cpr)

    # use a mask to indicate where the maximum is in original ReLU
    indexes = np.equal(r, pooled.repeat(p, axis=1).repeat(p,axis=2)).astype(int)
    # start processing indexes, indexes are flattened values of each pool
    # i.e. 2x2 pool, indexes from 0->3
    # shape: (4,16,16,omega)
    #indexes = indexes.reshape((indexes.shape[0], indexes.shape[1]*indexes.shape[2], 
    #                    indexes.shape[3], indexes.shape[4])).transpose([1,0,2,3])
    # shape: (4,16*16*omega)
    #indexes = indexes.reshape((indexes.shape[0], 
    #                    indexes.shape[1]*indexes.shape[2]*indexes.shape[3]))
    # shape: 1d array of 16*16*omega
    #indexes = np.argmax(indexes, axis=0)
    # shape: (64,16,16,1) final indexes
    #tot = indexes.shape[0]
    #indexes = indexes.reshape(pooled.shape[1],
    #        pooled.shape[2], pooled.shape[0], pooled.shape[3]).transpose(
    #                                        [2,0,1,3])
    return pooled, indexes

# backprop alogrithm
def backprop(height, k, sgd_weights, layers, batch_label, modifiers, cache, batch_size):
    # height indicates where the output is
    # k from height to 1. For example, k = [3,2,1]

    dX, dgamma, dbeta = [],[],[]

    # in the form (o - t) /dot/ h
    # at softmax layer
    if k == height:
        p = layers[k] - batch_label    # softmax gradient
        grad = (np.dot(p.T, layers[k-1])).T / BATCH    # dot product to hidden layer
    # in the form ((o - t) /dot/ W(2) * h * (1-h))^T /dot/ X
    #at hidden layers
    else:
        p = layers[height] - batch_label     # softmax grad
        
        # iterate through till k-th layer
        for i in range(height - 1, k - 1, -1):
            #print p.shape, sgd_weights[i][1:,:].T.shape
            p = np.dot(p, sgd_weights[i][1:,:].T)
            #sigmoid = np.multiply(layers[i][:,1:], 1 - layers[i][:,1:])
            #p = np.multiply(p, sigmoid)
            #print layers[k-1].shape
        
            p = np.multiply(p, (layers[i][:,1:] > 0))
        # batch normalization backprop
        if BN:
            dX, dgamma, dbeta = BN_backward(p, cache[k-1])
            #print dX.shape, layers[k-1].shape
            grad = (np.dot(dX.T, layers[k-1])).T/batch_size
        else:
            grad = (np.dot(p.T, layers[k-1])).T/batch_size

    return grad, dX, dgamma, dbeta

# calculate average cross-entropy loss over the batch size training example
def avg_CEloss(t, o):
    log_sum = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        log_sum[i] = -np.log(o[i, np.argmax(t[i])])

    return np.average(log_sum)

# calculate set loss (here is d)
def trainNetwork(d, weights, training, gamma, beta, params):
    # calculate from bot to top layer
    # --> all hidden layer nodes use sigmoid
    # ----> output layer nodes use softmax
    layers_return = []
    all_cache = []
    all_mean = []
    all_var = []
    #all_modifier = []

    # if doing dropout and while training
    if DROP and training:
        modifier = np.random.binomial(1, DROP_rate, d.shape)
        modifier[:,0] = 1
        base_layer = d * modifier
        #all_modifier.append(modifier)
        layers_return.append(base_layer)
    else:
        layers_return.append(d)

    for w in range(len(weights) - 1):
        if w == 0:
            cur_layer = np.dot(d, weights[w])
        else:
            cur_layer = np.dot(cur_layer, weights[w])

        # if applying BN, here
        if BN and training:
            # cur_layer normalized
            cur_layer, bn_cache, mean, var = BN_forward(cur_layer,
                                        gamma[w], beta[w])
            all_cache.append(bn_cache)
            all_mean.append(mean)
            all_var.append(var)
        # testing val or test
        elif BN and not training:
            cur_layer = (cur_layer - params['mean'][w]) / np.sqrt(params['var'][w] + 1e-8)
            cur_layer = gamma[w] * cur_layer + beta[w]

        # change to sigmoid
        #cur_layer = 1 / (1 + np.exp(-cur_layer))
        #cur_layer = np.insert(cur_layer, 0, 1, axis=1)

        # change to ReLU
        cur_layer = cur_layer * (cur_layer > 0)
        cur_layer = np.insert(cur_layer, 0, 1, axis=1)

        # as we are going to the 2nd layer, we dropout this
        if w == 0 and DROP and training:
            modifier = np.random.binomial(1,DROP_rate, cur_layer.shape)
            modifier[:,0] = 1
            cur_layer *= modifier
            #all_modifier.append(modifier)

        # insert bias term to cur_layer calculation
        layers_return.append(cur_layer)

    # when break out from the for loop above, we stopped at the output layer
    # calculation (e.g. 20000x101 X 101x19 not done, but no sigmoid/softmax yet)
    if len(weights) > 1:
        this_output = np.dot(cur_layer, weights[len(weights) - 1])
    # if there are no other hidden layer, straight from max pool to output
    else:
        this_output = np.dot(d, weights[len(weights) - 1])

    # change output to softmax
    maxed = np.max(this_output, axis=1)
    z = this_output - maxed.reshape((maxed.shape[0],1))
    this_output = np.exp(z)
    sum_output = np.sum(this_output, axis=1)
    sum_output = np.reshape(sum_output, (sum_output.shape[0], 1))
    this_output = this_output/sum_output

    layers_return.append(this_output)
    return layers_return, all_cache, all_mean, all_var


##################### helper functions end #######################
if '-h' in sys.argv:
    # print helpful tips
    print ('-t [FILE]\t\t\tTrain data file (Default: data/train.txt)')
    print ('-v [FILE]\t\t\tValidation data file (Default: data/val.txt)')
    print ('-l [0,1,..]\t\t\tNumber of layers (Default: 1)')
    print ('-n [20,100,..]\t\t\tNumber of hidden nodes (Default: 100)')
    print ('-b [1,5,..]\t\t\tSize of mini-batches (Default: 64)')
    print ('-r [1,2,..]\t\t\tNumber of rounds to run (Default: 1)')
    print ('-s [0.1,..]\t\t\tLearning rate for backprop (Default: 0.001)')
    print ('-m [0.0,0.5,..]\t\t\tMomentum for gradient (Default: 0.9)')
    print ('-e [10,50,..]\t\t\tNumber of epoches to run (Default: 100)')
    print ('--save [0,1]\t\t\tSave the result graphs and weights (Default: 1)')
    print ('--test [FILE]\t\t\tPerform testing on test data (Default: off)')
    print ('--BN \t\t\t\tPerform batch normalization (Default: 0)')
    print ('--DO (0, 1] \t\t\tPerform dropout with give rate (Default: 1)')
    print ('-fn [int]\t\t\tSpecify number of filters (Default: 1)')
    print ('-fs [int]\t\t\tSpecify size of filters (Default: 5x5)')
    print ('-ps [int]\t\t\tSpecify max pooling size (Default: 2x2)')
    print ('-name [string] [int]\t\tSpecify job name, and the # of trial.')
    print ('')
    exit()




# user specified input parameters or default
if '-t' in sys.argv:    # train data
    t_ind = sys.argv[sys.argv.index('-t') + 1]
else:
    t_ind = "data/train.txt"
if '-v' in sys.argv:    # val data
    v_ind = sys.argv[sys.argv.index('-v') + 1]
else:
    v_ind = "data/val.txt"
if '-l' in sys.argv:    # num of layer
    l_ind = int(sys.argv[sys.argv.index('-l') + 1])
else:
    l_ind = 1
if '-n' in sys.argv:    # num of hidden nodes
    n_ind = int(sys.argv[sys.argv.index('-n') + 1])
else:
    n_ind = 100
if '-b' in sys.argv:    # size of mini-batches
    BATCH = int(sys.argv[sys.argv.index('-b') + 1])
if '-r' in sys.argv:    # number of rounds to run
    ROUND = int(sys.argv[sys.argv.index('-r') + 1])
if '-s' in sys.argv:    # step of backprop
    RATE = float(sys.argv[sys.argv.index('-s') + 1])
if '-m' in sys.argv:    # momentum for gradient
    MOMENTUM = float(sys.argv[sys.argv.index('-m') + 1])
if '-e' in sys.argv:    # number of epoches
    EPOCH = int(sys.argv[sys.argv.index('-e') + 1])
if '--BN' in sys.argv:   # batch normalization
    BN = 1
if '--DO' in sys.argv:  # drop out
    DROP = 1
    # in fact is the keep rate, type 0.7 means DROP_rate = 0.3, which means
    # randoming 1 with prob 0.3 and 0 with 0.7
    DROP_rate = 1.0 - float(sys.argv[sys.argv.index('--DO') + 1])
if '-fs' in sys.argv:   # filter size
    filtersize = int(sys.argv[sys.argv.index('-fs') + 1])
if '-fn' in sys.argv:   # filter number
    features = int(sys.argv[sys.argv.index('-fn') + 1])
if '-ps' in sys.argv:   # pool size
    pooling = int(sys.argv[sys.argv.index('-ps') + 1])
if 'small' in sys.argv: # small set
    small = 1
if '-name' in sys.argv:
    fname = sys.argv[sys.argv.index('-name') + 1]
    trial_n = int(sys.argv[sys.argv.index('-name') + 2])


if '--save' in sys.argv:    # if saving data
    SAVE = int(sys.argv[sys.argv.index('--save') + 1])
if '--test' in sys.argv:    # if testing
    test_ind = sys.argv[sys.argv.index('--test') + 1]
    TEST = 1
else:
    test_ind = "No"

if fname == "cnn":
    print("define a proper name for the output graph.")
    exit()

print("Running Neural Network with the following settings: ")
print ("\tTraining data: {}\n\tValidation data: {}\n\tNumber of layers: {}".format(
                t_ind, v_ind, l_ind))
print ("\tSize of hidden layer: {}\n\tSize of mini batch: {}\n\tNumber of rounds: {}".format(
                n_ind, BATCH, ROUND))
print ("\tLearning Rate: {}\n\tMomentum: {}\n\tSave graph/weights: {}".format(RATE,
                MOMENTUM, SAVE))
print ("\tTest data: {}\n\tDropout rate: {}\n\tBatch Normalization: {}\n".format(test_ind,
                1 - DROP_rate, BN))

print ("Loading Data..")
start = time.time()
with open("sampledCIFAR10", "rb") as f:
    data = pickle.load(f)
    # each consists of a dict, with key->"labels" to labels,
    # key->"data" to data
    train, val, test = data["train"], data["val"], data["test"]

# assigning data/labels
train_data = train['data']
train_label = train['labels']
val_data = val['data']
val_label = val['labels']
test_data = test['data']
test_label = test['labels']

train_data = train_data.reshape((train_data.shape[0], 3, imgsize, imgsize)).transpose(0,2,3,1)
val_data = val_data.reshape((val_data.shape[0], 3, imgsize, imgsize)).transpose(0,2,3,1)
test_data = test_data.reshape((test_data.shape[0], 3, imgsize, imgsize)).transpose(0,2,3,1)

####### testing on smaller training etc
if small:
    np.random.seed(int(time.time())+1234)
    rand = np.arange(train_data.shape[0])
    rand = np.random.choice(rand, 2000, replace=False)
    train_data = train_data[rand]
    train_label = train_label[rand]
    rand = np.arange(val_data.shape[0])
    rand = np.random.choice(rand, 2000, replace=False)
    val_data = val_data[rand]
    val_label = val_label[rand]
    rand = np.arange(test_data.shape[0])
    rand = np.random.choice(rand, 2000, replace=False)    
    test_data = test_data[rand]
    test_label = test_label[rand]
print("Train data shape:", train_data.shape)

duration = time.time() - start
print("Finished loading data! Took {0:.2f} seconds\n".format(duration))

#plt.imshow(test_data[0])
#plt.show()

hidden_node = int(n_ind)  # number of hidden nodes
num_layer = int(l_ind)  # number of layers

# insert bias term to the front

# convert labels to vectors
train_label = convertLabel(train_label)
val_label = convertLabel(val_label)
test_label = convertLabel(test_label)

# layer dimension (include bias term)
dims = []
dims.append(len(train_data[0])) # append input dimension
for i in range(num_layer):  # append hidden layer dimension
    dims.append(hidden_node)
dims.append(OUT)    # append output layer dimension
print ("dimensions are:", dims)

# plot for the write-up
#exp = train[0:3]
#for item in exp:
#    pixels = np.array([int(x*256) for x in item[0]], dtype='uint8')
#    pixels = pixels.reshape((DIM,DIM*2))
#    
#    plt.title('Sum is %d' % item[1])
#    plt.imshow(pixels, cmap='gray')
#    plt.figure()
#plt.show()

for repeat in range(ROUND):
    print ("Start initializing weights..")
    start = time.time()
    seed = int(time.time()) + 1234  # initial seed for random

    # conv filter initialize weights
    # default is 1x10x10x3
    stdev = np.sqrt(1.0/(filtersize**2*3*features))
    filters = np.random.uniform(-2, 2, size=(features, filtersize, filtersize, 3))

    # im2col filters
    #filters = filters.reshape((filters.shape[0], filtersize**2 * 3))
    #print ("filters shape after im2col:", filters.shape)
    
    # weights
    # (included bias term)
    weights = []
    # output of max-pool to fully connected layer of size 100 ReLU
    weights.append(getWeights((imgsize//pooling)**2 * features + 1, n_ind, seed))

    # weights from ReLU layer to output
    weights.append(getWeights(n_ind + 1, OUT, seed))

    ############## original ###########
    # randomize initial weights (include bias term)
    #weights = [filters]
    #for i in range(num_layer + 1):
    #    weights.append(getWeights(dims[i] + 1, dims[i+1], seed))
    ####################################
    duration = time.time() - start
    print ("Finished weight initialization! Took {0:.2f} seconds".format(duration))
    print ("weight shapes are: ", [weights[x].shape for x in range(len(weights))])

    ############ stochastic gradient descent ###############
    sgd_weights = [np.copy(w) for w in weights]
    sgd_filters = np.copy(filters)
    best_filters = []
    best_val_loss = sys.maxsize  # prev loss initialized to maximum 
    best_train_loss = sys.maxsize
    best_val_class = 1
    best_train_class = 1
    best_gamma = []
    best_beta = []
    bset_params = []
    best_weights = []   # way to store the best weight
    best_epoch = 0      # best epoch number
    val_loss = []       # store all validation loss for graph
    train_loss  = []    # same for train loss
    val_error = []    # for another graph of classification ERROR
    train_error = []  # smae for train

    # initialize gamma, beta
    gamma = []
    beta = []
    bn_params = {}
    for i in range(l_ind):
        gamma.append(np.ones((1, weights[i].shape[1])))
        beta.append(np.zeros((1, weights[i].shape[1])))

    start = time.time()
    # mini-batch size of [BATCH] (default 32), train for at least 150 epoches or
    # early stopping when validation set cross-entropy starts to increase
    for z in range(EPOCH):
        # shuffle data before starting another epoch
        randomizer = np.arange(train_data.shape[0])
        np.random.seed(1234 + z * 2345)
        np.random.shuffle(randomizer)

        #print np.sum(train_data)

        # initialize momentum
        momentums = []
        filters_mom = []
        for m in range(len(sgd_weights)):
            momentums.append(np.zeros(sgd_weights[m].shape))
        filters_mom = np.zeros(sgd_filters.shape)

        #print randomizer
        train_data = train_data[randomizer]
        train_label = train_label[randomizer]
        
        #im2col_train = im2col_train[randomizer]
        #back_train = back_train[randomizer]

        # run a whole epoch = run all training data
        #print train_data.shape
        #print "Start training..."
        for l in range(0, train_data.shape[0], BATCH):
            modifiers = []

            # use im2col data
            if l + BATCH >= train_data.shape[0]:
                #batch = im2col_train[l:im2col_train.shape[0],:]
                batch = train_data[l:train_data.shape[0],:]
                batch_label = train_label[l:train_data.shape[0],:]
                #back_batch = back_train[l:back_train.shape[0],:]
            else:
                #batch = im2col_train[l:l+BATCH,:]
                batch = train_data[l:l + BATCH,:]
                batch_label = train_label[l: l + BATCH,:]
                #back_batch = back_train[l: l+ BATCH, :]

            batch_size = batch.shape[0]
            #print "batch size (original image batch):", batch.shape
            #print np.sum(batch)

            #------------------ convolutional steps -----------------#
            # STEP 1: run through the input data to convolutional forward
            # RETURN: a original image size tensor (i.e. batch_size x 32 x 32)
            # to be passed to ReLU function
            # rest are run in conv_forward function
            padded_layer, pooled, pooled_mask, biased_pooled = conv_forward(batch,
                                        padding, stride, sgd_filters, pooling)

            # expected to pass a 2D array, size (batch_size, num_pools *
            # num_filters). i.e. (64, 16x16x1)

            #--------------------------------------------------------#

            #print(biased_pooled.shape)
            # run the batch for stochastic gradient descent
            layers, all_cache, all_mean, all_var = trainNetwork(biased_pooled,
                                sgd_weights, 1, gamma, beta, bn_params)

            # store the running mean and var
            if 'mean' not in bn_params:
                bn_params['mean'] = all_mean
            else:
                bn_params['mean'] = [0.9 * bn_params['mean'][x] + 0.1 * all_mean[x]
                                    for x in range(len(all_mean))]
            if 'var' not in bn_params:
                bn_params['var'] = all_var
            else:
                bn_params['var'] = [0.9 * bn_params['var'][x] + 0.1 * all_var[x]
                                    for x in range(len(all_var))]
            
            
            height = len(layers) - 1   # the height in layers
            #for l in layers:
            #    print (l.shape)
            #exit()
            # backprop for convolutional layers
            intmed = layers[height] - batch_label
            intmed = np.dot(intmed, sgd_weights[1][1:,:].T) # to ReLU layer
            # at this point --> 64x100
            intmed = np.multiply(intmed, layers[height-1][:,1:])   # pass through ReLU
            # at this point --> 64x100
            intmed = np.dot(intmed, sgd_weights[0][1:,:].T)
            # at this point --> 64x9216
            intmed = intmed.reshape((intmed.shape[0], pooled.shape[1], pooled.shape[2],
                                pooled.shape[3]))
 
            # the backprop for the max pool (back to ReLU output) is literlly
            # the same as that of ReLU gradient (since only ReLU derivative 1
            # will only have effect on those cells > 0; the rest are 0)
            back_pooled = np.multiply(pooled_mask,
                    intmed.repeat(pooling, axis=1).repeat(pooling, axis=2))

            # now do grad for the filters. Basically a convolution operation
            # of X * back_pooled
            # ----- batch is padded already -----#
            f_grad = conv_backward(padded_layer, back_pooled, stride, filtersize,
                                    batch_size)
            #print(f_grad)
            sgd_filters = sgd_filters - RATE * (f_grad + MOMENTUM * filters_mom)
            filters_mom = f_grad
           

            # backprop based on the layer/output
            # through the top weights to the bottom weights
            for k in range(1, len(layers), 1):
                grad, dX, dgamma, dbeta = backprop(height, k, sgd_weights, layers,
                                    batch_label, modifiers, all_cache, batch_size)
                # if running batch normalization, update gamma/beta
                if BN and k < len(layers) - 1:
                    gamma[k-1] -= np.sum(dgamma, axis=0)
                    beta[k-1] -= np.sum(dbeta, axis=0)

                # update
                sgd_weights[k-1] = sgd_weights[k-1] - RATE * (grad +
                                        MOMENTUM * momentums[k-1])
                momentums[k-1] = grad
        
        #print (time.time() - start)
        #print pooled
        #print sgd_filters
        print ("Finished {} epoch(es), now evaluating..".format(z+1))
        # after each epoch update, print out new loss
        
        _,_,_,biased_pooled = conv_forward(train_data, padding,
                stride, sgd_filters, pooling)
        t_o = trainNetwork(biased_pooled, sgd_weights, 0, gamma, beta, bn_params)[0]
        train_output = t_o[len(t_o)-1]
        #print train_output
        _,_,_,biased_pooled = conv_forward(val_data, padding,
                stride, sgd_filters, pooling)
        v_o = trainNetwork(biased_pooled, sgd_weights, 0, gamma, beta, bn_params)[0]
        val_output = v_o[len(v_o)-1]
        #print val_output
        correct_train = 0
        correct_val = 0
        #print train_output.shape, val_output.shape
        for i in range(train_output.shape[0]):
            predict_train = np.argmax(train_output[i])
            if i < val_output.shape[0]:
                predict_val = np.argmax(val_output[i])
                if val_label[i, predict_val] == 1:
                    correct_val += 1

            if train_label[i, predict_train] == 1:
                correct_train += 1
        new_train_class = 100 * float(train_output.shape[0] - correct_train)/train_output.shape[0]
        new_val_class = 100 * float(val_output.shape[0] - correct_val)/val_output.shape[0]
        print("error prediction train: {}".format(new_train_class))
        print("error prediction val: {}".format(new_val_class))
        val_error.append(new_val_class)
        train_error.append(new_train_class)

        new_train_loss = avg_CEloss(train_label, train_output)
        new_val_loss = avg_CEloss(val_label, val_output)

        val_loss.append(new_val_loss)
        train_loss.append(new_train_loss)

        # find best epoch (minimum val loss)
        if new_val_loss < best_val_loss or new_val_class < best_val_class:
            best_val_loss = new_val_loss    # early stopping criteria
            best_train_loss = new_train_loss
            best_train_class = new_train_class
            best_val_class = new_val_class
            best_weights = list(sgd_weights)
            best_filters = np.copy(sgd_filters)
            best_epoch = z + 1

            best_gamma = gamma
            best_beta = beta
            best_params = bn_params
        #else:
        #    print "Early stopping because validation set error starts to increase."
        #    break
        print ("Train loss after {0} epoch: {1:.5f}".format(z+1, new_train_loss))
        print ("Validation loss after {0} epoch: {1:.5f}\n".format(z+1, new_val_loss))

    # finish SGD
    print ('STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED!.')
    print ('Total epoches: {}'.format(EPOCH))
    print ('Best epoch: {}'.format(best_epoch))
    print ('Best weight train cross-entropy loss: {}'.format(best_train_loss))
    print ('Best train error classification rate: {0:.2f}{1}'.format(best_train_class, "%"))
    print ('Best val cross-entropy loss: {}'.format(best_val_loss))
    print ('Best val error classification rate: {0:.2f}{1}'.format(best_val_class, "%"))

    # if testing
    if TEST:
        _,_,_,biased_pooled = conv_forward(test_data, padding,
            stride, best_filters, pooling)
        test_layers = trainNetwork(biased_pooled, best_weights, 0, best_gamma,
                                    best_beta, best_params)[0]
        test_output = test_layers[len(test_layers)-1]
        test_loss = avg_CEloss(test_label, test_output)
        test_correct = 0
        for i in range(test_output.shape[0]):
            predict_test = np.argmax(test_output[i])
            if test_label[i, predict_test] == 1:
                test_correct += 1
        test_error_rate = 100 * (float(test_output.shape[0] - test_correct) /
                                        test_output.shape[0])
        print ('Best weights on test data --------')
        print ('test cross-entropy loss: {}'.format(test_loss))
        print ('test error classification rate: {0:.2f}{1}'.format(test_error_rate, "%"))

    duration = time.time() - start
    print ("Training took {0} minutes {1:.2f} seconds\n".format(int(duration // 60),
                                duration % 60))
    
    print ("Number of filter maps: {}\nPooling size: {}\nLearning rate: {}".format(features,
                            pooling, RATE))

    # plot the data x-axis as num of epoch, y-axis as loss
    x_axis = [x for x in range(1, len(val_loss) + 1)]
    plt.xlabel("# of epoches")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train/Val Loss, {0} filters, {1}x{1} max pooling".format(features, pooling))
    plt.plot(x_axis, train_loss, 'b-', label='Train Loss')
    plt.plot(x_axis, val_loss, 'r-', label='Validation Loss')
    plt.plot([],[], '', label='Learning Rate is {0:.5f}'.format(RATE))
    plt.plot([],[], '', label='Momentum is {0:.5f}'.format(MOMENTUM))
    plt.axvline(x=best_epoch, linewidth=0.3, linestyle='--')
    plt.text(best_epoch + 1, best_val_loss, "Best val: {0:.2f}".format(best_val_loss))
    plt.text(best_epoch + 1, best_train_loss, "Best train: {0:.2f}".format(best_train_loss))
    plt.legend()
    if SAVE and TEST:
        plt.savefig("write_up/{0}_loss_{1}.png".format(fname, trial_n))
    else:
        plt.show()
    plt.cla()

    # plot for question b)
    plt.xlabel("# of epoches")
    plt.ylabel("Classification Error %")
    plt.title("Train/Val Error, {0} filters, {1}x{1} max pooling".format(features, pooling))
    plt.plot(x_axis, train_error, 'b-', label='Train Error %')
    plt.plot(x_axis, val_error, 'r-', label='Validation Error %')
    plt.plot([],[], '', label='Learning Rate is {0:.5f}'.format(RATE))  
    plt.plot([],[], '', label='Momentum is {0:.5f}'.format(MOMENTUM))
    plt.legend()
    if SAVE and TEST:
        plt.savefig("write_up/{0}_error_{1}.png".format(fname, trial_n))
    else:
        plt.show()
    plt.cla()
    
    # save the weights to a .txt file
    if SAVE and TEST:
        best_filters = best_filters.reshape(best_filters.shape[0], -1)
        np.savetxt('weights_{0}_{1}.txt'.format(fname, trial_n), best_filters)
