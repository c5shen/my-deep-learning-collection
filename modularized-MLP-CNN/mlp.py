import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle

RATE = 0.0001      # learning rate for stochastic G.D.
EPOCH = 100     # X epoches for SGD
BATCH = 64      # batch size
DIM = 32        # dimension of the image (single)
OUT = 10        # output layer number
ROUND = 1       # number of rounds to run different training (initial weights)
SAVE = 0        # by default, graph/weights are not saved
TEST = 1        # by default testing is disabled
MOMENTUM = 0.9  # by default, momentum is 0
BN = 0          # by default, not doing batch normalization
DROP = 0        # by default, no dropout
DROP_rate = 1   # by default, no dropout

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


# backprop alogrithm
def backprop(height, k, sgd_weights, layers, batch_label, modifiers, cache, batch_size):
    # height indicates where the output is
    # k from height to 1. For example, k = [3,2,1]

    dX, dgamma, dbeta = [],[],[]

    # in the form (o - t) /dot/ h
    # at softmax layer
    if k == height:
        p = layers[k] - batch_label    # softmax gradient
        grad = (np.dot(p.T, layers[k-1])).T / batch_size    # dot product to hidden layer
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

            # masking --> relu gradient
            p = np.multiply(p, (layers[i][:,1:] > 0))
        #print layers[k-1].shape
        
        # batch normalization backprop
        if BN:
            dX, dgamma, dbeta = BN_backward(p, cache[k-1])
            #print dX.shape, layers[k-1].shape
            grad = (np.dot(dX.T, layers[k-1])).T / batch_size
        else:
            grad = (np.dot(p.T, layers[k-1])).T / batch_size

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
    #all_mask = []
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

        # change to relu
        #if training:
            #all_mask.append((cur_layer > 0).astype(int))
            #inserting = np.insert(inserting, 0, 1, axis=1)
            #all_mask.append(inserting)
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
    this_output = np.dot(cur_layer, weights[len(weights) - 1])

    # change output to softmax
    maxed = np.max(this_output, axis=1)
    z = this_output - maxed.reshape((maxed.shape[0],1))
    this_output = np.exp(z)
    this_output = np.exp(this_output)
    sum_output = np.sum(this_output, axis=1)
    sum_output = np.reshape(sum_output, (sum_output.shape[0], 1))
    this_output = this_output/sum_output

    layers_return.append(this_output)
    return layers_return, all_cache, all_mean, all_var


##################### helper functions end #######################
if '-h' in sys.argv:
    # print helpful tips
    print '-t [FILE]\t\t\tTrain data file (Default: data/train.txt)'
    print '-v [FILE]\t\t\tValidation data file (Default: data/val.txt)'
    print '-l [0,1,..]\t\t\tNumber of layers (Default: 1)'
    print '-n [20,100,..]\t\tNumber of hidden nodes (Default: 100)'
    print '-b [1,5,..]\t\t\tSize of mini-batches (Default: 64)'
    print '-r [1,2,..]\t\t\tNumber of rounds to run (Default: 1)'
    print '-s [0.1,0.01,..]\tSize of learning rate for backprop (Default: 0.1)'
    print '-m [0.0,0.5,..]\t\tMomentum for gradient (Default: 0.0)'
    print '-e [10,50,..]\t\tNumber of epoches to run (Default: 150)'
    print '--save [0,1]\t\tSave the result graphs and weights (Default: 0)'
    print '--test [FILE]\t\tPerform testing on test data (Default: off)'
    print '--BN \t\t\t\tPerform batch normalization (Default: 0)'
    print '--DO (0, 1] \t\tPerform dropout with give rate (Default: 1)'
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


if '--save' in sys.argv:    # if saving data
    SAVE = int(sys.argv[sys.argv.index('--save') + 1])

if '--test' in sys.argv:    # if testing
    test_ind = sys.argv[sys.argv.index('--test') + 1]
    TEST = 1
else:
    test_ind = "No"

print "Running Neural Network with the following settings: "
print ("\tTraining data: %s\n\tValidation data: %s\n\tNumber of layers: %d") % (
                t_ind, v_ind, l_ind)
print ("\tSize of hidden layer: %d\n\tSize of mini batch: %d\n\tNumber of rounds: %d") % (
                n_ind, BATCH, ROUND)
print ("\tLearning Rate: %f\n\tMomentum: %f\n\tSave graph/weights: %d") % (RATE,
                MOMENTUM, SAVE)
print "\tTest data: %s\n\tDropout rate: %f\n\tBatch Normalization: %d\n" % (test_ind,
                1 - DROP_rate, BN)

print "Loading Data.."
start = time.time()
with open("sampledCIFAR10", "rb") as f:
    data = pickle.load(f)
    train, val, test = data["train"], data["val"], data["test"]

# assigning data/labels
train_data = train['data']
train_label = train['labels']
val_data = val['data']
val_label = val['labels']
test_data = test['data']
test_label = test['labels']

hidden_node = int(n_ind)  # number of hidden nodes
num_layer = int(l_ind)  # number of layers

train_data = np.insert(train_data, 0, 1, axis=1)    # bias term = 1
val_data = np.insert(val_data, 0, 1, axis=1)    # bias term = 1
test_data = np.insert(test_data, 0, 1, axis=1)  # bias term = 1
duration = time.time() - start
print "Finished loading data! Took %.02f seconds\n" % duration

# convert labels to vectors
train_label = convertLabel(train_label)
val_label = convertLabel(val_label)
test_label = convertLabel(test_label)

# layer dimension (include bias term)
dims = []
dims.append(len(train_data[0]) - 1) # append input dimension
for i in range(num_layer):  # append hidden layer dimension
    dims.append(hidden_node)
dims.append(OUT)    # append output layer dimension
print "dimensions are:", dims

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
    print "Start initializing weights.."
    start = time.time()
    # randomize initial weights (include bias term)
    weights = []
    seed = int(time.time()) + 1234  # initial seed for random
    for i in range(num_layer + 1):
        weights.append(getWeights(dims[i] + 1, dims[i+1], seed))
    for w in weights:
        print w.shape
    duration = time.time() - start
    print "Finished weight initialization! Took %.02f seconds\n" % duration

    ############ stochastic gradient descent ###############
    sgd_weights = [np.copy(w) for w in weights]
    best_val_loss = sys.maxint  # prev loss initialized to maximum 
    best_train_loss = sys.maxint
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
        np.random.shuffle(randomizer)

        #print np.sum(train_data)

        # initialize momentum
        momentums = []
        for m in range(len(sgd_weights)):
            momentums.append(np.zeros((
                    sgd_weights[m].shape[0],sgd_weights[m].shape[1])))

        #print randomizer
        train_data = train_data[randomizer,:]
        train_label = train_label[randomizer]

        # run a whole epoch = run all training data
        #print train_data.shape
        #print "Start training..."
        for l in range(0, train_data.shape[0], BATCH):
            modifiers = []

            if l + BATCH >= train_data.shape[0]:
                batch = train_data[l:train_data.shape[0],:]
                batch_label = train_label[l:train_data.shape[0],:]
            else:
                batch = train_data[l:l + BATCH,:]
                batch_label = train_label[l: l + BATCH,:]

            batch_size = batch.shape[0]
            #print np.sum(batch)
            # run the batch for stochastic gradient descent
            layers, all_cache, all_mean, all_var = trainNetwork(batch,
                                sgd_weights, 1, gamma, beta, bn_params)
            
            #for l in layers:
            #    print(l.shape)
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

            #print np.sum(layers[0])
            #if DROP:
            #    layers.insert(0, batch * modifiers[0])
            #else:
            #    layers.insert(0, batch)
            #print layers[0]
            #print layers[1]
            #print layers[2]
            #print modifiers
            #exit()
            #print len(modifiers),modifiers[0].shape, modifiers[1].shape
            #print "size of hidden layer --> %s" % (h_layer.shape,)
            #print "size of softmax layer --> %s" % (s_output.shape,)
            
            # backprop based on the layer/output
            # through the top weights to the bottom weights
            height = len(layers) - 1   # the height in layers 
            for k in range(1, len(layers), 1):
                grad, dX, dgamma, dbeta = backprop(height, k, sgd_weights, layers,
                                    batch_label, modifiers, all_cache, batch_size)
                #print grad.shape
                # if running batch normalization, update gamma/beta
                if BN and k < len(layers) - 1:
                    gamma[k-1] -= np.sum(dgamma, axis=0)
                    beta[k-1] -= np.sum(dbeta, axis=0)

                # update
                sgd_weights[k-1] = sgd_weights[k-1] - RATE * (grad +
                                        MOMENTUM * momentums[k-1])
                momentums[k-1] = grad
        
        print "Finished %d epoch(es), now evaluating.." % (z+1)
        # after each epoch update, print out new loss    
        t_o = trainNetwork(train_data, sgd_weights, 0, gamma, beta, bn_params)[0]
        train_output = t_o[len(t_o)-1]
        #print train_output
        v_o = trainNetwork(val_data, sgd_weights, 0, gamma, beta, bn_params)[0]
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
        print "error prediction train: %f" % new_train_class
        print "error prediction val: %f" % new_val_class
        val_error.append(new_val_class)
        train_error.append(new_train_class)

        new_train_loss = avg_CEloss(train_label, train_output)
        new_val_loss = avg_CEloss(val_label, val_output)

        val_loss.append(new_val_loss)
        train_loss.append(new_train_loss)

        # find best epoch (minimum val loss)
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss    # early stopping criteria
            best_train_loss = new_train_loss
            best_train_class = new_train_class
            best_val_class = new_val_class
            best_weights = list(sgd_weights)
            best_epoch = z + 1

            best_gamma = gamma
            best_beta = beta
            best_params = bn_params
        #else:
        #    print "Early stopping because validation set error starts to increase."
        #    break
        print "Train loss after %d epoch: %f" % (z+1, new_train_loss)
        print "Validation loss after %d epoch: %f\n" % (z+1, new_val_loss)

    # finish SGD
    print 'STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED!.'
    print 'Total epoches: %d' % EPOCH
    print 'Best epoch: %d' % best_epoch
    print 'Best weight train cross-entropy loss: %f' % best_train_loss
    print 'Best train error classification rate: %.02f%%' % best_train_class
    print 'Best val cross-entropy loss: %f' % best_val_loss
    print 'Best val error classification rate: %.02f%%' % best_val_class

    # if testing
    if TEST:
        test_layers = trainNetwork(test_data, best_weights, 0, best_gamma,
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
        print 'Best weights on test data --------'
        print 'test cross-entropy loss: %f' % test_loss
        print 'test error classification rate: %.02f%%' % test_error_rate

    duration = time.time() - start
    print "Training took %d minutes %.02f seconds\n" % (duration / 60,
                                duration % 60)

    # plot the data x-axis as num of epoch, y-axis as loss
    x_axis = [x for x in range(1, len(val_loss) + 1)]
    plt.xlabel("# of epoches")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train/Validation Loss Per Epoch (batch size = %d)" % BATCH)
    plt.plot(x_axis, train_loss, 'b-', label='Train Loss')
    plt.plot(x_axis, val_loss, 'r-', label='Validation Loss')
    plt.plot([],[], '', label='Learning Rate is %.02f' % RATE)
    plt.plot([],[], '', label='Momentum is %.02f' % MOMENTUM)
    plt.axvline(x=best_epoch, linewidth=0.3, linestyle='--')
    plt.text(best_epoch + 1, best_val_loss, "Best val: %.02f" % best_val_loss)
    plt.text(best_epoch + 1, best_train_loss, "Best train: %.02f" % best_train_loss)
    plt.legend()
    if SAVE and not TEST:
        plt.savefig("write_up/q5_a_%d.png" % (repeat + 1))
    else:
        plt.show()
    plt.cla()

    # plot for question b)
    plt.xlabel("# of epoches")
    plt.ylabel("Classification Error %")
    plt.title("Train/Validation Classification Error Per Epoch (batch size = %d" % BATCH)
    plt.plot(x_axis, train_error, 'b-', label='Train Error %')
    plt.plot(x_axis, val_error, 'r-', label='Validation Error %')
    plt.plot([],[], '', label='Learning Rate is %.02f' % RATE)  
    plt.plot([],[], '', label='Momentum is %.02f' % MOMENTUM)    
    plt.legend()
    if SAVE and not TEST:
        plt.savefig("write_up/q5_b_%d.png" % (repeat + 1))
    else:
        plt.show()
    plt.cla()
    
    # save the weights to a .txt file
    if SAVE and TEST:
        for i in range(len(best_weights)):
            np.savetxt('weights_%d_run%d.txt' % (i + 1, (repeat + 1)), best_weights[i])
