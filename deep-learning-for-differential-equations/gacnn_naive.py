"""
Naive GACNN model for learning a PDE using Neural network.
--> Chengze Shen, Senyu Tong <--
Currently tested on Laplace equation
Learning rate: 0.001
Momentum: 0.0
Epochs: 100
Points per epochs: all discrete points in domain D (20x20)
The results are shown using 3D graphs
"""

import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# constants
K = 20      # number of division to the domain
domain_x = 1.0  # boundary of x1
domain_y = 1.0  # boundary of x2
domain = 1.0    # boundary for generalized case

batch_size = 32     # batch size
EPOCH = 100         # number of epochs
rate = 0.001        # learning rate
drop_rate = 0.5     # drop out rate for neurons
num_layer = 1       # number of hidden layers
hidden_size = 100   # hidden layer size (# neurons)
OUT = 1             # output dimension
DROP = 0            # whether to dropout
save_weights = 0    # whether to save weights after training
load_weights = 0    # whether to load weights from current directory


# def the CNN version of the network
# 1 in-channel, 2 out-channels, kernel size 3
# 20 x 20 --> padding --> 22 x 22 --> conv --> 20 x 20
class Net(nn.Module):
    def __init__(self, num_filter=2):
        self.num_filter = num_filter
        super(Net, self).__init__()
        self.conv = nn.Conv2d(2, num_filter, 3, padding=1)
        self.pool = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(10 * 10, 64)
        self.fc2 = nn.Linear(64, K*K)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 10 * 10)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# define some toy first order PDE solution
# 3 * d/dx + d/dy = 0
def analytic_first_order(x):
    return (x[0] - 3*x[1]) * np.exp(-(x[0]-3*x[1])**2)

# initial condition for toy first order PDE solution
def init_first_order(x):
    return x * np.exp(-(x**2))

# analytical solution to Laplace equation
def analytic_solution(x):
    return ((1 / (np.exp(np.pi) - np.exp(-np.pi))) * 
            np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1])))

# right hand side of the equation (0, in Laplace equation's case)
def f(x):
    return 0.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# baseline model of one layer of sigmoid activation and one linear output neuron
def neural_network(W, x, if_train):
    layers = []
    #modifiers = []
    for w in range(len(W) - 1):
        if w == 0:
            cur_layer = np.dot(x, W[w])
        else:
            cur_layer = np.dot(cur_layer, W[w])
        
        # change to sigmoid
        cur_layer = sigmoid(cur_layer)
        #print(cur_layer.shape)
        #cur_layer = np.insert(cur_layer, 0, 1, axis=1)

        # drop out hidden neurons
        if DROP and if_train:
            cur_modifier = npr.binomial(1, drop_rate, cur_layer.shape)
            cur_modifier[:,0] = 1
            cur_layer *= cur_modifier
            #modifiers.append(cur_modifier)

        layers.append(cur_layer)

    # calculate the output layer: basically a linear combination
    output = np.dot(cur_layer, W[len(W) - 1])
    layers.append(output)
    return layers

# function to draw result based on current W
def graph_result(nx, ny, x_space, y_space, W, cur_epoch):
    surface = np.zeros((ny, nx))
    surface2 = np.zeros((ny, nx))

    # output analytic and trial solution
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = analytic_solution([x, y])            
            
            net_outt = neural_network(W, [x, y], 0)
            net_outt = net_outt[len(net_outt) - 1]
            surface2[i][j] = psy_trial([x, y], net_outt)
    
    # error
    errors = surface2 - surface
    
    # graph both analytical and NN solution on the same graph
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    #b_proxy = plt.Rectangle((0,0), 1, 1, fc='b')
    #y_proxy = plt.Rectangle((0,0), 1, 1, fc='y')
    surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.Oranges,
            linewidth=0, antialiased=False, alpha=1)
    surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    #ax.legend([y_proxy, b_proxy], ["analytical", "neural network"])
    ax.set_title("Result at epoch {}".format(cur_epoch))
    fig.savefig("Result_at_epoch_{}".format(cur_epoch), dpi=600)

    # graph error rate
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf3 = ax.plot_surface(X, Y, errors, rstride=1, cstride=1, cmap=cm.Reds,
            linewidth=0, antialiased=False)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$')
    ax.set_title("Error rate at epoch {}".format(cur_epoch))
    fig.savefig("Error_at_epoch_{}".format(cur_epoch), dpi=600)
    
# the boundary function A(x)
def A(x):
    return x[1] * np.sin(np.pi * x[0])

# the trial solution psy_t, A(x) + B(x)*N(x,p)
def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out

def psy_nonchange(x, net_out):
    return net_out

# loss function for trial solution
# input: W as weight vector, pairs as batch input data points (np array)
# output: batch loss sum
def loss_function(W, pairs):
    loss_sum = 0.
    
    # net_out of size BATCH*1
    layers = neural_network(W, pairs, 1)
    net_out = layers[len(layers) - 1]

    # iterate through all data points given (data points in batches)
    for i in range(0, pairs.shape[0]):
        psy_t = psy_trial(pairs[i], net_out[i][0])
        psy_t_jacobian = jacobian(psy_trial)(pairs[i], net_out[i][0])
        psy_t_hessian = jacobian(jacobian(psy_trial))(pairs[i], net_out[i][0])

        #print(psy_t_hessian)
        gradient_of_trial_d2x = psy_t_hessian[0][0]
        gradient_of_trial_d2y = psy_t_hessian[1][1]

        func = f(pairs[i]) # right part function

        err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
        loss_sum += err_sqr
       
    # return the batch loss sum
    return loss_sum

######## helper functions end ###########

if __name__ == "__main__":
    if '-dim' in sys.argv:  # input dimension
        input_dim = int(sys.argv[sys.argv.index('-l') + 1])
    else:
        input_dim = 2
    if '-l' in sys.argv:    # num of layer
        num_layer = int(sys.argv[sys.argv.index('-l') + 1])
    if '-n' in sys.argv:    # num of hidden nodes
        hidden_size = int(sys.argv[sys.argv.index('-n') + 1])
    if '-b' in sys.argv:    # size of mini-batches
        batch_size = int(sys.argv[sys.argv.index('-b') + 1])
    if '-e' in sys.argv:    # number of epoches
        EPOCH = int(sys.argv[sys.argv.index('-e') + 1])
    if 'save_weights' in sys.argv:
        save_weights = 1
    if 'load_weights' in sys.argv:
        load_weights = 1
    if '--DO' in sys.argv:  # drop out
        DROP = 1
        # in fact is the keep rate, type 0.7 means DROP_rate = 0.3, which means
        # randoming 1 with prob 0.3 and 0 with 0.7
        DROP_rate = 1.0 - float(sys.argv[sys.argv.index('--DO') + 1])
 

    ######## CNN version #########
    net = Net()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

    # generate discretized data points
    nx = K
    ny = K
    x_space = np.linspace(0, domain_x, nx)
    y_space = np.linspace(0, domain_y, ny)

    data = np.array([])
    for xi in x_space:
        for yi in y_space:
            #print(xi,yi)
            if data.shape[0] == 0:
                data = np.array([xi, yi])
            else:
                data = np.vstack((data,[xi, yi]))

    for epoch in range(5000):
        running_loss = 0.0
        optimizer.zero_grad()
        inputs = torch.Tensor(data.reshape((1,K,K,2)).transpose([0,3,1,2]))
        outputs = net(inputs)   # shape 1x400
        targets = torch.zeros(outputs[1].shape)
        # update LHS of loss for each output unit
        for i in range(data.shape[0]):
            targets[i] = analytic_first_order(data[i])
        
        loss = criterion(outputs[0] + outputs[1], targets)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        print(running_loss)
        # save fig
        if epoch == 4999 or epoch == 9 or epoch == 2499:
            outputs = outputs.reshape((2,K,K))
            #print(outputs.shape)
            surface = np.zeros((ny, nx))    # the GACNN outputs
            surface2 = np.zeros((ny, nx))    
            
            # get analytic solutions
            for i, x in enumerate(x_space):
                surface[i][0] = init_first_order(x)
                for j, y in enumerate(y_space):
                    surface2[i][j] = analytic_first_order([x, y])
                    surface[i][j] = outputs[0,i,j] + outputs[1,i,j]

            #print(surface[:,0])
            # get GACNN solutions
            #for j in range(1, len(y_space)):
                #print(outputs_x[:,j])
            #    surface[:,j] = surface[:,j-1] + outputs_x[:,j].detach().numpy() + outputs_y[:,j].detach().numpy()

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y = np.meshgrid(x_space, y_space)
            surf2 = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.Oranges,
                linewidth=0, antialiased=False, alpha=0.7)
            surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                linewidth=0, antialiased=False, alpha=0.5)
            ax.set_xlabel("x2")
            ax.set_ylabel("x1")
            ax.set_title("Result at epoch {}".format(epoch+1))
            ax.view_init(elev=20.,azim=324)
            #plt.show()
            fig.savefig("gacnn_naive_epoch{}".format(epoch+1), dpi=600)
            #if epoch == 9:
            #    for ii in range(0,360,2):
            #        ax.view_init(elev=20.,azim=ii)
            #        fig.savefig("test/movie{}.png".format(ii))
    exit()

    ##############################

    # number of intervals for input data --> (0,0), (0.05, 0.15) etc.
    dn = []
    nx = K
    ny = K
    test_nx = 2*K
    test_ny = 2*K

    dx = domain_x / nx
    dy = domain_y / ny

    input_space = []
    test_x_space = np.linspace(0, domain_x, test_nx)
    test_y_space = np.linspace(0, domain_y, test_ny)
    x_space = np.linspace(0, domain_x, nx)
    y_space = np.linspace(0, domain_y, ny)

    #for i in range(0, input_dim):
    #    dn.append(domain/K)
    #    input_space.append(np.linspace(0, domain, K))

    # initialize weights
    W = []
    dims = [input_dim]  # dims across all layers
    for i in range(0, num_layer):
        dims.append(hidden_size)
    dims.append(OUT)
    print("dimensions are: {}".format(dims))

    # load weights from current directory
    model_trained = 0
    if load_weights:
        model_trained = 1
        index = 0
        cur_weight = "W_{}.out".format(index)
        while(os.path.isfile(cur_weight)):
            W.append(np.loadtxt(cur_weight, delimiter=','))
            index += 1
            cur_weight = "W_{}.out".format(index)
        # reshape last weight
        W[len(W)-1] = W[len(W)-1].reshape((W[len(W)-1].shape[0],1))
    else:
        W = []
        for i in range(0, num_layer + 1):
            W.append(npr.randn(dims[i], dims[i+1]))
   
    # generate discretized data points
    data = np.array([])
    for xi in x_space:
        for yi in y_space:
            #print(xi,yi)
            if data.shape[0] == 0:
                data = np.array([xi, yi])
            else:
                data = np.vstack((data,[xi, yi]))
    # generate test data
    test_data = np.array([])
    for xi in test_x_space:
        for yi in test_y_space:
            if test_data.shape[0] == 0:
                test_data = np.array([xi, yi])
            else:
                test_data = np.vstack((test_data, [xi, yi]))

    # training the neural net, 100 iterations, learning rate 0.001
    start = time.time()
    test_losses = []
    train_losses = []
    for i in range(EPOCH):
        if model_trained:   # if already loaded trained weights, break
            break
        # graph at 1, 50, 100 epoch
        if (i == 0 or i == 49 or i == 99):
            graph_result(nx, ny, x_space, y_space, W, i)

        #print("epoch #{}".format(i+1))
        # create batch from all data randomed
        npr.shuffle(data)
        
        # run for each batch
        for b in range(0, data.shape[0], batch_size):
            if (b + batch_size > data.shape[0]):
                batch = data[b:data.shape[0],:]
            else:
                batch = data[b:b+batch_size,:]
            
            # record each batch-loss
            # use autograd for gradient calculation
            loss_grad =  grad(loss_function)(W, batch)

            for w in range(len(W)):
                W[w] = W[w] - rate * loss_grad[w]
        epoch_loss = loss_function(W, test_data) / test_data.shape[0]
        test_losses.append(epoch_loss)
        train_losses.append(loss_function(W, data)/data.shape[0])
        print("epoch {} train loss {}; test loss: {}".format(i+1, train_losses[-1], epoch_loss))

    duration = time.time() - start
    print("took {0:.2f} seconds to train.".format(duration))
    print(loss_function(W, data)/data.shape[0])
    
    # graph the epoch vs. loss
    epo = [i for i in range(1, EPOCH+1)]
    plt.plot(epo, test_losses, 'r-', label="test")
    plt.plot(epo, train_losses, 'b-', label="train")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("test loss")
    plt.show()
    exit()

    # save weights so next time don't need to train
    if save_weights:
        for w in range(len(W)):
            np.savetxt('W_{}.out'.format(w), W[w], delimiter=',')
