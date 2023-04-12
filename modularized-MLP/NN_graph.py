import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import random

BESTRUN='weights_1_f_3.txt'   # default best run (val loss: 1.05, train loss: 0.44)
DIM=28
if '-h' or '--help' in sys.argv:
    print '--dim [NUM] [NUM]\tSpecify your desired dimension for the graph output'
    print '\t\t\tyour dimension must match the input! (Default: 10 by 10)'
    print '--run [FILE]\t\tSpecify the weights file you want to visualize'
    exit()


if '--dim' in sys.argv:
    first_dim = int(sys.argv[sys.argv.index('--dim') + 1])
    second_dim = int(sys.argv[sys.argv.index('--dim') + 2])
    shape = (first_dim, second_dim)
else:
    shape = (10,10)

print shape
if '--run' in sys.argv:
    BESTRUN = sys.argv[sys.argv.index('--run') + 1]

# write out the weight(1) as 10x10 graph with each subgraph.dim = (28x56)
weights_1 = np.loadtxt(BESTRUN)

# remove the bias term from weights_1
weights_1 = weights_1[1:,:]
print weights_1.shape

print ("Loaded the weights for input layer -> hidden layer")

# reshape the array to: 100 columns -> 100 subimages (10x10),
# 1568 rows -> 28x56 size image
# Hence, the final size of image output -> 280x560
# first 28 rows -> first 56 columns (the original first column)


# normalize the data, then reshape the original weights
subgraphs = []
for i in range(weights_1.shape[1]):
    ceiling = np.max(weights_1[:,i])
    floor = np.min(weights_1[:,i])
    interval = ceiling - floor
    increment = abs(floor)
    
    weights_1[:,i] += increment
    weights_1[:,i] /= interval
    weights_1[:,i] *= 256

    subgraphs.append(np.reshape(weights_1[:,i], (DIM, 2*DIM)))
print len(subgraphs), subgraphs[0].shape

# convert the subgraphs to a complete graph
graph = np.zeros((DIM*shape[0], 2*DIM*shape[1]))
ind = 0
for i in range(shape[0]):
    layer = i * DIM
    for j in range(shape[1]):
        column = j * 2 * DIM
        graph[layer: layer + DIM, column:column + 2*DIM] = subgraphs[ind]
        ind += 1

#for i in range(len(subgraphs)):
#    layer = (i / shape[0]) * DIM  # indicate which layer it is
#    start_column = (i % shape[1]) * DIM * 2   # indicate the current column to start inserting
#    print layer, start_column
#    
#    graph[layer:layer+DIM,start_column:start_column+2*DIM] = subgraphs[i]

# plot for to show the weights
plt.title('Weight of Input Layer to Hidden Layer For the Best Run')
plt.imshow(graph, cmap='gray')
plt.show()


