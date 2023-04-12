# modularied-MLP
Fully modularized MLP with support to dropout and batch normalization.

The program is written in NN_calc.py and NN_graph.py.
NN_calc.py is used to train and test the neural network. NN_graph.py is used to
produce graphs for weights.

```
python NN_calc.py -h
```
This command will print out all customizable parameters.

```
python NN_calc.py           
```
This command will run the default setting for training the NN.
```
                            Default learning rate = 0.1,
                            epoches = 150,
                            batch size = 32,
                            output size = 19,
                            # of hidden nodes = 100,
                            # of layers = 1,
                            round of test = 5,
                            momentum = 0.0,
                            train data = 'data/train.txt',
                            val data = 'data/val.txt'.
```

Some special commands including:
```
                            '--save'    will save the weights of current run,
                            '--test [FILE]'   will read in test file and perform testing,
                            '--BN'    will enable batch normalization,
                            '--DO'    will enable dropout.
```
example run:
`python NN_calc.py -l 2 -n 300 -s 0.07 -m 0.5 -e 100 --DO --save 1 --test data/test.txt`

The above command will run the training with (2) layers, (300) hidden nodes at
each hidden layer, (0.07) learning rate, (0.5) momentum, (100) epoches, with
dropout, saving the best weights for model, reading in test file.



OUTPUT:
The code will output the result for each epoch, with training/val loss and
error rates. Then, at the end it will output two graphs: 1) train/val loss vs.
epoch numbers, 2) train/val error rate vs. epoch numbers. If '--test' is included,
The final print-screen will also include a report of the best epoch's performance
on the testing data.
