import backprop_data

import backprop_network
import matplotlib.pyplot as plt
import numpy as np



training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = backprop_network.Network([784, 40, 10])
sizes = [0.001, 0.01, 0.1, 1, 10, 100]
colors = ['b', 'g', 'r', 'c', 'm', 'y']
i = 0
training_error = []
training_loss = []
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['0.001', '0.01', '0.1', '1', '10', '100'])
for alpha in sizes:
    a,b,c = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=alpha, test_data=test_data)
    training_error.append(c)
    training_loss.append(b)
    plt.plot(np.arange(len(a)),a, colors[i], label = sizes[i])
    i+=1
plt.show()
plt.clf()

for i in range(len(training_error)):
    plt.plot(np.arange(len(training_error[i])),training_error[i], colors[i], label = sizes[i])

plt.legend(loc="upper left")
plt.show()
plt.clf()

for i in range(len(training_loss)):
    plt.plot(np.arange(len(training_loss[i])),training_loss[i], colors[i], label = sizes[i])

plt.legend(loc="upper left")
plt.show()
plt.clf()


training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

net = backprop_network.Network([784, 500, 10])
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

