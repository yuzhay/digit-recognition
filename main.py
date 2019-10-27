import sys, numpy
from network import Network

if __name__ != '__main__':
  sys.exit()

input_nodes = 784
hidden_nodes = 100
output_nodes  = 10

learning_grade = 0.3

net = Network(input_nodes, hidden_nodes, output_nodes, learning_grade)

train_file = open('data/train_100.csv', 'r')
train_list = train_file.readlines()
train_file.close()

for line in train_list:
  values = line.split(',')
  inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
  targets = numpy.zeros(output_nodes) + 0.01
  targets[int(values[0])] = 0.99
  net.train(inputs, targets)

test_file = open('data/test_10.csv', 'r')
test_list = test_file.readlines()
test_file.close()

for line in test_list:
  values = line.split(',')
  inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
  expected = values[0]
  actual = numpy.argmax(net.query(inputs))
  print("%s => %s" % (expected, actual))