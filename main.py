import sys
from network import Network

if __name__ != '__main__':
  sys.exit()

input_nodes, hidden_nodes, output_nodes  = 1, 1, 1
learning_grade = 0.3

net = Network(input_nodes, hidden_nodes, output_nodes, learning_grade)