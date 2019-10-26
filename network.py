import numpy
import scipy.special

class Network:

  def __init__(self, inodes, hnodes, onodes, lgrade):
    # Neural network schema:
    # input nodes | hidden nodes | output nodes

    self.inodes = inodes
    self.hnodes = hnodes
    self.onodes = onodes
    self.lgrade = lgrade
    self.afunc = lambda x: scipy.special.expit(x) 
    
    self.wih = numpy.random.normal(
      0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
    self.w_hidden_output = numpy.random.normal(
      0.0, pow(onodes, -0.5), (self.onodes, self.hnodes))
  
  def query(self, input_list):
    inputs = numpy.array(input_list, ndmin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.afunc(hidden_inputs)

    output_inputs = numpy.dot(self.who, hidden_outputs)
    output_outputs = self.afunc(output_inputs)

    return output_outputs

  def train(self):
    pass