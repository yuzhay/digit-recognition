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
    self.who = numpy.random.normal(
      0.0, pow(onodes, -0.5), (self.onodes, self.hnodes))
  
  def query(self, input_list):
    inputs = numpy.array(input_list, ndmin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.afunc(hidden_inputs)

    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.afunc(final_inputs)

    return final_outputs

  def train(self, input_list, target_list):
    inputs = numpy.array(input_list, ndmin=2).T
    targets = numpy.array(target_list, ndmin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.afunc(hidden_inputs)

    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.afunc(final_inputs)

    output_errors = targets - final_outputs
    hidden_errors = numpy.dot(self.who.T, output_errors)

    self.who += self.lgrade * numpy.dot(
      output_errors * final_outputs * (1 - final_outputs), 
      hidden_outputs.T)

    self.wih += self.lgrade * numpy.dot(
      hidden_errors * hidden_outputs * (1 - hidden_outputs),
      inputs.T)

