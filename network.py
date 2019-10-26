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
    self.afunction = lambda x: scipy.special.expit(x) 
    
    self.w_input_hidden = numpy.random.normal(
      0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
    self.w_hidden_output = numpy.random.normal(
      0.0, pow(onodes, -0.5), (self.onodes, self.hnodes))
  
  def query(self, input):
    pass

  def train(self):
    pass