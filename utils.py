import math
import numpy as np

def get_distance(pos_1, pos_2):
  """
  Calculate the Euclidean distance between two positions
  """

  x1, y1 = pos_1
  x2, y2 = pos_2
  dx = x1 - x2
  dy = y1 - y2
  return math.sqrt(dx**2 + dy**2)

def flatten(l):
  """
  Helper function for model datacollector for trade price
  collapses agent price list into one list
  """
  return [i for sl in l for i in sl]

def geometric_mean(list_of_prices):
  """
  Find the geometric mean of a list of prices
  """
  return np.exp(np.log(list_of_prices).mean())

def get_trade(agent):
  """
  For agent reportes in data collector

  Return list of trade partners and None for other agents
  """
  
  if type(agent) == Trader:
    return agent.trade_partners
  else: 
      return None