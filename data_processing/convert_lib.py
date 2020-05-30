"""Just some random things to make things easier."""

import os

class DatasetName(object):
  conll = 'conll12'
  preco = 'preco'

class Variation(object):

  classic = 'classic'
  gold = 'gold'

  goldconst = 'goldconst'
  predconst = 'predconst'

  goldsing = 'goldsing'
  predsing = 'predsing'

class DatasetSplit(object):
  train = 'train'
  dev = 'dev'
  test = 'test'
  ALL = [train, dev, test]


def create_dir(path):
  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s " % path)

