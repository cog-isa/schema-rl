import numpy as np
from model.inference import *
from model.featurematrix import FeatureMatrix

W = [np.zeros((5, 5)) for _ in range(3)]
R = [np.zeros((5, 5)) for _ in range(2)]

a = SchemaNetwork(W, R)