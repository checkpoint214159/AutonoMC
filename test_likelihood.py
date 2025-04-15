import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch


query_x = 0.5

mean_x = 1
var_x = 0.3

mean_y = 1
var_y = 0.3

cor_coeff = -0.5


posterior_mean = mean_y + cor_coeff * (var_y / var_x) * (query_x - mean_x)

print(posterior_mean)