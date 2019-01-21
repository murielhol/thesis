import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
import os


def value_at_risk(samples, alpha=5):
    samples.sort()
    var = np.percentile(samples, alpha)
    return var

def expected_shortfall(samples, alpha=5):
    '''
    expected left tail risk
    '''
    var = value_at_risk(samples, alpha=alpha)
    risky_samples = [s for s in samples if s < var]
    return np.mean(risky_samples)

def samples_at_risk(samples, alpha=5):
    '''
    expected left tail risk
    '''
    var = value_at_risk(samples, alpha=alpha)
    risky_samples = [s for s in samples if s < var]
    return risky_samples


def make_dirs(name):
    if not os.path.exists(name):
                os.makedirs(name)
    if not os.path.exists(name+'/results'):
                os.makedirs(name+'/results')
    if not os.path.exists(name+'/images'):
                os.makedirs(name+'/images')



