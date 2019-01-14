import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
import os


def ECDF(data, samples):
    ecdf = [np.sum(np.where( sample >= data, 1, 0)) for sample in samples]
    return np.divide(ecdf,len(data))

    
def KDE(data, samples, bandwidth=0.1):
    cdf = np.sum([norm(0).cdf((xi-data)/bandwidth) for xi in samples], axis=1)
    return np.divide(cdf,len(data))


def get_cdf(data, samples, method='ecdf', bandwidth='1.0'):
    if method.lower() == 'ecdf':
        return ECDF(data, samples)
    elif method.lower() == 'kde':
        return KDE(data, samples, bandwidth)
    else:
        print('unknown cdf method')
        return -1

def value_at_risk(samples, alpha=5):
    N = len(samples)
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

def expected_highfall(samples, alpha=95):
    '''
    expected right tail risk
    '''
    var = value_at_risk(samples, alpha=alpha)
    risky_samples = [s for s in samples if s > var]
    return np.mean(risky_samples)


def make_dirs(name):
    if not os.path.exists(name):
                os.makedirs(name)
    if not os.path.exists(name+'/results'):
                os.makedirs(name+'/results')
    if not os.path.exists(name+'/images'):
                os.makedirs(name+'/images')



