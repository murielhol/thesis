



def ECDF(data, samples):
    N = float(len(samples))
    ecdf = [np.sum(np.where( samples <= d, 1, 0))/N for d in data]
    return ecdf

    
def KDE(data, samples, bandwidth):
    cdf = sum(norm(0).cdf((data-xi)/bandwidth) for xi in samples)
    cdf = cdf/np.max(cdf)
    return cdf


def get_cdf(data, samples, method='ecdf', bandwidth='1.0'):
    if method.lower() = 'ecdf':
        return ECDF(data, samples)
    elif method.lower() = 'kde':
        return KDE(data, samples, bandwidth)
    else:
        print('unknown cdf method')
        return -1

def value_at_risk(samples, alpha=5):
    N = len(samples)
    returns = np.array([(samples[:,i] / samples[:,0]) - 1 for i in range(1, len(samples))])
    sorted_rets = returns.sort(axis=0)
    var = np.percentile(sorted_rets, 5)
    return var

def expected_shortfall(samples, alpha=5):
    var = value_at_risk(samples, alpha=alpha)
    risky_samples = [s for s in samples if ]


