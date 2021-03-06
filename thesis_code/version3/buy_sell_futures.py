import itertools
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots, use_dbplot_axis
import matplotlib.pyplot as plt

# Artemis needs:
# pip install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis

# To get, do:
#   git clone https://github.com/QUVA-Lab/artemis.git
#   cd artemis
#   pip install -e .
#   git checkout peter

def value_at_risk(samples, alpha=5):
    samples.sort()
    var = np.percentile(samples, alpha)
    return var

def expected_shortfall(current_price, samples, alpha=5):
    '''
    expected left tail risk
    '''
    if current_price == 0:
        current_price = 1e-10
    returns = (samples / current_price) - 1
    var = value_at_risk(returns, alpha=alpha)
    risky_samples = [s for s in returns if s < var]
    return np.mean(risky_samples)

def mark_trade(trade_time, trade_price, trade_type, color='k', label_prefix=''):
    plt.plot(trade_time, trade_price, marker='x' if trade_type == 'buy' else 'o', markersize=10, color=(0, 0, 0, 0), markeredgecolor=color, markeredgewidth=3, label=label_prefix+trade_type)


def get_next_or_none(iterable):
    try:
        return next(iterable)
    except StopIteration:  # there is no tsell
        return None


def transaction_chooser(futures_stream, transaction_cost, riskiness=0.1, initial_have_state = False):
    
    have_state = initial_have_state

    for t, futures in enumerate(futures_stream):
        expected_future = np.mean(futures, axis=0)  # (samples, time)
        x0 = expected_future[0]

        with hold_dbplots():
            dbplot(x0, 'x0')
            dbplot(futures.T, 'futures', plot_type='line')

        if not have_state:
            # check when and if there is a profitable sell moment
            t_sell = get_next_or_none(tau for tau, m in enumerate(expected_future) if m > x0 + transaction_cost
                                      and riskiness > -1*expected_shortfall(x0, futures[:, tau]))

            if t_sell is not None:
                # ES = expected_shortfall(x0, futures[:, t_sell])
                # check if until that sell moment arrives, there is a better buy moment
                if (t_sell == 1 or expected_future[1:t_sell].min() > x0):# and riskiness > (-1*ES):
                    print('BUY BUY BUY')
                    use_dbplot_axis('x0')
                    mark_trade(trade_time=t, trade_price=x0, trade_type='buy')
                    have_state = True


        else:
            # check if there is a moment when buying a new share is cheaper than keeping this one
            t_buy = get_next_or_none(tau for tau, m in enumerate(expected_future) if m < x0 - transaction_cost)
            # check if until that moment arrives, there is a better sell moment
            if t_buy is not None:
                if t_buy==1 or not expected_future[1:t_buy].max() > x0:
                    print('SELL SELL SELL')
                    use_dbplot_axis('x0')
                    mark_trade(trade_time=t, trade_price=x0, trade_type='sell')
                    have_state = False


def make_futures_stream(n_samples = 10, n_steps=100, driftiness = 0.1):
    
    for t in itertools.count(0):
        future_drift = np.random.randn(n_samples, n_steps) 
        future_drift[:, 0] = 0
        future_drift = np.cumsum(future_drift, axis=1) * driftiness

        futures = np.sin(np.arange(t, t+n_steps)/10. + future_drift)

        yield futures


if __name__ == '__main__':

    transaction_chooser(
        futures_stream=make_futures_stream(driftiness=0.1),
        transaction_cost=0.2,
        riskiness=0.1
    )