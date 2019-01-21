import itertools
from collections import namedtuple
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots, use_dbplot_axis
import matplotlib.pyplot as plt
Trade = namedtuple('Trade', ['time', 'price', 'type'])


class TradeTypes:
    BUY = 'buy'
    SELL = 'sell'

# To install artemis
# cd ~/whatever_directory_you_want_to_install_in/
# pip install -e git+http://github.com/QUVA-Lab/artemis.git@peter#egg=artemis


def mark_trade(trade: Trade, color='k', label_prefix='', ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(trade.time, trade.price, marker='x' if trade.type == TradeTypes.BUY else 'o', markersize=10, color=(0, 0, 0, 0), markeredgecolor=color, markeredgewidth=3, label=label_prefix+trade.type)


def transaction_chooser(futures_stream, transaction_cost, initial_have_state = False):
    """
    Given a stream which yields future forecast samples, generate transactions.
    :param futures_stream:
    :param transaction_cost:
    :param initial_have_state:
    :return:
    """
    have_state = initial_have_state
    history = []
    for t, futures in enumerate(futures_stream):
        # Futures is an (n_samples, n_steps) array of futures, with futures[:, 0] corresponding to n_samples copies of the current time-step.
        expected_future = np.mean(futures, axis=0)  # (samples, time)
        assert np.all(np.diff(futures[:, 0])==0), 'The zeroth time-step of the futures (corresponding to the present time) should be identical'
        x0 = expected_future[0]
        history.append(x0)

        with hold_dbplots(draw_every='0.05s'):
            data = np.concatenate([[np.array(history)]*len(futures), futures[:, 1:]], axis=1)
            dbplot(data.T, 'futures', plot_type=('line', dict(color='C0')))

        if not have_state:
            t_sell = next((tau for tau, m in enumerate(expected_future) if m > x0 + transaction_cost), None)  # Get next meeting criterion or None if it is never met.
            if t_sell is not None:
                if t_sell==1 or expected_future[:t_sell].min() >= x0:
                    print('BUY BUY BUY')
                    mark_trade(trade = Trade(time=t, price=x0, type=TradeTypes.BUY), ax = use_dbplot_axis('futures'))
                    have_state = True
        else:
            t_buy = next((tau for tau, m in enumerate(expected_future) if m < x0 - transaction_cost), None)
            if t_buy is not None:
                if t_buy==1 or not expected_future[:t_buy].max() > x0:
                    print('SELL SELL SELL')
                    mark_trade(trade = Trade(time=t, price=x0, type=TradeTypes.SELL), ax=use_dbplot_axis('futures'))
                    have_state = False


def make_sinusoidal_futures_stream(n_samples = 10, n_steps=100, frequency = 0.1, driftiness = 0.1):
    """
    Generate a stream of futures predictions.
    :param n_samples: Number if independent samples
    :param n_steps: Number of steps into future to predict.
    :param driftiness: The amount of random drift in our model (more means more randomness)
    :return Iterable[ndarray]: An array of the future predictions at time t, with futures[s, t] corresponding to the
        prediction of sample s, t-steps into the future.

    """
    for t in itertools.count(0):
        future_drift = np.random.randn(n_samples, n_steps) 
        future_drift[:, 0] = 0
        future_drift = np.cumsum(future_drift, axis=1) * driftiness
        futures = np.sin(np.arange(t, t+n_steps)*frequency + future_drift)
        yield futures


if __name__ == '__main__':

    transaction_chooser(
        futures_stream=make_sinusoidal_futures_stream(driftiness=0.1, frequency=0.1),
        transaction_cost=0.5,
    )
