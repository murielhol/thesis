import itertools
from collections import namedtuple
import numpy as np
from artemis.plotting.db_plotting import dbplot, hold_dbplots, use_dbplot_axis, DBPlotTypes
import matplotlib.pyplot as plt

from thesis_code.version1.generating_events_from_known_model import NoisySinusoidProbModel

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
    for t, (x0, futures) in enumerate(futures_stream):
        # Futures is an (n_samples, n_steps) array of futures, with futures[:, 0] corresponding to n_samples copies of the current time-step.
        expected_future = np.mean(futures, axis=0)  # (samples, time)
        history.append(x0)

        with hold_dbplots(draw_every='0.05s'):
            data = np.concatenate([[np.array(history)]*len(futures), futures], axis=1)
            dbplot(data.T, 'futures', plot_type=('line', dict(color='C0', axes_update_mode='expand')))

        if not have_state:
            t_sell = next((tau for tau, m in enumerate(expected_future) if m > x0 + transaction_cost), None)  # Get next meeting criterion or None if it is never met.
            if t_sell is not None:
                if t_sell==0 or expected_future[:t_sell].min() >= x0:
                    print('BUY BUY BUY')
                    mark_trade(trade = Trade(time=t, price=x0, type=TradeTypes.BUY), ax = use_dbplot_axis('futures'))
                    have_state = True
        else:
            t_buy = next((tau for tau, m in enumerate(expected_future) if m < x0 - transaction_cost), None)
            if t_buy is not None:
                if t_buy==0 or not expected_future[:t_buy].max() > x0:
                    print('SELL SELL SELL')
                    mark_trade(trade = Trade(time=t, price=x0, type=TradeTypes.SELL), ax=use_dbplot_axis('futures'))
                    have_state = False


def make_sinusoidal_futures_stream(n_samples = 10, n_steps=100, freq=1., seed=1234):
    """
    Generate a stream of futures predictions.
    :param n_samples: Number if independent samples
    :param n_steps: Number of steps into future to predict.
    :param driftiness: The amount of random drift in our model (more means more randomness)
    :return Iterable[Tuple[float, ndarray]]: Yields (current_sample, futures) futures[s, t] corresponding to the
        prediction of sample s, t+1 steps into the future.
    """
    model = NoisySinusoidProbModel(freq = freq)
    true_rng = np.random.RandomState(seed)
    sim_rngs = [np.random.RandomState(seed+i) for i in range(1, n_samples+1)]

    while True:
        sample = model.step(true_rng)
        futures = np.array([model.clone().simulate(n_steps=n_steps, rng=rng) for rng in sim_rngs])
        yield sample, futures


if __name__ == '__main__':

    transaction_chooser(
        futures_stream=make_sinusoidal_futures_stream(freq=2),
        transaction_cost=0.5,
    )
