from typing import Sequence, Tuple, Callable
import numpy as np


def compute_roi(timeseries: Sequence[float], events: Sequence[Tuple[str, int]]) -> float:
    """
    Given a timeseries and a list of events, compute the return on investment
    :param timeseries: A array of n_steps data points
    :param events: A List of tuples of events and the time indices at which they happen.
        e.g. [('buy', 5), ('sell', 10), ('buy', 22)]
    :return:
    """

    for event_type, time_step in events:
        if event_type=='buy':
            raise NotImplementedError()
        else:
            raise NotImplementedError()


class ProbModel:

    def step(self, x, rng: np.random.RandomState):
        raise NotImplementedError()

    def simulate(self, initial, n_steps, rng):
        simulation = []
        x_next = initial
        for t in range(n_steps):
            x_next = self.step(x_next, rng)
            simulation.append(x_next)

    def clone(self) -> 'ProbModel':
        """Return a clone of this model with the same state"""


def make_buy_sell_events(futures, transaction_cost) -> Sequence[Tuple[str, int]]:
    raise NotImplementedError('This is the hard part')


def evaluate_model(prob_model: ProbModel, event_function: Callable, n_run_steps, n_eval_steps, transaction_cost, n_simulations=10000, initial_seed=1234):

    rng = np.random.RandomState(initial_seed)

    context = prob_model.simulate(initial=0, n_steps=n_run_steps, rng=rng)
    initial_sample = context[-1]
    futures = []
    for i in range(n_simulations):
        this_model = prob_model.clone()
        future = this_model.simulate(initial=initial_sample, n_steps=n_eval_steps, rng = np.random.RandomState(initial_seed+i))
        futures.append(future)

    true_future = prob_model.simulate(initial=initial_sample, n_steps=n_run_steps, rng=rng)

    events = event_function(futures, transaction_cost=transaction_cost)

    roi = compute_roi(true_future, events)

    return roi


if __name__ == '__main__':
    evaluate_model(
        prob_model=...,
        event_function=make_buy_sell_events,
    )

