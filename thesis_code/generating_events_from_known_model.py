from typing import Sequence, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import copy

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

    def __init__(self, seed, dt=0.1, measurement_noise=0.05, x_noise = 0.1 , v_noise = 0.08):
        # system parameters
        np.random.seed(seed)
        self.freq = np.random.uniform(1.0,4.0)
        self.x_noise = x_noise
        self.v_noise = v_noise
        self.dt = dt
        self.measurement_noise=measurement_noise
        direction = np.random.choice([-1, 1])
        self.x, self.v = np.random.uniform(-1.0,1.0), direction*np.random.uniform(0.5,2.0)
        self.initial_energy = self.energy(self.x, self.v)
        
    def energy(self, x, v):
        return np.sqrt(1/(self.freq**2)*v**2 + x**2)

    def state(self):
        return [self.x, self.v]

    def step(self, rng: np.random.RandomState):
        self.v = self.v - self.dt * self.freq ** 2 * self.x + self.v_noise*rng.randn()
        self.x = self.x + self.v*self.dt + self.x_noise*rng.randn()
        current_energy = self.energy(self.x, self.v)
        self.x = self.x*self.initial_energy/current_energy
        self.v = self.v*self.initial_energy/current_energy
        m = self.x + self.measurement_noise*rng.randn()
        return m

    def simulate(self, n_steps, rng ):
        simulation = [self.step(rng) for _ in range(n_steps)]
        return simulation


    def clone(self) -> 'ProbModel':
        """Return a clone of this model with the same state"""
        return copy.deepcopy(self)


def make_buy_sell_events(futures, transaction_cost, strategy) -> Sequence[Tuple[str, int]]:
    pass
    


def evaluate_model(prob_model: ProbModel, event_function: Callable, n_run_steps = 10, n_eval_steps = 10, 
                   transaction_cost=1, n_simulations=1000, initial_seed=1234):

    rng = np.random.RandomState(initial_seed)
    context = prob_model.simulate(n_steps=n_run_steps, rng=rng)
    futures = []
    # state = prob_model.state()
    for i in range(n_simulations):
        this_model = prob_model.clone()
        # assert this_model.state() == state
        future = this_model.simulate(n_steps=n_eval_steps, rng = np.random.RandomState(initial_seed+i))
        futures.append(future)

    true_future = prob_model.simulate(n_steps=n_eval_steps, rng=rng)

    plt.plot(context, c='r')
    plt.plot((np.arange(n_run_steps, n_run_steps+n_eval_steps)*np.ones(np.shape(futures))).T, np.array(futures).T, alpha=0.5, c='b')
    plt.plot(np.arange(n_run_steps, n_run_steps+n_eval_steps), true_future, c='r')
    plt.show()

    events = event_function(futures, transaction_cost=transaction_cost)

    roi = compute_roi(true_future, events)

    return roi


if __name__ == '__main__':

    input_seq_len = 100
    output_seq_len = 50
    percentage_fee = 5

    experiment = 1
    evaluate_model(
        prob_model=ProbModel(experiment),
        event_function=make_buy_sell_events,
        n_run_steps = input_seq_len, 
        n_eval_steps = output_seq_len, 
        transaction_cost = percentage_fee
    )

