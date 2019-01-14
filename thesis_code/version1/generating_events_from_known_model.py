from typing import Sequence, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import copy
import os
import pandas as pd

from enum import Enum

from strategies import *


class strategies(Enum):
    now_or_never = 1
    if_you_do_it_do_it_good = 2

class order_types(Enum):
    market_order = 1
    limit_order = 2


def compute_roi(timeseries: Sequence[float], events: Sequence[Tuple[str, int]], transaction_cost, order_type = 'market_order') -> float:
    """
    Given a timeseries and a list of events, compute the return on investment
    :param timeseries: A array of n_steps data points
    :param events: A List of tuples of events, the order price and the time indices at which they happen.
        e.g. [('buy', 0.55, 5), ('sell', 1.2, 10), ('buy', 0.1, 22)]
    :return:
    """
    amount=0
    T = len(events)
    # at each tick check the action to be taken
    for i in range(T):
        
        event = events[i]
        if len(event) == 2: # if not hold
            buy = event[0]
            sell = event[1]
            if i+buy[-1]>T or i+sell[-1]>T:
                break
            # market order are time specific, and cost a fee
            if order_type == 'market_order':
                amount -= (timeseries[i+buy[-1]] * (1+transaction_cost))
                amount += (timeseries[i+sell[-1]] * (1-transaction_cost))
            elif order_type == 'limit_order':
                b = False
                for j in range(len(timeseries[i:i+10])):
                    # find the first buy oppertunity = when observed
                    # price is lower then what you are willing to pay 
                    if timeseries[i+j] <=buy[1] and not b:
                        amount -= timeseries[i+j]
                        b = True
                    # afterwards, find the first sell oppertunity = when observed
                    # price is higher then what you are asking for it
                    elif timeseries[i+j] >= sell[1] and b:
                        amount += timeseries[i+j]
                        break
    return amount



class ProbModel:

    def __init__(self, seed, dt=0.1, measurement_noise=0.05, x_noise = 0.1 , v_noise = 0.08):
        # system parameters
        np.random.seed(seed)
        self.seed = seed
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
        m = self.x + self.measurement_noise*rng.randn() + 2
        return m

    def simulate(self, n_steps, rng ):
        simulation = [self.step(rng) for _ in range(n_steps)]
        return simulation

    def clone(self) -> 'ProbModel':
        """Return a clone of this model with the same state"""
        return copy.deepcopy(self)


def make_buy_sell_events(futures, strategy, transaction_cost, reduce_risk, order_type) -> Sequence[Tuple[str, float, int]]:
    """
    tuple in form of: ('event', price, moment)
    """
    events = eval(strategy)(futures, reduce_risk, order_method=order_type, transaction_cost=transaction_cost)
    return events
    

def evaluate_model(prob_model: ProbModel, event_function: Callable, strategy: str, n_run_steps = 10, 
                    n_eval_steps = 10, transaction_cost=1, reduce_risk=0., order_type='market_order',
                    n_simulations=500, initial_seed=1234, make_fig=False):

    """
    Runs a simulation with the given ProbModel, and if make_fig is True it shows the simulations

    :return the total return on investment given the stratgy, for both the simulations and the actual future
    """
    # limit orders have no transaction costs
    if order_type == 'limit_order':
        transaction_cost = 0.0 
    rng = np.random.RandomState(initial_seed)

    context = prob_model.simulate(n_steps=1, rng=rng)
    events = []

    for t in range(n_run_steps):
        # state = prob_model.state()
        futures = []
        for i in range(n_simulations):
            this_model = prob_model.clone()
            # assert this_model.state() == state
            future = this_model.simulate(n_steps=n_eval_steps, rng = np.random.RandomState(initial_seed+i))
            future.insert(0, context[-1])
            futures.append(future)
        # plt.figure()
        # plt.plot(np.array(futures).T, c='c', alpha=0.5)
        event = event_function(futures, strategy, transaction_cost, reduce_risk, order_type)
        events.append(event)
        context+=prob_model.simulate(n_steps=1, rng=rng)
    roi = compute_roi(context, events, transaction_cost, order_type)

    target_roi = 0
    for t in range(n_run_steps):
        true_future = context[t:t+n_eval_steps]
        if strategy == 'if_you_do_it_do_it_good':
            target_roi += np.max([0, np.max(true_future)*(1-transaction_cost) - np.min(true_future)*(1+transaction_cost)])
        else:
            target_roi += np.max([0, np.max(true_future)*(1-transaction_cost) - true_future[0]*(1+transaction_cost)])
    
    print(roi, target_roi)

    if make_fig:
        plt.plot(context, label='true future')
        legend = True
        for t in range(n_run_steps):
            event = events[t]
            if event==2:
                buy = event[0]
                sell = event[1]
                if legend:
                    plt.scatter(t+buy[-1], 1, c='g', label='buy')
                    plt.scatter(t+sell[-1], 1, c='r', label='sell')
                    legend = False
                else:
                    plt.scatter(t+buy[-1], 1, c='g')
                    plt.scatter(t+sell[-1], 1, c='r')
            
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('example'+str(prob_model.seed)+'.png')
        plt.show()

    return roi, target_roi



def run(N = 10, show=True):

    """
    Runs N experiments, and possibly shows the buy and sell moments 
    prints the average return on investment for the simulations, and the one that is based on 
    the actual observed timeseries

    n_run_steps: number of ticks the time series is evaluated
    n_simulations: number of simulations per tick to choose buy/sell/hold
    n_eval_steps: the lenght of the rollout into the future
    reduce_risk: between 0 and 1, 0 is no risk reduction, 1 is
    """

    x_noise = 0.1
    v_noise = 0.08
    m_noise = 0.05
    reduce_risk = 1.0
    transaction_cost = 0.02
    output_seq_len = 10
    strategy = 1
    order_type = 2

    ROI = []
    TARGET_ROI = []

    for i in range(N):
        print('Running experiment', i, '...')
        roi, target_roi = evaluate_model(
            prob_model=ProbModel(i, measurement_noise=m_noise, x_noise = x_noise , v_noise = v_noise),
            event_function=make_buy_sell_events,
            strategy = str(strategies(strategy).name),
            n_run_steps = 200, 
            n_simulations = 500,
            n_eval_steps = output_seq_len, 
            transaction_cost = transaction_cost,
            reduce_risk = reduce_risk,
            order_type = str(order_types(order_type).name),
            make_fig = show
            )

        ROI.append(roi)
        TARGET_ROI.append(target_roi)
    print('mean roi: ', np.mean(ROI))
    print('mean target roi: ', np.mean(TARGET_ROI), '\n')

if __name__ == '__main__':


    run(N=10, show=True)



   

