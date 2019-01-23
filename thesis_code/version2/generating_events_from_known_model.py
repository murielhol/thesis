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
        expected_return = 1
        max_expected_return = 2


def compute_roi(timeseries: Sequence[float], events: Sequence[str], transaction_cost, show=False) -> float:
    """
    Given a timeseries and a list of events, compute the return on investment
    Note: only buys once every tick, but sells everything within 1 tick

    :param timeseries: A array of n_steps data points
    :param events: A List of events
        e.g. ['hold', buy', 'sell','buy']
    :return: total return made after excecuting the events on timeseries

    """
    roi = 0
    current_price = timeseries[0]
    returns = [(p/current_price)-1 for p in timeseries]
    final_returns = []
    bought = 0 # keep track of how much you have bought 
    wallet = 0 # keep track of how many "coins" you have bought
    plt.figure('events')
    plt.plot(timeseries)
    for i in range(len(events)):
        event = events[i]
        if event=='buy':
            bought += ((returns[i]+1)*(1+transaction_cost))
            wallet += 1
            plt.scatter(i, timeseries[i], c='g')
        if event=='sell' and wallet > 0:
            # final_returns.append(wallet*[timeseries[i]])
            plt.scatter(i, timeseries[i], c='r')
            roi +=  ((wallet*((returns[i]+1)*(1-transaction_cost))) - bought)
            bought = 0 # empty the wallet
            wallet = 0

    sharpe_ratio = roi / np.std(returns)
    if show:
        plt.show()
    return roi, sharpe_ratio

class ProbModel:
    """
    Harmonic Oscillator model
    :param seed: random see that determines the properties of the oscillator
    :param x_noise: noise of location transition, 0.08 is medium noisy
    :param v_noise: noise of velocity transition, 0.1 is medium noisy
    :param measurement_noise: noise of observation, 0.05 is light noisy
    """

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
        '''
        1 transition step, that changes the state [x, v] of the model
        '''
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


def make_buy_sell_events(futures, strategy, transaction_cost, reduce_risk) -> Sequence[Tuple[str, float, int]]:
    """
    :return string in form of: 'event'
    """
    event = eval(strategy)(futures, reduce_risk, transaction_cost=transaction_cost)
    return event
    

def evaluate_model(prob_model: ProbModel, event_function: Callable, strategy: str, n_run_steps = 10, 
                    n_eval_steps = 10, transaction_cost=1, reduce_risk=0., 
                    n_simulations=500, initial_seed=1234, make_fig=False):

    """
    Runs a simulation with the given ProbModel, and if make_fig is True it shows the simulations

    :return the total return on investment given the stratgy, for both the simulations and the actual future
    """

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
        event = event_function(futures, strategy, transaction_cost, reduce_risk)
        events.append(event)
        context+=prob_model.simulate(n_steps=1, rng=rng)

    true_events = []
    for t in range(n_run_steps):
        event = event_function([context[t:t+n_eval_steps+1]], strategy, transaction_cost, reduce_risk=0.0)
        true_events.append(event)

    roi, sharpe_ratio = compute_roi(context, events, transaction_cost)
    target_roi, target_sharpe_ratio = compute_roi(context, true_events, transaction_cost)

    if make_fig:
        plt.plot(context, label='true future')
        legend_buy = legend_sell =  True
        for t in range(n_run_steps):
            if events[t]=='buy':
                if legend_buy:
                    plt.scatter(t, context[t], c='g', label='buy')
                    legend_buy = False
                else:
                    plt.scatter(t, context[t], c='g')
            elif events[t]=='sell':
                if legend_sell:
                    plt.scatter(t, context[t], c='r', label='sell')
                    legend_sell = False
                else:
                    plt.scatter(t, context[t], c='r')
            elif events[t]=='hold':
                if legend_sell:
                    plt.scatter(t, context[t], c='y', label='hold')
                    legend_sell = False
                else:
                    plt.scatter(t, context[t], c='r')
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('example'+str(prob_model.seed)+'.png')
        plt.show()

    return roi, sharpe_ratio, target_roi, target_sharpe_ratio



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

    x_noise = 0.5
    v_noise = 0.25
    m_noise = 0.25
    reduce_risk = 0.5
    transaction_cost = 0.00
    output_seq_len = 5
    strategy = 2
    number_of_ticks = 200
    nuber_of_simulations = 200

    ROI = []
    SR = []
    TARGET_ROI = []
    TSR = []

    for i in range(N):
        print('Running experiment', i, '...')
        roi, sharpe_ratio, target_roi, target_sharpe_ratio = \
            evaluate_model(
                prob_model=ProbModel(i, measurement_noise=m_noise, x_noise = x_noise , v_noise = v_noise),
                event_function=make_buy_sell_events,
                strategy = str(strategies(strategy).name),
                n_run_steps = number_of_ticks, 
                n_simulations = nuber_of_simulations,
                n_eval_steps = output_seq_len, 
                transaction_cost = transaction_cost,
                reduce_risk = reduce_risk,
                make_fig = show
                )

        ROI.append(roi)
        SR.append(sharpe_ratio)
        TARGET_ROI.append(target_roi)
        TSR.append(target_sharpe_ratio)
    print('mean roi: ', np.mean(ROI), 'mean sharpe ratio: ', np.mean(SR))
    print('mean target roi: ', np.mean(TARGET_ROI), 'mean target sharpe ratio: ', np.mean(TSR), '\n')
    f =  open(str(strategy)+str(output_seq_len)+str(transaction_cost)+str(reduce_risk)+str(m_noise)\
        +str(x_noise)+str(v_noise)+'.txt', 'w')
    f.write('strategy: '+ str(strategy)+'\n')
    f.write('output_seq_len: '+str(output_seq_len)+'\n')
    f.write('transaction_cost: '+str(transaction_cost)+'\n')
    f.write('reduce_risk: '+str(reduce_risk)+'\n')
    f.write('m_noise: '+str(m_noise)+'\n')
    f.write('x_noise: '+str(x_noise)+'\n')
    f.write('v_noise: '+str(v_noise)+'\n')
    f.write('runs: '+ str(N)+'\n')
    f.write('nuber_of_simulations: '+str(nuber_of_simulations)+'\n')
    f.write('number_of_ticks: '+str(number_of_ticks)+'\n')
    f.write('roi: '+ str(ROI)+'\n')
    f.write('target roi: '+ str(target_roi)+'\n')
    f.write('sharpe_ratio: '+ str(sharpe_ratio)+'\n')
    f.write('target_sharpe_ratio: '+ str(target_sharpe_ratio)+'\n')
    f.close()
if __name__ == '__main__':

    run(N=10, show=False)



   

