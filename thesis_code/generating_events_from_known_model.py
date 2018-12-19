from typing import Sequence, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
plt.close()
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
    for event_type, price, time_step in events:
        if event_type=='buy':
            # market order are time specific, and cost a fee
            if order_type == 'market_order':
                amount -= (timeseries[time_step] * (1+transaction_cost))
            # limit orders are price specific and are free
            elif order_type == 'limit_order':
                options =  [p for p in timeseries if p<=price] 
                if options:
                    amount -= options[0]
        if event_type=='sell':
            if order_type == 'market_order':
                amount += (timeseries[time_step] * (1-transaction_cost))
            elif order_type == 'limit_order':
                options = [p for p in timeseries if p>=price]
                if options:
                    amount += options[0]
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


def make_buy_sell_events(futures, strategy, transaction_cost, reduce_risk) -> Sequence[Tuple[str, float, int]]:
    """
    tuple in form of: ('event', price, moment)
    """
    events = eval(strategy)(futures, reduce_risk, format='price', transaction_cost=transaction_cost)
    return events
    

def evaluate_model(prob_model: ProbModel, event_function: Callable, strategy: str, n_run_steps = 10, 
                    n_eval_steps = 10, transaction_cost=1, reduce_risk=False, order_type='market_order', 
                    n_simulations=1000, initial_seed=1234, make_fig=False):

    rng = np.random.RandomState(initial_seed)
    context = prob_model.simulate(n_steps=n_run_steps, rng=rng)
    futures = []
    # state = prob_model.state()
    for i in range(n_simulations):
        this_model = prob_model.clone()
        # assert this_model.state() == state
        future = this_model.simulate(n_steps=n_eval_steps, rng = np.random.RandomState(initial_seed+i))
        future.insert(0, context[-1])
        futures.append(future)

    true_future = prob_model.simulate(n_steps=n_eval_steps, rng = np.random.RandomState(initial_seed+n_simulations))
    true_future.insert(0, context[-1])

    events = event_function(futures, strategy, transaction_cost, reduce_risk)
    if events:
        roi = compute_roi(true_future, events, transaction_cost, order_type )
    else:
        roi = 0

    true_events = event_function([true_future], strategy, transaction_cost, reduce_risk=False)
    if true_events:
        target_roi = compute_roi(true_future, true_events, transaction_cost, order_type )
    else:
        target_roi = 0



    if make_fig:
        plt.figure()
        plt.plot(context, c='r')
        plt.plot((np.arange(n_run_steps, n_run_steps+n_eval_steps+1)*np.ones(np.shape(futures))).T, np.array(futures).T, alpha=0.5, c='b')
        plt.plot(np.arange(n_run_steps, n_run_steps+n_eval_steps+1), true_future, c='r')
        for event in events:
            color = 'm' if event[0] == 'buy' else 'g'
            plt.axvline(x=event[-1]+n_run_steps, label=event[0], c=color)
        for event in true_events:
            color = 'y' if event[0] == 'buy' else 'c'
            plt.axvline(x=event[-1]+n_run_steps, label='true_'+event[0], c=color)
        plt.legend()
        plt.savefig('experiment_'+strategy+'_'+str(prob_model.seed)+'.png')
        plt.show()


    return roi, target_roi




def run_experiments(trials, simulations_per_trial, strategy, order_type, transaction_cost, input_seq_len, output_seq_len,
                    x_noise, v_noise, m_noise, reduce_risk):
    transaction_cost = transaction_cost if order_type == 1 else 0.0

    results = [
        evaluate_model(
        prob_model=ProbModel(experiment, measurement_noise=m_noise, x_noise = x_noise , v_noise = v_noise),
        event_function=make_buy_sell_events,
        strategy = str(strategies(strategy).name),
        n_run_steps = input_seq_len, 
        n_simulations = simulations_per_trial,
        n_eval_steps = output_seq_len, 
        transaction_cost = transaction_cost,
        reduce_risk = reduce_risk,
        order_type = str(order_types(order_type).name),
        ) for experiment in range(trials)]

    estimates = [r[0] for r in results]
    targets = [r[1] for r in results]

    return estimates, targets



def strategy_vs_sample_size():

    make_dirs('strategy_vs_sample_size')

    x_noise = 0.1
    v_noise = 0.08
    m_noise = 0.05
    reduce_risk = False
    order_type = 1
    transaction_cost = 0.05
    input_seq_len = 100
    output_seq_len = 50
    trials = 500
    
    df = pd.DataFrame([])
    df['simulations_per_trial'] = [100, 500, 1000, 1500, 2000]

    colors = ['r', 'b']
    plt.figure('strategy_vs_sample_size')
    for strategy in [1,2]:
        print('strategy: ', str(strategies(strategy).name))
        errors = []
        for simulations_per_trial in [100, 500, 1000, 1500, 2000]:
            print('simulations_per_trial: ', simulations_per_trial)
            estimates, targets = run_experiments(trials, simulations_per_trial, strategy, order_type, transaction_cost, input_seq_len, output_seq_len,
                            x_noise, v_noise, m_noise, reduce_risk)            
            error = np.mean(np.subtract(targets,estimates))
            errors.append(error)
        plt.plot([100, 500, 1000, 1500, 2000], errors, c=colors[strategy-1], label = str(strategies(strategy).name) )
        df[str(strategies(strategy).name)] = errors
    plt.legend()
    df.to_csv('strategy_vs_sample_size/results/simulations_per_trial.png')
    plt.savefig('strategy_vs_sample_size/images/simulations_per_trial.png')
    plt.show()


def order_type_vs_strategy():

    make_dirs('order_type_vs_strategy')

    x_noise = 0.1
    v_noise = 0.08
    m_noise = 0.05
    reduce_risk = True
    input_seq_len = 100
    output_seq_len = 50
    trials = 500
    simulations_per_trial = 500
    transaction_cost = 0.02
    
    df = pd.DataFrame([])

    colors = ['r', 'b', 'g', 'm']
    i = 0
    plt.figure('order_type_vs_strategy')
    for strategy in [1,2]:
        print('strategy: ', str(strategies(strategy).name))
        for order_type in [1,2]:
            print('order_type: ', str(order_types(order_type).name))
            estimates, targets = run_experiments(trials, simulations_per_trial, strategy, order_type, transaction_cost, input_seq_len, output_seq_len,
                            x_noise, v_noise, m_noise, reduce_risk)            
            error = np.mean(np.subtract(targets,estimates))
            plt.bar(i, error, label = str(strategies(strategy).name)+'/'+str(order_types(order_type).name) )
    plt.legend()
    plt.savefig('order_type_vs_strategy/images/order_type_vs_strategy.png')
    plt.show()


def strategy_vs_transaction_cost():

    make_dirs('strategy_vs_transaction_cost')

    x_noise = 0.1
    v_noise = 0.08
    m_noise = 0.05
    reduce_risk = False
    order_type = 1
    input_seq_len = 100
    output_seq_len = 50
    trials = 500
    simulations_per_trial = 500
    
    df = pd.DataFrame([])
    df['transaction_cost'] = np.arange(10)/100.0

    colors = ['r', 'b']
    plt.figure('strategy_vs_transaction_cost')
    for strategy in [1,2]:
        print('strategy: ', str(strategies(strategy).name))
        errors = []
        for transaction_cost in np.arange(10)/100.0:
            print('transaction_cost: ', transaction_cost)
            estimates, targets = run_experiments(trials, simulations_per_trial, strategy, order_type, transaction_cost, input_seq_len, output_seq_len,
                            x_noise, v_noise, m_noise, reduce_risk)            
            error = np.mean(np.subtract(targets,estimates))
            errors.append(error)
        plt.plot(np.arange(10)/100.0, errors, c=colors[strategy-1], label = str(strategies(strategy).name) )
        df[str(strategies(strategy).name)] = errors
    plt.legend()
    df.to_csv('strategy_vs_transaction_cost/results/transaction_cost.csv')
    plt.savefig('strategy_vs_transaction_cost/images/transaction_cost.png')
    plt.show()

def noise_vs_risk():

    make_dirs('noise_vs_risk')

    x_noise_initial = 0.1
    v_noise_initial = 0.08
    m_noise = 0.05
    order_type = 1
    transaction_cost = 0.05
    input_seq_len = 100
    output_seq_len = 50
    trials = 500
    simulations_per_trial = 500
    strategy = 1
    
    df = pd.DataFrame([])
    df['extra_noise'] = np.arange(0,20,2)/10.0
    
    colors = ['r', 'b']
    i = 0
    plt.figure('noise_vs_risk')
    for reduce_risk in [False,True]:
        i+=1
        print('reduce_risk: ', reduce_risk)
        errors = []
        for extra_noise in np.arange(0,20,2)/10.0:
            print('extra_noise: ', extra_noise)
            v_noise = v_noise_initial * (1.0+extra_noise)
            x_noise = x_noise_initial * (1.0+extra_noise)
            estimates, targets = run_experiments(trials, simulations_per_trial, strategy, order_type, transaction_cost, input_seq_len, output_seq_len,
                            x_noise, v_noise, m_noise, reduce_risk)            
            error = np.mean(np.subtract(targets,estimates))
            errors.append(error)
        plt.plot(np.arange(0,20,2)/10.0, errors, c=colors[i-1], label = str(reduce_risk) )
        df[str(strategies(strategy).name)] = errors
    plt.legend()
    df.to_csv('noise_vs_risk/results/noise.csv')
    plt.savefig('noise_vs_risk/images/noise.png')
    plt.show()

def show_an_example():
    x_noise = 0.1
    v_noise = 0.08
    m_noise = 0.05
    reduce_risk = False
    order_type = 1
    transaction_cost = 0.05
    input_seq_len = 100
    output_seq_len = 50
    strategy = 1

    evaluate_model(
        prob_model=ProbModel(10, measurement_noise=m_noise, x_noise = x_noise , v_noise = v_noise),
        event_function=make_buy_sell_events,
        strategy = str(strategies(strategy).name),
        n_run_steps = input_seq_len, 
        n_simulations = 50,
        n_eval_steps = output_seq_len, 
        transaction_cost = transaction_cost,
        reduce_risk = reduce_risk,
        order_type = str(order_types(order_type).name),
        make_fig = True
        )

if __name__ == '__main__':

    noise_vs_risk()


   

