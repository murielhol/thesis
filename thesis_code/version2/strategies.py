import numpy as np
import itertools

from utils import *


def expected_return(futures, reduce_risk=0.9, order_method = 'market_order', transaction_cost=0.0):

    """
    This strategy computes for each simulation the expected return
    The expected expected return is used for decision making

    If you want to reduce risk, the expected shortfall is used instead of the 
    expected expected return, which is the expectation of the 5 percent worst 
    expected returns 
    """

    reduce_risk = np.min([1., reduce_risk]) # should be between 0 and 1
    current_price = futures[0][0]

    expected_returns = [np.mean([(f/current_price)-1 for f in future]) for future in futures]

    if reduce_risk>0:
        expected_return = expected_shortfall(expected_returns, alpha=reduce_risk*10.)
    else:
        expected_return = np.mean(expected_returns)
    if expected_return < 0:
        return 'sell'
    # both buying and selling costs money, so double the transaction cost
    elif expected_return > (2*transaction_cost):
        return 'buy'
    else:
        return 'hold'

def max_expected_return(futures, reduce_risk=0.9, order_method = 'market_order', transaction_cost=0.0):

    """
    This strategy computes for each simulation the expected return
    The expected expected return is used for decision making

    If you want to reduce risk, the expected shortfall is used instead of the 
    expected expected return, which is the expectation of the 5 percent worst 
    expected returns 
    """
    T = len(futures[0])
    reduce_risk = np.min([1., reduce_risk]) # should be between 0 and 1
    current_price = futures[0][0]
    time_slices = [[future[t] for future in futures] for t in range(1,T)]
    # risk can be reduced by asuming that at each time step, the futures below the value at risk
    # for that time step, will happen
    if reduce_risk>0:
        time_slices = [samples_at_risk(time_slice, alpha=(1.- reduce_risk)*100.) for time_slice in time_slices]
    max_exp_ret = np.max([np.mean(np.divide(time_slice, current_price) - 1) for time_slice in time_slices])
    
    if max_exp_ret < 0:
        return 'sell'
    # both buying and selling costs money, so double the transaction cost
    elif max_exp_ret > (2*transaction_cost):
        return 'buy'
    else:
        return 'hold'
