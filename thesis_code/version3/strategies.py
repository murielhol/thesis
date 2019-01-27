import numpy as np
import itertools

from utils import *


def find_best_future(futures, reduce_risk=0.9, have_stat = 'True', transaction_cost=0.0):

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

