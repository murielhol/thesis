import numpy as np
import itertools


from utils import *

def now_or_never(futures, reduce_risk=0.0, order_method = 'market_order', transaction_cost=0.0):

    """
    This strategy buys now when futures indicate that the price will go up in the next T time steps
    It computes the optimal selling time/price and place a sell order for this timestamp/price
    """

    T = len(futures[0])
    time_slices = [[future[t] for future in futures] for t in range(1,T)]
    reduce_risk = np.min([1., reduce_risk]) # should be between 0 and 1
    current_price = futures[0][0]

    # in order to reduce risk, take a weighted average between expectation and left tail
    if reduce_risk>0:
        penalty = [expected_shortfall(time_slice) for time_slice in time_slices]
        penalty.insert(0, current_price)
        futures = [[(1-reduce_risk)*f[i] + reduce_risk*penalty[i] for i in range(T)] for f in futures]
    # market orders are time specific
    # find for each time step what the expected profit is, sell at the time step with
    # highest expected profit
    if order_method == 'market_order':
        deals  = [np.mean(np.subtract(time_slice,current_price)) for time_slice in time_slices]
        selling_moment = np.argmax(deals)
        best_deal = deals[selling_moment]
        selling_price = best_deal+current_price
    # limit orders are price specific
    # find for each future the expected profit, sell at this price
    elif order_method == 'limit_order':
        best_deal  = np.mean([np.max(future[1:])-future[0] for future in futures])
        selling_price = best_deal+current_price

        selling_moment = 0

    # check if the profit you expect is enough to exceed transaction free
    if best_deal > (transaction_cost*current_price + transaction_cost*selling_price):
        return [('buy', current_price, 0), 
            ('sell', selling_price, selling_moment+1)]
    else:
        return [('hold', 0, 0)]

def if_you_do_it_do_it_good(futures, reduce_risk=0.9, order_method = 'market_order', transaction_cost=0.0):

    """
    This strategy computes the optimal buy and sell time/price and place a sell order for this timestamp/price
    """


    T = len(futures[0])
    time_slices = [[future[t] for future in futures] for t in range(1,T)]
    reduce_risk = np.min([1., reduce_risk]) # should be between 0 and 1
    current_price = futures[0][0]

    # in order to reduce risk at selling time, take a weighted average between expectation and left tail
    # in order to reduce risk at buying time, take a weighted average between expectation and right tail
    if reduce_risk>0:
        penalty_buy = [expected_shortfall(time_slice) for time_slice in time_slices]
        penalty_buy.insert(0, current_price)
        penalty_sell = [expected_highfall(time_slice) for time_slice in time_slices]
        penalty_sell.insert(0, current_price)
        futures_high = [[(1-reduce_risk)*f[i] + reduce_risk*penalty_sell[i] for i in range(T)] for f in futures]
        futures_low = [[(1-reduce_risk)*f[i] + reduce_risk*penalty_buy[i] for i in range(T)] for f in futures]
    else:
        futures_low = futures_high = futures

    best_deals, best_buys, best_sells, buy_moments = [],[],[],[]

    if order_method == 'market_order':
        buys = np.mean([[future[t] for future in futures_low] for t in range(1,T)], axis=1)
        sells = np.mean([[future[t] for future in futures_high] for t in range(1,T)], axis=1)
        # for each optional buy moment, find best selling moment
        options = [(np.max(buys[i+1:]), sells[i], np.max(buys[i+1:])-sells[i], i, np.argmax(buys[i+1:])) for i in range(T-2)]
        # select the best combination
        winner = max(options,key=lambda item:item[2])
        best_deal = winner[2]
        buying_price = winner[1]
        selling_price = winner[0]
        buy_moment = winner[3]
        sell_moment = winner[4]+buy_moment


    elif order_method == 'limit_order':
        for future_low, future_high in zip(futures_low,futures_high):
            # for each optional buy moment, find best selling moment
            options = [(np.max(future_low[i+1:]), future_high[i], np.max(future_low[i+1:])-future_high[i]) for i in range(T-2)]
            # select the best combination
            winner = max(options,key=lambda item:item[2])
            best_deals.append(winner[2])
            best_buys.append(winner[1])
            best_sells.append(winner[0])

        buying_price = np.mean(best_buys)
        selling_price = np.mean(best_sells)
        best_deal = np.mean(best_deals)
        buy_moment = sell_moment = 0 # moments dont matter anymore in limit orders

    if best_deal > (transaction_cost*buying_price + transaction_cost*selling_price):
        # if np.shape(futures)[0] == 1:
        return [('buy', buying_price, buy_moment), 
            ('sell', selling_price, sell_moment)]
    else:
        return [('hold', 0, 0)]




     