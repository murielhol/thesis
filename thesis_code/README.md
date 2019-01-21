# The purpose

For a stochastic autoregressive model, find a method to convert the sampled roll-outs into buy/sell/hold decisions.
The method is validated by computing the total return on investment. 

The stochastic autoregressive model used for simulation is a harmonic oscillator. 

To be runned are the run functions in generate_events_from_known_model.py

I would say for now use version 2 (explained below)

## Version 1: this version plans ahead, by finding the best buy and sell moment in the next N steps. Thus it ensures there is a buy AND sell moment in the rolled out future. So high frequency trading. 

Two different strategy types:
    * now_or_never: at each tick you can buy, or not, but you can not postpone this 
    * if_you_do_it_do_it_good: at each tick you can buy, or not, or buy later

Two different order types:
    * market_order: place two orders, one with a time stamp to buy and one with a time stamp to sell
    * limit_order: place two orders, one with a price to buy and one with a price to sell

The risk of the price being lower then expected can be reduced by underestimating the roll-outs

### example results:
- mean of 10 runs, each run has 200 ticks and at each tick 100 roll-outs:
    - now_or_never and market_order: mean roi:  52.31 , mean target roi:  43.05
    - now_or_never and limit_order: mean roi:  20.658, mean target roi:  52.357
    - if_you_do_it_do_it_good and market_order: mean roi:  27.461, mean target roi:  89.967
    - if_you_do_it_do_it_good and limit_order: mean roi:  -87.231, mean target roi:  105.919


## Version 2: this version makes decisions on the go (it does buys without knowing if it can sell later)

strategy: 

- for each future, compute expected return
- compute mean expected return
    * <= 0, sell
    * => current_price + transaction_cost, buy
    * else hold

- risk can be reduced by replacing the expected return with the expected shortfall


### example results:
 - mean of 10 runs, each run has 200 ticks and at each tick 100 roll-outs:
    mean roi:  80.59
    mean target roi:  78.73
