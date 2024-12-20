# Non-Stationary Time Series Analysis

_Just some Montecarlo simutations to understand it..._

_Just run it and you will see..._

Some technical details following Enders (2014, p. 183):

- In code `1 - simulations_slide_5.py`, the `"Process 1"` have a deterministic trend because of $|\phi = 0 | < 1$.
- In code `1 - simulations_slide_5.py`, the `"Process 2"` have a deterministic trend because of $|\phi = 0.5 | < 1$.
- In code `1 - simulations_slide_5.py`, the `"Process 3"` have a deterministic trend because of $|\phi = 0.5 | < 1$.
- In code `2.0.1 - stochastic_trend_(levels_of_the_first-difference_stochastic_trend_slide_6).py`, the `"Process 2"` have a stochastic trend because of $|\phi = 1 | = 1$.
- In code `2.0.1 - stochastic_trend_(levels_of_the_first-difference_stochastic_trend_slide_6).py`, the `"Process 3"` have a stochastic trend because of $|\phi = 1 | = 1$.
- In code `2.1.1 - exponential_growth_levels.py`, the `"Process 2"` have hyper-exponential growth because of $|\phi = 2 | > 1$.
- In code `2.1.1 - exponential_growth_levels.py`, the `"Process 3"` have hyper-exponential growth because of $|\phi = 2 | > 1$.


## References

Enders, W. (2014). “Applied Econometric Time Series, 4th Edition”. Wiley Series in Probability and Statistics.
[https://www.wiley.com/en-us/Applied+Econometric+Time+Series%2C+4th+Edition-p-9781118808566](https://www.wiley.com/en-us/Applied+Econometric+Time+Series%2C+4th+Edition-p-9781118808566)

