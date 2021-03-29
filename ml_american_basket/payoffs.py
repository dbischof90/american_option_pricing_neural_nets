import math as m
import numpy as np

def vanilla_call(x, tau, K, r):
    return m.exp(-r*tau) * np.maximum(x - K, 0).ravel()

def vanilla_put(x, tau, K, r):
    return m.exp(-r*tau) * np.maximum(K - x, 0).ravel()

def sum_put(x, tau, K, r):
    return vanilla_call(x.sum(axis=0), tau, K, r)

def strangle_spread(x, tau, r, K1, K2, K3, K4):
    l1 = - vanilla_put(x, tau, K1,r)
    l2 = vanilla_put(x, tau, K2,r)    
    l3 = vanilla_call(x, tau, K3, r)
    l4 = - vanilla_call(x, tau, K4, r)
    return l1 + l2 + l3 + l4
