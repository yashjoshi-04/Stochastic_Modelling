import pandas as pd
import numpy as np
from scipy.stats import norm

# ------------------------------------------------_Black_Scholes_Model_------------------------------------------------#

# Parameters defined:
# - option_type: str, type of option ('vanilla_call', 'vanilla_put','cash_call', 'cash_put', 'asset_call', 'asset_put')
# - S: float, current stock price
# - K: float, strike price
# - T: float, time to maturity (in years)
# - R: float, risk-free interest rate (annualized)
# - sigma: float, volatility of the underlying asset (annualized)
# - q: float, continuous dividend yield (default = 0)
# - Returns: float, option price based on type of option.

def Black_Scholes_Model (option_type, S, K, T, R, sigma, q):
    
    # Cumulative distribution function and Probability density function of d1 and d2 calculated below
    d1 = (np.log(S / K) + (R - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Option pricing formula is input below:
    if option_type == 'vanilla_call':
        return S * np.exp(-q * T) * norm.norm.cdf(d1) - K * np.exp(-R * T) * norm.norm.cdf(d2)
    
    elif option_type == 'vanilla_put':
        return K * np.exp(-R * T) * norm.norm.cdf(-d2) - S * np.exp(-q * T) * norm.norm.cdf(-d1)
    
    elif option_type == 'digital_cash_call':
        return np.exp(-R * T) *norm.norm.cdf(d2)
    
    elif option_type == 'digital_cash_put':
        return np.exp(-R * T) *norm.norm.cdf(d2)
    
    elif option_type == 'digital_asset_call':
        return S * np.exp( -q * T) * norm.norm.cdf(d1)
    
    elif option_type == 'digital_asset_put':
        return S * np.exp( -q * T) * norm.norm.cdf(-d1)
    
    else: ValueError(' Invalid Option type -  Choose from the below options a. vanilla_call b. vanilla_put c. digital_cash_call d. digital_cash_put e. digital_asset_call f. digital_asset_put')


# Just for testing below: Input random parameters and  check outcomes

option_type = 'vanilla_call'
S = 100
K = 110
T = 0.3
R = 0.47
sigma = 0.8
q = 0

print(Black_Scholes_Model('vanilla_call', S, K, T, R, sigma, q))


# ------------------------------------------------Bachelier's_Model------------------------------------------------#

# Parameters defined:
# - option_type: str, type of option ('vanilla_call', 'vanilla_put','cash_call', 'cash_put', 'asset_call', 'asset_put')
# - S: float, current stock price
# - T: float, time to maturity (in years)
# - R: float, risk-free interest rate (annualized)
# - sigma: float, volatility of the underlying asset (annualized)
# - K: float, strike price

def Bacheliers_Model (option_type, S, T, R, sigma, K):
    
    # Cumulative distribution function and Probability density function of d calculated below
    d = (S - K) / (sigma * np.sqrt(T))
    
    # Option Pricing Formula is input below:
    if option_type == 'vanilla_call':
        return np.exp(-R * T) * (((S - K)* norm.norm.cdf(d)) + ((sigma * T) * norm.norm.pdf(d)))
    
    elif option_type == 'vanilla_put':
        return np.exp(-R * T) * (((K - S)* norm.norm.cdf(-d)) + ((sigma * T) * norm.norm.pdf(d)))
    
    # The cash-or-nothing call pays a fixed cash amount if the asset price S exceeds the strike K at maturity.
    elif option_type == 'digital_cash_call':
        return np.exp(-R * T) * norm.norm.cdf(d)
    
    elif option_type == 'digital_cash_put':
        return np.exp(-R * T) * norm.norm.cdf(-d)
    
    # The asset-or-nothing call pays the value of the asset if S > K at maturity.
    elif option_type == 'digital_asset_call':
        return S * norm.norm.cdf(d)
    
    elif option_type == 'digital_asset_put':
        return S * norm.norm.cdf(-d)


# Just for testing below: Input random parameters and  check outcomes

option_type = 'digital_asset_call'
S = 100
K = 130
T = 2
R = 0.47
sigma = 0.8

print(Bacheliers_Model('digital_asset_call', S, K, T, R,  sigma))


# ------------------------------------------------Black's_Model------------------------------------------------#

# Parameters defined:
# - option_type: str, type of option ('vanilla_call', 'vanilla_put','cash_call', 'cash_put', 'asset_call', 'asset_put')
# - S: float, current stock price
# - T: float, time to maturity (in years)
# - R: float, risk-free interest rate (annualized)
# - F: float, forward price - F = S*np.exp(r*T)
# - K: float, strike price
# - sigma: float, volatility of the underlying asset (annualized)

def Blacks_Model (option_type, S, T, R, F, sigma):
    
    # Calculating d1 and d2 and the respective Cumulative Distribution Function below:
    d1 = (np.log(F/K) + ((sigma ** 2)*T)/2) / (sigma * np.sqrt(T))
    d2 = (np.log(F/K) - ((sigma ** 2)*T)/2) / (sigma * np.sqrt(T))
    
    if option_type == 'vanilla_call':
        return np.exp(-R * T) * ((F * norm.norm.cdf(d1)) - (K * norm.norm.cdf(d2)))
    
    if option_type == 'vanilla_put':
        return np.exp(-R * T) * ((K * norm.norm.cdf(-d2)) - (F * norm.norm.cdf(-d1)))
    
    if option_type == 'digital_cash_call':
        return np.exp(-R * T) * norm.norm.cdf(d2)
    
    if option_type == 'digital_cash_put':
        return np.exp(-R * T) * norm.norm.cdf(-d2)
    
    if option_type == 'digital_asset_call':
        return F * norm.norm.cdf(d1)
    
    if option_type == 'digital_asset_put':
        return F * norm.norm.cdf(-d1)


# Just for testing below: Input random parameters and  check outcomes

option_type = 'digital_asset_call'
S = 100
K = 120
T = 0.3
R = 0.47
sigma = 0.8

print(Blacks_Model('digital_asset_call', S, K, T, R,  sigma))


# ------------------------------------------------Displaced_Diffusion_Model------------------------------------------------#

# Parameters defined:
# - option_type: str, type of option ('vanilla_call', 'vanilla_put','cash_call', 'cash_put', 'asset_call', 'asset_put')
# - S: float, current stock price
# - T: float, time to maturity (in years)
# - R: float, risk-free interest rate (annualized)
# - F: float, forward price - F = S*np.exp(r*T)
# - K: float, strike price
# - sigma: float, volatility of the underlying asset (annualized)
# - B: Beta, float, value between 0 and 1. If the value tends to 1, the Displaced - Diffusion model tends to be the same as Blacks Model.
# - A: Alpha, float, value is (1-B/B)*F

def Displaced_Diffusion_Model(option_type, S, T, R, F, sigma, B):
    
    # Calculating d1 and d2 and the respective Cumulative Distribution Function below:
    d1 = (np.log((F + (((1-B)/B) * F))/(K + (((1-B)/B) * F))) + ((sigma ** 2)*T)/2) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    
    if option_type == 'vanilla_call':
        return np.exp(-R * T) * ((F/B * norm.norm.cdf(d1)) - ((K + (((1-B)/B) * F)) * norm.norm.cdf(d2)))
    
    if option_type == 'vanilla_put':
        return np.exp(-R * T) * (((K + (((1-B)/B) * F)) * norm.norm.cdf(-d2)) - (F/B * norm.norm.cdf(-d1)))
    
    if option_type == 'digital_cash_call':
        return np.exp(-R * T) * norm.norm.cdf(d2)
    
    if option_type == 'digital_cash_put':
        return np.exp(-R * T) * norm.norm.cdf(-d2)
    
    if option_type == 'digital_asset_call':
        return F/B * norm.norm.cdf(d1)
    
    if option_type == 'digital_asset_put':
        return F/B * norm.norm.cdf(-d1)


# Just for testing below: Input random parameters and  check outcomes

option_type = 'digital_asset_call'
S = 100
K = 120
T = 0.3
R = 0.47
sigma = 0.8
B = 0.45

print(Displaced_Diffusion_Model('digital_asset_call', S, K, T, R,  sigma, B))

