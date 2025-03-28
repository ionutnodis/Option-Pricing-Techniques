

#author:@ionutnodis
#time: 2025-03-22


#--- Import libraries ---#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


#--- Comparison of Option Pricing Techniques ---#

#---- FFT -----#
# Characteristic function for BS model
def bs_char_func(u, S0, r, T, sigma):
    """
    Computes the characteristic function for the Black-Scholes model.

    This function calculates the Fourier transform of the log of the stock price 
    under the risk-neutral measure, which is a key component in option pricing 
    using the FFT method.

    Parameters:
    - u: Complex argument for the characteristic function
    - S0: Spot price of the underlying asset
    - r: Risk-free interest rate
    - T: Time to maturity (in years)
    - sigma: Volatility of the underlying asset

    Returns:
    - The value of the characteristic function for the given parameters
    """
    
    
    i = 1j
    return np.exp(i * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * u**2 * T)

# Carr-Madan FFT pricing
def fft_option_price(S0, K, r, T, sigma, alpha=1.5, N=2**12, eta=0.25):
    """
    Computes the option price using the Carr-Madan FFT method.

    This function implements the Carr-Madan approach to option pricing, which 
    leverages the characteristic function of the underlying asset's price process 
    and the Fast Fourier Transform (FFT) to efficiently compute option prices.

    Parameters:
    - S0: Spot price of the underlying asset
    - K: Strike price of the option
    - r: Risk-free interest rate
    - T: Time to maturity (in years)
    - sigma: Volatility of the underlying asset
    - alpha: Damping factor to ensure convergence (default is 1.5)
    - N: Number of FFT points (default is 2^12)
    - eta: Spacing of the integration grid (default is 0.25)

    Returns:
    - The interpolated option price for the given strike price (K)
    """
    
    lambd = 2 * np.pi / (N * eta)
    b = np.log(K) - N * lambd / 2
    u = np.arange(N) * eta
    v = b + np.arange(N) * lambd

    i = 1j
    phi = bs_char_func(u - (alpha + 1) * i, S0, r, T, sigma)
    simpson_weights = (3 + (-1) ** np.arange(N)) / 3
    integrand = (
        np.exp(-r * T)
        * phi
        / (alpha**2 + alpha - u**2 + i * (2 * alpha + 1) * u)
        * np.exp(i * u * (-b))
        * eta
        * simpson_weights
    )

    fft_values = np.fft.fft(integrand).real
    strikes = np.exp(v)
    prices = np.exp(-alpha * v) / np.pi * fft_values
    return np.interp(K, strikes, prices)


#---- Black-Scholes -----#

# Black-Scholes pricing
def bs_call_price(S0, K, r, T, sigma):
    """
    Calculate European Call option prices using Black-Scholes-Merton formula.

    The function takes the following input Parameters:

        S0(float): Spot price of the underlying asset
        K (float): Strike price 
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying asset (annualized)

    And outputs the prices of European Call Option according to the Black-Scholes-Merton Model.
 
    """
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


#--- Binomial Tree ---#

# Binomial tree pricing
def binomial_call_price(S0, K, r, T, sigma, N=1000):
    """
    Computes the price of a European call option using the Binomial Tree model.

    This implementation supports dividend payments at specified times, making it
    suitable for pricing options on dividend-paying stocks.

    Parameters:
    - S0 : float : Initial stock price
    - K : float : Strike price
    - T : float : Time to maturity (in years)
    - r : float : Risk-free interest rate (annualized)
    - sigma : float : Volatility of the underlying stock (annualized)
    - N : int : Number of steps in the binomial tree

    Returns:
    - float : Option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    ST = np.array([S0 * u**j * d**(N - j) for j in range(N + 1)])
    option_values = np.maximum(ST - K, 0)

    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
    return option_values[0]


#--- Monte Carlo ---#

# Function to compute the option price using Monte Carlo
def monte_carlo_call_price(S0, K, r, T, sigma, simulations=10000):
    """
    Computes the price of a European call option using the Monte Carlo simulation method.

    This function generates multiple random paths for the underlying asset price
    based on the Geometric Brownian Motion model. It then calculates the payoff
    for each path and discounts it back to the present value to estimate the option price.

    Parameters:
    - S0: Spot price of the underlying asset
    - K: Strike price of the option
    - r: Risk-free interest rate
    - T: Time to maturity (in years)
    - sigma: Volatility of the underlying asset
    - simulations: Number of Monte Carlo simulations (default is 10,000)

    Returns:
    - The estimated price of the European call option
    """
    
    Z = np.random.standard_normal(simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# --- Option Pricing using numerical integration ----# 
#Function to compute put options via numerical integration 
#We assume that the stock price follows a lognoral distribution 

#Function to Compute the lognormal density 
def logNormal(S, r, q, sig, S0, T): 
    """
    Computes the lognormal density function for a given stock price.

    This function calculates the probability density of the stock price at time T
    under the assumption that it follows a lognormal distribution.

    Parameters:
    - S: Stock price at time T
    - r: Risk-free interest rate
    - q: Dividend yield
    - sig: Volatility of the underlying asset
    - S0: Initial stock price
    - T: Time to maturity (in years)

    Returns:
    - The value of the lognormal density function for the given parameters
    """ 
    f = np.exp(-((np.log(S/S0) - (r-q-0.5*sig**2)*T)**2)/(2*sig**2*T))/(S*np.sqrt(2*np.pi*sig**2*T))
    return f

#Parameters for the Put 
S0 = 100
K = 90
r = 0.05
q = 0.02
sig = 0.25
T = 1.0

#Function to compute the numerical integration for the put 

def numerical_integral_put(r, q, S0, K, sig, T, N): 
    
    #temporary values
    eta = 0.0 #spacing of the integration grid
    priceP = 0.0 #price of the put option
    
    #discount factor 
    df = np.exp(-r*T)
    #step size
    eta = 1. *K/N
    #vector of stock prices
    S = np.arange(1, N+1)*eta
    #vector of weights
    w = np.ones(N)*eta
    w[0] = eta / 2
    #lognormal density
    logN = np.zeros(N)
    logN = logNormal(S, r, q, sig, S0, T)
    #numerical integral 
    sumP = np.sum((K-S)*logN*w)
    #price of the put option
    priceP = df*sumP
    return eta, priceP

#Now we can use the previous function to price the put for different values of N = 2^n , n=1,2,...,15

#Define a vector with all values of N 
n_min = 1
n_max = 15
n_vec = np.arange(n_min, n_max+1, dtype=int)

#Compute numerical integration for all values of N
start = time.time()

#Storing the results in a Pandas DataFrame

def results_to_dataframe(n_vec, eta_vec, put_vec) -> pd.DataFrame:
    """
    Saves the results of numerical integration into a pandas DataFrame.

    Parameters:
    - n_vec: Array of N values (powers of 2)
    - eta_vec: Array of eta values (step sizes)
    - put_vec: Array of put option prices

    Returns:
    - A pandas DataFrame containing the results
    """
    results_df = pd.DataFrame({
        'N': [f'2^{n}' for n in n_vec],
        'eta': eta_vec,
        'P_0': put_vec
    })
    return results_df

# Calculate eta_vec based on n_vec
eta_vec = K / (2 ** n_vec)

# Compute put_vec using numerical integration for all values of N
put_vec = np.array([numerical_integral_put(r, q, S0, K, sig, T, N=2**n)[1] for n in n_vec])

# Save the results into a DataFrame
integration_results = results_to_dataframe(n_vec, eta_vec, put_vec)

end = time.time()
print("Time for numerical integration:", end - start)



# --- Comparing all Option Pricing Techniques ---#

#Parameters and pricing across strikes 

S0 = 100 
r = 0.05
T = 1.0
sigma = 0.2
q = 0.0 #no dividends

#Array of strike prices
K_values = np.linspace(60, 140, 50)


#Dictionary to store the prices and times
fft_prices, bs_prices, binomial_prices, mc_prices, ni_prices = [], [], [], [], []
fft_times, bs_times, binomial_times, mc_times, ni_times = [], [], [], [], []


#Loop over to compute the prices and times
for K in K_values:
    start = time.time()
    bs_prices.append(bs_call_price(S0, K, r, T, sigma))
    bs_times.append(time.time() - start)

    start = time.time()
    fft_prices.append(fft_option_price(S0, K, r, T, sigma))
    fft_times.append(time.time() - start)

    start = time.time()
    binomial_prices.append(binomial_call_price(S0, K, r, T, sigma))
    binomial_times.append(time.time() - start)

    start = time.time()
    mc_prices.append(monte_carlo_call_price(S0, K, r, T, sigma))
    mc_times.append(time.time() - start)
    
    # Put-Call parity and pricing using numerical integration
    start = time.time()
    _, put_price = numerical_integral_put(r, q, S0, K, sigma, T, N=2**15)
    call_price = put_price + S0 * np.exp(-q * T) - K * np.exp(-r * T)
    ni_prices.append(call_price)
    ni_times.append(time.time() - start)

# --- Plot Prices ---
plt.figure(figsize=(12, 6))
plt.plot(K_values, bs_prices, label="Black-Scholes", linestyle="--")
plt.plot(K_values, fft_prices, label="FFT")
plt.plot(K_values, binomial_prices, label="Binomial Tree", linestyle=":")
plt.plot(K_values, mc_prices, label="Monte Carlo", linestyle="-.")
plt.plot(K_values, ni_prices, label="Numerical Integration", linestyle="-")
plt.title("European Call Option Pricing: All Methods")
plt.xlabel("Strike Price (K)")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Computation Times ---
plt.figure(figsize=(12, 6))
plt.plot(K_values, bs_times, label="BS Time")
plt.plot(K_values, fft_times, label="FFT Time")
plt.plot(K_values, binomial_times, label="Binomial Time")
plt.plot(K_values, mc_times, label="MC Time")
plt.plot(K_values, ni_times, label="NI Time")
plt.title("Computation Time per Method")
plt.xlabel("Strike Price (K)")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()

# --- Determine Fastest Method ---
average_times = {
    "Black-Scholes": np.mean(bs_times),
    "FFT": np.mean(fft_times),
    "Binomial": np.mean(binomial_times),
    "Monte Carlo": np.mean(mc_times),
    "Numerical Integration": np.mean(ni_times)
}

fastest_method = min(average_times, key=average_times.get)
print("Fastest method on average:", fastest_method)


# --- Performance Evaluation Section --- #
# --- Accuracy vs Runtime ---
# We consider BSM to be the baseline and compare it with the other techniques

def compute_accuracy_vs_runtime(S0, K_values, r, T, sigma, bs_fn, fft_fn, binomial_fn, mc_fn, ni_fn):
    """
    Computes the accuracy and runtime of different option pricing methods.
    
    """
    
    # Initialize list to store results
    results = []

    for K in K_values:
        # Black-Scholes
        bs_start = time.time()
        bs_price = bs_fn(S0, K, r, T, sigma)
        bs_time = time.time() - bs_start

        # FFT
        fft_start = time.time()
        fft_price = fft_fn(S0, K, r, T, sigma)
        fft_time = time.time() - fft_start

        # Binomial
        binomial_start = time.time()
        binomial_price = binomial_fn(S0, K, r, T, sigma)
        binomial_time = time.time() - binomial_start

        # Monte Carlo
        mc_start = time.time()
        mc_price = mc_fn(S0, K, r, T, sigma)
        mc_time = time.time() - mc_start

        # Numerical Integration
        q = 0.0
        ni_start = time.time()
        _, put_price = ni_fn(r, q, S0, K, sigma, T, 2**14)
        ni_price = put_price + S0 * np.exp(-q * T) - K * np.exp(-r * T)
        ni_time = time.time() - ni_start

        results.append({
            'K': K,
            'BS Price': bs_price, 'FFT Price': fft_price, 'Binomial Price': binomial_price,
            'MC Price': mc_price, 'NI Price': ni_price,
            'BS Time': bs_time, 'FFT Time': fft_time, 'Binomial Time': binomial_time,
            'MC Time': mc_time, 'NI Time': ni_time,
            'FFT_Error': abs(fft_price - bs_price),
            'Binomial_Error': abs(binomial_price - bs_price),
            'MC_Error': abs(mc_price - bs_price),
            'NI_Error': abs(ni_price - bs_price),
        })

    return pd.DataFrame(results)


# --- Plotting Function ---
def plot_accuracy_vs_runtime(df):
    """
    Plots absolute error vs. runtime for all methods compared to Black-Scholes.
    """
    plt.figure(figsize=(12, 6))

    plt.scatter(df['FFT Time'], df['FFT_Error'], label='FFT', marker='o')
    plt.scatter(df['Binomial Time'], df['Binomial_Error'], label='Binomial', marker='s')
    plt.scatter(df['MC Time'], df['MC_Error'], label='Monte Carlo', marker='^')
    plt.scatter(df['NI Time'], df['NI_Error'], label='Numerical Integration', marker='x')

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Absolute Error vs Black-Scholes")
    plt.title("Accuracy vs Runtime of Option Pricing Methods")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# --- Run Accuracy vs Runtime ---
if __name__ == "__main__":
    K_values = np.linspace(60, 140, 40)

    df_results = compute_accuracy_vs_runtime(
        S0=100,
        K_values=K_values,
        r=0.05,
        T=1.0,
        sigma=0.2,
        bs_fn=bs_call_price,
        fft_fn=fft_option_price,
        binomial_fn=binomial_call_price,
        mc_fn=monte_carlo_call_price,
        ni_fn=numerical_integral_put
    )

    plot_accuracy_vs_runtime(df_results)