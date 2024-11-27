# QF620 Stochastic Modelling in Finance

---

## 📖 **Project Overview**
This project explores advanced option pricing, calibration, and hedging techniques under various stochastic models. The primary focus is on vanilla and digital options, model calibration using implied volatility smiles, and dynamic delta hedging strategies.

---

## 🧩 **Part I: Analytical Option Formulae**
### **Objective**
- Analyze vanilla and digital options using the **Black-Scholes**, **Bachelier**, **Black**, and **Displaced-Diffusion** models.
- Evaluate the impact of different assumptions on option prices.

### **Key Equations**
The standard normal distribution:
\[
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}, \quad \Phi(z) = \int_{-\infty}^z \phi(z') dz'.
\]

The Black-Scholes formula for a call option:
\[
C_{BS,v} = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2),
\]
where:
\[
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}.
\]

### **Model Comparisons**
| **Model**               | **Vanilla Call Price** | **Vanilla Put Price** | **Digital Cash-or-Nothing Call Price** |
|--------------------------|------------------------|-----------------------|---------------------------------------|
| Black-Scholes            | \( C_{BS,v} \)       | \( P_{BS,v} \)       | \( C_{BS,c} \)                       |
| Bachelier                | \( C_{Ba,v} \)       | \( P_{Ba,v} \)       | \( C_{Ba,c} \)                       |
| Displaced-Diffusion      | \( C_{D,v} \)        | \( P_{D,v} \)        | \( C_{D,c} \)                        |

---

## 📊 **Part II: Model Calibration**

### **Objective**
Calibrate **Displaced-Diffusion (DD)** and **SABR** models to match market-implied volatility smiles observed in **SPX** and **SPY** options.

### **Calibration Parameters**
1. **Displaced-Diffusion (DD)**:
   \[
   \min_{\sigma, \beta} \sum_{i=1}^n (\epsilon_i^D)^2
   \]

2. **SABR Model**:
   \[
   \min_{\alpha, \rho, \nu} \sum_{i=1}^n (\epsilon_i^{SABR})^2
   \]

### **Implied Volatility Smiles**
- **SPX** and **SPY** demonstrate typical volatility smiles, with higher implied volatilities for OTM puts.
- Calibration reveals **SABR's** superior ability to capture market dynamics compared to DD.

| Parameter | SPX (17 Days) | SPX (45 Days) | SPY (17 Days) | SPY (45 Days) |
|-----------|---------------|---------------|---------------|---------------|
| \(\alpha\) | 1.212         | 1.817         | 0.665         | 0.908         |
| \(\rho\)   | -0.301        | -0.404        | -0.412        | -0.489        |
| \(\nu\)    | 5.460         | 2.790         | 5.250         | 2.729         |

---

## 🔄 **Part III: Static Replication**
### **Payoff Function**
The exotic derivative payoff:
\[
h(S_T) = S_T^{\frac{1}{3}} + 1.5 \cdot \ln(S_T) + 10.0.
\]

The valuation formula:
\[
V_0 = e^{-rT}h(F) + \int_{F}^0 h''(K)P(K)dK + \int_{F}^\infty h''(K)C(K)dK.
\]

### **Model-Free Integrated Variance**
\[
E = 2 e^{rT} \left(\int_{F}^0 \frac{P(K)}{K^2}dK + \int_{F}^\infty \frac{C(K)}{K^2}dK\right).
\]

---

## 📈 **Part IV: Dynamic Hedging**
### **Dynamic Delta Hedging**
The delta hedge for a call option under Black-Scholes:
\[
\phi_t = \Phi\left(\frac{\ln\left(\frac{S_t}{K}\right) + \left(r + \frac{1}{2}\sigma^2\right)(T-t)}{\sigma \sqrt{T-t}}\right).
\]

### **Simulation Results**
- **Hedging Errors**:
  - \(N = 21\): Higher variability.
  - \(N = 84\): Smoother error trajectories and reduced variability.
- **PnL Distribution**:
  - Standard deviation decreases from \( \sigma = 0.43 \) (daily rebalancing) to \( \sigma = 0.22 \) (sub-daily rebalancing).

---

## 🔍 **Insights**
1. **Displaced-Diffusion Model**:
   - Improved over Black-Scholes but limited by its simplicity.
2. **SABR Model**:
   - Captures volatility clustering and tail behavior more effectively.
3. **Dynamic Hedging**:
   - Increased rebalancing frequency reduces hedging errors but incurs higher transaction costs.

---

## 🛠 **Technologies Used**
- Python (NumPy, SciPy, Statsmodels)
- Jupyter Notebook
- Monte Carlo Simulations

---

## 📜 **References**
- Hagan, P. S., et al. (2002). "Managing Smile Risk." Wilmott Magazine.
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities."

---

## 💬 **Contact**
For questions or collaborations, please reach out to **Yash Ashish Kumar Joshi** at [yashak.j.2024@mqf.smu.edu.sg].
