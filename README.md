# Option Pricing Techniques

This project implements and compares several techniques for pricing European call options, focusing on both computational efficiency and pricing accuracy.

## 📈 Techniques Compared

- **Black-Scholes Model** (analytical baseline)
- **Fast Fourier Transform (FFT)**
- **Binomial Tree Model**
- **Monte Carlo Simulation**
- **Numerical Integration**

Each method is benchmarked against the Black-Scholes model in terms of:

- ✅ Accuracy
- ⏱ Runtime
- 📊 Error vs Runtime tradeoff score

## 🔬 Goals

- Understand tradeoffs in option pricing algorithms
- Visualize accuracy vs speed performance
- Identify optimal methods under different constraints

## 📊 Sample Outputs

- Accuracy vs Runtime plot (log-log scale)
- Best-performing method based on normalized tradeoff score

## 🛠 How to Run

1. Clone the repo:
   ```bash
   git clone git@github.com:your-username/Option-Pricing-Techniques.git
   cd Option-Pricing-Techniques

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies: 
   ```bash
   pip install -r requirements.txt 

4. Run the main script: 
   ```bash
   python option_pricing_techniques.py 

## File Structure 
option_pricing_techniques.py   # Core implementations and comparison logic
requirements.txt               # Dependencies
README.md                      # You're reading it

## Future Enhancements 
1. Add implied volatility surfaces 
2. Extend support for exploring pricing techniques for American  options or other path dependent derivatives 
3. Memory profiling and scalability tests 

## Author 
Ionut Nodis
