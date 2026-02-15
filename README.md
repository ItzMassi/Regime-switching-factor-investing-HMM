# Regime-Switching Factor Investing with Hidden Markov Models

A quantitative investment strategy that uses Hidden Markov Models (HMM) to detect market regimes and dynamically switches between factor-based investment strategies to optimize risk-adjusted returns.

## Overview

Financial markets exhibit different behavioral patterns—bull markets characterized by steady growth and low volatility, and bear markets marked by heightened uncertainty and negative returns. This project leverages unsupervised machine learning (HMM) to identify these latent market states and adapts the investment strategy accordingly:

- **Bull Regime** → AQR-style multi-factor strategy (Value, Momentum, Quality, Investment)
- **Bear Regime** → Cash (risk-free rate)

## Key Features

- **Regime Detection**: 2-state Gaussian HMM trained on daily returns and rolling volatility
- **Rolling Window Training**: 10.7-year (~2,707 trading days) rolling window for adaptive model updates
- **Confidence Filtering**: State probability threshold (>80%) prevents spurious regime switches
- **Factor Strategies**: Multiple academic factor models (FF3, Carhart, AQR, Value)
- **Backtesting Framework**: Full out-of-sample backtesting with performance metrics

## Methodology

### 1. Feature Engineering
The HMM is trained on two features extracted from SPY (S&P 500 ETF):
- **Daily Returns**: Close-to-close percentage changes
- **Rolling Volatility**: 10-day MSE-based volatility measure

### 2. Regime Classification
The trained HMM classifies each trading day into one of two states:
| State | Characteristics | Strategy |
|-------|----------------|----------|
| Bull | Higher mean returns, lower volatility | AQR Multi-Factor |
| Bear | Lower/negative returns, higher volatility | Cash |

### 3. Confidence Thresholds
To avoid overtrading during uncertain periods, strategy switches only occur when the HMM's `predict_proba` returns a state probability greater than 80%. This ensures high-confidence regime classification before triggering any strategy change.

## Project Structure

```
.
├── backtest.py          # Main backtesting engine
├── factor_analysis.py   # Factor strategy analysis per regime
├── hmm_model.py         # HMM wrapper class
├── data_fetcher.py      # Data retrieval and feature calculation
└── CLAUDE.md            # Development guidelines
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `data_fetcher.py` | Fetches SPY price data via Yahoo Finance, calculates daily returns and rolling volatility |
| `hmm_model.py` | Wrapper around `hmmlearn.GaussianHMM` for regime detection |
| `factor_analysis.py` | Loads Fama-French factors, evaluates strategy performance by regime |
| `backtest.py` | Runs out-of-sample backtest with rolling HMM training and dynamic strategy switching |

## Installation

```bash
# Clone the repository
git clone https://github.com/ItzMassi/Regime-switching-factor-investing-HMM.git
cd Regime-switching-factor-investing-HMM

# Install dependencies
pip install numpy pandas yfinance pandas_datareader hmmlearn scikit-learn scipy matplotlib tqdm
```

## Usage

### Run the Backtest
```bash
python backtest.py
```
Executes the regime-switching strategy from 2018 to present, comparing against SPY benchmark.

### Analyze Factor Performance by Regime
```bash
python factor_analysis.py
```
Evaluates Sharpe ratios and returns of different factor strategies within each regime (2007-2017 training period).

### HMM Training Diagnostics
```bash
python hmm_model.py
```
Trains the HMM and displays state statistics (means, covariances, state distribution).

## HMM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Components | 2 | Number of hidden states (Bull/Bear) |
| Covariance | Full | Full covariance matrices per state |
| Iterations | 75 | Maximum EM iterations |
| Window | 2,707 days | Rolling training window (~10.7 years) |

## Data Sources

- **[Yahoo Finance](https://finance.yahoo.com/)** (via `yfinance`): SPY historical OHLCV data
- **[Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)** (via `pandas_datareader`): Fama-French 5 factors + Momentum

## Factor Strategies

| Strategy | Factors | Use Case |
|----------|---------|----------|
| Market | Mkt-RF | Baseline equity exposure |
| Fama-French 3 | Mkt-RF + SMB + HML | Classic value/size tilts |
| Carhart | FF3 + Momentum | Adds momentum premium |
| Value | Mkt-RF + HML | Pure value exposure |
| AQR | Mkt-RF + HML + Mom + RMW + CMA | Comprehensive multi-factor |

## Dependencies

- `numpy`
- `pandas`
- `yfinance`
- `pandas_datareader`
- `hmmlearn`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `tqdm`

## References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
- Carhart, M. M. (1997). On persistence in mutual fund performance.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- Wang, Lin, Mikhelson (2020). Regime-Switching Factor Investing with Hidden Markov Models


## License

This project is for educational and research purposes.
