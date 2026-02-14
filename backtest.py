import numpy as np
import pandas as pd
from data_fetcher import fetch_data, calculate_features
from hmm_model import HMM
from factor_analysis import load_ff
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def get_state_map(hmm, X_train):
    states = hmm.predict(X_train)
    means = []
    vols = []
    for i in range(hmm.model.n_components):
        mask = (states == i)
        means.append(X_train[mask, 0].mean())
        vols.append(X_train[mask, 1].mean())
    sorted_indices  = np.argsort(means)
    state_map = {
        sorted_indices[0] : 'Bear',
        sorted_indices[1] : 'Bull',
    }
    return state_map


def calculate_pdf_confidence(hmm, state, obs):
    mean = hmm.model.means_[state]
    cov = hmm.model.covars_[state]
    mu_ret = mean[0]
    mu_vol = mean[1]
    var_ret = cov[0,0] # Diagonal element of the covariance matrix
    var_vol = cov[1,1] # Diagonal element of the covariance matrix
    std_ret = np.sqrt(var_ret)
    std_vol = np.sqrt(var_vol)
    
    pdf_ret = stats.norm.pdf(obs[0], loc = mu_ret, scale = std_ret)
    pdf_vol = stats.norm.pdf(obs[1], loc = mu_vol, scale = std_vol)

    return pdf_ret, pdf_vol

def run_backtest(start_date = '2018-01-01', window_size = 2707):
    full_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=window_size * 2) # Buffer
    df = fetch_data(start=full_start_date.strftime('%Y-%m-%d'))
    df = calculate_features(df).dropna()

    ff_data = load_ff(start_date = '2007-01-01', end_date = '2023-12-31')
    ff_data.columns = [c.strip() for c in ff_data.columns]

    # Align
    common_idx = df.index.intersection(ff_data.index)
    df = df.loc[common_idx]
    ff_data = ff_data.loc[common_idx]

    # Starting iteration
    test_dates = df.index[df.index >= pd.to_datetime(start_date)]
    print(f'Starting backtest from {start_date} ({len(test_dates)} trading days...)')

    # Init portfolio
    current_strategy = 'RF'
    portfolio_value = 100.0
    values = []
    decisions = []

    strat_map = {
        'Bull': 'AQR',
        'Bear': 'Fama-French3'
    }

    t_start = df.index.get_loc(test_dates[0])

    for t in tqdm(range(t_start, len(df))):
        date = df.index[t]

        train_start = t - window_size
        train_data = df.iloc[train_start:t]
        X_train = train_data[['Daily_return', 'MSE_Vol']].values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        hmm = HMM(n_components = 2, n_iter = 75, random_state = 42)
        hmm.model.fit(X_train)

        state_labels = get_state_map(hmm, X_train)

        last_obs = X_train[-1].reshape(1,-1)
        current_state_idx = hmm.model.predict(last_obs)[0]
        predicted_regime = state_labels[current_state_idx]

        pdf_ret, pdf_vol = calculate_pdf_confidence(hmm, current_state_idx, last_obs[0])

        confident = (pdf_ret > 0.5) and (pdf_vol > 0.3)

        if confident:
            target_strategy = strat_map[predicted_regime]
            if target_strategy != current_strategy:
                current_strategy = target_strategy
        else: 
            pass 

        todays_factors = ff_data.loc[date]
        rf = todays_factors['RF']
        mkt = todays_factors['Mkt-RF']
        smb = todays_factors['SMB']
        hml = todays_factors['HML']
        mom = todays_factors['Mom']
        rmw = todays_factors['RMW']
        cma = todays_factors['CMA']

        if current_strategy == 'Market':
            day_ret = mkt + rf
        elif current_strategy == 'Fama-French3':
            day_ret = mkt + smb + hml + rf
        elif current_strategy == 'Carhart':
            day_ret = mkt + smb + hml + mom + rf
        elif current_strategy == 'Value':
            day_ret =  mkt + hml + rf
        elif current_strategy == 'AQR':
            day_ret = mkt + hml + mom + rmw + cma + rf
        elif current_strategy == 'Cash':
            day_ret = rf
        else:
            day_ret = 0
        
        portfolio_value *= (1+ day_ret)

        values.append({
            'Date': date,
            'Portfolio_Value': portfolio_value,
            'Strategy': current_strategy,
            'Regime': predicted_regime,
            'Confident': confident,
            'State_idx': current_state_idx
        })
    results_df = pd.DataFrame(values).set_index('Date')

    strat_daily_ret = results_df['Portfolio_Value'].pct_change().dropna()

    spy_series = df.loc[results_df.index, 'Close']
    spy_daily_ret = spy_series.pct_change().dropna()

    analysis_dates = strat_daily_ret.index
    rf_daily = ff_data.loc[analysis_dates, 'RF']

    strat_excess = strat_daily_ret - rf_daily
    spy_excess = spy_daily_ret - rf_daily

    strat_sharpe = 0.0
    if strat_excess.std() != 0:
        strat_sharpe = (strat_excess.mean() / strat_excess.std()) * np.sqrt(252)
    
    spy_sharpe = 0.0
    if spy_excess.std() != 0:
        spy_sharpe = (spy_excess.mean() / spy_excess.std()) * np.sqrt(252)
    
    strat_end_value = results_df['Portfolio_Value'].iloc[-1]

    print('\n---Backtest Results (Jan 2018 - Dec 2023)---')
    print(f'Strategy Sharpe Ratio: {strat_sharpe:.2f}')
    print(f'SPY Sharpe Ratio: {spy_sharpe:.2f}')
    print(f'Final Portfolio Value: {strat_end_value:.2f}')

    # Visualisation
    plt.figure(figsize=(12, 6))
    spy_curve = df.loc[results_df.index, 'Close']
    spy_curve = spy_curve / spy_curve.iloc[0] * 100
    plt.plot(results_df.index, results_df['Portfolio_Value'], label='HMM Strategy', linewidth=2)
    plt.plot(results_df.index, spy_curve, label='SPY Benchmark', linestyle='--', alpha=0.7)
    plt.title('HMM Dynamic Strategy vs SPY (2018-Present)')
    plt.ylabel('Portfolio Value (Start=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_results.png')
    print("\nSaved plot to backtest_results.png")
    
    return results_df

if __name__ == '__main__':
    df_res = run_backtest()