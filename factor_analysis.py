import pandas as pd
import pandas_datareader.data as web
import numpy as np 
from data_fetcher import fetch_data, calculate_features
from hmm_model import HMM
from sklearn.preprocessing import StandardScaler

def load_ff(start_date, end_date):
    # (Mkt-RF, SMB, HML, RMW, CMA, RF)
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start_date, end_date)[0]
    # Momentum
    mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start_date, end_date)[0]
    # Let's combine them
    factors = pd.concat([ff5,mom], axis = 1)
    factors = factors / 100

    return factors.dropna()

def analyze_performance(hmm_states, factor_data):
    # Align data by time index to ensure we have the exact same period 
    common_idx = hmm_states.index.intersection(factor_data.index) # Finds the overlap dates in hmm and factor
    states = hmm_states.loc[common_idx] # Filter hmm for common dates
    factors = factor_data.loc[common_idx] # Filter factors for common dates

    factors.columns = [c.strip() for c in factors.columns]

    rf = factors['RF']
    mkt_rf = factors['Mkt-RF']
    smb = factors['SMB']
    hml = factors['HML']
    rmw = factors['RMW']
    cma = factors['CMA']
    mom = factors['Mom']

    # Market (baseline)
    ret_market = mkt_rf + rf
    # FF3
    ret_ff = mkt_rf + smb + hml + rf
    # Carhart
    ret_carhart = mkt_rf + smb + hml + mom + rf
    # Value
    ret_value = mkt_rf + hml + rf
    # AQR
    ret_aqr = mkt_rf + hml + mom + rmw + cma + rf

    strategies = pd.DataFrame({
        'Fama-French3' : ret_ff,
        'Carhart' : ret_carhart,
        'Value' : ret_value,
        'AQR': ret_aqr,
        'Market': ret_market,
        'RF': rf
    }, index = common_idx)
    
    strategies['State'] = states

    results = []
    unique_states = np.sort(strategies['State'].unique()) # Sort the regimes, so the loop runs in order

    for state  in unique_states:
        state_data = strategies[strategies['State'] == state]
        row = {'State': state, 'Count': len(state_data)}
        for strategy in ['Fama-French3', 'Carhart', 'Value', 'AQR','Market']:
            ret = state_data[strategy]
            avg_ret = ret.mean() * 252
            sharpe = ((ret - rf).mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
            row[f'{strategy}_Sharpe'] = sharpe
            row[f'{strategy}_Return'] = avg_ret
        results.append(row)
    return pd.DataFrame(results)
 
if __name__ == '__main__':
    df = fetch_data(start = '2007-01-01', end = '2017-12-31')
    df = calculate_features(df).dropna()

    train_df = df.copy()
    X_train = train_df[['Daily_return', 'MSE_Vol']].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    hmm = HMM()
    hmm.train(X_train)

    states = hmm.predict(X_train)
    train_df['HMM_state'] = states

    state_means = train_df.groupby('HMM_state')['Daily_return'].mean() 
    print('\nState Mean Daily Return:')
    print(state_means)

    ff_data = load_ff('2007-01-01','2017-12-31')
    if not ff_data.empty:
        perf = analyze_performance(train_df['HMM_state'], ff_data)
        print('\nSharpe Ratio per regime:')
        print(perf.set_index('State')[['Fama-French3_Sharpe','Carhart_Sharpe','Value_Sharpe','AQR_Sharpe', 'Market_Sharpe']])
        print('\nAnnualized Return per regime:')
        print(perf.set_index('State')[['Fama-French3_Return', 'Carhart_Return', 'Value_Return', 'AQR_Return', 'Market_Return']])
        print('\nBest performing strategy per regime:')
        for _, row in perf.iterrows():
            state = row['State']
            sharpes = {k.replace('_Sharpe', ''): v for k, v in row.items() if '_Sharpe' in k} # and 'Market' not in k
            best_strat = max(sharpes, key = sharpes.get)
            print(f'State {state} ({state_means[state]*100:.4f}% daily avg): Best = {best_strat} (Sharpe: {sharpes[best_strat]:.2f})')
    else:
        print('Failed to retrieve Fama-French data.')