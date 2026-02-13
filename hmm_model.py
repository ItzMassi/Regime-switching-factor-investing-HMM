from hmmlearn.hmm import GaussianHMM
from data_fetcher import fetch_data, calculate_features
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class HMM:
    def __init__(self, n_components = 2, covariance_type = 'full', n_iter = 75, random_state = 42):
        self.model = GaussianHMM(n_components = n_components,
                                 covariance_type = covariance_type,
                                 n_iter = n_iter,
                                 random_state = random_state)
    
    def train(self, data):
        print(f'Training the model with {self.model.n_components} components.')
        self.model.fit(data)
        print('Model converged:', self.model.monitor_.converged)

    def predict(self, data):
        return self.model.predict(data)
    
    def state_stats(self):
        return self.model.means_, self.model.covars_


if __name__ == '__main__':
    df = fetch_data(start = '2007-01-01', end = '2025-12-31')
    df = calculate_features(df).dropna()

    # Filtering for training window
    train_df = df.loc['2007-01-01':'2017-12-31'].copy()
    print(train_df.head())

    if not train_df.empty:
        X_train = train_df[['Daily_return', 'MSE_Vol']].values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        hmm_model = HMM()
        hmm_model.train(X_train)

        means, covars = hmm_model.state_stats()

        print('\nState Means (Daily Return and Volatility), as number of standard deviations from the average:')
        for i,mean in enumerate(means):
            print(f'State: {i}: {mean}')
        
        print('\nState Covariances (diagonal elements roughly):')
        for i,cov in enumerate(covars):
            print(f'State: {i}: \n{cov}')

        hidden_states = hmm_model.predict(X_train)
        unique, counts = np.unique(hidden_states, return_counts = True)
        print("\nState Distribution in Training Set:")
        print(dict(zip(unique, counts)))