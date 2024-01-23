import pandas as pd


def return_train_test_data(data, n_test, xy=False, ohe=False):
    
    if xy:
        X = data.drop(["Appliances", "Appliances_24"], axis = 1)
        y = enhanced_data.Appliances_24

        if ohe:
            X = pd.get_dummies(X, dtype=np.uint8)
        
        X_train, y_train = X.iloc[:-n_test], y.iloc[:-n_test]
        X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]

        return X_train, X_test, y_train, y_test

    else:
        train, test = data.iloc[:-n_test], data.iloc[-n_test:]
        return train, test