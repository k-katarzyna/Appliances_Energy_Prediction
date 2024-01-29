# test for most stable time series split

n_test = 2*144*7
cv_params_set = [
    {"n_splits": 5, "max_train_size": 4*n_test, "test_size": n_test, "gap": 0},
    {"n_splits": 5, "max_train_size": 4*n_test, "test_size": n_test, "gap": n_test},
    {"n_splits": 5, "max_train_size": 3*n_test, "test_size": n_test, "gap": 0},
    {"n_splits": 5, "max_train_size": 3*n_test, "test_size": n_test, "gap": n_test},
    {"n_splits": 5, "max_train_size": None, "test_size": n_test, "gap": 0},
    {"n_splits": 5, "max_train_size": None, "test_size": n_test, "gap": n_test},
    {"n_splits": 10, "max_train_size": 2*n_test, "test_size": int(0.5*n_test), "gap": 0},
    {"n_splits": 10, "max_train_size": 2*n_test, "test_size": int(0.5*n_test), "gap": n_test},
    {"n_splits": 10, "max_train_size": 4*n_test, "test_size": int(0.5*n_test), "gap": 0},
    {"n_splits": 10, "max_train_size": 4*n_test, "test_size": int(0.5*n_test), "gap": n_test},
    {"n_splits": 10, "max_train_size": None, "test_size": int(0.5*n_test), "gap": 0},
    {"n_splits": 10, "max_train_size": None, "test_size": int(0.5*n_test), "gap": n_test}
]