XGB_CLASSIFIER_PARAM_GRID = {
    'n_estimators': [1200,1400,1600],
    'max_depth': [5,6,7],
    'min_child_weight': [1,2],
    'learning_rate': [0.05,0.1],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
    'gamma': [0],
    'reg_alpha': [0.1,0.2,0.5],
    'reg_lambda': [1.0,2.0],
}

XGB_REGRESSOR_PARAM_GRID = {
    'n_estimators': [1000, 1200, 1400],
    'max_depth': [5,8,12],
    'learning_rate': [0.01,0.05],
    'subsample': [1],
    'colsample_bytree': [0.4,1],
    'gamma': [0],
    'reg_alpha': [0.1,0.5],
    'reg_lambda': [1.0,2.0],
    'min_child_weight': [1,3],
}


def use_XGB_Models(X_train,y_train,X_test,y_test):
    classifier = XGBClassifier(
        use_label_encoder = False,
        random_state=17,
        eval_metrics='logloss',
    )
    regressor = XGBRegressor(
        use_label_encoder=False,
        random_state=17,
    )
    grid_search = GridSearchCV(
        estimator= regressor,
        param_grid=XGB_REGRESSOR_PARAM_GRID,
        scoring='neg_mean_squared_error',
        cv=4,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train,y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters for classifier: ",grid_search.best_params_)
    y_pred = best_model.predict(X_test)
    metrics = regression_results(y_test, y_pred)
    print(metrics)

def regression_results(y_test,y_pred):
    print(y_test,y_pred)