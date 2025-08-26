import pandas as pd
from model import RegressionModel


joined = pd.read_csv('/Users/jasonluo/Documents/nbaProj/data/transformed_data/joined.csv')
X = joined[['Age', 'G', 'GS', 'MP',
       'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
       'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
       'PF', 'PTS']]
y = joined['income']


if __name__ == "__main__":

    model = RegressionModel(X = X, y = y)
    test_size = 0.33
    X_train, X_test, y_train, y_test, y_pred = model.split_train_predict(test_size=test_size)

    model.get_accuracy(y_test, y_pred)
    model.get_feature_corrs(X)
    model.cross_val(n_splits=10)

    rf_params = {
        'n_estimators': [100, 300], 
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [20, 30],
        'max_features': ['sqrt', 'log2']
    }

    best_params = model.tune_hyperparameters(hypermarameters=rf_params, X_train=X_train, y_train=y_train)
    print(best_params)