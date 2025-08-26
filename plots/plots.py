import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_corr_matrix(X, output_path):
    """Plots and saves correlation matrix for give dataframe
    
    Parameters:
    - X: dataframe of all numerical features
    - output_path: string to denote where to save the plot"""
    plt.clf()
    corr = X.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize = [25,15])
    plt.title("Correlation Matrix For Features")
    sns.heatmap(corr, mask=mask, annot = True, square=True, fmt=".2f", cbar = True)
    plt.savefig(f"{output_path}/corr_matrix.png", dpi = 200)


def plot_residuals(y_test, y_pred, output_path):
    """Plots and saves a residual plot of predicted values vs residuals
    
    Parameters:
    - y_test: numpy array of testing data
    - y_pred: numpy array of predictions
    - output_path: string to denote where to save the plot"""
    plt.clf()
    residuals = y_test - y_pred
    plt.figure(figsize = [10,3])
    plt.title("Residual Plot")
    sns.regplot(x = y_pred, y = residuals, lowess=True, line_kws=dict(color="r"))
    plt.ylabel("Residuals")
    plt.xlabel("Predicted Values")
    plt.savefig(f'{output_path}/residual_plot.png', dpi = 200)