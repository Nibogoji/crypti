import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier
import re
from sklearn.metrics import f1_score, mean_absolute_error


def custom_pca(data, variance_explained = False, n_pcs = False):

    X = data

    covM = np.cov(X, rowvar= False)

    eigen_val, eigen_vec = np.linalg.eigh(covM)

    sort_idx = np.argsort(eigen_val)[::-1]
    sorted_vals = eigen_val[sort_idx]
    sorted_vectors = eigen_vec[:, sort_idx]
    var_vals = np.cumsum(sorted_vals / sum(sorted_vals))

    if n_pcs:

        pcs = sorted_vectors[:,:n_pcs]

    if variance_explained:

        explained_var_idx = len(var_vals)-1
        for i,v in enumerate(var_vals):
            if v >= variance_explained:
                explained_var_idx = i
                break
        pcs = sorted_vectors[:, :explained_var_idx]

    projection = np.dot(pcs.transpose(), X.transpose())
    X_reduced = projection.transpose()

    pc_receipt = np.dot(X_reduced.transpose()-X_reduced.mean(), X - X.mean())
    pc_receipt = pc_receipt.transpose()

    pc_receipt = abs(pc_receipt)/(abs(pc_receipt).sum(axis=0))
    importance = sorted_vals/sum(sorted_vals)
    importance = importance[:n_pcs]
    if n_pcs >= 3:

        residuals = abs(np.sum((X**2).transpose(), axis= 0 )-np.sum(projection**2, axis=0))

    else:

        pc_receipt_df = pd.DataFrame(pc_receipt, columns=['Pc {}'.format(i) for i in range(n_pcs)])
        most_important_indexes = []
        feats_importance = []
        pcs_space = pcs.transpose()
        for _ in range(int(12 / n_pcs)):
            imp = [i for i in pc_receipt_df.max(axis=0)]
            feats_importance += list(imp * importance)
            most_important_indexes += [i for i in pc_receipt_df.idxmax(axis=0)]
            pc_receipt_df = pc_receipt_df.drop(most_important_indexes[-pcs_space.shape[0]:], axis=0)

        residuals = abs(
            np.sum((X[:, most_important_indexes] ** 2).transpose(), axis=0) - np.sum(projection ** 2, axis=0))

    return X_reduced, var_vals, pc_receipt, pcs.transpose(), residuals, importance


def run_xgboost_test(dataset, val_start, type = 'classification'):

    x_train, x_test = dataset[:, :-1], dataset[val_start:, :-1]
    y_train, y_test = dataset[:, -1], dataset[val_start:, -1]

    print('y train ', y_train)

    if type == 'classification':

        xgb = XGBClassifier(max_depth=12, learning_rate=0.01, n_estimators=100,
                            objective='binary:logistic', booster='gbtree',eval_metric = 'error', use_label_encoder= True)
        xgb.fit(x_train, y_train)
        prediction = xgb.predict(x_test)
        kpi = f1_score(y_test,prediction, average='weighted')

    if type == 'regression':

        xgb = XGBRegressor()
        xgb.fit(x_train, y_train)
        prediction = xgb.predict(x_test)
        kpi = mean_absolute_error(y_test,prediction)



    return kpi, xgb



def run_mlp_test(dataset, val_start, rs=1):


    x_train, x_test = dataset[:, :-1], dataset[val_start:, :-1]
    y_train, y_test = dataset[:, -1], dataset[val_start:, -1]

    model = MLPClassifier(hidden_layer_sizes=(86, 86), random_state=rs, solver='adam', shuffle=False, max_iter=50000)
    # Train the model on the whole data set
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    f1score = sklearn.metrics.f1_score(y_test, predictions, average='weighted')

    return model, f1score


def plot_classifier(X, Y, val_start, test_start, model, target_size, title ='', validation = False):


    # Calculate
    h = np.max(X[:,0])/500
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot diagram
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, cmap='ocean', alpha=0.25)
    ax.contour(xx, yy, Z, colors='w', linewidths=0.4)

    if validation:

        scatter = ax.scatter(X[: val_start, 0], X[: val_start, 1], c=Y[: val_start],
                             s=target_size[:, :val_start], cmap='Spectral', label='Training')

        ax.scatter(X[val_start: test_start, 0], X[val_start: test_start, 1], c=Y[val_start: test_start],
                   s=target_size[:, val_start: test_start], cmap='Spectral', marker='v', label='Validation')

        ax.scatter(X[test_start:, 0], X[test_start:, 1], c=Y[test_start:], s=target_size[:, test_start:],
                   cmap='Spectral', marker='+', label='Test')

    else:

        scatter = ax.scatter(X[: test_start, 0], X[: test_start, 1], c=Y[: test_start], s=target_size[:, :test_start], cmap='Spectral', label ='Training')
        ax.scatter(X[test_start: val_start, 0], X[test_start: val_start, 1], c=Y[test_start: val_start], s=target_size[:, test_start: val_start], cmap='Spectral', marker ='v', label ='Insample test')
        ax.scatter(X[val_start:, 0], X[val_start:, 1], c=Y[val_start:], s=target_size[:, val_start:], cmap='Spectral', marker='+', label ='Outsample Test')


    leg1 = ax.legend(loc='lower center')
    ax.add_artist(leg1)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

    b = []
    for j in labels:
        b.append([float(s) for s in re.findall(r'-?\d+\.?\d*', j)])

    b = np.asarray(b)
    b = [i[0] / 100 for i in b]

    c = ['$\\mathdefault{' + str(i) + '}$ %' for i in b]
    leg2 = ax.legend(handles, c, loc="upper right", title="Range")
    ax.add_artist(leg2)

    leg3 = ax.legend(*scatter.legend_elements(prop='colors'),
                        loc="lower right", title="0 : Negative R \n"
                                                 "1 : Positive R")
    ax.add_artist(leg3)
    plt.title(title)

    return fig

def make_inertial_target(df):
    position = 'Short'
    df5 = df[(df.HOLDING_MARKET_DAYS == 5) & (df.RANKED_PRODUCT == 'Italy Y1') & (
                df.TRADING_PERIOD_OPEN_POS == '1_Morning') &
           (df.TRADING_PERIOD_CLOSE_POS == '1_Morning') & (df.POS_DIRECTION == position) & (df.IS_HELD == 'NO')]
    df10 = df[(df.HOLDING_MARKET_DAYS == 10) & (df.RANKED_PRODUCT == 'Italy Y1') & (
                df.TRADING_PERIOD_OPEN_POS == '1_Morning') &
           (df.TRADING_PERIOD_CLOSE_POS == '1_Morning') & (df.POS_DIRECTION == position) & (df.IS_HELD == 'NO')]
    df15 = df[(df.HOLDING_MARKET_DAYS == 15) & (df.RANKED_PRODUCT == 'Italy Y1') & (
                df.TRADING_PERIOD_OPEN_POS == '1_Morning') &
           (df.TRADING_PERIOD_CLOSE_POS == '1_Morning') & (df.POS_DIRECTION == position) & (df.IS_HELD == 'NO')]

    df5 = df5.loc[:'2021-04-08']
    df5 = df5.iloc[:,46:]
    df5 = df5.select_dtypes(include=['float64'])

    returns_columns = [i for i in df5.columns if 'return' in i]
    returns_columns.append('IS_LIQUID_PRODUCT')

    df5 = df5.drop(returns_columns,axis =1)
    df5 = df5.dropna()

    features5 = df5.iloc[:, :-1]
    target5 = df5.iloc[:, -1].to_numpy()

    df10 = df10.loc[:'2021-04-08']
    df10 = df10.iloc[:,46:]
    df10 = df10.select_dtypes(include=['float64'])

    returns_columns = [i for i in df10.columns if 'return' in i]
    returns_columns.append('IS_LIQUID_PRODUCT')

    df10 = df10.drop(returns_columns,axis =1)
    df10 = df10.dropna()

    features10 = df10.iloc[:, :-1]
    target10 = df10.iloc[:, -1].to_numpy()

    df15 = df15.loc[:'2021-04-08']
    df15 = df15.iloc[:,46:]
    df15 = df15.select_dtypes(include=['float64'])

    returns_columns = [i for i in df15.columns if 'return' in i]
    returns_columns.append('IS_LIQUID_PRODUCT')

    df15 = df15.drop(returns_columns,axis =1)
    df15 = df15.dropna()

    features15 = df15.iloc[:, :-1]
    target15 = df15.iloc[:, -1].to_numpy()

    Z = np.zeros(features15.shape[0])
    avg_returns = np.zeros(features15.shape[0])
    # if features5.shape[0] == features10.shape[0] == features15.shape[0]:
    for i in range(features15.shape[0]):
        if features5.iloc[i,:].all() == features10.iloc[i,:].all() == features15.iloc[i,:].all():
            avg_returns[i] = (target5[i]+target10[i]+target15[i])/3
            if target5[i]+target10[i]+target15[i]>0:
                Z[i] = 1
            else:
                Z[i] = 0
    Z = np.array(Z)
    avg_returns = np.array(avg_returns)

    return features15,Z, avg_returns

def make_categorical_target(data, plot= True):

    features = data.iloc[:, :-1]
    target = data.iloc[:, -1].to_numpy()
    labels = ['negative r', 'positive r']

    Z = [0 if i < 0 else 1 for i in target]
    Z = np.array(Z)

    state_values = np.empty((len(Z), 2))

    if plot:

        fig, ax = plt.subplots(figsize=(10, 5))

        for i in range(2):
            state_values[:, i] = np.nan
            state_values[Z == i, i] = target[Z == i]
            plt.plot( state_values[:, i], label=labels[i])

        plt.legend()
        plt.title("Binary Classes")
        plt.show()

    return Z, features, target

