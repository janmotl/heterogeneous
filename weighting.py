import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler


# Initialization
skf = StratifiedKFold(n_splits=10)
norm = StandardScaler()
pt = PowerTransformer()
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
neigh = KNeighborsClassifier(n_neighbors=3, metric='precomputed', n_jobs=4)
cat_weights = np.arange(0.0, 10.0, 0.1)
logger = []
np.seterr(all='ignore')
pd.set_option('display.width', 1600)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def cat_dist(X, Y, ohe=False, zscore_normalized=False, cardinality_normalized=False):
    weight = None

    if ohe:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe.fit(X)
        X = ohe.transform(X)
        Y = ohe.transform(Y)

    if zscore_normalized:
        norm = StandardScaler()
        norm.fit(X)
        X = norm.transform(X)
        Y = norm.transform(Y)

    if cardinality_normalized:
        cardinality = np.zeros(np.size(Y, 1))
        for col in range(np.size(Y, 1)):
            cardinality[col] = len(np.unique(Y[:, col]))
        weight = 2.0 / (1.0-1.0/cardinality)

    nrow_x = np.size(X,0)
    nrow_y = np.size(Y, 0)
    dist = np.zeros((nrow_x, nrow_y), float)
    for row_x in range(nrow_x):
        for row_y in range(nrow_y):
            if cardinality_normalized:
                dist[row_x, row_y] = np.sum(weight * (X[row_x,:] != Y[row_y,:]))
            else:
                dist[row_x, row_y] = np.sum((X[row_x, :] != Y[row_y, :]))
    return dist

# Dataset list
openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
filtered = datalist.query('NumberOfClasses == 2')
filtered = filtered.query('NumberOfInstances < 5000')
filtered = filtered.query('NumberOfInstances > 30')
filtered = filtered.query('NumberOfFeatures < 120')
filtered = filtered.query('NumberOfNumericFeatures > 1')
filtered = filtered.query('NumberOfSymbolicFeatures > 2')
filtered = filtered.query('NumberOfMissingValues == 0')
filtered = filtered.query('did <= 41521') # close duplicates follow
filtered = filtered.query('did not in [4329, 902, 891, 862, 771, 479, 465]') # close duplicates
filtered = filtered.query('did <= 40705') # actually only numeric

for did in filtered.did:
    try:
        # Download dataset
        dataset = openml.datasets.get_dataset(did)
        X, y, categorical_indicator, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='array')
        is_categorical = categorical_indicator
        is_continuous = np.bitwise_not(categorical_indicator)
        print('Dataset', dataset.name, did, flush=True)  # For progress indication

        # Split
        fold = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold += 1

            # Normalize numerical features. Since power transform can easily overflow, we first z-score normalize the data. It still overflows when the optimizer guesses ridiculous lambdas, but the optimization process correctly steers away from these values
            norm.fit(X_train[:, is_continuous])
            X_train[:, is_continuous] = norm.transform(X_train[:, is_continuous])
            X_test[:, is_continuous] = norm.transform(X_test[:, is_continuous])

            pt.fit(X_train[:, is_continuous])
            con_train = pt.transform(X_train[:, is_continuous])
            con_test = pt.transform(X_test[:, is_continuous])

            num_dist_train = cdist(con_train, con_train, metric='cityblock')    # sqeuclidean or cityblock
            num_dist_test = cdist(con_test, con_train, metric='cityblock')

            # Categorical features
            cat_dist_train = cat_dist(X_train[:, is_categorical], X_train[:, is_categorical], cardinality_normalized=False)
            cat_dist_test = cat_dist(X_test[:, is_categorical], X_train[:, is_categorical], cardinality_normalized=False)

            # Test different rescaling of categorical features
            for cat_weight in cat_weights:
                X_train = num_dist_train + cat_weight * cat_dist_train  # no need to take root-square as root-square is a monotonous function on non-negative numbers
                X_test = num_dist_test + cat_weight * cat_dist_test

                # Classify
                neigh.fit(X_train, y_train)
                prediction = neigh.predict(X_test)
                probs = neigh.predict_proba(X_test)

                kappa = metrics.cohen_kappa_score(y_test, prediction)
                auc = metrics.roc_auc_score(y_test, probs[:,1])
                brier = metrics.brier_score_loss(y_test, probs[:,1])

                logger.append([dataset.name, did, fold, cat_weight, kappa, auc, brier])
    except RuntimeWarning and UserWarning and NotImplementedError:
        continue

result = pd.DataFrame(logger, columns=['dataset', 'did', 'fold', 'cat_weight', 'kappa', 'auc', 'brier'])
result.to_csv('~/Downloads/results.csv')

# Analysis - ranking
agg = result.groupby(by=['did', 'cat_weight']).mean().groupby(by='did').rank(method='average').groupby(by='cat_weight').mean()
agg.index = cat_weights
agg = agg.drop('fold',axis=1)

plt.style.use('ggplot')
plt.clf()
plt.title('Manhattan distance + cat_weight * Hamming distance')
plt.plot(agg.index, agg, marker='o')
plt.legend(['Kappa', 'AUC', 'Brier'])
plt.xlabel('cat_weight')
plt.ylabel('Rank')
plt.savefig('manhattan.png', bbox_inches='tight')

# Analysis - avg
agg2 = result.groupby(by=['cat_weight']).mean()
agg2 = agg2['auc'] + agg2['kappa'] - agg2['brier']

plt.figure()
plt.title('Manhattan distance + cat_weight * Hamming distance')
plt.plot(cat_weights, agg2, marker='o')
plt.xlabel('cat_weight')
plt.ylabel('Average accuracy')
plt.savefig('manhattan_avg.png', bbox_inches='tight')

# Analysis - bootstrap
nrow = len(np.unique(result['did']))
acc = np.zeros((len(cat_weights), 1))
plt.figure()
for repeat in range(100):
    selected = np.random.choice(nrow, size=nrow, replace=True)
    agg = result.loc[result['did'].isin(filtered.iloc[selected].index), ['auc','did', 'cat_weight']].groupby(by=['did', 'cat_weight']).mean().groupby(by='did').rank(method='average').groupby(by='cat_weight').mean()
    plt.plot(cat_weights, agg, color='gray')
    acc += agg.values
plt.plot(cat_weights, acc/100, color='black')
