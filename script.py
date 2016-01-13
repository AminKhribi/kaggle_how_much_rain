import random
import functools
import itertools
import copy
import multiprocessing
import concurrent.futures

from scipy import stats

# from sknn import mlp

from sklearn import svm, cross_validation, metrics, kernel_approximation, linear_model, preprocessing, decomposition, feature_selection, neighbors

import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt

# from adk_utils import monitored_mp
import amin_mp as monitored_mp

from sklearn import cross_validation, metrics, kernel_approximation, linear_model, preprocessing, multiclass
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from scipy.stats import randint as sp_randint


columns = ['Id', 'minutes_past', 'radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th', 'Ref_5x5_90th',
           'RefComposite', 'RefComposite_5x5_10th', 'RefComposite_5x5_50th', 'RefComposite_5x5_90th',
           'RhoHV', 'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th',
           'Zdr','Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th',
           'Kdp', 'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th',
           'Expected']


var_columns = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th', 'Ref_5x5_90th',
               'RefComposite', 'RefComposite_5x5_10th', 'RefComposite_5x5_50th', 'RefComposite_5x5_90th',
               'RhoHV', 'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th',
               'Zdr','Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th',
               'Kdp', 'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th']

diff_columns = ['Ref',
               'RefComposite',
               'RhoHV',
               'Zdr',
               'Kdp']



def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum

def transform_features(x):
    """
    Transform features grouped by Id
    """

    # len_1 = len(x)
    # len_2 = sum(~x['Ref'].isnull())

    # x = x[~x['Ref'].isnull()]

    # if len_2:

    x = x.sort('minutes_past', ascending=True)
    est = marshall_palmer(x['Ref'], x['minutes_past'])

    # x['na_counts'] = (len_1 - len_2) / len_1
    x['len'] = len(x)

    x['est'] = est

    minutes = list(x['minutes_past'].values)

    x['min_q_25'] = minutes[int(len(minutes) / 4)]
    x['min_q_50'] = minutes[int(len(minutes) / 2)]
    x['min_q_75'] = minutes[int(3 * len(minutes) / 4)]
    x['min_q_100'] = minutes[-1]

    for feature in var_columns:
        # x[feature] = x[feature].replace('NaN', x[feature].mean())
        # x[feature] = x[feature].replace('NaN', x[feature].median())

        values = np.array(x[feature]) - np.mean(x[feature])

        x[feature + '_mean'] = x[feature].mean()
        x[feature + '_std'] = x[feature].std() if len(x) > 1 else 0

        if feature in diff_columns:
            x[feature + '_t0'] = values[0]
            x[feature + '_t25'] = values[int(len(minutes) / 4)]
            x[feature + '_t50'] = values[int(len(minutes) / 2)]
            x[feature + '_t75'] = values[int(3 * len(minutes) / 4)]
            x[feature + '_t100'] = values[-1]

            x[feature + '_diff_25'] = sum(np.diff(values[:int(len(minutes) / 4)]))
            x[feature + '_diff_50'] = sum(np.diff(values[int(len(minutes) / 4):int(len(minutes) / 2)]))
            x[feature + '_diff_75'] = sum(np.diff(values[int(len(minutes) / 2):int(3 * len(minutes) / 4)]))
            x[feature + '_diff_100'] = sum(np.diff(values[int(3 * len(minutes) / 4):]))

            x[feature + '_ratio_25'] = values[int(len(minutes) / 4)] - values[0]
            x[feature + '_ratio_50'] = values[int(len(minutes) / 2)] - values[int(len(minutes) / 4)]
            x[feature + '_ratio_75'] = values[int(3 * len(minutes) / 4)] - values[int(len(minutes) / 2)]
            x[feature + '_ratio_100'] = values[-1] - values[int(3 * len(minutes) / 4)]

            x[feature + '_max_25'] = max(abs(np.diff(values[:int(len(minutes) / 4)]))) if len(np.diff(values[:int(len(minutes) / 4)])) > 1 else 0
            x[feature + '_max_50'] = max(abs(np.diff(values[int(len(minutes) / 4):int(len(minutes) / 2)]))) if len(np.diff(values[int(len(minutes) / 4):int(len(minutes) / 2)])) > 1 else 0
            x[feature + '_max_75'] = max(abs(np.diff(values[int(len(minutes) / 2):int(3 * len(minutes) / 4)]))) if len(np.diff(values[int(len(minutes) / 2):int(3 * len(minutes) / 4)])) > 1 else 0
            x[feature + '_max_100'] = max(abs(np.diff(values[int(3 * len(minutes) / 4):]))) if len(np.diff(values[int(3 * len(minutes) / 4):])) > 1 else 0

    x.drop_duplicates(subset='Id', inplace=True)

    return x

def group_by(x=None, feature=None, func=None):
    return x.groupby(feature, group_keys=False).apply(func)

"""
Different set of features to learn with
"""

cols_base = ['radardist_km', 'est', 'len',
              'min_q_25', 'min_q_50', 'min_q_75', 'min_q_100',
              'Ref_mean', 'Ref_std',
              'Ref_t0', 'Ref_t25', 'Ref_t50', 'Ref_t75', 'Ref_t100',
              'RefComposite_mean', 'RefComposite_std',
              'RhoHV_mean', 'RhoHV_std',
              'Zdr_mean', 'Zdr_std',
              'Kdp_mean', 'Kdp_std']

cols_org = ['radardist_km', 'est', 'len',
              'min_q_25', 'min_q_50', 'min_q_75', 'min_q_100',
              'Ref_mean', 'Ref_std',
              'Ref_t0', 'Ref_t25', 'Ref_t50', 'Ref_t75', 'Ref_t100',
              'RefComposite_mean', 'RefComposite_std',
              'RefComposite_t0', 'RefComposite_t25', 'RefComposite_t50', 'RefComposite_t75', 'RefComposite_t100',
              'RhoHV_mean', 'RhoHV_std',
              'RhoHV_t0','RhoHV_t25', 'RhoHV_t50', 'RhoHV_t75', 'RhoHV_t100',
              'Zdr_mean', 'Zdr_std',
              'Zdr_t0', 'Zdr_t25', 'Zdr_t50','Zdr_t75', 'Zdr_t100',
              'Kdp_mean', 'Kdp_std',
              'Kdp_t0', 'Kdp_t25', 'Kdp_t50', 'Kdp_t75', 'Kdp_t100',
              'RefComposite_5x5_10th_mean', 'RefComposite_5x5_50th_mean', 'RefComposite_5x5_90th_mean',
              'RefComposite_5x5_10th_std', 'RefComposite_5x5_50th_std', 'RefComposite_5x5_90th_std',
              'Ref_5x5_10th_std', 'Ref_5x5_50th_std', 'Ref_5x5_90th_std',
              'Ref_5x5_10th_mean', 'Ref_5x5_50th_mean', 'Ref_5x5_90th_mean',
              'RhoHV_5x5_10th_mean', 'RhoHV_5x5_50th_mean', 'RhoHV_5x5_90th_mean',
              'RhoHV_5x5_10th_std', 'RhoHV_5x5_50th_std', 'RhoHV_5x5_90th_std',
              'Zdr_5x5_10th_mean', 'Zdr_5x5_50th_mean', 'Zdr_5x5_90th_mean',
              'Zdr_5x5_10th_std', 'Zdr_5x5_50th_std', 'Zdr_5x5_90th_std',
              'Kdp_5x5_10th_mean', 'Kdp_5x5_50th_mean', 'Kdp_5x5_90th_mean',
              'Kdp_5x5_10th_std', 'Kdp_5x5_50th_std', 'Kdp_5x5_90th_std']

diff_cols = ['Ref_diff_25', 'Ref_diff_50', 'Ref_diff_75', 'Ref_diff_100',
              'RefComposite_diff_25', 'RefComposite_diff_50', 'RefComposite_diff_75', 'RefComposite_diff_100',
              'RhoHV_diff_25','RhoHV_diff_50', 'RhoHV_diff_75', 'RhoHV_diff_100',
              'Zdr_diff_25', 'Zdr_diff_50', 'Zdr_diff_75','Zdr_diff_100',
              'Kdp_diff_25', 'Kdp_diff_50', 'Kdp_diff_75', 'Kdp_diff_100']

ratio_cols = ['Ref_ratio_25', 'Ref_ratio_50', 'Ref_ratio_75', 'Ref_ratio_100',
              'RefComposite_ratio_25', 'RefComposite_ratio_50', 'RefComposite_ratio_75', 'RefComposite_ratio_100',
              'RhoHV_ratio_25','RhoHV_ratio_50', 'RhoHV_ratio_75', 'RhoHV_ratio_100',
              'Zdr_ratio_25', 'Zdr_ratio_50', 'Zdr_ratio_75','Zdr_ratio_100',
              'Kdp_ratio_25', 'Kdp_ratio_50', 'Kdp_ratio_75', 'Kdp_ratio_100']

max_cols = ['Ref_max_25', 'Ref_max_50', 'Ref_max_75', 'Ref_max_100',
              'RefComposite_max_25', 'RefComposite_max_50', 'RefComposite_max_75', 'RefComposite_max_100',
              'RhoHV_max_25','RhoHV_max_50', 'RhoHV_max_75', 'RhoHV_max_100',
              'Zdr_max_25', 'Zdr_max_50', 'Zdr_max_75','Zdr_max_100',
              'Kdp_max_25', 'Kdp_max_50', 'Kdp_max_75', 'Kdp_max_100']

learn_cols_knn = ['radardist_km', 'len',
                  'min_q_25', 'min_q_50', 'min_q_75', 'min_q_100',
                  'Ref_mean', 'Ref_std',
                  'Ref_t0', 'Ref_t25', 'Ref_t50', 'Ref_t75', 'Ref_t100']


df = pd.read_csv('train.csv', usecols=columns)

df_learn = df

df_learn = df_learn.reset_index()

df_learn = df_learn[~df_learn['Ref'].isnull()]

########################
print('apply functions...')
# n_workers = multiprocessing.cpu_count()
n_workers = 200

learn_ids = list(df_learn['Id'].unique())
part = [[int(k * (len(learn_ids) / n_workers)), int((k + 1) * (len(learn_ids) / n_workers))] for k in range(n_workers)]

max_idx = 40
security_mem = 3e9
sleep_time = 3

my_args = {'feature': 'Id', 'func': transform_features}
range_args = {'x': [df_learn[df_learn['Id'].isin(learn_ids[r[0]:r[1]])] for r in part]}

dfs = monitored_mp.bound_mp(my_func=group_by, my_args=my_args, range_args=range_args, max_idx=max_idx,
                            sleep_time=sleep_time, security_mem=security_mem, name='', daemon=False, verbose=True)

df_learn = pd.concat(dfs, ignore_index=True)

df_learn = df_learn.replace('inf', 1e5)
df_learn = df_learn.replace('-inf', -1e5)

df_learn.drop_duplicates(subset='Id', inplace=True)

#####################

all_learn_ids = list(df_learn['Id'].unique())

learn_cols_diff = copy.copy(cols_org)
learn_cols_diff.extend(diff_cols)

learn_cols_full = copy.copy(cols_org)
learn_cols_full.extend(diff_cols)
learn_cols_full.extend(max_cols)

learn_cols = cols_org

#####################
## PARAMETERS #######
#####################

ada=False
drop_outliers=False
with_test_neighbors=False
norm_x=True, norm_y=True

"""
seperate learn and test ids
"""

random.shuffle(all_learn_ids)

learn_ids = all_learn_ids[:int(0.8 * len(all_learn_ids))]
test_ids = all_learn_ids[int(0.8 * len(all_learn_ids)):]

df_learn_local = df_learn[df_learn['Id'].isin(learn_ids)]
df_test_local = df_learn[df_learn['Id'].isin(test_ids)]

# df_learn_local = df_learn_local[df_learn_local['Expected'] < 70]
# df_test_local = df_test_local[df_test_local['Expected'] < 70]

df_learn_local = df_learn_local[list(set(learn_cols).union(['Expected', 'Id']))].dropna()
# print("dropped {}% of learn data".format( 100 * (1 - (len(df_learn_local) / 200000)) ))

# df_learn_local = df_learn_local[:130000]

# df_test_local = df_test_local[list(set(learn_cols).union(['Id', 'Expected']))].replace('NaN', 0)
# df_test_local = df_test_local[list(set(learn_cols).union(['Expected', 'Id']))].dropna()

"""
try filling missing values by nearest-neighbors
"""

print('filling test set..')

nan_indexes = []
df_test_local[learn_cols].isnull().apply(lambda x: nan_indexes.append(x.name) if True in x.values else None, axis=1)

df_test_nan = df_test_local.ix[nan_indexes]
df_test_not_nan = df_test_local.ix[list(set(df_test_local.index).difference(nan_indexes))]

nan_indexes = []
df_learn_local[learn_cols].isnull().apply(lambda x: nan_indexes.append(x.name) if True in x.values else None, axis=1)

df_learn_nan = df_learn_local.ix[nan_indexes]
df_learn_not_nan = df_learn_local.ix[list(set(df_learn_local.index).difference(nan_indexes))]

knn = neighbors.NearestNeighbors(n_neighbors=1)

knn.fit(df_learn_not_nan[learn_cols_knn])

df_test_nan['knn'] =  knn.kneighbors(df_test_nan[learn_cols_knn], return_distance=False)
df_test_nan = df_test_nan.reset_index(drop=True)

df_neighbors = df_learn_not_nan.iloc[df_test_nan['knn'].values]
df_neighbors = df_neighbors.reset_index(drop=True)

df_test_nan.update(df_neighbors, overwrite=False)

df_test_local = pd.concat([df_test_nan, df_test_not_nan])


############################
######### OUTLIER CLF ######
############################
"""
Run a mann whitney test to find which feature has different distribution from the target
"""

df_learn_local['outlier'] = 0
df_learn_local.loc[df_learn_local[df_learn_local['Expected'] > 70].index, 'outlier'] = 1

df_test_local['outlier'] = 0
df_test_local.loc[df_test_local[df_test_local['Expected'] > 70].index, 'outlier'] = 1

# feature analysis
##################

print("feature -- median -- pval -- reject")
vars_table = []
for feature in learn_cols:
    feature_out = df_learn_local[df_learn_local['outlier'] == 1][feature].values
    feature_in = df_learn_local[df_learn_local['outlier'] == 0][feature].values

    random.shuffle(feature_in)
    feature_in = feature_in[:len(feature_out)]

    mw_stat, p_val = stats.mannwhitneyu(feature_out, feature_in, use_continuity=False)
    
    feature_dict = {'feature': feature,
                    'p_value': p_val,
                    'reject': True if p_val < 0.05 else False,
                    'median_diff': abs(np.median(feature_in) - np.median(feature_out)) }

    vars_table.append(feature_dict)
    print("{} -- {} -- {} "
          .format(feature,
                  feature_dict['median_diff'],
                  feature_dict['reject']))


sorted_table = sorted([i for i in vars_table if i['reject']], key=operator.itemgetter('median_diff'))

learn_cols_outliers = [i['feature'] for i in sorted_table if i['median_diff'] > 0.1]

###################
"""
regressor on outliers
"""

X_learn_clf = df_learn_local[learn_cols]
Y_learn_clf = df_learn_local['outlier']

X_test_clf = df_test_local[learn_cols]
Y_test_clf = df_test_local['outlier']

scaler_clf = preprocessing.StandardScaler()
X_learn_clf = scaler_clf.fit_transform(X_learn_clf)
X_test_clf = scaler_clf.transform(X_test_clf)

# Feature selection
ref = feature_selection.RFECV(DecisionTreeClassifier(), step=1, cv=cross_validation.StratifiedKFold(Y_learn_clf),
                                            scoring='roc_auc',
                                            estimator_params=None, verbose=5)

ref.fit(X_learn_clf, Y_learn_clf)
important_features = [i for j, i in enumerate(learn_cols) if ref.support_[j] ]

X_learn_clf = X_learn_clf[:, ref.support_]
X_test_clf = X_test_clf[:, ref.support_]

####################

rs = RandomizedSearchCV(DecisionTreeClassifier(),
                     param_distributions={'max_depth': stats.randint(10, 200),
                                         'max_features': ['log2', 'auto', 'sqrt'],
                                         'class_weight': [None, 'balanced'] },
                    n_iter=20,
                    n_jobs=4,
                    verbose=5,
                    scoring='roc_auc',
                    iid=False,
                    cv=cross_validation.StratifiedKFold(Y_learn_clf),
                    pre_dispatch=2,
                    error_score=1) 

# rs.fit(X_learn_clf, Y_learn_clf)
# classifier = rs.best_estimator_


classifier = DecisionTreeClassifier()
classifier.fit(X_learn_clf, Y_learn_clf)

Y_clf_hat = classifier.predict(X_test_clf)
Y_clf_proba = classifier.predict_proba(X_test_clf)

df_test_local['outlier_hat'] = Y_clf_hat

score_clf = metrics.roc_auc_score(Y_test_clf, Y_clf_hat, average='weighted')
print("clf roc score {}".format(score_clf))

cf_clf = metrics.confusion_matrix(Y_test_clf, Y_clf_hat)
print(cf_clf)


#############################
##### outlier regressor #####
#############################
"""
learn a regressor only on outliers
"""

df_learn_outlier = df_learn_local[df_learn_local['outlier'] == 1]
df_test_outlier = df_test_local[df_test_local['outlier_hat'] == 1]

X_learn_out = df_learn_outlier[learn_cols]
Y_learn_out = np.log(df_learn_outlier['Expected'])

X_test_out = df_test_outlier[learn_cols]
Y_test_out = np.log(df_test_outlier['Expected'])

scaler_out = preprocessing.StandardScaler()
X_learn_out = scaler_out.fit_transform(X_learn_out)
X_test_out = scaler_out.transform(X_test_out)

regressor_out = ExtraTreesRegressor(n_jobs=4)
regressor_out.fit(X_learn_out, Y_learn_out)

Y_out_hat = regressor_out.predict(X_test_out)
df_test_outlier['y_hat'] = np.exp(Y_out_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_out), np.exp(Y_out_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_out), [np.exp(np.median(Y_test_out))] * len(Y_test_out))
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

#
"""
learn a regressor on non outliers
"""

df_learn_nor = df_learn_local[df_learn_local['outlier'] == 0]
df_test_nor = df_test_local[df_test_local['outlier_hat'] == 0]

X_learn_nor = df_learn_nor[learn_cols]
Y_learn_nor = np.log(df_learn_nor['Expected'])

X_test_nor = df_test_nor[learn_cols]
Y_test_nor = np.log(df_test_nor['Expected'])

scaler_out = preprocessing.StandardScaler()
X_learn_nor = scaler_out.fit_transform(X_learn_nor)
X_test_nor = scaler_out.transform(X_test_nor)

regressor_nor = ExtraTreesRegressor(n_jobs=4)
regressor_nor.fit(X_learn_nor, Y_learn_nor)

Y_nor_hat = regressor_nor.predict(X_test_nor)
df_test_nor['y_hat'] = np.exp(Y_nor_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_nor), np.exp(Y_nor_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_nor), [np.exp(np.median(Y_test_nor))] * len(Y_test_nor))
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

df_test_local = pd.concat([df_test_outlier, df_test_nor], axis=0)

"""
score of combined regressors and outlier classifier
"""

print(' -- ALL -- ')

score = metrics.mean_absolute_error(df_test_local['Expected'], df_test_local['y_hat'])
score_median = metrics.mean_absolute_error(df_test_local['Expected'], [df_learn_local['Expected'].median()] * df_test_local['Id'].nunique())
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

print(' -- OUTLIERS -- ')
score_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > 70]['Expected'], df_test_local[df_test_local['Expected'] > 70]['y_hat'])
score_median_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > 70]['Expected'], [df_test_local[df_test_local['Expected'] > 70]['Expected'].median()] * df_test_local[df_test_local['Expected'] > 70]['Id'].nunique())
print("score: ", score_high, " -- score median", score_median_high, " -- ratio ", (score_median_high - score_high) / score_median_high, ' -- perc', round(100 * sum(df_test_local['Expected'] > 70) / len(df_test_local), 2)  )

print(' -- NORMAL -- ')
score_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < 70]['Expected'], df_test_local[df_test_local['Expected'] < 70]['y_hat'])
score_median_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < 70]['Expected'], [df_test_local[df_test_local['Expected'] < 70]['Expected'].median()] * df_test_local[df_test_local['Expected'] < 70]['Id'].nunique())
print("score: ", score_normal, " -- score median", score_median_normal, " -- ratio ", (score_median_normal - score_normal) / score_median_normal, ' -- perc', round(100 * sum(df_test_local['Expected'] < 70) / len(df_test_local), 2))


############################
####### CLUSTERING #########
############################

"""
cluster by values of marshall_palmer estimation
"""

def apply_func(sub_df=None, func=None):
    sub_df = sub_df.apply(func, axis=1)
    return sub_df


def apply_class(feature=None, list_class=None, x=None):

    val = x[feature]

    idx = 0
    while val > list_class[idx]:
        idx += 1
        if idx >= len(list_class):
            break

    x[feature + '_tr'] = idx

    return x

values = df_learn_local['est']

list_class = [values.quantile(i) for i in [0.25, 0.5, 0.75]]

# df_learn_local = df_learn_local.apply(functools.partial(support_scopes.apply_class, 'est', list_class), axis=1)

max_idx = 5
sleep_time = 0
security_mem = 1e9

n_workers = multiprocessing.cpu_count()
part = [[int(k * (len(df_learn_local) / n_workers)), int((k + 1) * (len(df_learn_local) / n_workers))] for k in range(n_workers)]

my_args = {'func': functools.partial(apply_class, 'est', list_class)}
range_args = {'sub_df': [df_learn_local[r[0]:r[1]] for r in part]}

# my_args = {'func': fixed_flag_mod}
# range_args = {'sub_df': [df_to_scope[r[0]:r[1]] for r in part]}

dfs = monitored_mp.bound_mp(my_func=apply_func, my_args=my_args, range_args=range_args,
                            max_idx=max_idx, sleep_time=sleep_time, security_mem=security_mem, name='',
                            daemon=False, verbose=False)

df_learn_local = pd.concat(dfs, ignore_index=False)
df_learn_local = df_learn_local.reset_index()

"""
learn a classifier on precedant clusters (useless since clustering was just a deterministic transformation..)
"""

X_learn_clf = df_learn_local[:int(0.8 * len(df_learn_local))][learn_cols]
Y_learn_clf = df_learn_local[:int(0.8 * len(df_learn_local))]['est_tr']

X_test_clf = df_learn_local[int(0.8 * len(df_learn_local)):][learn_cols]
Y_test_clf = df_learn_local[int(0.8 * len(df_learn_local)):]['est_tr']

scaler_clf = preprocessing.StandardScaler()
X_learn_clf = scaler_clf.fit_transform(X_learn_clf)
X_test_clf = scaler_clf.transform(X_test_clf)

def func_score(est, X,y):
    y_hat = est.predict(X)
    cf_clf = metrics.confusion_matrix(y, y_hat)
    cf_clf = cf_clf / np.sum(cf_clf, 0)
    score_clf = np.sum(cf_clf) - np.sum(np.diag(cf_clf.diagonal()))
    return score_clf

# rs = RandomizedSearchCV(DecisionTreeClassifier(),
#                      param_distributions={'max_depth': stats.randint(10, 200),
#                                          'max_features': ['log2', 'auto', 'sqrt'],
#                                          'class_weight': [None, 'balanced'] },
#                     n_iter=20,
#                     n_jobs=4,
#                     verbose=5,
#                     scoring=func_score,
#                     iid=False,
#                     cv=cross_validation.StratifiedKFold(Y_learn_clf),
#                     pre_dispatch=2,
#                     error_score=1) 

# rs.fit(X_learn_clf, Y_learn_clf)
# classifier = rs.best_estimator_


# classifier = multiclass.OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=4)
classifier = DecisionTreeClassifier()
# classifier = ensemble.GradientBoostingClassifer()
classifier.fit(X_learn_clf, Y_learn_clf)

Y_clf_hat = classifier.predict(X_test_clf)
df_test_local['est_tr_hat'] = Y_clf_hat

cf_clf = metrics.confusion_matrix(Y_test_clf, Y_clf_hat)
cf_med = metrics.confusion_matrix(Y_test_clf, [random.choice([0, 1, 2]) for i in range(len(Y_test_clf))])

cf_clf = cf_clf / np.sum(cf_clf, 0)
cf_med = cf_med / np.sum(cf_med, 0)

score_clf = np.sum(cf_clf) - np.sum(np.diag(cf_clf.diagonal()))
score_med = np.sum(cf_med) - np.sum(np.diag(cf_med.diagonal()))

print("clf score {}".format(score_clf))
print("rand score {}".format(score_med))

#########################
#### LEARNING ON CL #####
#########################

# class_0
df_learn_0 = df_learn_local[df_learn_local['est_tr'] == 0]
df_test_0 = df_test_local[df_test_local['est_tr_hat'] == 0]

X_learn_out = df_learn_0[learn_cols]
Y_learn_out = np.log(df_learn_0['Expected'])

X_test_out = df_test_0[learn_cols]
Y_test_out = np.log(df_test_0['Expected'])

scaler_out = preprocessing.StandardScaler()
X_learn_out = scaler_out.fit_transform(X_learn_out)
X_test_out = scaler_out.transform(X_test_out)

regressor_out = ExtraTreesRegressor(n_jobs=4)
regressor_out.fit(X_learn_out, Y_learn_out)

Y_out_hat = regressor_out.predict(X_test_out)
df_test_0['y_hat'] = np.exp(Y_out_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_out), np.exp(Y_out_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_out), [np.exp(np.median(Y_test_out))] * len(Y_test_out))
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

# class_1
df_learn_1 = df_learn_local[df_learn_local['est_tr'] == 0]
df_test_1 = df_test_local[df_test_local['est_tr_hat'] == 0]

X_learn_out = df_learn_1[learn_cols]
Y_learn_out = np.log(df_learn_1['Expected'])

X_test_out = df_test_1[learn_cols]
Y_test_out = np.log(df_test_1['Expected'])

scaler_out = preprocessing.StandardScaler()
X_learn_out = scaler_out.fit_transform(X_learn_out)
X_test_out = scaler_out.transform(X_test_out)

regressor_out = ExtraTreesRegressor(n_jobs=4)
regressor_out.fit(X_learn_out, Y_learn_out)

Y_out_hat = regressor_out.predict(X_test_out)
df_test_1['y_hat'] = np.exp(Y_out_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_out), np.exp(Y_out_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_out), [np.exp(np.median(Y_test_out))] * len(Y_test_out))
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

# class_2
df_learn_2 = df_learn_local[df_learn_local['est_tr'] == 0]
df_test_2 = df_test_local[df_test_local['est_tr_hat'] == 0]

X_learn_out = df_learn_2[learn_cols]
Y_learn_out = np.log(df_learn_2['Expected'])

X_test_out = df_test_2[learn_cols]
Y_test_out = np.log(df_test_2['Expected'])

scaler_out = preprocessing.StandardScaler()
X_learn_out = scaler_out.fit_transform(X_learn_out)
X_test_out = scaler_out.transform(X_test_out)

regressor_out = ExtraTreesRegressor(n_jobs=4)
regressor_out.fit(X_learn_out, Y_learn_out)

Y_out_hat = regressor_out.predict(X_test_out)
df_test_2['y_hat'] = np.exp(Y_out_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_out), np.exp(Y_out_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_out), [np.exp(np.median(Y_test_out))] * len(Y_test_out))
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

# bring it all together
df_test_local = pd.concat([df_test_outlier, df_test_nor], axis=0)


###########################
### CLASSIC LEARNING ######
###########################

# model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
# model_ransac.fit(X, y)
# inlier_mask = model_ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)


X_learn = df_learn_local[learn_cols]
Y_learn = np.log(df_learn_local['Expected'])

X_test = df_test_local[learn_cols]
Y_test = np.log(df_test_local['Expected'])

print('test ids ', df_test_local['Id'].nunique())
print("learn ids: ", df_learn_local['Id'].nunique())


if norm_x:
    print('normalizing...')
    scaler_x = preprocessing.StandardScaler()
    # scaler_x = preprocessing.MinMaxScaler()
    X_learn = scaler_x.fit_transform(X_learn)
    X_test = scaler_x.transform(X_test)

if norm_y:
    scaler_y = preprocessing.StandardScaler()
    # scaler_y = preprocessing.MinMaxScaler()
    Y_learn = scaler_y.fit_transform(Y_learn)
    Y_test = scaler_y.transform(Y_test)

# Feature selection
ref = feature_selection.RFECV(ExtraTreesRegressor(n_jobs=4), step=1, cv=3, scoring='mean_absolute_error',
                                            estimator_params=None, verbose=5)

ref.fit(X_learn, Y_learn)
important_features = [i for j, i in enumerate(learn_cols) if ref.support_[j] ]

X_learn = X_learn[:, ref.support_]
X_test = X_test[:, ref.support_]

# learning curve
train_sizes, train_scores, test_scores = learning_curve.learning_curve(ExtraTreesRegressor(),
                                            X_learn, Y_learn,
                                            train_sizes=array([ 0.1, 0.33, 0.55, 0.78, 1. ]),
                                            cv=3, scoring='mean_absolute_error',
                                            exploit_incremental_learning=False, n_jobs=4,
                                            pre_dispatch='all', verbose=0)



print('learning...')
if ada:
    regressor = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
else:
    regressor = ExtraTreesRegressor(n_jobs=4)

# rs = RandomizedSearchCV(regressor, param_distributions={
#                     'n_estimators': stats.randint(10, 500),
#                     'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 'auto', 'sqrt', 'log2'],
#                     'max_depth': stats.randint(10, 100)},
#                     n_jobs=4,
#                     verbose=5,
#                     scoring=metrics.mean_absolute_error,
#                     iid=False,
#                     cv=4,
#                     pre_dispatch=4) 

regressor.fit(X_learn, Y_learn)

########################
########################

print('prediction...')
Y_hat = regressor.predict(X_test)

if norm_y:
    df_test_local['y_hat'] = np.exp(scaler_y.inverse_transform(Y_hat))
else:
    df_test_local['y_hat'] = np.exp(Y_hat)

print('compute scores...')


############################
########### SCORES #########
############################

print(' -- ALL -- ')

score = metrics.mean_absolute_error(df_test_local['Expected'], df_test_local['y_hat'])
score_median = metrics.mean_absolute_error(df_test_local['Expected'], [df_learn_local['Expected'].median()] * df_test_local['Id'].nunique())
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

print(' -- OUTLIERS -- ')
score_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > 70]['Expected'], df_test_local[df_test_local['Expected'] > 70]['y_hat'])
score_median_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > 70]['Expected'], [df_test_local[df_test_local['Expected'] > 70]['Expected'].median()] * df_test_local[df_test_local['Expected'] > 70]['Id'].nunique())
print("score: ", score_high, " -- score median", score_median_high, " -- ratio ", (score_median_high - score_high) / score_median_high, ' -- perc', round(100 * sum(df_test_local['Expected'] > 70) / len(df_test_local), 2)  )

print(' -- NORMAL -- ')
score_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < 70]['Expected'], df_test_local[df_test_local['Expected'] < 70]['y_hat'])
score_median_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < 70]['Expected'], [df_test_local[df_test_local['Expected'] < 70]['Expected'].median()] * df_test_local[df_test_local['Expected'] < 70]['Id'].nunique())
print("score: ", score_normal, " -- score median", score_median_normal, " -- ratio ", (score_median_normal - score_normal) / score_median_normal, ' -- perc', round(100 * sum(df_test_local['Expected'] < 70) / len(df_test_local), 2))



