# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, ShuffleSplit, validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
import xgboost as xgb

seed = 25000

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

# load data

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

# 只考虑开店营业的数据
trainset_openstore = train_raw.query('Open!=0')

# 得到每家店每日客户数的中间值，并将其定义为store popularity
store_popularity = pd.DataFrame(trainset_openstore.groupby(['Store'])['Customers'].median())
store_popularity.columns = ['store_popularity']

# 购买力 = median(Sales) / median(customers) per store
consumption = pd.DataFrame(trainset_openstore.groupby(['Store'])['Sales'].median() / trainset_openstore.groupby(['Store'])['Customers'].median())
consumption.columns = ['consumption_power']
# 将store popularity放入store信息内
store = pd.merge(store, store_popularity, left_on='Store', right_index=True)
store = pd.merge(store, consumption, left_on='Store', right_index=True)
# 将门店数据合并到训练集，测试集中
trainset = pd.merge(train_raw, store, on='Store')
testset = pd.merge(test_raw, store, on='Store')


# Data Cleaning

def clean_data(dataset):
    # 去除缺失值占比大的特征
    dataset = dataset.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                            'Promo2SinceWeek', 'PromoInterval', 'Promo2SinceYear'], axis=1)

    # 将Date转换为datetime对象
    dataset['Date'] = pd.to_datetime(dataset['Date'])

    # 创建'Month','Year'特征
    dataset['Month'] = dataset['Date'].map(lambda x: x.month)
    dataset['Year'] = dataset['Date'].map(lambda x: x.year)

    # StateHoliday中的typo进行修正，并将当日不是节日设置为0
    dataset.loc[dataset['StateHoliday'] == '0', 'StateHoliday'] = 0

    # 将当日是节日，则设置为1，不管是什么节日
    dataset.loc[dataset['StateHoliday'] != 0, 'StateHoliday'] = 1
    dataset['StateHoliday'] = dataset['StateHoliday'].astype('uint8')

    # 进行独热编码
    for x in ['StoreType', 'Assortment', 'DayOfWeek', 'Month']:
        dataset = dataset.join(pd.get_dummies(dataset[x], prefix=x))

    # 对testset中缺失的月份用0填充
    for month in range(1, 13):
        head = 'Month_' + str(month)
        if head not in dataset.columns:
            dataset[head] = 0

    if 'Customers' in dataset.columns:
        dataset = dataset.drop(['Customers'], axis=1)

    if 'Date' in dataset.columns:
        dataset = dataset.drop(['Date'], axis=1)

    dataset = dataset.drop(['StoreType', 'Assortment', 'DayOfWeek', 'Month'], axis=1)  # ,'Store'

    return dataset


trainset = clean_data(trainset)
testset = clean_data(testset)

# 只对开店营业的数据进行学习
trainset = trainset.loc[trainset['Open'] == 1, :]
trainset = trainset.drop(['Open'], axis=1)

# 去除trainset中Sales=0的异常值 54/844392
trainset = trainset.query('Sales!=0')

testset = testset.set_index('Id')
testset.sort_index(inplace=True)

# 将Sales设置为标签
Y_train = trainset.pop("Sales")

# 设置训练集
X_train = trainset

# 对标签进行对数转换
Y_train = np.log(Y_train + 1)

features = list(X_train.columns)

features.remove('Store')

# xgboost model

# development_set for grid cv, evaluation set for evaluation
X_dev, X_eval, Y_dev, Y_eval = train_test_split(X_train, Y_train, test_size=0.33, shuffle=True, stratify=X_train['Store'], random_state=seed)

# 设置cross-validation用的loss function


def rmspe_exp(y_true, y_predict):
    y_true = np.exp(y_true) - 1
    y_predict = np.exp(y_predict) - 1
    w = np.zeros(y_true.shape, dtype=float)
    indice = y_true != 0
    w[indice] = 1. / (y_true[indice]**2)
    score = np.sqrt(np.mean(w * (y_true - y_predict)**2))
    return score


rmspe_exp_error = make_scorer(rmspe_exp, greater_is_better=False)

# 设置cross-validation策略
cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=seed)


# default XGBoost model

# default params:
# ```python
# default_params = {
#     'objective':'reg:linear',
#     'max_depth':3,
#     'learning_rate':0.1,
#     'n_estimators':100,
#     'colsample_bytree':1,
#     'subsample':1
# }
# ```

default_reg = XGBRegressor(objective='reg:linear', seed=seed)

# 初始XGBoost的learning curve
train_default_sizes, train_default_scores, test_default_scores = learning_curve(
    default_reg, X_eval[features], Y_eval, train_sizes=np.linspace(0.1, 1, 8), cv=cv, scoring=rmspe_exp_error,
    verbose=100, shuffle=True, random_state=seed)

# 将learning curve数据导出以便graph.ipynb绘图
learn_curve_default = pd.DataFrame({
    'train_sizes': train_default_sizes,
    'train_scores_mean': np.mean(train_default_scores, axis=1),
    'train_scores_std': np.std(train_default_scores, axis=1),
    'test_scores_mean': np.mean(test_default_scores, axis=1),
    'test_scores_std': np.std(test_default_scores, axis=1)
})
learn_curve_default = np.abs(learn_curve_default)

learn_curve_default.to_json('./json/learn_curve_default.json')


# parameter tuning

# 根据domain knowledge 设置参数范围
params = {
    'n_estimators': np.arange(100, 650, 50),
    'max_depth': [4, 5, 6, 7, 8],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'subsample': [0.5, 0.6, 0.7, 0.8, ],
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
}


# 使用线性的回归树
reg = XGBRegressor(objective='reg:linear', seed=seed)
# 进行随机参数选择
bst_grid = RandomizedSearchCV(estimator=reg, param_distributions=params, cv=cv,
                              scoring=rmspe_exp_error, verbose=100, n_iter=80, random_state=seed)

bst_grid.fit(X_dev[features], Y_dev)

print('best param_sets', bst_grid.best_params_)

# 将cv过程结果写入json文件以便graph.ipynb绘图
cv_result = pd.DataFrame(bst_grid.cv_results_)
cv_result.index = cv_result.index + 1

cv_result.to_json('./json/cvresults_random.json')

# predict and test --- optimized model


def rmspe_xg(y_predict, y_true):
    y_true = np.exp(y_true) - 1
    y_predict = np.exp(y_predict) - 1
    w = np.zeros(y_true.shape, dtype=float)
    indice = y_true != 0
    w[indice] = 1. / (y_true[indice]**2)
    score = np.sqrt(np.mean(w * (y_true - y_predict)**2))
    return 'rmspe_xg', score


# 最佳参数组合
param_origin_xgb = {
    'objective': 'reg:linear',
    'max_depth': 8,
    'eta': 0.25,  # 同learning_rate
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'seed': seed
}
n_rounds = 550  # 同n_estimators
# dtrain = xgb.DMatrix(X_train,Y_train)  # 训练模型

dtrain = xgb.DMatrix(X_train[features], Y_train)
dtest = xgb.DMatrix(testset[features])

origin_reg = xgb.train(param_origin_xgb, dtrain, n_rounds, feval=rmspe_xg)  # 预测testset

# 预测


def output_csv(estimator, name, benchmark=False):
    predict_name = 'prdict' + name
    output_name = 'sample_submission_' + name
    if benchmark:
        testset[predict_name] = estimator.predict(X_test_linear)
    else:
        testset[predict_name] = estimator.predict(dtest)

    testset[predict_name] = np.exp(testset[predict_name]) - 1

    ind = testset[predict_name] < 0
    testset.loc[ind, predict_name] = 0

    # 对关店的数据直接预测Sales=0
    Open_0_indice = testset['Open'] == 0
    testset.loc[Open_0_indice, predict_name] = 0

    test_submission = pd.DataFrame({'Id': list(testset.index),
                                    "Sales": testset[predict_name]})

    test_submission.to_csv(r'./score_sheet/{}.csv'.format(output_name), index=False)
    return None


output_csv(origin_reg, 'optimized')

# 将特征重要性信息导入json文件，以便后续graph.ipynb绘图
feature_list, gain_list = zip(*[(key, value) for (key, value) in origin_reg.get_score().items()])
feature_importance = pd.DataFrame({'Feature': feature_list, 'Gain': gain_list})
feature_importance.to_json(r'./json/feature_importance.json')


# Linear regression(benchmark)

features_linear = features.copy()

# 消除多重共线问题
for x in ['StoreType_d', 'Assortment_c', 'DayOfWeek_7', 'Month_12']:
    features_linear.remove(x)

# 缺失值用均值替代
X_train_linear = X_train[features_linear].apply(lambda x: x.fillna(x.mean()), axis=0)
X_test_linear = testset[features_linear]
X_test_linear = X_test_linear.apply(lambda x: x.fillna(x.mean()), axis=0)

# 线性回归
linear_reg = LinearRegression()

linear_reg.fit(X_train_linear, Y_train)

output_csv(linear_reg, 'linear', benchmark=True)
