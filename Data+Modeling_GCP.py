
# coding: utf-8

# # Feature Selection & Modelling

# In this notebook, I will be using decision tree to identify the feature importance and use them as weights for the columns.

# In[1]:


# Base
import pandas as pd
import numpy as np

# Model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
get_ipython().magic(u'matplotlib inline')


# In[2]:


training = pd.read_pickle('eda_training.pickle')
test = pd.read_pickle('eda_test.pickle')
data_dict = pd.read_pickle('data_dict.pickle')


# In[3]:


data_dict


# In[4]:


X = training.drop('dlq',axis=1)
y = training['dlq']


# In[5]:


res_X = pd.read_pickle('SMOTE_res_x.pickle')
res_y = pd.read_pickle('SMOTE_res_y.pickle')
X_test = pd.read_pickle('scaled_X_test.pickle')
y_test = pd.read_pickle('y_test.pickle')
scaled_test = pd.read_pickle('scaled_test.pickle')


# In[6]:


# Setting AUC as scoring metrics

from sklearn.metrics import roc_auc_score, make_scorer

def roc_auc_score_proba(y_true, proba):
    return roc_auc_score(y_true, proba[:, 1])

auc = make_scorer(roc_auc_score_proba, needs_proba=True)# Decision Tree


# # Decision Tree

# I will be using decision tree for feature selection using feature importance.

# In[4]:


# dtc_params = {
#     'max_depth':[1,2,3,4,5,6,7,8],
#     'max_features':[None,'log2','sqrt',2,3,4],
#     'min_samples_split':[2,3,4,5,10,15,20,25]
# }


# dtc_gs = GridSearchCV(DecisionTreeClassifier(random_state=2), dtc_params, cv=3, verbose=1,scoring=auc)
# dtc_gs.fit(res_X, res_y)


# In[12]:


# feature_importance = dtc_gs.best_estimator_.feature_importances_


# In[15]:


# feature_importance.dump('feature_importance.pickle')


# In[7]:


feature_importance = pd.read_pickle('feature_importance.pickle')


# In[8]:


pd.DataFrame(feature_importance,index=X.columns).sort_values(by=0,ascending=False)


# In[9]:


weighted_X = np.multiply(res_X,feature_importance)
weighted_X_test = np.multiply(X_test,feature_importance)
weighted_test = np.multiply(scaled_test,feature_importance)


# # Logistic Regression

# In[11]:


# logregcv = LogisticRegressionCV(n_jobs=-1, random_state=2, max_iter=200, cv=3,\
#                                 scoring=auc)
# logregcv.fit(weighted_X, res_y)


# In[18]:


# print ('Training Score: {}'.format(logregcv.score(weighted_X,res_y)))   # training score


# In[20]:


# print ('Test Score: {}'.format(logregcv.score(weighted_X_test,y_test)))   # test score


# In[21]:


# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.

#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.

#     title : string
#         Title for the chart.

#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.

#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.

#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.

#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - An object to be used as a cross-validation generator.
#           - An iterable yielding train/test splits.

#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.

#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,random_state=2)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt


# In[22]:


# title = 'Logistic Regression CV'
# estimator = logregcv

# from sklearn.utils import shuffle
# X, y = shuffle(weighted_X, res_y)

# plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))


# Our cross validation score is plateau-ing throughout. This shows that the validation set is unable to learn much from the training data. The training score decrease by only a little bit. It seems that our model is underfitting. We can try to increase the model complexity to see if we can improve on the results. 

# # Kaggle Submission

# In[97]:


# test_predict = logregcv.predict_proba(weighted_test)
# test_predict = pd.DataFrame(test_predict)[1]


# In[102]:


# submission = pd.DataFrame(np.arange(1, len(test_predict) + 1))
# submission[1] = test_predict
# submission = submission.rename({0:'Id',1:'Probability'},axis=1)


# In[106]:


# submission.set_index('Id').to_csv('submission_2.csv')


# <img src="https://i.imgur.com/weWdohG.png">

# My first Kaggle submission using logistic regression cv provided a score of <b>0.857848 (Private)</b> and <b>0.852654 (Public)</b>

# # XGBoost

# In[47]:


# plot learning curve
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.grid_search import GridSearchCV


parameters = {'learning_rate': [0.005,0.01,0.025,0.05],
              'max_depth': [4,6,8,10],
              'min_child_weight': [2,4,6,8,10],
              'gamma': [1,2,4,6,8,10,12],
              'subsample': [0.7,0.75,0.8,0.85],
              'colsample_bytree': [0.15,0.3,0.45,0.6,0.85],
              'n_estimators': [500]} #number of trees, change it to 1000 for better results}

# fit model
# model = XGBClassifier(max_depth=5, learning_rate=0.025, n_estimators=100, silent=True, \
#                       objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, \
#                       gamma=0.65, min_child_weight=10, max_delta_step=1.8, subsample=0.8, colsample_bytree=0.4, \
#                       colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, \
#                       random_state=2, seed=None, missing=None)

estimator = XGBClassifier()
model = GridSearchCV(estimator,parameters, n_jobs=-1,cv=5,verbose=True,scoring='roc_auc')

eval_set = [(weighted_X,res_y), (np.array(weighted_X_test),np.array(y_test))]
model.fit(weighted_X,res_y)

from sklearn.externals import joblib
joblib.dump(model, 'xgboost_model.pkl') 

# , eval_metric=["auc"], eval_set=eval_set, verbose=True,early_stopping_rounds=50)

# make predictions for test data
# y_pred = model.predict(np.array(weighted_X_test))
# predictions = [round(value) for value in y_pred]

# # evaluate predictions
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(np.array(y_test), predictions)

# print("AUC: %.2f%%" % (auc * 100.0))

# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['auc'])
# x_axis = range(0, epochs)

# # plot auc
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['auc'], label='Train')
# ax.plot(x_axis, results['validation_1']['auc'], label='Test')
# ax.legend()
# pyplot.ylabel('AUC')
# pyplot.title('XGBoost AUC')
# pyplot.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




