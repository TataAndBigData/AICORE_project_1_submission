#!/usr/bin/env python
# coding: utf-8

# In[174]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from itertools import combinations 


# ### Part 1: Get the wine dataset¶
# Describe the dataset
# 
# Load in the training and test splits of the dataset

# In[2]:


df_train = pd.read_csv('/Users/eiwi/Practical-ML-DS/Chapter 1. Machine Learning Foundations/DATA/winequality-red-train.csv',index_col=0)
df_test = pd.read_csv('/Users/eiwi/Practical-ML-DS/Chapter 1. Machine Learning Foundations/DATA/winequality-red-test.csv', index_col=0)


# In[3]:


# The charts showing the relationship between each feature and the target
fig, ax = plt.subplots(len(df_train.columns),1, sharey = True, figsize = (18,18))
for i in range(len(df_train.columns)):
    current_col = df_train[df_train.columns[i]]
    ax[i].barh(df_train['quality'], current_col)
    ax[i].set_xlabel(str(df_train.columns[i]))
    ax[i].set_ylabel('quality')
plt.show()


# In[4]:


f, ax = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(df_train.corr(), annot=True)


# In[5]:


print("All the features given in this dataset are numeric.")
print('The most correlated features are: fixed acidity, density and citric acid, total sulfur dioxide and free sulfur dioxide, negatively correlated: fixed acidity and pH, citric and volatile acidity.')


# In[6]:


print('The Quality paremeter is mostly correlated to alcohol (positively) and volatile acidity (negatively).')


# In[7]:


# Separate target and predictors:


# In[8]:


target_train = df_train.pop('quality')
target_test = df_test.pop('quality')


# In[9]:


## Further split train dataset into train and validation sets

X_train, X_validation, y_train, y_validation = train_test_split(df_train, target_train, test_size=0.33, random_state=42)


# In[10]:


# Set the indices in the new datasets as the subquential range of numbers 

def reset_indices(df,target):
    lst_idx = []
    for idx,_ in enumerate(df.iloc[:,0]):
        lst_idx.append(idx)
    df['index'] = lst_idx
    df.set_index(df['index'], inplace = True)
    df.drop(columns='index', inplace=True)
    target.index = lst_idx
    return df, target


# In[11]:


X_train, y_train = reset_indices(X_train, y_train)


# In[14]:


X_validation, y_validation = reset_indices(X_validation, y_validation)


# In[15]:


X_test, y_test = reset_indices(df_test, target_test)


# In[16]:


print(len(X_train))
print(len(y_train))
print(len(X_validation))
print(len(y_validation))
print(len(X_test))
print(len(y_test))


# In[17]:


print('The data: train set of {} rows, validation set of {} rows, test set of {} rows.'.format(len(X_train), len(X_validation), len(df_test)), end = '\n')
print('Number of predictors: {}. List of predictors: {}.  Target: {}. Possible classes : {}.'.format(len(X_train.columns), ', '.join(X_train.columns), 'column "quality"', str(set(target_train))), end = '\n')
print('Missing values for train set: {}, for validation set: {}, for test set: {}.'.format(X_train.isna().sum().sum(), X_validation.isna().sum().sum(), df_test.isna().sum().sum()))


# In[18]:


# The data requires scaling due to the difference in units:
X_train.describe()


# In[19]:


# Data scaling 
scaler = StandardScaler()
X_train_std = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_validation_std = pd.DataFrame(scaler.fit_transform(X_validation),  columns = X_validation.columns)
X_test_std = pd.DataFrame(scaler.fit_transform(df_test), columns = df_test.columns)


# In[20]:


fig, ax = plt.subplots(1,2, figsize = (12,8))
ax[0].plot(X_train)
ax[0].set_title('Features before scaling')
ax[1].plot(X_train_std)
ax[1].set_title('Features after scaling')
plt.legend(X_train.columns, loc="lower center", bbox_to_anchor=(0, -0.3),  ncol= 3)
plt.show()


# In[21]:


# Now, the data is ready to be fed to the model. 
# I will treat this as a classification problem and will try several basic models to
# get an idea of the most suitable algorithm and hyper parameters.


# ### Part 2: Fit models to the wine dataset and test performance
# Make sure you are comfortable with passing the data through a model.
# 
# Evaluate the performance of the model. 
# 
# Make sure you are testing it on the right set of data.
# 

# In[23]:


results_dict = {'model':[],
                'params':[],
                'train_score':[],
                'validation_score':[]
               }


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models_lst = [
    LogisticRegression(random_state=42),
    SVC(gamma=2, random_state=42),
    MLPClassifier(max_iter=1000, random_state=42),
    KNeighborsClassifier(n_jobs=-1),
    DecisionTreeClassifier(random_state=42),
   ]


# In[25]:


params = [
    {'penalty':('l1','l2', 'elasticnet', 'none'), 'C':(0.025, 0.01,1,10), 'solver': ('newton-cg', 'lbfgs', 'liblinear','sag','saga')},
    {'C':(0.025, 0.01,1,10), 'kernel':('linear', 'rbf'), 'decision_function_shape':('ovr','ovo')},
    {'activation':('relu', 'logistic'), 'solver':('sgd','lbfgs'), 'alpha':(0.001,0.01,0.1,1), 'learning_rate': ('adaptive','constant')},
    {'n_neighbors':(3,5,10), 'weights':('distance','uniform'), 'algorithm':('auto','ball_tree','kd_tree')},
    {'criterion':('gini','entropy'), 'splitter':('best','random'),'max_depth':(5,10,25), 'min_samples_split':(2,3,5)}
]


# In[26]:


for i in range(len(models_lst)):
    model = models_lst[i]
    parameters = params[i]
    grid_search_model = GridSearchCV(model, parameters)
    grid_search_model.fit(X_train,y_train)
    train_score =  grid_search_model.best_score_
    validation_score = grid_search_model.best_estimator_.score(X_validation,y_validation)
    results_dict['model'].append(grid_search_model.best_estimator_)
    results_dict['params'].append(grid_search_model.best_params_)
    results_dict['train_score'].append(train_score)
    results_dict['validation_score'].append(validation_score)


# In[27]:


results_dict


# In[28]:


classifiers_comparison = pd.DataFrame.from_dict(results_dict)


# In[29]:


classifiers_comparison.to_csv('/Users/eiwi/Practical-ML-DS/Chapter 1. Machine Learning Foundations/PROJECT/classifiers_comparison.csv')


# In[30]:


classifiers_comparison


# In[31]:


best_train_score = classifiers_comparison['train_score'].max()
print('best_train_score: ', best_train_score)
print('best_train_score_model: ', classifiers_comparison['model'][classifiers_comparison['train_score']==best_train_score], end = '\n')
 
print('\n')    

second_best_train_score = list(classifiers_comparison['train_score'].sort_values(ascending = False))[1]
print('second_best_train_score: ', second_best_train_score)
print('second_best_train_score_model: ', classifiers_comparison['model'][classifiers_comparison['train_score']==second_best_train_score], end = '\n')

print('\n')    

best_validation_score = classifiers_comparison['validation_score'].max()
print('best_validation_score: ', best_validation_score)
print('best_validation_score_model: ', classifiers_comparison['model'][classifiers_comparison['validation_score']==best_validation_score], end = '\n')
 
print('\n')    

second_best_validation_score = list(classifiers_comparison['validation_score'].sort_values(ascending = False))[1]
print('second_best_validation_score: ', second_best_validation_score)
print('second_best_validation_score_model: ', classifiers_comparison['model'][classifiers_comparison['validation_score']==second_best_validation_score], end = '\n')


# In[32]:


print('So far the best models were {} and {}, so I will try to combine them into an ensemble.'.format('DecisionTreeClassifier', 'KNeighborsClassifier' ))


# ### Part 3: Improve your performance by ensembling some of your models
# Combine the results of more than one model
# 
# Evaluate the performance of the ensemble

# In[138]:


from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier


# In[139]:


ensembles_results = {
    'ensemble':[],
    'train_score':[],
    'validation_score':[]
}


# In[140]:


# First, I will combine DecisionTreeClassifier + KNeighborsClassifier 
# using ensemble called VotingClassifier a) with hard vote b) with soft vote


# In[141]:


# a) VotingClassifier with a hard vote: the classifier uses 
# predicted class labels for majority rule voting.


# In[142]:


knn_estimator =  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                       metric_params=None, n_jobs=-1, n_neighbors=10, p=2)

dtc_estimator = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                         max_depth=25, max_features=None, max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=3,
                         min_weight_fraction_leaf=0.0,splitter='best')


# In[143]:


hard_voting_ensemble = VotingClassifier(estimators=[('knn_estimator',knn_estimator), ('dtc_estimator',dtc_estimator)],
                                   voting='hard')


# In[144]:


hard_voting_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('VotingClassifier_hard')
ensembles_results['train_score'].append(hard_voting_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(hard_voting_ensemble.score(X_validation,y_validation))


# In[145]:


# b) VotingClassifier with a soft vote: predicts the class label based on the argmax 
# of the sums of the predicted probabilities


# In[146]:


soft_voting_ensemble = VotingClassifier(estimators=[('knn_estimator',knn_estimator), ('dtc_estimator',dtc_estimator)],
                                   voting='soft')


# In[147]:


soft_voting_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('VotingClassifier_soft')
ensembles_results['train_score'].append(soft_voting_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(soft_voting_ensemble.score(X_validation,y_validation))


# In[148]:


# Secondly, I will use StackingClassifier with the same DecisionTreeClassifier + KNeighborsClassifier 

stacking_ensemble = StackingClassifier(estimators=[('knn_estimator',knn_estimator), ('dtc_estimator',dtc_estimator)])

stacking_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('StackingClassifier')
ensembles_results['train_score'].append(stacking_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(stacking_ensemble.score(X_validation,y_validation))


# In[149]:


# Compare StackingClassifier with final AdaBoostClassifier
ada_stacking_ensemble = StackingClassifier(estimators=[('knn_estimator',knn_estimator), 
                                                       ('dtc_estimator',dtc_estimator)],
                                           final_estimator=None,
                                           stack_method='auto',
                                           n_jobs=-1)

ada_stacking_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('StackingClassifier_AdaBoost')
ensembles_results['train_score'].append(ada_stacking_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(ada_stacking_ensemble.score(X_validation,y_validation))


# In[150]:


# Then compare ensemble perfomance with a BaggingClassifier 
# using multiple KNeighborsClassifiers OR DecisionTreeClassifiers


# In[151]:


# a) BaggingClassifier with DecisionTreeClassifier


# In[152]:


dtc_bagging_ensemble = BaggingClassifier(base_estimator = dtc_estimator,
                                         n_estimators=100,
                                         random_state=42)

dtc_bagging_ensemble.fit(X_train, y_train)


# In[153]:


ensembles_results['ensemble'].append('DTC_BaggingClassifier')
ensembles_results['train_score'].append(dtc_bagging_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(dtc_bagging_ensemble.score(X_validation,y_validation))


# In[154]:


# b) BaggingClassifier with KNeighborsClassifiers


# In[155]:


knn_bagging_ensemble = BaggingClassifier(base_estimator=knn_estimator,
                                     n_estimators=100,
                                     random_state=42)


# In[156]:


knn_bagging_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('KNN_BaggingClassifier')
ensembles_results['train_score'].append(knn_bagging_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(knn_bagging_ensemble.score(X_validation,y_validation))


# In[157]:


# As DecisionTreeclassifier was the most cussessful in previous ensembles,
# now I will try out RandomForestClassifier


# In[158]:


rfc_ensemble = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_ensemble.base_estimator = dtc_estimator


# In[159]:


rfc_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('RandomForestClassifier')
ensembles_results['train_score'].append(rfc_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(rfc_ensemble.score(X_validation,y_validation))


# In[160]:


#Compare with AdaBoostClassifier based on DecisionTreeClassifiers only 
# (as KNeighborsClassifier doesn't support sample_weight)


# In[161]:


dtc_ada_ensemble = AdaBoostClassifier(base_estimator= dtc_estimator, n_estimators=50, 
                                      learning_rate=1.0, algorithm='SAMME.R', 
                                      random_state=42)
dtc_ada_ensemble.fit(X_train, y_train)

ensembles_results['ensemble'].append('AdaBoost_dtc_ensemble')
ensembles_results['train_score'].append(dtc_ada_ensemble.score(X_train, y_train))
ensembles_results['validation_score'].append(dtc_ada_ensemble.score(X_validation, y_validation))


# In[162]:


# Now let's compare these ensembles


# In[163]:


ensembles_results


# In[164]:


ensembles_comparison = pd.DataFrame.from_dict(ensembles_results)


# In[165]:


ensembles_comparison


# In[166]:


# Conclusion: the highest validation score of 0.822695 was achieved with DTC_BaggingClassifier
# The second best score of 0.817967 was achieved with RandomForestClassifier
# Surprisingly, VotingClassifier and StackingClassifier combining different base models
# have worse perfomanse than ensembles with 1 estimator (RandomForestClassifier, DTC_BaggingClassifier)


# In[362]:


dtc_ada_ensemble.__class__.__name__


# In[324]:


lst = list(combinations([3,6,7], 2))
for t in lst:
    print(list(t))
    
(1,2)


# ### Part 4: Write an algorithm to find the best combination of models and hyperparameters.
# 
# There are an infinite number of ways that you could combine different models with different hyperparameters, but some will perform better than others.
# 
# List the different parameters which you test over, as well as the ranges which you test
# 
# Describe the search procedure which your algorithm implements

# In[484]:


from sklearn.model_selection import cross_val_score


# In[533]:


def algo_name(est):
    return est.__class__.__name__
def algo_voting_soft_classifier(estimators_list, param_grid_list, N, X, y, scoring='f1'):
    """
        This function searches for best combination of model and hyperparameters to be used in VotingClasifier(soft)
    
        estimators_list — list of non-fitted estimators
        param_grid_list — list of param grids for GridSearch, must be same size as estimators_list
        N — number of models to use in ensemble
        X — features
        y — target
        scoring — scoring function name (string)
    """
    
    # Lists must be of same size
    assert len(estimators_list) == len(param_grid_list)
    
    # Can't use more estimators than we have
    assert N <= len(estimators_list)
    
    # We will keep list of best fitted estimators
    best_estimators = []
    
    # Let's GridSearch em all
    for estimator, param_grid in zip(estimators_list, param_grid_list):
        print("Searching best params for " + algo_name(estimator))
        gs = GridSearchCV(estimator, param_grid, scoring=scoring)
        gs.fit(X,y)
        best_estimators.append(gs.best_estimator_)
        print("Best score = " + str(gs.best_score_))
        
    # Here we have all possible combinations of N estimators
    all_combinations = list(combinations(best_estimators, N))
    
    # Keep track of best ensemble yet
    top_score = 0
    top_ensemble = None
    
    # Go over all combinations
    for t in all_combinations:
        voting_estimators = []
        voting_estimators_names = []
        
        # Construct list of estimators for VotingClassifier
        for estimator in list(t):
            name = algo_name(estimator)
            voting_estimators_names.append(name)
            voting_estimators.append((name, estimator))
        
            
        print(f"\n\nChecking combination of {N} estimators:")
        print("\t\n".join(voting_estimators_names))
        # Cross-validating ensemble of N estimators
        soft_voting_ensemble = VotingClassifier(estimators=voting_estimators, voting='soft')
        cross_vals = cross_val_score(soft_voting_ensemble, X,y, scoring=scoring)
        
        
        # Selecting top ensemble
        current_score = max(cross_vals)
        
        
        print(f"Score ({scoring}) of this combination is {current_score}")
        if (current_score>top_score):
            top_score = current_score
            top_ensemble = soft_voting_ensemble
       
    return top_ensemble        


# In[534]:


estimators_list = [
    LogisticRegression(random_state=42, n_jobs=-1),
    SVC(gamma=2, random_state=42, probability=True),
#     MLPClassifier(max_iter=1000, random_state=42),
#     KNeighborsClassifier(n_jobs=-1),
    DecisionTreeClassifier(random_state=42)
]

param_grids_list = [
    {'penalty':('l1','l2', 'elasticnet', 'none'), 'C':(0.025, 0.01,1,10), 'solver': ('newton-cg', 'lbfgs', 'liblinear','sag','saga')},
    {'C':(0.025, 0.01,1,10), 'kernel':('linear', 'rbf'), 'decision_function_shape':('ovr','ovo')},
#     {'activation':('relu', 'logistic'), 'solver':('sgd','lbfgs'), 'alpha':(0.001,0.01,0.1,1), 'learning_rate': ('adaptive','constant')},
#     {'n_neighbors':(3,5,10), 'weights':('distance','uniform'), 'algorithm':('auto','ball_tree','kd_tree')},
    {'criterion':('gini','entropy'), 'splitter':('best','random'),'max_depth':(5,10,25,50), 'min_samples_split':(2,3,5,10)}
]


# In[535]:


demo_X = X_train[:100]
demo_y = y_train[:100]


# In[536]:


import warnings
warnings.filterwarnings("ignore")
best_voter = algo_voting_soft_classifier(estimators_list, param_grids_list, 2, demo_X, demo_y, scoring='accuracy')
warnings.filterwarnings("default")


# In[518]:


best_voter


# In[519]:


warnings.filterwarnings("ignore")
best_voter_3 = algo_voting_soft_classifier(estimators_list, param_grids_list, 3, demo_X, demo_y, scoring='accuracy')
warnings.filterwarnings("default")


# In[520]:


best_voter_3


# ### Part 5: Present your results
# 
# The final part of the project requires you to attempt to summarise your work. It should be clear how we could replicate the results by implementing exactly the same ensemble (models and hyperparameters).
# 
# Please try to communicate and display your results with any graphs or charts.
# If you have any insights into why certain ensembles or models perform better of worse than others, and would like to write a paragraph to explain this, we'd love to read it!
# Please also write a summary paragraph that describes the best permutation that you found.

# In[ ]:


# Based on my observation, the individual models  show lower score than ensembles


# In[676]:


print( 'The average score among individual models is equal to {}, maximum validation score - {} (provided by {}).'.format(round(classifiers_comparison['validation_score'].mean(),3), round(classifiers_comparison['validation_score'],3).max(), str(classifiers_comparison[classifiers_comparison['validation_score'] == classifiers_comparison['validation_score'].max()]['model'])[5:27]))


# In[710]:


print('The average score among ensembles is equal to {}, maximum validation score - {} (provided by {}).'.format(round(ensembles_comparison['validation_score'].mean(),3),
                        round(ensembles_comparison['validation_score'].max(),3),
                        str(ensembles_comparison[ensembles_comparison['validation_score'] == ensembles_comparison['validation_score'].max()]['ensemble'])[5:27]))


# In[ ]:


# The most powerful individual models are DecisionTreeClassifier and KNeighborsClassifier.
# The most powerful ensembles are DTC_BaggingClassifier and RandomForestClassifier (both are based on DecisionTreeClassifier)


# In[ ]:


# Single models VS ensembles


# In[718]:


fig, ax = plt.subplots(1, 2, figsize = (16,6), sharey = True)

ax[0].bar(classifiers_comparison.index, classifiers_comparison['train_score'], width = -0.3, align = 'edge', color='navy')
ax[0].bar(classifiers_comparison.index, classifiers_comparison['validation_score'], width = 0.3, align = 'edge', color='orange')
ax[0].set_title('Score of the individual models')

ax[1].bar(ensembles_comparison.index, ensembles_comparison['train_score'], width = -0.3, align = 'edge', color='navy')
ax[1].bar(ensembles_comparison.index, ensembles_comparison['validation_score'], width = 0.3, align = 'edge' , color='orange' )
ax[1].set_title('Score of the ensembles')


plt.title(label = 'Difference in score between individual models and ensembles', loc='left')
plt.show()


# In[719]:


# Interestingly, while individual models tend to show slightly higher score on the validation set,
# while ensembles tend to do the opposite: their train score is higher than validation one.


# In[720]:


# It is also interesting to compare the ensembles combining the different models VS homogeneous ones


# In[727]:


ensembles_comparison


# In[725]:


fig, ax = plt.subplots()
ax.set_title('Ensembles score')
ax.bar(ensembles_comparison.index, ensembles_comparison['train_score'], width = -0.3, align = 'edge', color='navy')
ax.bar(ensembles_comparison.index, ensembles_comparison['validation_score'], width = 0.3, align = 'edge' , color='orange')
ax.set_ylabel('Score')
ax.set_xlabel('Ensemble')


# In[ ]:


# After performing a series of test, I made a conclusion that the best possible score can be achieved with
# the ensemble methods that combine multiple versions of the same estimator (BaggingClassifier, RandomForestClassifier, ) rather than 
# the ones combining a list of differents estimators (VotingClassifier and StackingClassifier).


# In[ ]:


# So for the further iteartions I would use BaggingClassifier or RandomForestClassifier.
# The key for reproducibility the would be to stick to the repeatable method for every step of data analysis:
# so it is reasonable to create the standard functions for data processing and cleaning, scaling and modelling.
# Also, in order to replicate the results I would pass an argument 'random_state = 42' to the models 
# which allows to get the same results while running the model again.


# ### Part 6:
# A stakeholder asks you which features most affect the response variable (output). Describe how you would organise a test to determine this.

# In[521]:


# I would consider the influence of each feature to the output as a change that occurs in target value
# with change of the output with one unit


# In[525]:


get_ipython().system("['Approach for features importance calculation']('Change_in_var.png')")


# In[745]:


# I would test this approach on desicion-tree based model in order to compare the results with the built function:
feature_importances = list(dtc_estimator.fit(demo_X, demo_y).feature_importances_)
features = demo_X.columns


# In[757]:


feat_imp = pd.DataFrame()
feat_imp['features'] = features
feat_imp['feature_importances'] = feature_importances


# In[762]:


feat_imp.sort_values(by = 'feature_importances', ascending = False)


# In[ ]:


# The inbuilt function suggests that alcohol has the strongest effect on the target (quality of  wine).

