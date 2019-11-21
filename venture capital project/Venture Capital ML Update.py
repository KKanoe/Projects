#Import core packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np


#Drop column not perceived to be features
drop_cols = ['company_permalink','company_category_code','company_country_code','company_state_code','company_city','company_region',
             'founded_month','founded_quarter','first_funding_at','first_funding_at','last_funding_at','last_milestone_at',
             'acquirer_permalink','acquirer_name','acquirer_category_code','acquirer_country_code','acquirer_state_code','acquirer_region',
             'acquirer_city','acquired_at','acquired_month','acquired_quarter','acquired_year','price_currency_code',
             'price_amount','founded_year','founded_at','unique_name','first_to_last_fund_days','last_to_acq_days',
             'found_to_acq_days','first_fund_to_acq_days','funding_total_usd','rddt_1','rddt_2','rddt_3','rddt_4','rddt_5',
             'rddt_6','rddt_7','rddt_8','rddt_9','rddt_10','rddt_11','rddt_12','rddt_13','rddt_14']

ml_df = comp_df.drop(columns=drop_cols).set_index('company_name')

#Format day diff columns
#All nan's due to date differences only occuring at second round, drop in founded to funding days as replacement (should drop negative days)
ml_df['rddt_diff_1'] = ml_df['found_to_fund_days'].dt.days 
ml_df = ml_df.drop(columns='found_to_fund_days') 

#Change negative values where funding date preceded founding date
ml_df['rddt_diff_1'] = np.where(ml_df['rddt_diff_1'] < 0, 0, ml_df['rddt_diff_1'])

#Replace Nan's with zeroes (skipping status column)
ml_df.iloc[:, 1:] = ml_df.iloc[:, 1:].fillna(0)

#Calculate total funding from rounds (this ensures our accounting balances with the rds data instead of using provided funding total)
start_col_amt = ml_df.columns.get_loc('rdamt_1')
end_col_amt =  ml_df.columns.get_loc('rdamt_14') + 1
funding_total_usd = ml_df.iloc[:, start_col_amt:end_col_amt].sum(axis=1)
ml_df.insert(end_col_amt, 'funding_total_usd', funding_total_usd)

#Format df to various versions (i.e. filtering based on criteria)
#Fix negative category codes where -1 (missing category originally) 
max_cat_code = comp_df['category_code'].max() + 1
ml_df['category_code'] = np.where(ml_df['category_code'] == -1.0, max_cat_code, ml_df['category_code'])

#Scale/transform data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Convert df to array and perform one-hot encoding for target variable 'status'
tgt_enc = LabelEncoder().fit(ml_df['status'])
tgt_encoded = tgt_enc.transform(ml_df['status'])
np.unique(tgt_enc.inverse_transform(tgt_encoded))
np.unique(tgt_encoded)
ml_df = ml_df.drop(columns='status')

X = ml_df.values
Y = tgt_encoded

scaled_X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.3, random_state=0)

#Feature Selection
#Feature Importance with Forest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

model_exTree = ExtraTreesClassifier()
model_exTree.fit(X_train, y_train)

model_RandTree = RandomForestClassifier()
model_RandTree.fit(X_train, y_train)

#Visualization and ouput
#Extra Trees
features_exTree = {}

for feature, importance in zip(ml_df.columns, model_exTree.feature_importances_):
    features_exTree[feature] = importance

importances_exTree = pd.DataFrame.from_dict(features_exTree, orient='index').rename(columns={0: 'Gini-Importance'})
importances_exTree.sort_values(by='Gini-Importance', ascending=False).plot(kind='bar', rot=90)
plt.title('Feature Importance - ExtraTrees Classifier')
plt.tight_layout()
plt.show()

features_RandTree = {}

for feature, importance in zip(ml_df.columns, model_RandTree.feature_importances_):
    features_RandTree[feature] = importance

importances_RandTree = pd.DataFrame.from_dict(features_RandTree, orient='index').rename(columns={0: 'Gini-Importance'})
importances_RandTree.sort_values(by='Gini-Importance', ascending=False).plot(kind='bar', rot=90)
plt.title('Feature Importance - Random Forest Classifier')
plt.tight_layout()
plt.show()

#Implement Tree Model
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel

#Subset original df using identified highest importance features
sfm_RandTree = SelectFromModel(model_RandTree, threshold=0.05)
sfm_RandTree.fit(X_train, y_train)

#Print selected features
selected_feat = []

for feature_list_index in sfm_RandTree.get_support(indices=True):
    print(ml_df.columns[feature_list_index])
    selected_feat.append(ml_df.columns[feature_list_index])

X_important_train = sfm_RandTree.transform(X_train)
X_important_test = sfm_RandTree.transform(X_test)

#Perform RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Number of Trees
n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=30)]

#Number of features at each split
max_features = ['auto','sqrt']

#Max depth of each tree
max_depth = [int(x) for x in np.linspace(start=5, stop=100, num=20)]
max_depth.append(None)

#Min samples at each split
min_samples_split = [int(x) for x in np.linspace(start=2, stop=100, num=10)]

#Minimun samples at base leaf
min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=100, num=10)]

#Method of selecting samples for training each tree
bootstrap = [True, False]

#Place all parameter ranges in grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

#Instantiate instance of RandomForestClassifier
rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, 
                               random_state=42, n_jobs=-1)

#Fit to training set with selected features
rf_random.fit(X_important_train, y_train)

#Best parameters from randomized search cv
rf_random.best_params_

#Implemenet Grid Search with identified best parameters from randomized search    
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
              'bootstrap': [True],
              'max_depth': [None],
              'max_features': ['auto'],
              'min_samples_leaf': [60, 67, 74],
              'min_samples_split': [50, 56, 62],
              'n_estimators': [100, 283, 500, 1000]
}

# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2)  

# Fit the grid search to the data
grid_search.fit(X_important_train, y_train)

#View results of grid_search
grid_search.best_params_
best_grid = grid_search.best_estimator_

#Base Model Evaluation
base_model = RandomForestClassifier(n_estimators=10, random_state=4)
base_model.fit(X_important_train, y_train)
base_model_pred = base_model.predict(X_important_test)

#RandomizedSearch Tuned Model Evaluation 
tuned_model = RandomForestClassifier(**rf_random.best_params_, random_state=5)
tuned_model.fit(X_important_train, y_train)
tuned_model_pred = tuned_model.predict(X_important_test)

#Grid Search Tuned Model
best_grid.fit(X_important_train, y_train)
best_grid_pred = best_grid.predict(X_important_test)

#Train and create prediction values for full dataset to compare to selected features
tuned_model.fit(X_train, y_train)
tuned_model_pred_full = tuned_model.predict(X_test)

best_grid.fit(X_train, y_train)
best_grid_pred_full = best_grid.predict(X_test)

#Base Model Score versus Parameter tuned models
print("Base Model Accuracy Score: %s" % round(accuracy_score(y_test, base_model_pred),2))
print("Tuned Model Accuracy Score: %s" % round(accuracy_score(y_test, tuned_model_pred),2))
print("Tuned Model Accuracy Score - Full Dataset: %s" % round(accuracy_score(y_test, tuned_model_pred_full),2))
print("Best Grid Model Accuracy Score: %s" % round(accuracy_score(y_test, best_grid_pred),2))
print("Best Grid Model Accuracy Score - Full Dataset: %s" % round(accuracy_score(y_test, best_grid_pred_full),2))
print("Best RCV Parameters: %s", rf_random.best_params_)
print("Best GSCV Parameters: %s", best_grid)

#Graph confusion matrix for full dataset and dataset using top features
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
mat1 = confusion_matrix(y_test, base_model_pred)
mat2 = confusion_matrix(y_test, tuned_model_pred)

sns.heatmap(mat1.T, square=True, annot=True, fmt='d', cbar=False, ax=ax1)
ax1.set_title('Confusion Matrix - Base Model (accuracy = %s percent)' 
              % round(accuracy_score(y_test, base_model_pred),2), fontsize=15)
ax1.set_xlabel('True Label', fontsize=13)
ax1.set_ylabel('Predicted Label', fontsize=13)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False, ax=ax2)
ax2.set_title('Confusion Matrix - Tuned Model (accuracy = %s percent)' % 
              round(accuracy_score(y_test, tuned_model_pred),2), fontsize=15)
ax2.set_xlabel('True Label', fontsize=13)
ax2.set_ylabel('Predicted Label', fontsize=13)
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top- 0.5)

plt.show()

#Print Status Labels and Numbers for reference
print("Target Labels", np.unique(tgt_enc.inverse_transform(tgt_encoded)), '\n' ,"Label Numbers", np.unique(tgt_encoded))

#Classification Report
print("Random Forest - Full Dataset:" + '\n' + classification_report(y_test, y_pred_RandTree))
print("Random Forest - Important Dataset:" + '\n' + classification_report(y_test, y_pred_RandTree_important))
#Classification Report
print("Random Forest - Base Model:" + '\n' + classification_report(y_test, base_model_pred, target_names=np.unique(tgt_enc.inverse_transform(tgt_encoded))))
print("Random Forest - Tuned Model:" + '\n' + classification_report(y_test, tuned_model_pred, target_names=np.unique(tgt_enc.inverse_transform(tgt_encoded))))

#Plot decision trees
from glob import glob
import PIL
import pydotplus
from IPython.display import display, Image
from sklearn.tree import export_graphviz

def save_decision_trees_as_png(clf, iteration, feature_name, target_name):
    file_name = "vc_" + str(iteration) + ".png"
    dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=feature_name,
            class_names=target_name,
            rounded=True,
            proportion=False,
            precision=2,
            filled=2,
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.set_size('"20,20!"')
    graph.write_png('C:/datascience/springboard/projects/Venture Capital/data/Exported Data/%s' % file_name)
    print("Decision Tree {} saved as png file".format(iteration + 1))

feature_names = selected_feat
target_names = np.unique(tgt_enc.inverse_transform(tgt_encoded))
    
for i in range(len(tuned_model.estimators_)):
    save_decision_trees_as_png(tuned_model.estimators_[i], i, feature_names, target_names)
    
images = [ PIL.Image.open(f) for f in glob('./*.png') ]

for im in images:
    #display(Image(filename=im.filename, retina=True))
    im.save('C:/datascience/springboard/projects/Venture Capital/data/Exported Data/Decision Tree %s.png' )

#Appendix
#Visual Tuning Parameters
#Parameter Tuning for n_estimators. Using selected features.
n_estimators = range(1, 202, 5)

train_results = []
#test_results = []

for estimator in n_estimators:
    #Training data
    rf = RandomForestClassifier(n_estimators=estimator, random_state=1, n_jobs=-1)
    rf.fit(X_important_train, y_train)
    train_pred = rf.predict(X_important_train)
    train_score = round(accuracy_score(y_train, train_pred),2)
    train_results.append(train_score)

#Plot Results
train_line = plt.plot(n_estimators, train_results, color='blue', label = 'Train Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy Scores')
plt.show()

#Parameter Tuning for max_depth. Using selected features and n_estimators=15.
rf_depth = range(1, 50, 1)

train_results = []
test_results = []

for depth in rf_depth:
    #Training data
    rf = RandomForestClassifier(max_depth=depth, n_estimators=50, n_jobs=-1)
    rf.fit(X_important_train, y_train)
    train_pred = rf.predict(X_important_train)
    train_score = round(accuracy_score(y_train, train_pred),2)
    train_results.append(train_score)

    #Test Data
    test_pred = rf.predict(X_important_test)
    test_score = round(accuracy_score(y_test, test_pred),2)
    test_results.append(test_score)

#Plot Results
train_line = plt.plot(rf_depth, train_results, color='blue', label = 'Train Score')
test_line = plt.plot(rf_depth, test_results, color='red', label = 'Test Score')
plt.legend()
plt.xlabel('Max Depth of Random Forest')
plt.ylabel('Accuracy Scores')
plt.show()

#Parameter Tuning for min_samples_split and min_samples_leaf. Using selected features and n_estimators=50.
min_sample = np.arange(0.05, 0.50, 0.05)

train_results = []
test_results = []

for sample in min_sample:
    #Training data
    #I used same code to test leafs, just changed min_samples_split to min_samples_leaf (changed max step value to 0.5 as well)
    rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=sample, n_jobs=-1)
    rf.fit(X_important_train, y_train)
    train_pred = rf.predict(X_important_train)
    train_score = round(accuracy_score(y_train, train_pred),2)
    train_results.append(train_score)

#Plot Results
train_line = plt.plot(min_sample, train_results, color='blue', label = 'Train Score')
test_line = plt.plot(min_sample, test_results, color='red', label = 'Test Score')
plt.legend()
plt.xlabel('Minimum Sample')
plt.ylabel('Accuracy Scores')
plt.show()

