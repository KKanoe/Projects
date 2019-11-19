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

#array = ml_df.values
#tgt_col = ml_df.columns.get_loc('status')

X = ml_df.values
Y = tgt_encoded

scaled_X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.3, random_state=0)

#Feature Selection
#Feature Importance with Forest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

model_exTree = ExtraTreesClassifier(max_depth=5, n_estimators=10, random_state=0)
model_exTree.fit(X_train, y_train)

model_RandTree = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=0, n_jobs=-1)
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
sfm_RandTree = SelectFromModel(model_RandTree, threshold=0.10)
sfm_RandTree.fit(X_train, y_train)

#Print selected features
selected_feat = []

for feature_list_index in sfm_RandTree.get_support(indices=True):
    print(ml_df.columns[feature_list_index])
    selected_feat.append(ml_df.columns[feature_list_index])

X_important_train = sfm_RandTree.transform(X_train)
X_important_test = sfm_RandTree.transform(X_test)

model_RandTree_important = RandomForestClassifier(max_depth=10, n_estimators=5, random_state=0, n_jobs=-1)
model_RandTree_important.fit(X_important_train, y_train)

#Compare original dataset score to important dataset score
y_pred_RandTree = model_RandTree.predict(X_test)
y_pred_RandTree_important = model_RandTree_important.predict(X_important_test)

#Graph confusion matrix
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
mat1 = confusion_matrix(y_test, y_pred_RandTree)
mat2 = confusion_matrix(y_test, y_pred_RandTree_important)

sns.heatmap(mat1.T, square=True, annot=True, fmt='d', cbar=False, ax=ax1)
ax1.set_title('Confusion Matrix - Full Features (accuracy = %s percent)' % round(accuracy_score(y_test, y_pred_RandTree),2))
ax1.set_xlabel('True Label')
ax1.set_ylabel('Predicted Label')
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False, ax=ax2)
ax2.set_title('Confusion Matrix - Important Features (accuracy = %s percent)' % round(accuracy_score(y_test, y_pred_RandTree_important),2))
ax2.set_xlabel('True Label')
ax2.set_ylabel('Predicted Label')
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top- 0.5)

plt.show()

#Classification Report
print("Random Forest - Full Dataset:" + '\n' + classification_report(y_test, y_pred_RandTree))
print("Random Forest - Important Dataset:" + '\n' + classification_report(y_test, y_pred_RandTree_important))

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
    graph.write_png(file_name)
    print("Decision Tree {} saved as png file".format(iteration + 1))

feature_names = selected_feat
target_names = np.unique(tgt_enc.inverse_transform(tgt_encoded))
    
for i in range(len(model_RandTree_important.estimators_)):
    save_decision_trees_as_png(model_RandTree_important.estimators_[i], i, feature_names, target_names)
    
images = [ PIL.Image.open(f) for f in glob('./*.png') ]

for im in images:
    display(Image(filename=im.filename, retina=True))




clf_tree = tree.DecisionTreeClassifier(max_depth=5)

x_train, x_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.3, random_state=42)

clf_tree = clf_tree.fit(scaled_X, Y)

plt.figure(figsize=(20,18))
tree.plot_tree(clf_tree.fit(scaled_X, Y))




#Univariate Selection (Won't take negative values)
from sklearn.feature_selection import SelectKBest, chi2

n_features = len(ml_df.columns) - 1
test = SelectKBest(score_func=chi2, k=n_features)
fit = test.fit(scaled_X, Y)

#Recursive Feature Selection (endless loop warning)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3) #N top features to select
fit = rfe.fit(scaled_X, Y)

print('Num Features: %d' % fit.n_features_)
print('Selected Features: %s' % fit.support_)
print('Feature Ranking: %s' % fit.ranking_)

#Principle Component Analysis (talk about appropriateness for this project)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
fit = pca.fit(scaled_X)

print('Explained Variance: %s' % fit.explained_variance_ratio_)
