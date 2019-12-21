#Import packages
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
from dateutil.relativedelta import relativedelta


#Read-in JSON files, then convert to pandas dataframe
with open('C:/datascience/springboard/mini projects/ultimate_challenge/logins.json') as read_file:
    login_data = json.load(read_file)
    login_data = pd.DataFrame.from_dict(login_data, orient='columns')

#Count number of logins in 15 minute intervals
login_data['login_time'] = pd.to_datetime(login_data['login_time'])
login_data.loc[:, 'login_ct'] = 1
login_data.sort_values('login_time')
login_data = login_data.set_index('login_time')
login_data = login_data.resample('15min').sum().reset_index()

#Isolate times for comparison (15min intervals)
login_data.loc[:, 'time'] = [d.time() for d in login_data['login_time']]
login_data['time'] = login_data['time'].apply(lambda x: x.strftime("%H:%M"))
time_data = login_data.groupby('time')['login_ct'].agg('sum').reset_index()

#Visualize Demand
#Peak times of demand
fig, ax = plt.subplots(figsize=(15,9)) 

ax.plot_date(time_data['time'], time_data['login_ct'], linestyle='-')
plt.title('Login Counts in 15 Minute Intervals', fontsize=16, fontweight='bold')
plt.xlabel('Login Time (HH:MM)', fontsize=14, fontweight='bold')
plt.ylabel('Login Count', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=14)
plt.tight_layout()
plt.show()

#Part 3 Predictive Modeling
#Load city data and convert to dataframe
with open('C:/datascience/springboard/mini projects/ultimate_challenge/ultimate_data_challenge.json') as read_file:
    city_data = json.load(read_file)
    city_data = pd.DataFrame.from_dict(city_data, orient='columns')
    
#Convert columns to datetime
city_data['last_trip_date'] = pd.to_datetime(city_data['last_trip_date'])
city_data['signup_date'] = pd.to_datetime(city_data['signup_date'])
city_data.loc[:, 'ultimate_black_user'] = np.where(city_data['ultimate_black_user'] == True, 1, 0)

#Format df
float_conv = city_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
int_conv = city_data.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
object_conv = city_data.select_dtypes(include=['object'])

#Determine if unique values in object column are less than 50% of total values. If less, change to category, otherwise leave as object
for col in object_conv.columns:
    num_unique_values = len(object_conv[col].unique())
    num_total_values = len(object_conv[col])
    if num_unique_values / num_total_values < 0.5:
        object_conv.loc[:, col] = object_conv[col].astype('category')
    else:
        object_conv.loc[:, col] = object_conv[col]
#Overlay converted columns dtypes on original city_data. This acheived as estimated 90% reduction in memory usage
city_data[float_conv.columns] = float_conv
city_data[int_conv.columns] = int_conv
city_data[object_conv.columns] = object_conv

#Remove unneeded variables
del(col, float_conv, int_conv, num_total_values, num_unique_values, object_conv)

#Deal with nans
#driver rtg
mean_rtg = city_data['avg_rating_of_driver'].mean()
city_data.loc[:, 'avg_rating_of_driver'] = np.where(city_data['avg_rating_of_driver'].isnull(), mean_rtg, city_data['avg_rating_of_driver'])

#rider rtg
mean_rtg_driver = city_data['avg_rating_by_driver'].mean()
city_data.loc[:, 'avg_rating_by_driver'] = np.where(city_data['avg_rating_by_driver'].isnull(), mean_rtg_driver, city_data['avg_rating_by_driver'])

#phone (only 0.8% loss of data)
city_data = city_data.dropna(subset=['phone'])

#Explore
#Look at potential time features to assess ranges
scat_mat = city_data.drop(columns=['city','phone','ultimate_black_user'])

#Plot using scattermatrix
pd.plotting.scatter_matrix(scat_mat, alpha=0.2, figsize=(15, 15))
plt.show()

#Look at potential time features to assess ranges
dist_df = city_data[['trips_in_first_30_days','avg_rating_of_driver','avg_surge','weekday_pct','avg_dist','avg_rating_by_driver']]

for col in dist_df.columns:

    fig, ax = plt.subplots(figsize = (9,4))
    plt.style.use('ggplot')
    counts, bins, patches = ax.hist(dist_df[col])
    ax.set_xticks(bins)
    ax.tick_params(axis='x', direction='out', labelsize=12)
    ax.tick_params(axis='y', direction='out', labelsize=12)
    ax.set_title(col, fontsize=18, weight='bold')
    ax.set_xlabel(col, fontsize=14, weight='bold')
    ax.set_ylabel('Count', fontsize=14, weight='bold')

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for patch, count, x in zip(patches, counts, bin_centers):
        #Get x and y placement of label
        y_value = patch.get_height()
        x_value = patch.get_x() + patch.get_width() / 2

        #Number of points between bar and label
        space = 3

        #Vertical alignment of positive values
        va = 'bottom'

        #If value of bar is negative: place label below
        if y_value < 0:
            space *= -1
            va = 'top'

        #Use y-value as label and format number with zero decimal place
        label = "{:.0f}".format(y_value)

        #Create annotation
        plt.annotate(label, (x_value, y_value), xytext=(0, space), textcoords='offset points', ha='center', va=va, size=12)

        #Label the issuer percentages 
        percent = '%00.2f%%' % (100* float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'), xytext=(0, -45), textcoords='offset points', 
                    va='top', ha='center', size=12)

    plt.tight_layout()
    
#Determine "Active" user
#Use 30 days from most recent trip as no specific dates given only preceding 30 days
cutoff_date = city_data['last_trip_date'].max() - relativedelta(days=30)
city_data.loc[:, 'active_user'] = np.where(city_data['last_trip_date'] >= cutoff_date, 1, 0)

#Calculate time as customer  - This dominated other features and not sure it should be included.
#city_data.loc[:, 'days_as_customer'] = (city_data['last_trip_date'] - city_data['signup_date']).dt.days

#Encode Categoricals (I found this method simpler than Scikit Learn's Binarizer function)
#Create new df that's prepped for ML usage
ml_df = city_data.copy()

#City
ml_df['kings_landing'] = np.where(ml_df['city'].str.contains("King's Landing"), 1, 0)
ml_df['astapor'] = np.where(ml_df['city'].str.contains("Astapor"), 1, 0)
ml_df['winterfell'] = np.where(ml_df['city'].str.contains("Winterfell"), 1, 0)

#Phone
ml_df['iphone'] = np.where(ml_df['phone'].str.contains("iPhone"), 1, 0)
ml_df['android'] = np.where(ml_df['phone'].str.contains("Android"), 1, 0)

#Drop Encoded columns
ml_df = ml_df.drop(columns=['city','signup_date','last_trip_date','phone'])

#Import ML Packages
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel

#Encode tgt column, specify X,Y values, scale X values, Split data into training and test sets
tgt_enc = LabelEncoder().fit(ml_df['active_user'])
tgt_encoded = tgt_enc.transform(ml_df['active_user'])
ml_df = ml_df.drop(columns='active_user')

X = ml_df.values
Y = tgt_encoded

scaled_X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Instantiate RandomForestClassifier with base parameters
model_RandTree = RandomForestClassifier(random_state=0, n_jobs=-1)
model_RandTree.fit(X_train, y_train)

#Visualize feature importance
features_RandTree = {}

for feature, importance in zip(ml_df.columns, model_RandTree.feature_importances_):
    features_RandTree[feature] = importance

importances_RandTree = pd.DataFrame.from_dict(features_RandTree, orient='index').rename(columns={0: 'Gini-Importance'})
importances_RandTree = importances_RandTree.sort_values(by='Gini-Importance', ascending=True)
importances_RandTree  = importances_RandTree[importances_RandTree['Gini-Importance'] >= 0.01]

fig, ax = plt.subplots(figsize=(15,12))
plt.barh(importances_RandTree.index, importances_RandTree['Gini-Importance'])
plt.title('Feature Importance above 1% - Random Forest Classifier', fontsize=16)
ax.set_ylabel('Features', fontsize=14)
ax.set_yticklabels(importances_RandTree.index, fontsize=12)
ax.set_xlabel('Gini-Importance', fontsize=14)
plt.tight_layout()
plt.show()

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

# Create the parameter grid based on the results of random search 
param_grid = {
              'bootstrap': [True],
              'max_depth': [25, 50, 85],
              'max_features': ['sqrt'],
              'min_samples_leaf': [1, 2, 5, 10],
              'min_samples_split': [50, 100, 150],
              'n_estimators': [500, 1000, 1500]
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

#Classification Report
print("Random Forest - Base Model:" + '\n' + classification_report(y_test, base_model_pred))
print("Random Forest - Tuned Model:" + '\n' + classification_report(y_test, tuned_model_pred))
