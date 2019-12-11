#Import packages
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#Load Data
engagement = pd.read_csv('C:/datascience/springboard/mini projects/relax_challenge/takehome_user_engagement.csv',
                         parse_dates=['time_stamp'], infer_datetime_format=True)

users = pd.read_csv('C:/datascience/springboard/mini projects/relax_challenge/takehome_users.csv', encoding="ISO-8859-1",
                    parse_dates=['creation_time'], infer_datetime_format=True)


adopt_df = engagement

#Change time_stamp to string object to remove duplicates (ignoring hours:mins:seconds) 
adopt_df['time_stamp'] = adopt_df['time_stamp'].dt.date
adopt_df = adopt_df.drop_duplicates(subset=['user_id','time_stamp'])
adopt_df = adopt_df.sort_values(['user_id','time_stamp'])

#Count number of daily logins for each week
#Convert back to datetime
adopt_df['time_stamp'] = pd.to_datetime(adopt_df['time_stamp'])

#Calculate day differences and rolling logins over 3 day window
adopt_df['day_diff'] = adopt_df.groupby('user_id')['time_stamp'].transform(lambda x: x.diff())
adopt_df['day_diff'] = adopt_df['day_diff'].dt.days

#Rolling window sums time days btw logins and requires at least three days, but no more than 7 days
adopt_df = adopt_df.groupby('user_id')['day_diff'].rolling(min_periods=3, window=3).sum().reset_index()
adopt_df = adopt_df[(adopt_df['day_diff'] <= 7) & (adopt_df['day_diff'] >= 3)]

#Send adopted user_ids to list for comparison to users dataframe
adopted_user_ids = adopt_df['user_id'].unique().tolist() 

#Place 0 of 1 in newly created adopted user column as target
users.loc[:, 'adopted_user'] = users['object_id'].isin(adopted_user_ids).astype(int)

#Deal with nan's in users df
#Swap in last engagement date for nan's in users df
engagement = engagement.sort_values(['user_id','time_stamp'])
last_session_nan = engagement.groupby('user_id')['time_stamp'].last().reset_index()

#Merge last session nan with users df
last_session_nan = last_session_nan.rename(columns={'user_id':'object_id','time_stamp':'last_known_login'})
users = pd.merge(users, last_session_nan, on='object_id', how='outer')

#Swap in last_session_nan value where na existed in user df - THIS DIDN'T PRODUCE ANY NEW RECORDS
users['last_session_creation_time'] = np.where(users['last_session_creation_time'].isnull(), 
                                               users['last_known_login'], users['last_session_creation_time'])

#Drop as no longer needed
users = users.drop(columns=['last_known_login'])

#Replace na with zero for invited by user id and last_session_creation_time (i.e. not invited by another user or have never logged in)
users[['last_session_creation_time','invited_by_user_id']] = users[['last_session_creation_time','invited_by_user_id']].fillna(0)

#Create new feature: days(int) between creation time and last_session_creation_time
users['last_session_creation_time'] = users['last_session_creation_time'].apply(lambda x: datetime.fromtimestamp(x))

#Calculate days (as int) between creation date and last login
users.loc[:, 'days_creation_login'] = (users['last_session_creation_time'] - users['creation_time']).dt.days

#Replace negative values with zero (by-product of never logging in)
users['days_creation_login'] = np.where(users['days_creation_login'] < 0, 0, users['days_creation_login'])

#Encode categoricals in users df
users['creation_source_code'] = users['creation_source'].astype('category').cat.codes

#Isolate feature and target columns
users_ml = users[['adopted_user','creation_source_code','days_creation_login','invited_by_user_id','org_id',
                  'enabled_for_marketing_drip','opted_in_to_mailing_list']]

#Scale/transform data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#Convert df to array and perform one-hot encoding for target variable 'status'
tgt_enc = LabelEncoder().fit(users_ml['adopted_user'])
tgt_encoded = tgt_enc.transform(users_ml['adopted_user'])
np.unique(tgt_enc.inverse_transform(tgt_encoded))
np.unique(tgt_encoded)
users_ml = users_ml.drop(columns='adopted_user')

X = users_ml.values
Y = tgt_encoded

#option to scale data
#scaled_X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

model_RandTree = RandomForestClassifier(random_state=0, n_jobs=-1)
model_RandTree.fit(X_train, y_train)

#Visualize feature importance
features_RandTree = {}

for feature, importance in zip(users_ml.columns, model_RandTree.feature_importances_):
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

