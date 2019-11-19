import pandas as pd
import glob
import re
import ntpath
from functools import reduce
import datetime as dt

#Place filenames in list
filenames=glob.glob('C:/datascience/springboard/projects/Venture Capital/data/*.csv')

#Load files and place each in list
df_container = []
df_names = []
flat_dfs = []


#The primary goal for wrangling this dataset is to get all relevant data at the Company level (i.e. one row per company)
#This approach results in the need to take the Investments/Rounds df's from long to wide format for select columns 
#Loop through filenames
for file in filenames:
    
    #Extract filename to format specific dfs
    df_name = re.sub(r'.csv',"", ntpath.basename(file))
    df_names.append(df_name)
    #Crunchbase companies df needs additional column formatting in order to merge with other dfs 
    if df_name == 'crunchbase-companies':
        
        #Load df (special encoding given characters in dataset)
        df = pd.read_csv(file, encoding="ISO-8859-1") 
        
        #Format column names
        df = df.rename(columns={'permalink':'company_permalink','name':'company_name','category_code':'company_category_code',
                                'country_code':'company_country_code','state_code':'company_state_code','region':'company_region',
                                'city':'company_city'})
        
        #Drop rows where company name or status (target) is missing
        df = df.dropna(subset=['company_name'])
        
        #Append to empty df list
        df_container.append(df)
    
    elif df_name == 'crunchbase-acquisitions':
        
        #Load df (special encoding given characters in dataset)
        df = pd.read_csv(file, encoding="ISO-8859-1")
       
        #Drop rows where company name is missing
        df = df.dropna(subset=['company_name'])
        
        #Create new status columns for merging later
        df.loc[:, 'status'] = 'acquired'
        
        #Append to empty df list
        df_container.append(df)
    
    elif df_name == 'crunchbase-investments':
        
        #Load df (special encoding given characters in dataset)
        df = pd.read_csv(file, encoding="ISO-8859-1")
       
        #Drop rows where company name is missing
        df = df.dropna(subset=['company_name','funded_at'])
        
        #Create investor score
        #Format amount to be more reader friendly and sort for faster operations
        df.loc[:, 'raised_amount_usd'] = df['raised_amount_usd']/1000000
        df = df.sort_values(['investor_name','funded_at'])
        
        #Calculate cumulative amount raised and rank based on amount raised at that given date 
        df.loc[:, 'investor_cum_raised_amt'] = df.groupby('investor_name')['raised_amount_usd'].transform('cumsum')
        df.loc[:, 'num_investors'] = df.groupby(['company_name','funded_at'])['investor_name'].transform('count')
        df['investor_pct'] = df['investor_cum_raised_amt'].rank(pct=True)
        
        #Create unique name for each unique funding round (pivot table use)
        df = df.sort_values(['company_name','funded_at'])
        df.loc[:, 'counter'] = df.groupby('company_name')['funded_month'].transform(lambda x: pd.CategoricalIndex(x).codes)+1
        df['unique_investors'] = 'invscore_' + df['counter'].astype(str)
        df['unique_investors2'] = 'invct_' + df['counter'].astype(str)
        
        #Calculate investor score for each round for given date
        df.loc[:, 'co_investor_score'] = round(df.groupby(['company_name','unique_investors'])['investor_pct'].transform('mean'), 3) #Should this be by round?

        #Append to empty df list
        df_container.append(df)
        
        #Flat format df 
        #Investor Scores
        inv_piv = pd.pivot_table(df, values='co_investor_score', index='company_name', columns='unique_investors').reset_index()
        
        #Number of investors in each round
        inv_piv2 = pd.pivot_table(df, values='num_investors', index='company_name', columns='unique_investors2').reset_index()
        
        #Append wide format dfs to list
        flat_dfs.append(inv_piv)
        flat_dfs.append(inv_piv2)
        
    elif df_name == 'crunchbase-rounds':
        
        #Load df (special encoding given characters in dataset)
        df = pd.read_csv(file, encoding="ISO-8859-1")
       
        #Drop rows where company name is missing
        df = df.dropna(subset=['company_name'])
       
        #Fromat/Create new columns for merging later
        df.loc[:, 'raised_amount_usd'] = df['raised_amount_usd'] / 1000000
        
        #Append to empty list
        df_container.append(df)
        
        #Format to wide format df to merge with company level data
        #Transform/Format second wide format df for rounds data
        df = df[['company_name','raised_amount_usd','funding_round_type','funded_at']]
        
        #Implement counter to create unique company names (i.e. duplicate company names for rounds don't allow for pivot)
        df = df.sort_values(['company_name','funded_at'])
        df['counter'] = df.groupby('company_name').cumcount()+1
        df['unique_name'] = df['company_name'] + '_' + df['funding_round_type'] + '_' + df['counter'].astype(str)
        df['unique_rds_amt'] = 'rdamt_' + df['counter'].astype(str)
        df['unique_rds_dt'] = 'rddt_' + df['counter'].astype(str)
        df['unique_rds_dt_diff'] = 'rddt_diff_' + df['counter'].astype(str)
        
        #Convert funded at column to dateteime and caluclate day differences between rounds
        df['funded_at'] = df['funded_at'].apply(pd.to_datetime)
        df['rd_day_diff'] = df.groupby('company_name')['funded_at'].transform(lambda x: x.diff())
        df['rd_day_diff'] = df['rd_day_diff'].dt.days
        
        #Create pivot tables
        rds_piv = pd.pivot_table(df, values='raised_amount_usd', index='company_name', columns='unique_rds_amt').reset_index()
        rds_piv2 = df.pivot(values='funded_at', index='unique_name', columns='unique_rds_dt').reset_index()
        rds_piv3 = df.pivot(values='rd_day_diff', index='company_name', columns='unique_rds_dt_diff').reset_index()
        
        #Re-arrange order of columns 
        rds_piv = rds_piv[['company_name','rdamt_1','rdamt_2','rdamt_3','rdamt_4','rdamt_5','rdamt_6','rdamt_7','rdamt_8','rdamt_9',
                           'rdamt_10', 'rdamt_11', 'rdamt_12', 'rdamt_13','rdamt_14']]
        
        rds_piv2 = rds_piv2[['unique_name','rddt_1','rddt_2','rddt_3','rddt_4','rddt_5','rddt_6','rddt_7','rddt_8','rddt_9',
                             'rddt_10', 'rddt_11', 'rddt_12', 'rddt_13','rddt_14']]
        
        rds_piv3 = rds_piv3[['company_name','rddt_diff_1','rddt_diff_2','rddt_diff_3','rddt_diff_4','rddt_diff_5','rddt_diff_6','rddt_diff_7','rddt_diff_8','rddt_diff_9',
                             'rddt_diff_10', 'rddt_diff_11', 'rddt_diff_12', 'rddt_diff_13','rddt_diff_14']]
       
        #Prepare and merge pivoted df's with company name
        df = df[['unique_name','company_name']]
        df = pd.merge(df, rds_piv2, on='unique_name').set_index(['company_name','unique_name'])
        df = df.groupby('company_name').apply(lambda x: x.ffill(axis=0).bfill(axis=0)).reset_index()
        df = df.drop_duplicates('company_name')
        
        #Merge first and second wide format df
        df = pd.merge(df, rds_piv3, on='company_name', how='inner')
        df = pd.merge(df, rds_piv, on='company_name', how='inner')
        
        #Append flat df to empty list
        flat_dfs.append(df)
       
    else:
        
        continue

#Merge Acquisitons and Companies dataframes   
comp_df = pd.merge(df_container[1], df_container[0], on=['company_name','company_permalink','company_category_code',
                                                         'company_country_code','company_state_code','company_region',
                                                         'company_city','status'], how='outer')

#Merge wide format dfs 
flat_df = reduce(lambda left, right: pd.merge(left, right, on='company_name', how='outer'), flat_dfs)

#Combine wide format df with company df
comp_df = pd.merge(comp_df, flat_df, on='company_name', how='outer')

#Drop duplicate company names
comp_df = comp_df.drop_duplicates('company_name')

#Format date columns
comp_df[['founded_at','first_funding_at','last_funding_at','acquired_at','last_milestone_at']] = comp_df[['founded_at','first_funding_at','last_funding_at',
                                                                                                          'acquired_at','last_milestone_at']].apply(pd.to_datetime, format = '%Y/%m/%d') #mm/dd/yyyy

#Format merged df
float_conv = comp_df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
int_conv = comp_df.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
object_conv = comp_df.select_dtypes(include=['object'])

#Determine if unique values in object column are less than 50% of total values. If less, change to category, otherwise leave as object
for col in object_conv.columns:
    num_unique_values = len(object_conv[col].unique())
    num_total_values = len(object_conv[col])
    if num_unique_values / num_total_values < 0.5:
        object_conv.loc[:, col] = object_conv[col].astype('category')
    else:
        object_conv.loc[:, col] = object_conv[col]
#Overlay converted columns dtypes on original comp_df. This acheived as estimated 90% reduction in memory usage
comp_df[float_conv.columns] = float_conv
comp_df[int_conv.columns] = int_conv
comp_df[object_conv.columns] = object_conv

#Remove unneeded variables
del(col, df, df_name, file, flat_df, flat_dfs, float_conv, int_conv, inv_piv, num_total_values, num_unique_values, object_conv)
   
#Calculate time features
comp_df.loc[:, 'found_to_fund_days'] = comp_df['first_funding_at'] - comp_df['founded_at']
comp_df.loc[:, 'first_to_last_fund_days'] = comp_df['last_funding_at'] - comp_df['first_funding_at'] 
comp_df.loc[:, 'last_to_acq_days'] = comp_df['acquired_at'] - comp_df['last_funding_at']
comp_df.loc[:, 'found_to_acq_days'] = comp_df['acquired_at'] - comp_df['founded_at']
comp_df.loc[:, 'first_fund_to_acq_days'] = comp_df['acquired_at'] - comp_df['first_funding_at']

#Codify feature categorical columns 
comp_df['category_code'] = comp_df['company_category_code'].cat.codes
comp_df['region_code'] = comp_df['company_region'].cat.codes

comp_df.to_csv('C:/datascience/springboard/projects/Venture Capital/data/Exported Data/Company summary df.csv', encoding="ISO-8859-1")   
      
