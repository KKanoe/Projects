import pandas as pd
import glob
import re
import ntpath
from functools import reduce
from pandas.tseries.offsets import BMonthEnd

#Place filenames in list
filenames= glob.glob('C:/datascience/springboard/projects/Rental Property ROI/data/csv/*.csv')

#Loop through filenames and store each df in a list. I choose this method to allow for efficient/flexible loading of additional data if I wished at a later time. 
file_container = []
file_names = []

for idx, file in enumerate(filenames):
    
    if idx == 0: #We only need (RegionName, City, State, Metro, CountyName, SizeRank columns for the first df)
        #Read files and extract filename for reference
        df_name = re.sub(r'.csv',"", ntpath.basename(file))
        df = pd.read_csv(file)
        #Re-orient the dataframe from wide format (dates as columns) to long format (dates as rows)
        df.set_index(['RegionID','RegionName','City','State','Metro','CountyName','SizeRank'], inplace=True)
        df = df.stack(level=0).reset_index().rename(columns={'level_7':'date', 0:df_name})
        #Format selected columns
        df.loc[:, 'date'] = pd.to_datetime(df['date'], format = '%Y-%m')  
        df.loc[:, 'Metro'] = df['Metro'].str.replace(r'/|-', "_").str.replace(r'.',"")
        file_names.append(df_name) #Storing filenames in list provides name reference for file_container
        file_container.append(df)
        del(df, df_name, file)
    else: 
        #Read files and extract filename for reference 
        df_name = re.sub(r'.csv',"", ntpath.basename(file))
        df = pd.read_csv(file)
        #This drops duplicate columns names, which will be useful for joining into one df later
        df = df.drop(columns=['RegionName','City','State','Metro','CountyName','SizeRank'])
        #Re-orient the dataframe from wide format (dates as columns) to long format (dates as rows)
        df.set_index('RegionID', inplace=True)
        df = df.stack(level=0).reset_index().rename(columns={'level_1':'date', 0:df_name})
        #Format selected columns
        df.loc[:, 'date'] = pd.to_datetime(df['date'], format = '%Y-%m')
        file_names.append(df_name) #Storing filenames in list provides name reference for file_container
        file_container.append(df)
        del(df, df_name, file)
        
#Compress df to preserve memory and convert column types for merging
for df in file_container:
    float_conv = df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    int_conv = df.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
    object_conv = df.select_dtypes(include=['object'])

    #Determine if unique values in object column are less than 50% of total values. If less, change to category, otherwise leave as object
    for col in object_conv.columns:
        num_unique_values = len(object_conv[col].unique())
        num_total_values = len(object_conv[col])
        if num_unique_values / num_total_values < 0.5:
            object_conv.loc[:, col] = object_conv[col].astype('category')
        else:
            object_conv.loc[:, col] = object_conv[col]
    #Overlay converted columns dtypes on original df. This acheived as estimated 90% reduction in memory usage
    df[float_conv.columns] = float_conv
    df[int_conv.columns] = int_conv
    df[object_conv.columns] = object_conv
    
#Combine dfs into one df for comparison
rental_data = reduce(lambda left, right: pd.merge(left, right, on=['RegionID','date'], how='outer'), file_container)

del(col, df, file_container, file_names, filenames, float_conv, idx, int_conv, num_total_values, num_unique_values, object_conv)

#Clean and refine combined data
#The most important data points for the analysis is Median Values (SqFt) and Rental Prices (SqFt) and
#Rental data wasn't available until 2010-11, so that's the starting date I used
rental_data = rental_data.sort_values(['RegionID','date'])
rental_data = rental_data[rental_data['date'] >= '2011-10-01']

#Next step - check for Nan's - I iterate through these lines of code to keep checking for NaN's after some filtering in the next section
#Gives a nice summary and we can see many nan's, which warrants further inspection
#rental_data.isnull().sum() #ARIPerSqFt is a key dataset for me and has highest na count, so I'm going to use that as my na_count target
#months_of_data = rental_data['date'].nunique() #Determine what a complete dataset looks like for each RegionID (91 months in this case)
#na_rows = rental_data[(rental_data['Neighborhood_MedianValuePerSqft_AllHomes'].isnull()) | (rental_data['Neighborhood_ZriPerSqft_AllHomes'].isnull())] #Return all na rows for each key columns of data
rental_data.loc[:, 'rent_count'] = rental_data.groupby('RegionID')['Neighborhood_ZriPerSqft_AllHomes'].transform('count') 
rental_data.loc[:, 'value_count'] = rental_data.groupby('RegionID')['Neighborhood_MedianValuePerSqft_AllHomes'].transform('count') 

#NaN filtering - Some NaN's remain in other columns, but I'm choosing to ignore them for this analysis since I'm primarily concerned with the SqFt data
rental_data = rental_data[rental_data['rent_count'] != 0] #Approx. 28% of the data didn't have any rental (per Sq Ft) information. This isn't alarming because many neighborhoods don't have a prevalant rental market. Reduced NaN's from 200K+ to 1731.
rental_data = rental_data[rental_data['value_count'] != 0] #Approx. 14% of the data didn't have any value (per Sq Ft) information. This is more difficult to explain away.
rental_data = rental_data.dropna(axis=0, subset=['Neighborhood_ZriPerSqft_AllHomes','Neighborhood_MedianValuePerSqft_AllHomes']) # I decided to drop the remaining incomplete columns. I lost less than 1% of additional data and the incomplete datasets removed potential bias in the data. Alternatively, I could have inferred some data, but felt it wouldn't have added significantly to the analysis.

#Load/Format/Merge Mortgage Data 
rates = pd.read_csv('C:/datascience/springboard/projects/Rental Property ROI/data/MortgageRateConventionalFixed.csv')
rates = rates.rename(columns={'Date':'date', 'MortgageRateConventionalFixed':'mort_rate'})
rates['mort_rate'] = rates['mort_rate'].transform(lambda x: x.ffill().bfill()) #Periodic days were missing mortgage rates in the data. Filling with recent rate is not likely to introduce significant error and will help improve overall analysis (Positives > Negatives).
rates.loc[:,'date'] = pd.to_datetime(rates['date'], format = '%Y-%m-%d')
rates = rates.groupby('date')['mort_rate'].mean().reset_index() #Take average of four daily measurements
rates = rates[rates['date'] >= '2011-10-01']
rental_data.loc[:, 'date'] = pd.to_datetime(rental_data['date'], format = '%Y-%m') + BMonthEnd(0) #Converts existing dates to business end of month (not beg. of month). I tried to do this earlier, but it screwed up merge.
rental_data = pd.merge(rental_data, rates, on=['date']) #No NaN's on merge

#Load/Format/Merge Tax Data
states = rental_data['State'].sort_values().unique().tolist() #Send all state codes I need for data to list (these are names of each sheet in excel file with County Property Tax Data)
df_cont = []

for i in states:
    df = pd.read_excel('C:/datascience/springboard/projects/Rental Property ROI/data/CountyPropertyTaxes.xls', skiprows=2, sheet_name=i)
    df = df.rename(columns={'Unnamed: 0':'county_fip', 'Unnamed: 1':'CountyName', 'Unnamed: 2':'avg_home_val','Unnamed: 3':'avg_re_tax',
                            'County Average':'county_avg', 'High-Low Ratio':'hi_lo_ratio', 'Highest Tract':'highest_tract'})
    df.loc[:, 'State'] = str(i) 
    df['CountyName'] = df['CountyName'].str.split(',', expand=True) #Remove full state name with county and keep county name only
    df_cont.append(df)

tax_data = pd.concat(df_cont)
tax_data = tax_data.dropna(axis=0, subset=['CountyName'])

#Minor text changes to get merges to conform
tax_data['CountyName'] = tax_data['CountyName'].str.replace('St.', 'Saint').str.replace('city','City')
tax_data['CountyName'] = tax_data['CountyName'].str.replace('Anchorage Municipality','Anchorage Borough')

#Merge tax data with rental data
rental_data = pd.merge(rental_data, tax_data, on=['CountyName','State']) #By doing an inner join here, we lose about 4% of our data due to tax data not being available for all counties in the rental data
rental_data_short = rental_data[['date','RegionID','State','City','CountyName','RegionName','SizeRank','Neighborhood_MedianValuePerSqft_AllHomes',
                                 'Neighborhood_ZriPerSqft_AllHomes','rent_count','value_count','mort_rate','avg_re_tax','avg_home_val',
                                 'County Average','High-Low Ratio','Highest Tract']]

#Send files to csv for easy import later
rental_data.to_csv('C:/datascience/springboard/projects/Rental Property ROI/data/Rental Data.csv') #This file contains 7 addiitonal columns of data, which contain varying levels of NaN's
rental_data.to_csv('C:/datascience/springboard/projects/Rental Property ROI/data/Rental Data Clean.csv', index=False) #This file contains all columns I plan on using with no NaN's








