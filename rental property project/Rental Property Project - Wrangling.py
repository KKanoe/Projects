import pandas as pd
import glob
import re
import ntpath
from functools import reduce
from pandas.tseries.offsets import BMonthEnd

#Place filenames in list
filenames=glob.glob('C:/datascience/springboard/projects/Rental Property ROI/data/csv/*.csv')

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
#The most important data points for the analysis is Median Values and Rental Prices
#Rental data wasn't available until 2010-11, so that's the starting date I used
rental_data = rental_data.sort_values(['RegionID','date'])
rental_data = rental_data[rental_data['date'] >= '2011-10-01']

#Next step - check for Nan's - I iterate through these lines of code to keep checking for NaN's after some filtering in the next section
#Gives a nice summary and we can see many nan's, which warrants further inspection
rental_data.isnull().sum() #ARIPerSqFt is a key dataset for me and has highest na count, so I'm going to use that as my na_count target
#Forward fill any missing months (likely not to be big gaps so previous month is good estimate). No backfill as each Region may have varying starting points.
rental_data['Neighborhood_Zhvi_SingleFamilyResidence'] = rental_data.groupby('RegionID')['Neighborhood_Zhvi_SingleFamilyResidence'].transform(lambda x: x.ffill())
rental_data['Neighborhood_Zri_SingleFamilyResidenceRental'] = rental_data.groupby('RegionID')['Neighborhood_Zri_SingleFamilyResidenceRental'].transform(lambda x: x.ffill())
rental_data['Neighborhood_PriceToRentRatio_AllHomes'] = rental_data.groupby('RegionID')['Neighborhood_PriceToRentRatio_AllHomes'].transform(lambda x: x.ffill()) #This field is no longer be provided by Zillow for free
rental_data.loc[:, 'rent_count'] = rental_data.groupby('RegionID')['Neighborhood_Zhvi_SingleFamilyResidence'].transform('count') 
rental_data.loc[:, 'value_count'] = rental_data.groupby('RegionID')['Neighborhood_Zri_SingleFamilyResidenceRental'].transform('count') 

#NaN filtering - Some NaN's remain in other columns, but I'm choosing to ignore them for this analysis since I'm primarily concerned with the SqFt data
rental_data = rental_data[rental_data['rent_count'] != 0] #Approx. 28% of the data didn't have any rental information. This isn't alarming because many neighborhoods don't have a prevalant rental market. Reduced NaN's from 200K+ to 1731.
rental_data = rental_data[rental_data['value_count'] != 0] #Approx. 14% of the data didn't have any value information. This is more difficult to explain away.
rental_data = rental_data.dropna(axis=0, subset=['Neighborhood_Zhvi_SingleFamilyResidence','Neighborhood_Zri_SingleFamilyResidenceRental','RegionName']) # I decided to drop the remaining incomplete columns. I lost less than 1% of additional data and the incomplete datasets removed potential bias in the data. Alternatively, I could have inferred some data, but felt it wouldn't have added significantly to the analysis.

#Load/Format/Merge Mortgage Data 
rates = pd.read_csv('C:/datascience/springboard/projects/Rental Property ROI/data/MortgageRateConventionalFixed.csv')
rates = rates.rename(columns={'Date':'date', 'MortgageRateConventionalFixed':'mort_rate'})
rates['mort_rate'] = rates['mort_rate'].transform(lambda x: x.ffill().bfill()) #Periodic days were missing mortgage rates in the data. Filling with recent rate is not likely to introduce significant error and will help improve overall analysis (Positives > Negatives).
rates['mort_rate'] = rates['mort_rate'] / 100
rates.loc[:,'date'] = pd.to_datetime(rates['date'])
rates = rates.groupby('date')['mort_rate'].mean().reset_index() #Take average of four daily measurements
rates = rates[rates['date'] >= '2011-10-01']
rental_data.loc[:, 'date'] = pd.to_datetime(rental_data['date'], format = '%Y-%m') + BMonthEnd(0) #Converts existing dates to business end of month (not beg. of month). I tried to do this earlier, but it screwed up merge.

rental_data = pd.merge(rental_data, rates, on=['date'], how='inner') #No NaN's on merge

#Load/Format/Merge Tax Data
states = rental_data['State'].sort_values().unique().tolist() #Send all state codes I need for data to list (these are names of each sheet in excel file with County Property Tax Data)
df_cont = []

for i in states:
    df = pd.read_excel('C:/datascience/springboard/projects/Rental Property ROI/data/CountyPropertyTaxes.xls', skiprows=2, sheet_name=i)
    df = df.rename(columns={'Unnamed: 0':'county_fip', 'Unnamed: 1':'CountyName', 'Unnamed: 2':'avg_home_val','Unnamed: 3':'avg_re_tax',
                            'County Average':'county_avg', 'High-Low Ratio':'hi_lo_ratio', 'Lowest Tract ':'lowest_tract', 'Highest Tract':'highest_tract'})
    df.loc[:, 'State'] = str(i) 
    df['CountyName'] = df['CountyName'].str.split(',', expand=True) #Remove full state name with county and keep county name only
    df_cont.append(df)

tax_data = pd.concat(df_cont)
tax_data = tax_data.dropna(axis=0, subset=['CountyName'])

#Minor text changes to get merges to conform
tax_data['CountyName'] = tax_data['CountyName'].str.replace('St.', 'Saint').str.replace('city','City')
tax_data['CountyName'] = tax_data['CountyName'].str.replace('Anchorage Municipality','Anchorage Borough')
tax_data['county_avg'] = tax_data['county_avg'] / 1000 

#Merge tax data with rental data
rental_data = pd.merge(rental_data, tax_data, on=['CountyName','State'], how='inner') #By doing an inner join here, we lose about 4% of our data due to tax data not being available for all counties in the rental data

#Load State Insurance Data
ins_data = pd.read_csv('C:/datascience/springboard/projects/insurance_data.csv', encoding='windows-1252')
rental_data = pd.merge(rental_data, ins_data, on=['State'], how='inner')

#Adjust estimated insurance expenses (adjusts for size of properties by Region and for time)
ins_mon_adj = []

for i in states:
    df = rental_data[rental_data['State'] == i]
    df.loc[:, 'state_avg_medval'] = df.groupby('date')['Neighborhood_Zhvi_SingleFamilyResidence'].transform('mean')
    #Calculates state avg by date (adjusts for value relative to state average)
    df.loc[:, 'size_adj_pct'] = df['Neighborhood_Zhvi_SingleFamilyResidence'] / df['state_avg_medval'] 
    #Divides historical value by most recent value (corresponds with quoted monthly insurance for 2019)
    df.loc[:, 'time_adj_pct'] = df.groupby('RegionID')['Neighborhood_Zhvi_SingleFamilyResidence'].apply(lambda x: (x / x.iloc[-1])) 
    #Adjusts statewide average insurance rates for value of home in specified region as well as time
    df.loc[:, 'adj_mo_ins'] = -df['monthly_insurance'] * df['size_adj_pct'] * df['time_adj_pct'] #Uses 
    df = df[['date','RegionID','state_avg_medval','size_adj_pct','time_adj_pct','adj_mo_ins']]
    ins_mon_adj.append(df)

#Concatenate amortisation dfs and merge with rental df
adj_ins_data = pd.concat(ins_mon_adj)
rental_data = pd.merge(rental_data, adj_ins_data, on=['RegionID','date'], how='inner')

#Calculate monthly change in value (rent and median home values)
rental_data = rental_data.sort_values(['RegionID','date'])
rental_data['zhvi_sfh_pct_chg'] = rental_data.groupby('RegionID')['Neighborhood_Zhvi_SingleFamilyResidence'].transform(lambda x: x.pct_change())
rental_data['zri_sfh_pct_chg'] = rental_data.groupby('RegionID')['Neighborhood_Zri_SingleFamilyResidenceRental'].transform(lambda x: x.pct_change())
rental_data['zhvi_sfh_pchg'] = rental_data.groupby('RegionID')['Neighborhood_Zhvi_SingleFamilyResidence'].transform(lambda x: x.diff())
rental_data['zri_sfh_pchg'] = rental_data.groupby('RegionID')['Neighborhood_Zri_SingleFamilyResidenceRental'].transform(lambda x: x.diff())

#Calculate New Price to Rent Ratio for Single Family Homes
rental_data['price_to_rent_sfh'] = rental_data['Neighborhood_Zhvi_SingleFamilyResidence'] / rental_data['Neighborhood_Zri_SingleFamilyResidenceRental']

#Send files to csv for easy import later
rental_data.to_csv('C:/datascience/springboard/projects/Rental Property ROI/data/Rental Data Original.csv', index=False) 

#Load Latest Data and Connect new features for clustering portion. This is to be used for ad-hoc data additions
file = 'C:/datascience/springboard/projects/Rental Property ROI/data/Listings Sales/Sale_Counts_Neighborhood.csv'

df = pd.read_csv(file, encoding='windows-1252')
df_name = re.sub(r'.csv',"", ntpath.basename(file))
#Re-orient the dataframe from wide format (dates as columns) to long format (dates as rows)
df.set_index(['RegionID', 'RegionName', 'StateName', 'SizeRank'], inplace=True)
df = df.stack(level=0).reset_index().rename(columns={'level_4':'date', 0:df_name})
#Subset to latest date
df = df[df['date'] == '2018-12']
#Format selected columns
df.loc[:, 'date'] = pd.to_datetime(df['date'], format = '%Y-%m')  

df.loc[:, 'Metro'] = df['Metro'].str.replace(r'/|-', "_").str.replace(r'.',"")
file_names.append(df_name) #Storing filenames in list provides name reference for file_container
file_container.append(df)
del(df, df_name, file)


#Place filenames in list
filenames=glob.glob('C:/datascience/springboard/projects/Rental Property ROI/data/csv/Additional Files/*.csv')

file_container = []
file_names = []

for file in filenames:
    #Read files and extract filename for reference
    df_name = re.sub(r'.csv',"", ntpath.basename(file))
    df = pd.read_csv(file)
    #Re-orient the dataframe from wide format (dates as columns) to long format (dates as rows)
    df.set_index(['RegionID','RegionName','RegionType','StateName','SizeRank','MSA','MSARegionID'], inplace=True)
    df = df.stack(level=0).reset_index().rename(columns={'level_7':'date', 0:df_name})
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
bs_idx_data = reduce(lambda left, right: pd.merge(left, right, on=['RegionID','RegionName','RegionType','StateName','SizeRank','MSA','MSARegionID','date'], how='outer'), file_container)

del(col, df, file_container, file_names, filenames, float_conv, int_conv, num_total_values, num_unique_values, object_conv)

bs_idx_data = bs_idx_data[bs_idx_data['date'] == '2019-05-01']
