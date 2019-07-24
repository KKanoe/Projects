import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import itertools
from scipy import stats
import datetime
from datetime import date

#Load rental data
rental_data = pd.read_csv('C:/datascience/springboard/projects/Rental Property ROI/data/Rental Data Update.csv', parse_dates=['date'], infer_datetime_format=True) 

#Calculate rental income
VACANT_MOS = 1 #Assumed months of vacancy (1 month is standard) 
rental_data.loc[:, 'gross_rent'] = rental_data['Neighborhood_Zri_SingleFamilyResidenceRental'] * ((12 - VACANT_MOS) / 12)

#Calculate mortgage related expenses
region_list = rental_data['RegionID'].unique().tolist() 
date_list = rental_data['date'].map(pd.Timestamp.date).unique().tolist()#Use a month end reference for available dates

#Scalar values
YEARS = 30 #Mortgage length (years). Only have 30yr mortgage data right now
DOWN_PMT = 0.20 #Initial amount down on purchase (percent)

#Empyt list stores amortisation dfs for each RegionID
mort_df_list = [] 

#Creates amortisation df for each region given a purchase date (end-of month - 'YYYY-MM-DD')
def mort_pmt(df, purchase_date_eom):
    #Loops through each region ID
    purchase_date_eom = pd.to_datetime(purchase_date_eom)
    for i in region_list:
        df_region = df[df['RegionID'] == i].reset_index()
        df_region = df_region[df_region['date'] >= purchase_date_eom]
        #Skips region if data not available beginning on given purchase date 
        if df_region['date'].iloc[0] != purchase_date_eom: 
            continue
        else:
            #Calculate mortgage related components (period, principal, interest, remaining balance)
            #Dynamic scalar values
            purch_price = df_region['Neighborhood_Zhvi_SingleFamilyResidence'].iloc[0]
            mort_rate = df_region['mort_rate'].iloc[0] #Consider adding some refinancing feature
            mort_amt = (purch_price * (1-DOWN_PMT))
            #Rolling columns for key mortgage components (period, interest, principal)
            df_region.loc[:, 'period'] = np.arange(1, len(df_region)+1, 1) 
            df_region.loc[:, 'prin_pmt'] = np.ppmt(mort_rate/12, df_region['period'], YEARS*12, mort_amt)  
            df_region.loc[:, 'int_pmt'] = np.ipmt(mort_rate/12, df_region['period'], YEARS*12, mort_amt) 
            df_region.loc[:, 'cum_prin'] = df_region['prin_pmt'].cumsum()
            #Ensures balance doesn't go above loan amt 
            df_region.loc[:, 'cum_prin'] = df_region['cum_prin'].clip(lower=-mort_amt)
            df_region.loc[:, 'mort_balance'] = mort_amt + df_region['cum_prin']
            #Calculate depreciation and after-sale taxes
            df_region.loc[:, 'cum_deprec'] = (purch_price / (-27.5 * 12)) * df_region['period'] #Depreciation limit set at 27.5 years
            #Ensures principal, interest payments go to zero when paid and depreciation doesn't exceed purchase price
            df_region.loc[:, 'cum_deprec'] = np.where(df_region['cum_deprec'] <= -purch_price, -purch_price, df_region['cum_deprec'])
            df_region.loc[:, 'prin_pmt'] = np.where(df_region['mort_balance'].shift(1) == 0, 0, df_region['prin_pmt'])
            df_region.loc[:, 'int_pmt'] = np.where(df_region['mort_balance'].shift(1) == 0, 0, df_region['int_pmt'])    
            #Filters out relevant columns
            df_region.loc[:, 'book_value'] = purch_price + df_region['cum_deprec']
            df_region = df_region[['date','RegionID','period','prin_pmt','int_pmt','cum_prin','mort_balance','cum_deprec','book_value']]
            mort_df_list.append(df_region)

#Concatenate amortisation dfs and merge with rental df
mort_pmt(rental_data, '2014-01-31')
mort_data = pd.concat(mort_df_list)
rental_data = pd.merge(rental_data, mort_data, on=['RegionID','date'], how='inner')

#Operating Expenses
OP_EX = 250 #USD per month (includes estimated monthly repair/capex expenses)
MGMT_FEE_PCT = 0.08 #Percent of rent
MULTIPLIER = 0.80 #Used to calculate assessed value given market value (typically 60% to 80%)

rental_data.loc[:, 'mgmt_fee'] = rental_data['gross_rent'] * -MGMT_FEE_PCT
rental_data.loc[:, 'oper_exp'] = rental_data['size_adj_pct'] * -OP_EX
rental_data.loc[:, 'taxes'] = (rental_data['Neighborhood_Zhvi_SingleFamilyResidence'] * MULTIPLIER) * (-rental_data['county_avg']/12) 

#Net Cash Flow
rental_data.loc[:, 'net_cf'] = rental_data['gross_rent'] + rental_data['int_pmt'] + rental_data['adj_mo_ins'] + rental_data['mgmt_fee'] + rental_data['oper_exp'] + rental_data['taxes']

#After-tax proceeds (from sale at any given month)
CG_TAX_RATE = 0.25
DEP_RECAP_RATE = 0.25
SALES_COMM = 0.05 

#include improvements somewhere?
rental_data.loc[:, 'gross_sale_proceeds'] = (rental_data['Neighborhood_Zhvi_SingleFamilyResidence'] * (1-SALES_COMM))
rental_data.loc[:, 'gain_loss'] = rental_data['gross_sale_proceeds'] - rental_data['book_value']
rental_data.loc[:, 'dep_recap'] = np.where(rental_data['gain_loss'] > 0, DEP_RECAP_RATE * rental_data['cum_deprec'], 0)
rental_data.loc[:, 'cap_gain'] = rental_data['gain_loss'] + rental_data['dep_recap']
rental_data.loc[:, 'capgl_tax'] = rental_data['cap_gain'] * CG_TAX_RATE #Can only deduct loss of up to $25K/year if income below $100K
rental_data.loc[:, 'after_tax_proceeds'] = rental_data['gross_sale_proceeds'] - rental_data['mort_balance'] - rental_data['capgl_tax'] 
rental_data.loc[:, 'cfs_plus_sale'] = rental_data['after_tax_proceeds'] + rental_data['net_cf'] 

rental_data = rental_data.reset_index()
rental_data = rental_data.set_index('RegionName')
rental_view = rental_data.loc[['Southeast Como']].reset_index()

region_list = rental_data['RegionID'].unique().tolist() 
#Calculate IRR. Input dataframe and month of expected sale of property 
region_dfs = []

def irr_calc(df, sale_month):
    #Iterate through each region 
    for i in region_list:
        df_new = df[df['RegionID'] == i].reset_index()
        if sale_month > len(df_new):
            print('Not enough data!')
            break
        else:        
            initial_inv = df_new['Neighborhood_Zhvi_SingleFamilyResidence'].iloc[0] * -DOWN_PMT
            sale_cf = df_new['cfs_plus_sale'].iloc[sale_month-1]
            cf_list = [initial_inv]
        #Create running list of cfs to input into np.irr function
        for index, row in df_new.iterrows():
            if index < sale_month-1:
                cf_list.append(row['net_cf'])
                df_new.loc[index, 'cf_irr'] = np.irr(cf_list)
            else: 
                cf_list.append(row['cfs_plus_sale'])
                df_new.loc[index, 'cf_irr'] = np.irr(cf_list)   
        region_dfs.append(df_new)

rental_data_new = pd.concat(region_dfs)
          
#As described in observation one above. Let's begin by comparing each state to see if any signifcant differences occur amongst
#price-to-rent ratios. Additionally, removing all properties with price-to-rent ratios above 14 (lower is better).
rental_subset = rental_data[(rental_data['Neighborhood_PriceToRentRatio_AllHomes'] <= 1000) & 
                            (rental_data['State'] == 'MI')].sort_values('date')

#Populate list of differences between statistics for all possible combinations (mean, std)
#Create empty list for each combination metric
mean_diff = []
std_diff = []
feature_combo = []

#Loop through df and perform desired calculation for each column 
def feature_comp(df, feature_col):     
    
    #Get rollup statistics for desired feature (mean, std, samp_var, count
    df_agg = df.groupby(feature_col)['net_cf'].agg(['mean', 'std','count']).reset_index()
    df_agg['samp_var'] = df_agg['std'] **2 / df_agg['count']
    
    #Iterate through all combinations of given feature calculating differences for mean, std 
    for idx, col in enumerate(df_agg.columns): 
        for a, b in itertools.combinations(df_agg[col], 2):
            if col == feature_col:
                feature_combo.append((a, b))
            elif col == str('mean'):
                mu_diff = a - b 
                mean_diff.append(mu_diff)
            elif col == str('samp_var'):
                sigma_diff = float(np.sqrt([a + b])) 
                std_diff.append(sigma_diff)
            else:
                continue
#Input desired df and feature_col
feature_comp(rental_subset, 'RegionID')
                
#Form new df for each combination, then calculate z_score, p_value, and determine is Ho can be rejected.
feature_df = pd.DataFrame({'feature_combo':feature_combo, 'mean_diff':mean_diff, 'std_diff':std_diff})  
feature_df.loc[:, 'z_score'] = feature_df['mean_diff'] / feature_df['std_diff']
feature_df.loc[:, 'p_value'] = feature_df['z_score'].apply(lambda x: round(stats.norm.sf(abs(x)),4)*2) #Two-tailed
feature_df.loc[:, 'reject_null'] = feature_df['p_value'].apply(lambda x: 'Yes' if x < 0.01 else 'No') #Significance @ 1%

#Print summary df where null hypothesis was rejected
print(df_agg)
reject_null_df = feature_df[feature_df['reject_null'] == 'Yes']
print('Percent significant of total combinations: %.3f' % (len(reject_null_df) / len(mean_diff)))
print(reject_null_df[['feature_combo','z_score','p_value','reject_null']].sort_values('z_score'))

