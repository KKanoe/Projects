import matplotlib.pyplot as plt
    import seaborn as sns; sns.set(color_codes=True)

#unpack df's in list to more obvious names
acq_df = df_container[0]
co_df = df_container[1]
inv_df = df_container[2]
rds_df = df_container[3]
rds_df.loc[:, 'raised_amount_usd'] = rds_df.loc[:, 'raised_amount_usd']/1000000

del(df_container)

#See what companies in acquisitions, but not companies df
comp_list = co_df['company_name'].unique().tolist()
acq_list = acq_df['company_name'].unique().tolist()

acq_ni_comp = [x for x in acq_list if x in comp_list] 

#Calculate summary data
num_cos = len(co_df['company_name'].unique()) #Companies
num_investors = len(inv_df['investor_name'].unique()) #Investors
num_acquirer = len(acq_df['acquirer_name'].unique()) #Acquirers

#Notable value counts
status_cts = comp_df['status'].value_counts() #(Operating, Acquired, IPO, Closed)
co_reg = co_df['company_region'].value_counts() #Regions by Company
acq_reg = acq_df['acquirer_region'].value_counts() #Regions by Acquirer
inv_reg = inv_df['investor_region'].value_counts() #Regions by Investor
inv_reg = inv_df['investor_name'].value_counts() #Investors by name
co_cat_ct = co_df['company_category_code'].value_counts() #Company unique category count
top_acquirers = acq_df['acquirer_name'].value_counts()
funding_type_inv = inv_df['funding_round_type'].value_counts()
funding_type = rds_df['funding_round_type'].value_counts().reset_index()
funding_type = rds_df.groupby('funding_round_type')['raised_amount_usd'].agg({'rd_count':'count', 'avg_amt':'mean', 'tot_amt':'sum'})
funding_type = funding_type.round(0)
funding_type.loc[:, 'tot_amt'] = funding_type.groupby('funding_round_type').sum()

#Plot each group
rds_df['raised_amount_usd'].hist(by=rds_df['funding_round_type'])
plt.show()

#Distribution graphs
melt_data = pd.melt(rds_df, id_vars='funding_round_type', value_vars='').dropna(subset=['value'])

fig, ax = plt.subplots(figsize = (12,9))
plt.style.use('ggplot')
counts, bins, patches = ax.hist(melt_data['value'])
ax.set_xticks(bins)
ax.tick_params(axis='x', direction='out', labelsize=12)
ax.tick_params(axis='y', direction='out', labelsize=12)
ax.set_title('Price to Rent Ratio Distribution (Past 91 months)', fontsize=18, weight='bold')
ax.set_xlabel('Price to Rent Ratio', fontsize=14, weight='bold')
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

plt.subplots_adjust(bottom=0.15)


#Visualize numerical data in scatter matrix
num_df = vc_df[['company_name','company_category_code','','','','','']]


time_df_sub = comp_df[comp_df['found_to_fund_yrs'] >= 0]

for col in time_df_sub.columns:

    fig, ax = plt.subplots(figsize = (9,4))
    plt.style.use('ggplot')
    counts, bins, patches = ax.hist(time_df_sub[col], range=[-20, 20])
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