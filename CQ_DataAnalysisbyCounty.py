#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# the file that contains all clients included from the first processing
pre = pd.read_csv(r'...')
# the file that contains all clients that are placed in the queue after preliminary analysis
demo = pd.read_csv(r'...')
# the file that contains all clients that have enrolled in a program
outcome = pd.read_csv(r'...')

fig_size = (16,12)

#printing columns so easier to copy columns
#uncomment if needed
#print(pre.columns)
#print(demo.columns)
#print(outcome.columns)

# Delete duplicate ClientID data by keeping the first entry. Not the best way since some entries differed by columns like HHSize
# But the best way for the dataset

print('outcome count before dropping duplicates: ' + str(outcome['ClientID'].count()))
print('queue count before dropping duplicates: ' + str(demo['ClientID'].count()))

outcome.drop_duplicates(subset='ClientID', keep='first', inplace=True)
demo.drop_duplicates(subset='ClientID', keep='first', inplace=True)

print('outcome count after dropping duplicates: ' + str(outcome['ClientID'].count()))
print('queue count after dropping duplicates: ' + str(demo['ClientID'].count()))

#merge the demo file to outcome using left join
outcome_demo_merged = outcome.merge(how='left', on='ClientID', right=demo)

# turn the merged file to a csv file and save it in the desired location
outcome_demo_merged.to_csv(r'...')

#merging resulted in empty values in some columns as they were already empty in demo file
print('outcome_demo_merged count: ' + str(outcome_demo_merged['ClientID'].count()))
#print(outcome_demo_merged.iloc[:,64:70].head(100))
#print(outcome_demo_merged['Gender'].count())

# Not Applicable is if the data is missing an entry i.e. due to DV/Anon history
# Missing is if the person did not put in the data when submitting the form


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = fig_size

width=0.4

# could have done using sort_values() or groupby() but the sorting ceates a problem bc mix of integers and string
# method to print bar graphs of the files for  data with single cols
def for_single_cols(df, col, df2):
    # number in outcome
    arr = []
    # number in queue
    arr2 = []
    
    #getting all unique values from cols
    unique_vals=df[col].unique()
    print('unique_vals: ' + str(unique_vals))
    unique_vals2=df2[col].unique()
    print('unique_vals2: ' + str(unique_vals2))
    vals = np.append(unique_vals2, unique_vals)
    print('appended vals: ' + str(vals))
    vals_list = list(dict.fromkeys(vals))
    print('final unique array: ' + str(vals_list))
    
    #keeping the count for all unique values
    for index in range(len(vals_list)):
        arr.append((df[col] == vals_list[index]).sum())
        arr2.append((df2[col] == vals_list[index]).sum())
        vals_list[index] = str(vals_list[index])
        
    total=0
    total2=0
    for nums in arr:
        total = total + nums
    for nums in arr2:
        total2 = total2 + nums
        
    print('total in outcome: ' + str(total))
    print('total in queue: ' + str(total2))
    
    #plot the bar graph
    n = np.arange(len(vals_list))
    ax = plt.gca()
    ax.bar(n, arr, color = '#DA7C30', width=0.4, label='Enrolled')
    ax.bar(n+width, arr2, color = '#396AB1', width=0.4, label='On Queue')
    ax.set_xlabel(col)
    ax.set_ylabel('Number of people')
    ax.set_title('Distribution by ' + col, fontsize=18)
    ax.set_xticks(n+width/2)
    ax.set_xticklabels(vals_list, rotation=90)
    ax.legend(fontsize=14)
    
    #print the actual numbers in the bar graph by getting the patches of axs
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
            
    total = sum(totals)
        
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width()/2, i.get_height()+.3, str(i.get_height()), fontsize=14)
    
    plt.grid(axis='y')
    plt.show()
    

# remember to sort the values before sending it as a parameter for ascending graph
# also remember to replace blanks as not applicable if necessary
for_single_cols(df=outcome_demo_merged.sort_values('HHSize').fillna('Not Applicable'), col='HHSize', 
                df2=demo.sort_values('HHSize').fillna('Not Applicable'))
for_single_cols(df=outcome_demo_merged.sort_values('ScoreTotal').fillna('Not Applicable'), col='ScoreTotal',
               df2=demo.sort_values('ScoreTotal').fillna('Not Applicable'))
# TODO TYPE DESC - What is it???
for_single_cols(df=outcome_demo_merged.sort_values('Type Desc').fillna('Not Applicable'), col='Type Desc',
               df2=demo.sort_values('Type Desc').fillna('Not Applicable'))
for_single_cols(df=outcome_demo_merged.sort_values('DVIntake').fillna('Not Applicable'), col='DVIntake',
               df2=demo.sort_values('DVIntake').fillna('Not Applicable'))
for_single_cols(df=outcome_demo_merged.sort_values('Veteran Status').fillna('Not Applicable'), col='Veteran Status',
               df2=demo.sort_values('Veteran Status').fillna('Not Applicable'))
for_single_cols(df=outcome_demo_merged.sort_values('Provider').fillna('Not Applicable'), col='Provider',
               df2=demo.sort_values('Provider').fillna('Not Applicable'))

# method for bar graphs of multiple vols i.e. categorical turned to multiple kinda
def for_multiple_cols(df, cols, desc, df2):
    # array to hold outcome count values
    arr=[]
    # array to hold queue count values
    arr2=[]
    for vals in cols:
        arr.append((df[vals]==1).sum())
        arr2.append((df2[vals]==1).sum())
    
    missing = (df[cols[0]].isnull()).sum()
    missing2 = (df2[cols[0]].isnull()).sum()
    cols.append('Not Applicable')
    arr.append(missing)
    arr2.append(missing2)
    
    total=0
    for nums in arr:
        total = total + nums
    print('total in outcome: ' + str(total))
    total2=0
    for nums in arr2:
        total2 = total2 + nums
    print('total in demo: ' + str(total2))
    
    n = np.arange(len(cols))
    ax = plt.gca()
    ax.bar(n, arr, color = '#DA7C30', width=0.4, label='Enrolled')
    ax.bar(n+width, arr2, color = '#396AB1', width=0.4, label='On Queue')
    ax.set_xlabel(desc)
    ax.set_ylabel('Number of people')
    ax.set_title('Distribution by ' + desc, fontsize=18)
    ax.set_xticks(n+width/2)
    ax.set_xticklabels(cols, rotation=90)
    ax.legend(fontsize=14)
    
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
            
    total = sum(totals)
        
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width()/2, i.get_height()+.3, str(i.get_height()), fontsize=14)
    
    plt.grid(axis='y')
    plt.show()

    
# remember to put missing for missing data in certain entries; done in the method called i.e. for_multiple_cols
# TODO fix the two missing bars that show up for two different reasons for some graphs below
for_multiple_cols(df=outcome_demo_merged, cols=['FAMType0-3', 'FAMType4-8', 'FAMType9+'], desc = 'Family Size',
                 df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['Race: White', 'Race: Black', 'Race: Asian', 'Race: Indian',
       'Race: Hawaiian', 'Race: DK/R', 'Race: Missing'], desc = 'Race', df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['Gender: Male', 'Gender: Female', 'Gender: Transgender',
       'Gender: Not Identified', 'Gender: DK/R', 'Gender: Missing'], desc = 'Gender', df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['Ethnicity: Hispanic', 'Ethnicity: Non-Hispanic', 'Ethnicity: DK/R',
       'Ethnicity: Missing'], desc = 'Ethnicity', df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['Age: Under 18',
       'Age: 18 - 24', 'Age: 25 - 30', 'Age: 31 - 40', 'Age: 41 - 50',
       'Age: 51 - 61', 'Age: Over 62+', 'Age: DK/R', 'Age: Missing'], desc = 'Age Group', df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['Chronic: Yes', 'Chronic: No'], desc = 'Chronic Homelessness Status', df2=demo)
for_multiple_cols(df=outcome_demo_merged, cols=['MentalHealth: Yes', 'MentalHealth: No',
       'MentalHealth: Unknown'], desc = 'Mental Health Diagnosis', df2=demo)


# In[ ]:


# filter by rapid rehousing
rrh = outcome_demo_merged[outcome_demo_merged['ProgramTypeDesc']=='Rapid Rehousing']
print(outcome_demo_merged['ProgramTypeDesc'].unique())
print(rrh['ProgramTypeDesc'].unique())
# filter by permanent supportive housing
psh = outcome_demo_merged[outcome_demo_merged['ProgramTypeDesc'].isin(['Permanent Supportive Housing', 'Permanent Housing'])]


# In[ ]:


# Days to Enrollment and Days to House Analysis by Box Plot
plt.rcParams['figure.figsize'] = fig_size

# used Labelncoder earlier but not needed, can uncomment if required
#from sklearn.preprocessing import LabelEncoder

import matplotlib.patches as mpatches

# make boxplots for days to enrollment
def plot_box_enroll(df, cols, by):
    
    print(by)
    
    #filtering by rrh and psh
    rrh = outcome_demo_merged[outcome_demo_merged['ProgramTypeDesc']=='Rapid Rehousing']
    psh = outcome_demo_merged[outcome_demo_merged['ProgramTypeDesc'].isin(['Permanent Supportive Housing', 'Permanent Housing'])]
    dfc = df.copy()
    
    for col in cols:
        lst = []
        label = []
        
        for val in dfc[col].unique():
            #append data values
            lst.append(rrh[rrh[col]==val][by].to_numpy())
            label.append(str(val) + '_RRH')
            lst.append(psh[psh[col]==val][by].to_numpy())
            label.append(str(val) + '_PSH')
        
        fig, axs = plt.subplots()
    
        bp_dict = axs.boxplot(lst, showmeans=True, meanline=True, labels=label)
        #print(bp_dict.keys())
        
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1]
            axs.text(x + .15, y, '%.1f' % y, horizontalalignment='center', color='orange') # draw above, centered
        
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[1]
            axs.text(x + .15,y, '%.1f' % y, horizontalalignment='center', color='red')
            x, y = line.get_xydata()[2]
            axs.text(x + .15,y, '%.1f' % y, horizontalalignment='center', color='red')
            
        for line in bp_dict['means']:
            x, y = line.get_xydata()[0]
            axs.text(x - .15,y, '%.1f' % y, horizontalalignment='center', color='green')
            
        for line in bp_dict['whiskers']:
            x, y = line.get_xydata()[1]
            axs.text(x - .40,y, '%.1f' % y, horizontalalignment='center', color='blue')
        
        axs.set_title('Box Plot by ' + col, fontsize = 18)
        
        #create legend by creating patches
        blue_patch = mpatches.Patch(color='blue', label='MIN/MAX VALUES')
        green_patch = mpatches.Patch(color='green', label='MEAN VALUE')
        orange_patch = mpatches.Patch(color='orange', label='MEDIAN VALUE')
        red_patch = mpatches.Patch(color='red', label='1ST/3RD QUARTILE VALUES')
        plt.legend(handles=[blue_patch,green_patch,orange_patch,red_patch], fontsize=14)
        
        plt.xticks(rotation='vertical')
        plt.grid(axis='y')
        plt.show()
        
print('581 - Permanent Supportive Housing \t 618 - Rapid Rehousing')
print()
plot_box_enroll(outcome_demo_merged, ['VISPDATType', 'DV/Anon', 'Veteran', 'MentalHealth_x', 'Provider'], by='DaystoEnroll')
plot_box_enroll(outcome_demo_merged, ['VISPDATType', 'DV/Anon', 'Veteran', 'MentalHealth_x', 'Provider'], by='DaystoHoused')


# In[ ]:


import seaborn as sns
cols = ['VISPDATType', 'DV/Anon', 'Veteran', 'MentalHealth_x']
sns.set_style("whitegrid")
sns.set_context("paper")

def scatter(df, cols, by):
    print('581 - Permanent Supportive Housing \t 618 - Rapid Rehousing')
    for col in cols:
        sns.catplot(x=col, y=by, col='ServiceNeedsCodeID', data=df, kind='strip', jitter=False)
        
scatter(outcome_demo_merged,cols,'DaystoEnroll')
scatter(outcome_demo_merged,cols,'DaystoHoused')


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context("paper")

# create box plots for provider
def boxplot_provider(df, cols):
    print('581 - Permanent Supportive Housing \t 618 - Rapid Rehousing')
    for col in cols:
        
        sns.catplot(x=col, y='Provider', kind="box", col='ServiceNeedsCodeID', data=df)
        #plt.gcf().set_size_inches(30, 10)
        
        
boxplot_provider(df=outcome_demo_merged, cols=['DaystoEnroll', 'DaystoHoused'])


# In[ ]:


import math

bin_size = 25

# Find the first digit 
def firstDigit(n) : 
  
    # Remove last digit from number 
    # till only one digit is left 
    while n >= 10:  
        n = n / 10; 
      
    # return the first digit 
    return int(n)

def histogram_byfilter(df, filter_col, by_col):
    df = df.fillna({filter_col:'Not Applicable'})
    unique_vals = df[filter_col].unique()
    for vals in unique_vals:
        new_df = df[df[filter_col]==vals]
        max = new_df[by_col].max()
        end  = (firstDigit(max)+1)*(math.pow(10,(int(math.log10(max)))))
        plt.hist(new_df[by_col], bins=np.arange(0,end,bin_size))
        ax=plt.gca()
        ax.set_xlabel(by_col)
        ax.set_ylabel('Number of people')
        plt.title('Histogram for ' + str(vals), fontsize=18)
        
        for i in ax.patches:
            ax.text(i.get_x() + i.get_width()/2, i.get_height()+.02, str(i.get_height()), fontsize=14)
        
        plt.show()
        
histogram_byfilter(outcome_demo_merged, 'ServiceNeedsCodeID', by_col = 'DaystoEnroll')
histogram_byfilter(outcome_demo_merged, 'ServiceNeedsCodeID', by_col='DaystoHoused')
histogram_byfilter(outcome_demo_merged, 'Provider', by_col='DaystoHoused')
histogram_byfilter(outcome_demo_merged, 'Provider', by_col='DaystoHoused')


# In[ ]:


import pandas as pd

county = '...'


print('Gender, Race and Ethnicity Population of ' + county)
state_sex_race = pd.read_csv(r'C:\Users\Intern\Desktop\Data Analysis\data by counties\state_sex_race.csv')
#print(state_sex_race['CTYNAME'].unique())
state_sex_race = state_sex_race.drop([0])
#print(state_sex_race.columns)
# filter by latest population estimate
state_sex_race = state_sex_race[state_sex_race['Year.id'] == 'est72018']
# filter by county selected
state_sex_race = state_sex_race[state_sex_race['GEO.display-label'] == county]
state_sex_race = state_sex_race.drop(['Year.id', 'Year.display-label',  'Sex.display-label',
        'Hisp.display-label', 'GEO.id', 'GEO.id2',
       'GEO.display-label'], axis=1)
print()
print('LEGEND')
print('wa - White American')
print('ba - Black American')
print('ia - American Indian and Alaska Native')
print('aa - Asian')
print('na - Native Hawaiian and Other Pacific Islander')
print()
print(state_sex_race.head(100))
print('\nfrom 2018 Population Estimates')
print('\n' * 5)


print('HHSize numbers of ' + county)
print()
state_hhsize = pd.read_csv(r'C:\Users\Intern\Desktop\Data Analysis\data by counties\state_hhsize.csv')
state_hhsize_num = state_hhsize[state_hhsize['GEO.display-label'] == county]
#print(state_hhsize.head(100))

for i in np.arange(3,19):
    print(state_hhsize.iat[0,i] + ': ' + state_hhsize_num.iat[0,i])
#print(state_hhsize.head(100))
print('\nfrom 2013-2017 American Community Survey 5-Year Estimates')
print('\n' * 5)


print('Veteran Population of ' + county)
print()
state_veteran = pd.read_csv((r'C:\Users\Intern\Desktop\Data Analysis\data by counties\state_veteran.csv'))
state_veteran_num = state_veteran[state_veteran['GEO.display-label'] == county]
#print(state_veteran_num.head())

print(state_veteran.iat[0,3] + ': ' + state_veteran_num.iat[0,3])
print(state_veteran.iat[0,4] + ': ' + state_veteran_num.iat[0,4])
print('\nfrom 2013-2017 American Community Survey 5-Year Estimates')
print('\n' * 5)


print('Age Group Distribution of ' + county)
print()
state_age = pd.read_csv((r'C:\Users\Intern\Desktop\Data Analysis\data by counties\state_age.csv'))
state_age_num = state_age[state_age['GEO.display-label'] == county]
for i in np.arange(3,102,3):
    print(state_age.iat[0,i] + ': ' + state_age_num.iat[0,i])
print('\nfrom 2018 Population Estimates')
print('\n' * 5)


# In[ ]:




