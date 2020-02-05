#!/usr/bin/env python
# coding: utf-8

# In[1]:
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# In[3]:


file = "data_files/revenue.xlsx"


# In[4]:


revenue_df = pd.read_excel(file)
revenue_df.head()


# In[5]:


revenue_df.isnull().sum()


# In[6]:



# %%
file = "data_files/revenue.xlsx"


# %%
revenue = pd.read_excel(file)
revenue


# %%
revenue.isnull().sum()


# %%
grad_rate = pd.read_excel("data_files/grad_rate.xlsx")
grad_rate.head()


# In[7]:


grad_rate_1 = pd.read_excel("data_files/grad_rate.xlsx")
grad_rate_1.head()


# In[8]:


grad_rate.shape


# In[9]:


grad_rate_df = grad_rate.rename(columns={"Unnamed: 0":"States", "Unnamed: 1":"2011", "Unnamed: 2":"2012", "Unnamed: 3":"2013", "Unnamed: 4":"2014", "Unnamed: 5":"2015", "Unnamed: 6":"2016", "Unnamed: 7": "Average" })
grad_rate_df.head()


# In[29]:


grad_rate_df_1 = grad_rate_df
# grad_rate_renamed = grad_rate_df_1.rename(columns={: "State"})
grad_rate_renamed_df = grad_rate_df_1.set_index('States')
grad_rate_renamed_df.head()


# In[30]:


clean_grad_rate = grad_rate_df_1.rename(columns={'States':'State'})


# In[21]:


grad_rate_renamed_df.loc['Oklahoma'].transform(lambda x: x.fillna(x.mean()))


# In[22]:


grad_rate_renamed_df.loc['Idaho'].transform(lambda x: x.fillna(x.mean()))


# In[23]:


grad_rate_renamed_df.loc['Kentucky'].transform(lambda x: x.fillna(x.mean()))


# In[24]:


us_grad_rate_1 = pd.DataFrame(grad_rate_1.mean())
us_grad_rate_1


# In[25]:


us_grad_rate_df = us_grad_rate_1.rename(columns={0:'Grad_Rate'})
us_grad_rate_df


# In[26]:


us_grad_rate_df.plot(kind='line', label='Grad_Rate', figsize=(14,7), color='b', marker='^')
plt.xlabel('Years')
plt.ylabel('graduation rate in %')
plt.title('Aggregated US Gradulation Rate')

plt.legend(loc='best')
plt.grid()
plt.savefig('Aggregated US Gradulation Rate')


# In[27]:


# %%
grad_rate_df = grad_rate.rename(columns={"Unmaned: 0":"States", "Unnamed: 1":"2010", "Unnamed: 2":"2011", "Unnamed: 3":"2012", "Unnamed: 4":"2013", "Unnamed: 5":"2014", "Unnamed: 6":"2015", "Unnamed: 7":"2016" })
grad_rate_df.head()


# %%
grad_rate_df_1 = grad_rate_df.drop([0,1])
grad_rate_renamed = grad_rate_df_1.rename(columns={"Unnamed: 0": "State"})
grad_rate_renamed_df = grad_rate_renamed.set_index('State')
grad_rate_renamed_df.head()


# %%
grad_rate_renamed_df.loc['Oklahoma'].transform(lambda x: x.fillna(x.mean()))


# %%
grad_rate_renamed_df.loc['Idaho'].transform(lambda x: x.fillna(x.mean()))


# %%
grad_rate_renamed_df.loc['Kentucky'].transform(lambda x: x.fillna(x.mean()))


# %%
us_grad_rate = pd.DataFrame(grad_rate_renamed.mean())
us_grad_rate_df = us_grad_rate.rename(columns={0:'Grad_Rate'})
us_grad_rate_df


# %%
us_grad_rate_df.plot(kind='line', label='Grad_Rate', figsize=(14,7), color='b')
plt.xlabel('Year')
plt.ylabel('Graduation Percentage')
plt.title('US Graduation Rate')
plt.grid()
plt.legend(loc='best')


# %%



# %%
teacher_salaries = pd.read_excel("data_files/teacher_salaries_1.xlsx").round(0)
teacher_salaries_new = teacher_salaries.drop([0])
teacher_salaries_new.head()


# In[28]:


# %%
teacher_salaries_new_1 = teacher_salaries_new.drop(columns=['2007','2008','2009', '2017', '2018'], axis=1)
teacher_salaries_new_1.head()


# In[31]:


grad_salary_df = pd.merge(clean_grad_rate, teacher_salaries_new_1, on='State', how='outer')
grad_salary_df.head()


# In[38]:


grad_salary_renamed = grad_salary_df.rename(columns={2011 : '2011_grad_rate',
                                                 2012 : '2012_grad_rate', 2013 : '2013_grad_rate',
                                                 2014 : '2014_grad_rate', 2015 : '2015_grad_rate',
                                                 2016 : '2016_grad_rate', '2010':'2010_salary','2011':'2011_salary',
                                                 '2012':'2012_salary', '2013':'2013_salary', '2014':'2014_salary',
                                                 '2015':'2015_salary', '2016':'2016_salary'}) 
# %%
grad_salary_df = pd.merge(grad_rate_renamed, teacher_salaries_new_1, on='State', how='outer')
grad_salary_df.head()


# %%
grad_salary_renamed = grad_salary_df.rename(columns={'2010_x' : '2010_grad_rate', '2011_x' : '2011_grad_rate',
                                                 '2012_x' : '2012_grad_rate', '2013_x' : '2013_grad_rate',
                                                 '2014_x' : '2014_grad_rate', '2015_x' : '2015_grad_rate',
                                                 '2016_x' : '2016_grad_rate', '2010_y':'2010_salary','2011_y':'2011_salary',
                                                 '2012_y':'2012_salary', '2013_y':'2013_salary', '2014_y':'2014_salary',
                                                 '2015_y':'2015_salary', '2016_y':'2016_salary'}) 

grad_salary_renamed.head()


# In[33]:


grad_salary_renamed.iloc[2,6]= 80


# In[39]:


grad_salary_df = grad_salary_renamed[['State', '2011_grad_rate', '2011_salary', 
# %%
grad_salary_renamed.iloc[2,6]= 80


# %%
grad_salary_df = grad_salary_renamed[['State', '2010_grad_rate', '2010_salary', '2011_grad_rate', '2011_salary', 
                                     '2012_grad_rate', '2012_salary', '2013_grad_rate', '2013_salary',
                                     '2014_grad_rate', '2014_salary', '2015_grad_rate', '2015_salary',
                                     '2016_grad_rate', '2016_salary']]

grad_salary_df.head()


# In[40]:


grad_salary_df.head()


# In[41]:


# %%
grad_salary_df.head()


# %%



# %%
pupil_spending = pd.read_excel("data_files/per_pupil_spending.xlsx")
pupil_spending.head()


# In[42]:
# In[ ]:

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import plotly
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:
# In[2]:


# Revenue Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","revenue.xlsx")
revenue = pd.read_excel(dataFilePath)


# In[ ]:
# In[3]:


# Salary Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","teacher_salaries_1.xlsx")
teacher_salaries = pd.read_excel(dataFilePath).round(0)
teacher_salaries.drop([0])

sal1 = "data_files/teacher_salaries_1.xlsx"
sal2="data_files/gradvssal.csv"
salaries=pd.read_excel(sal1)
salaries2=pd.read_csv(sal2, index_col="States")

teacher_salaries


# In[ ]:


# Region Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","Region Defintions.xlsx")
regionData = pd.read_excel(dataFilePath)


# In[4]:


# Graduation Rate Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","grad_rate.xlsx")
grad_rate = pd.read_excel(dataFilePath)

grad1= "data_files/grad_rates.xlsx"
grad2= "data_files/grad_rates.xlsx"
gradrate=pd.read_excel(grad1)
gradrate2=pd.read_excel(grad2, index_col="States")


# In[ ]:


# Pupil Spending Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","per_pupil_spending.xlsx")
pupil_spending = pd.read_excel(dataFilePath)
pupil_spending


# In[5]:

# Ratio Data
path = os.getcwd()
dataFilePath = os.path.join(path,"data_files","teacher_student_ratio.xlsx")
ratio = pd.read_excel(dataFilePath)

ratio1="data_files/teacher_student_ratio.xlsx"
ratio2="data_files/Ratios.csv"
teachratio=pd.read_excel(ratio1)
teachratio2=pd.read_csv(ratio2, index_col="States")

ratio


# In[ ]:


grad_rate_df = grad_rate.rename(columns={"Unmaned: 0":"States", "Unnamed: 1":"2010", "Unnamed: 2":"2011", "Unnamed: 3":"2012", "Unnamed: 4":"2013", "Unnamed: 5":"2014", "Unnamed: 6":"2015", "Unnamed: 7":"2016" })
grad_rate_df.head()


# In[ ]:


grad_rate_df_1 = grad_rate_df.drop([0,1])
grad_rate_renamed = grad_rate_df_1.rename(columns={"Unnamed: 0": "States"})
grad_rate_renamed = grad_rate_renamed.set_index('States')
grad_rate_renamed.head()


# In[ ]:


# Get full data for state
yearValues = pd.DataFrame(grad_rate_renamed.loc['Oklahoma'].transform(lambda x: x.fillna(x.mean())))
# Drop incomplete data
grad_rate_renamed = grad_rate_renamed.drop(index='Oklahoma')
# Flatten the year data
stateData = yearValues.T
# Add new data and sort by state name
grad_rate_renamed = grad_rate_renamed.append(stateData).sort_index()
# Print results
grad_rate_renamed


# In[ ]:


yearValues = grad_rate_renamed.loc['Idaho'].transform(lambda x: x.fillna(x.mean()))
# Drop incomplete data
grad_rate_renamed = grad_rate_renamed.drop(index='Idaho')
# Flatten the year data
stateData = yearValues.T
# Add new data and sort by state name
grad_rate_renamed = grad_rate_renamed.append(stateData).sort_index()
# Print results
grad_rate_renamed


# In[ ]:


yearValues = grad_rate_renamed.loc['Kentucky'].transform(lambda x: x.fillna(x.mean()))
# Drop incomplete data
grad_rate_renamed = grad_rate_renamed.drop(index='Kentucky')
# Flatten the year data
stateData = yearValues.T
# Add new data and sort by state name
grad_rate_renamed = grad_rate_renamed.append(stateData).sort_index()
# Print results
grad_rate_renamed


# In[ ]:


pupil_spending_renamed = pupil_spending.rename(columns={"2007":"2007_PPS", "2008":"2008_PPS", "2009":"2009_PPS",
                                                       "2010":"2010_PPS", "2011":"2011_PPS", "2012":"2012_PPS", "2013":"2013_PPS",
                                                       "2014":"2014_PPS", "2015":"2015_PPS", "2016":"2016_PPS", 
                                                       "Unnamed: 2":"2007 pct_change", "Unnamed: 4":"2008 pct_change",
                                                       "Unnamed: 6":"2009 pct_change","Unnamed: 8":"2010 pct_change",
                                                       "Unnamed: 10":"2011 pct_change", "Unnamed: 12":"2012 pct_change",
                                                       "Unnamed: 14":"2013 pct_change", "Unnamed: 16":"2014 pct_change",
                                                       "Unnamed: 18":"2015 pct_change", "Unnamed: 20":"2016 pct_change"})

pupil_spending_renamed.head()


# In[43]:
# In[ ]:


pupil_spending_df = pupil_spending_renamed.drop([0,1])
pupil_spending_df.head()


# In[44]:
# In[ ]:


pupil_spending_df.isnull().sum()


# In[45]:
ratio = pd.read_excel("data_files/teacher_student_ratio.xlsx")
ratio.head()


# In[46]:
# In[ ]:


# ratio_df = ratio.drop(['Unnamed: 1', 2007, 'Unnamed: 3', 'Unnamed: 4', 2008, 'Unnamed: 6', 'Unnamed: 7', 2009, 'Unnamed: 9','Unnamed: 10', 2010, 'Unnamed: 12'],axis=1)
# ratio_df.head()

# In[ ]:
ratio

# In[47]:
# In[ ]:

# Testing code, eliminating column renames ::SPD020420
ratio_renamed_df = ratio
# ratio_renamed_df = ratio_df.rename(columns={"Unnamed: 0":"State", "Unnamed: 13":"2011_staff", 2011:"2011_enrollment", 
#                                      "Unnamed: 15":"2011_ratio", "Unnamed: 16":"2012_staff", 2012:"2012_enrollment", 
#                                       "Unnamed: 18":"2012_ratio", "Unnamed: 19":"2013_staff", 2013:"2013_enrollment",
#                                      "Unnamed: 21":"2013_ratio", "Unnamed: 22":"2014_staff", 2014:"2014_enrollment",
#                                      "Unnamed: 24":"2014_ratio",  "Unnamed: 25":"2015_staff", 2015:"2015_enrollment",
#                                      "Unnamed: 27":"2015_ratio", "Unnamed: 28":"2016_staff", 2016:"2016_enrollment",
#                                      "Unnamed: 30":"2016_ratio"}) 
                                     
# ratio_renamed_df.head()


# In[48]:
# In[ ]:


ratio_cleaned_df = ratio_renamed_df.drop([0])
ratio_cleaned_df.head()


# In[49]:
# In[ ]:


ratio_cleaned_df.isnull().sum()


# In[50]:


math_reading = pd.read_excel("data_files/math_reading.xlsx").round(2)
math_reading.head()


# In[51]:


math_reading_df = math_reading.set_index('State')
math_reading_df.head()


# In[52]:


math_df = math_reading_df.drop(['2007_reading', '2009_reading', '2011_reading', '2013_reading', '2015_reading'],axis=1)
math_df.head()


# In[53]:


math_avg = math_df.mean(axis=1)
math_avg.head()


# In[54]:


reading_df = math_reading_df.drop(['2007_math', '2009_math', '2011_math', '2013_math', '2015_math'],axis=1)
reading_df.head()


# In[55]:


reading_avg = math_reading_df.mean(axis=1)
reading_avg.head()


# In[56]:


math_reading_avg = pd.DataFrame({"Math_Average":math_avg, "Reading_Average":reading_avg})
math_reading_avg.head()


# In[57]:


math_reading_avg.plot(kind='bar', figsize=(16,8))


# In[58]:


math_reading_avg = math_reading_df.mean()
math_reading_avg


# In[59]:


math_avg_7 = math_reading_df['2007_math'].mean()
math_avg_9 = math_reading_df['2009_math'].mean()
math_avg_11 = math_reading_df['2011_math'].mean()
math_avg_13 = math_reading_df['2013_math'].mean()
math_avg_15 = math_reading_df['2015_math'].mean()
math_average = [math_avg_7, math_avg_9, math_avg_11, math_avg_13, math_avg_15]
math_average


# In[60]:


reading_avg_7 = math_reading_df['2007_reading'].mean()
reading_avg_9 = math_reading_df['2009_reading'].mean()
reading_avg_11 = math_reading_df['2011_reading'].mean()
reading_avg_13 = math_reading_df['2013_reading'].mean()
reading_avg_15 = math_reading_df['2015_reading'].mean()
reading_average = [reading_avg_7, reading_avg_9, reading_avg_11, reading_avg_13, reading_avg_15]
reading_average


# In[61]:


avg_df = pd.DataFrame(math_average, reading_average)
avg_df['Years'] = [2007, 2009, 2011, 2013, 2015]

avg_df


# In[62]:


avg_df_1 = avg_df.set_index('Years')
avg_df_1


# In[63]:


avg_df_1['reading'] = reading_average
avg_df_1


# In[64]:


avg_df_2 = avg_df_1.rename(columns={0:'math'})
avg_df_2


# In[65]:


avg_df_2.plot(kind='bar', figsize=(16,8))
plt.ylabel('Average Score')
plt.title('Average Math & Reading score in Georgia')
plt.show()


# In[66]:


revenue_grouped = revenue_df.groupby('YEAR')
revenue_grouped_df = pd.DataFrame(revenue_grouped['TOTAL_REVENUE'].sum()/1000000)
revenue_grouped_df


# In[67]:


expenditure_grouped = revenue_df.groupby('YEAR')
expenditure_grouped_df = pd.DataFrame(revenue_grouped['TOTAL_EXPENDITURE'].sum()/1000000)
expenditure_grouped_df


# In[68]:


x = expenditure_grouped_df.index
revenue = revenue_grouped_df['TOTAL_REVENUE']
expenditure = expenditure_grouped_df['TOTAL_EXPENDITURE']


# In[69]:


revenue = plt.plot(x, revenue, marker='o', color='blue', linewidth=2, label='US Total Revenue')
expenditure = plt.plot(x, expenditure, marker='+', color='red', linewidth=2, label='US Total Expenditure')
plt.xlabel('Years')
plt.ylabel('Billion Dollars')
plt.legend(loc='best')
plt.title('US Total revenue vs US total expenditure from 2007-2016')
plt.grid()
plt.savefig("US Total Revenue vs US Total Expenditure")


# In[70]:


revenue_grouped_df.plot(kind='line', label='US Total_Revenue(billion)', figsize=(14,7), color='b', marker='o', linewidth=2, linestyle='-')
plt.ylabel("Revenue in Billion Dollars")
plt.title('US Total Revenue from 2007-2016')
plt.grid()
plt.savefig('US Total Revenue from 2007-2016')


# In[71]:


revenue_grouped_df_1 = revenue_grouped_df


# In[72]:



revenue_df_2 = revenue_grouped_df_1.drop([2007, 2008, 2009, 2010], axis=0)
revenue_df_2


# In[80]:


graduation_rate = [79.97, 81.06, 79.97, 82.99, 83.84, 84.58]
revenue_df_2['Grad_Rate'] = graduation_rate

# In[81]:


fig = plt.figure()
ax = revenue_df_2['TOTAL_REVENUE'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
plt.ylabel('Revenue in Billion Dollars')
plt.xlabel('Years')
ax2 = ax.twinx()
ax2.plot(revenue_df_2['Grad_Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation rate')
# plt.ylim(76,88)
plt.title('Average Graduation Rate Compared to Revenue in US')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Revenue in US')
plt.show()


# In[164]:


revenue_grouped_state = revenue_df.groupby(['STATE'])


# In[165]:


revenue_grouped_state_df = pd.DataFrame(revenue_grouped_state['TOTAL_REVENUE'].mean()*10)
revenue_grouped_state_df.head()


# In[166]:


exp_grouped_state_df = pd.DataFrame(revenue_grouped_state['TOTAL_EXPENDITURE'].mean()*10)
exp_grouped_state_df.head()


# In[167]:


rev_exp_df = pd.merge(revenue_grouped_state_df, exp_grouped_state_df, on='STATE', how='outer')
rev_exp_df.head()


# In[168]:


rev_exp_dif = rev_exp_df


# In[169]:


rev_exp_dif['DIFFERENCE'] = (rev_exp_df['TOTAL_REVENUE'] - rev_exp_df['TOTAL_EXPENDITURE'])/1000
rev_exp_dif.head()


# In[170]:


rev_exp_dif_df = rev_exp_dif.drop(columns=['TOTAL_REVENUE', 'TOTAL_EXPENDITURE'], axis=1)
rev_exp_dif_df.head()


# In[171]:


rev_exp_dif_df.columns


# In[172]:


rev_exp_dif_df['DIFFERENCE'].plot(kind='barh', figsize=(10,25),
                    color=(rev_exp_dif_df['DIFFERENCE'] > 0).map({True: 'g',
                                                    False: 'r'}))

plt.xlabel('Thousand Dollars')
plt.title('States Deficit on Education')
plt.savefig('States Deficit on Education')


# In[173]:


rev_exp_df[['TOTAL_REVENUE', 'TOTAL_EXPENDITURE']].plot(kind='barh', figsize=(10,25))
plt.xlabel('Billion Dollars')
plt.savefig('Total Revenue and Expenditure for all the states')


# In[174]:


revenue_grouped_state_df.plot(kind='barh', figsize=(10,25), color='green')
plt.xlabel('Revenue in billion')


# In[175]:


exp_grouped_state_df.plot(kind='barh', figsize=(10,25), color='red')
plt.xlabel('Expenditures in billion')


# In[178]:


ga_numbers = revenue_df.loc[revenue_df['STATE']=='GEORGIA']
ga_numbers_df = ga_numbers[['YEAR', 'TOTAL_REVENUE', 'TOTAL_EXPENDITURE']]
ga_numbers_df


# In[179]:


ga_numbers_df_1 = ga_numbers_df.set_index('YEAR')
ga_numbers_df_1


# In[181]:


ga_numbers_df_1.plot(kind='bar', figsize=(14,6))
plt.ylabel('10 Billion Dollars')
plt.title('Total Revenue and Expenditure for Georgia')
plt.savefig('Total Revenue and Expenditure for Georgia')
plt.show()


# In[133]:


clean_grad_rate.State = clean_grad_rate.State.astype(str).str.upper()


# In[134]:


clean_grad_rate_df = clean_grad_rate
clean_grad_rate_df.head()


# In[135]:


clean_grad_rate_df['Agg'] = clean_grad_rate.mean(axis=1)


# In[136]:


grad_rate_cleaned = clean_grad_rate[['State', 'Agg']].set_index('State')
grad_rate_cleaned.head()


# In[137]:


pupil_spending_renamed = pupil_spending_df.rename(columns={'STATE':'State'})
pupil_spending_renamed.head()


# In[138]:


pupil_spending_df_1 = pupil_spending_renamed[['State', 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]].set_index('State')
pupil_spending_df_1.head()


# In[139]:


pupil_spending_df_1['Avg'] = pupil_spending_df_1.mean(axis=1)
pupil_spending_df_1.head()


# In[140]:


student_spending_df_2 = pupil_spending_df_1[['Avg']]
student_spending_df_2.head()


# In[141]:


student_spending_df_2['Avg'].plot(kind='bar', figsize=(16,8), label='Per Student Spending')
plt.ylabel('Per Student Spending')
plt.title('Average Per Student Spendning of Every State')
plt.savefig('Average Per Student Spendning of Every State')

plt.show()


# In[142]:


student_spending_df_2['Agg'] = grad_rate_cleaned['Agg']
student_spending_df_2.head()


# In[1]:


fig = plt.figure()
ax = student_spending_df_2['Avg'].plot(kind='bar', figsize=(16,8), label='Per Student Spending')
plt.ylabel('Average Spending Per Student in Dollars')
ax2 = ax.twinx()
ax2.plot(student_spending_df_2['Agg'].values, linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Grade')
plt.title('Graduation Rate compate to Per Student Spending per state')
plt.savefig('Graduation Rate compate to Per Student Spending per state')
plt.grid()
plt.show()


# In[144]:


pupil_spending_us = round(pupil_spending_df_1.mean(),2)
student_spending_us = pd.DataFrame(pupil_spending_us)
student_spending_us_df = student_spending_us.drop(['Avg'])
student_spending_renamed = student_spending_us_df.rename(columns={0:'Avg_spent_per_student in US'})
student_spending_renamed


# In[145]:


student_spending_renamed.plot(kind='line', marker='o', linestyle='-', linewidth=2, color='g', figsize=(14,7))
plt.xlabel('Years')
plt.ylabel('Dollars')
plt.title('Average per student spending in US')
plt.grid()
plt.savefig('Average per student spending in US')
plt.show()


# In[146]:


grad_rate_spending = student_spending_renamed


# In[147]:


grad_spending_df = grad_rate_spending.drop([2007, 2008, 2009, 2010])
grad_spending_df


# In[148]:


grad_spending_df['Grad Rate'] = graduation_rate
grad_spending_df


# In[149]:


fig = plt.figure()
ax = grad_spending_df['Avg_spent_per_student in US'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
plt.ylabel('Avg_spent_per_student in US')
plt.xlabel('Years')
ax2 = ax.twinx()
ax2.plot(grad_spending_df['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation rate')
plt.title('Average Graduation Rate Compared to Avg_spent_per_student in US')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Avg_spent_per_student in US')
plt.show()


# In[150]:


clean_grad_rate_df.head(2)


# In[151]:


ga_grad_rate_df = grad_rate_1.loc[grad_rate_1['States']=='Georgia']
ga_grad_rate_df


# In[152]:


# ga_grad_rate = clean_grad_rate_df.loc[grad_rate_df['State']=='GEORGIA']
ga_grad_sorted_df = ga_grad_rate_df.set_index('States')
# ga_grad_sorted_df = ga_grad_sorted.drop(['Average', 'Agg'], axis=1)
ga_grad_sorted_df


# In[153]:


ga_grad_rate_df_1 = ga_grad_sorted_df.T
ga_grad_renamed_df = ga_grad_rate_df_1.rename(columns={'States':'Year', 'Georgia':'Graduation Rate in Georgia'})
ga_grad_renamed_df


# In[154]:


ga_grad_renamed_df.plot(kind='line', marker='s', linestyle='-', color = 'red', linewidth = 2, label='Georgia_Graduation_Rate')
plt.ylabel('Graduation Rate')
plt.xlabel('Years')
plt.title('Georgia Graduation Rate from 2011-2016')
plt.grid()
plt.savefig('Georgia Graduation Rate from 2011-2016')
plt.show()


# In[155]:


ga_student_spending = pd.DataFrame(pupil_spending_df_1.loc['GEORGIA'])
ga_student_spending_rename = ga_student_spending.rename(columns={'GEORGIA':'Georgia Per Student Spending'})
ga_student_spending_df = ga_student_spending_rename.drop([2007, 2008, 2009, 'Avg'])
ga_student_spending_df


# In[156]:


teacher_salaries_new.head(2)


# In[157]:


teacher_salaries_us = round(teacher_salaries.mean(),2)
teacher_salaries_us_df = pd.DataFrame(teacher_salaries_us)
teacher_salaries_us_df.head()


# In[158]:


teacher_salaries_renamed = teacher_salaries_us_df.rename(columns={0:'Average Teacher Salary in US'})
teacher_salaries_renamed.head()


# In[159]:


teacher_salaries_renamed.plot(kind='line', marker='^', color='g', linewidth=2)
plt.xlabel('Years')
plt.ylabel('Dollars')
plt.title('Average Teacher Salary in US')
plt.grid()
plt.savefig('Average Teacher Salary in US')
plt.show


# In[160]:


grad_rate_salary = teacher_salaries_renamed
grad_rate_salary


# In[161]:


grad_rate_salary_df = grad_rate_salary.drop(['2007', '2008', '2009', '2010', '2017', '2018'])
grad_rate_salary_df


# In[162]:


grad_rate_salary_df['Grad_Rate'] = graduation_rate
grad_rate_salary_df


# In[163]:


fig = plt.figure()
ax = grad_rate_salary_df['Average Teacher Salary in US'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
plt.ylabel('Avg Teacher Salary in US')
plt.xlabel('Years')
ax2 = ax.twinx()
ax2.plot(grad_rate_salary_df['Grad_Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation rate')
plt.title('Average Graduation Rate Compared to Average Teacher Salary in US')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Average Teacher Salary in US')
plt.show()


# In[164]:


teacher_salaries_sorted = teacher_salaries_new.set_index('State')
teacher_salaries_sorted.head()


# In[165]:


teacher_salaries_sorted['Avg'] = round(teacher_salaries_sorted.mean(axis=1),2)
teacher_salaries_sorted.head()


# In[166]:


teacher_salaries_states = teacher_salaries_sorted[['Avg']]
teacher_salaries_states.head()


# In[167]:


teacher_salaries_states.plot(kind='bar', figsize=(14,7), color='y')
plt.ylabel('Average Salary in Dollars')
plt.title('Average Teachers Salary in all the States in US')
plt.grid()
plt.savefig('Average Teachers Salary in all the States in US')
plt.show()


# In[168]:


teacher_salaries_ga = teacher_salaries.loc[teacher_salaries['State'] == 'Georgia']
teacher_salaries_ga_clean = teacher_salaries_ga.drop(columns=['2007', '2008', '2009'], axis=1)
teacher_salaries_ga_df = teacher_salaries_ga_clean.set_index('State')
teacher_salaries_ga_df


# In[169]:


teacher_salaries_ga_df_1 = teacher_salaries_ga_df.T
teacher_salaries_ga_df_2 = teacher_salaries_ga_df_1.drop(['2017', '2018'])
teacher_salaries_ga_renamed_df = teacher_salaries_ga_df_2.rename(columns={'Georgia': 'Average Teachers Salary in Georgia'})
teacher_salaries_ga_renamed_df


# In[170]:


teacher_salaries_ga_renamed_df.plot(kind='line', label='Teachers Salaries in GA', marker='^', linestyle='-', color='b')
plt.legend(loc='best')
plt.xlabel('Years')
plt.ylabel('Dollars')
plt.title('Average Teachers salary in Georgia from 2010-2016')
plt.grid()
plt.savefig('Average Teachers salary in Georgia from 2010-2016')
plt.show()


# In[171]:


revenue_fed = revenue_df[['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenue_fed.head()


# In[172]:


revenue_avg = pd.DataFrame(revenue_fed.mean())
revenue_avg


# In[173]:


explode=(0.1,0,0)
revenue_avg.plot(kind='pie', explode=explode, autopct="%1.1f%%", shadow=True, subplots=True, figsize=(14,7))
plt.axis('equal')
plt.savefig('Revenue Distribution in US')
plt.show()


# In[174]:


revenue_ga = revenue_df[['STATE', 'YEAR', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenue_ga.head()


# In[175]:


revenue_ga_df = revenue_ga.loc[revenue_ga['STATE']=='GEORGIA']
rev_ga_df = revenue_ga_df.drop(['YEAR'], axis=1)
rev_ga_avg = rev_ga_df.mean()
rev_ga_df = pd.DataFrame(rev_ga_avg)
rev_ga_df                       


# In[176]:


explode = (0.1,0,0)
rev_ga_df.plot(kind='pie', explode=explode, autopct = "%1.1f%%", shadow=True, subplots=True, figsize=(14,7))
plt.axis('equal')
plt.savefig('Revenue distribution in Georgia')
plt.show()


# In[177]:


rev_exp_df_1 = revenue_df[['STATE', 'YEAR', 'TOTAL_REVENUE', 'TOTAL_EXPENDITURE']]
rev_exp_df_1.head()


# In[178]:


student_spending_df_2.head()


# In[179]:


grad_rate_cleaned.head()


# In[180]:


x_grad = grad_rate_cleaned['Agg']
y_spend = student_spending_df_2['Avg']
(slope, intercept, rvalue, pvalue, stderr) = linregress(x_grad,y_spend)
regress_values = x_grad * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(x_grad,regress_values,"r-")
plt.annotate(line_eq,(65,11200), fontsize=15,color="red")
plt.scatter(x_grad,y_spend)
print(f'The rvalue is {rvalue}')
plt.ylabel('Average Student Spending')
plt.xlabel('Average student Graduation Rate')
plt.show()


# In[181]:


ga_student_spending_df.head()


# In[182]:


teacher_salaries_ga_df


# In[183]:


teacher_salaries_ga_df_1 = teacher_salaries_ga_df.T
teacher_salaries_ga_df_2 = teacher_salaries_ga_df_1.drop(['2017', '2018'])
teacher_salaries_ga_df_2


# In[184]:


x_salary = teacher_salaries_ga_df_2['Georgia']
x_salary


# In[193]:


ga_grad_rate = clean_grad_rate_df.loc[clean_grad_rate_df['State']=='GEORGIA']
ga_grad_sorted = ga_grad_rate.set_index('State')
ga_grad_sorted_df = ga_grad_sorted.drop(['Agg'], axis=1)
ga_grad_sorted_df


# In[194]:


ga_grad_sorted_df_1 = ga_grad_sorted_df.T
ga_grad_sorted_df_1


# In[195]:


# ga_grad_df_2 = ga_grad_sorted_df_1.drop(['Average'])
# ga_grad_df_2


# # In[196]:


# ga_grad_df_2['Teacher Salary'] = teacher_salaries_ga_df_2['Georgia']
# ga_grad_df_2

# In[ ]:
# More code to fix issue where values have been edited
# already in testing

ga_grad_df_2['Teacher Salary'] = teacher_salaries_ga_df_2['Georgia']

ga_grad_df_2
# In[197]:


fig = plt.figure()
ax = ga_grad_df_2['Teacher Salary'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
plt.ylabel('Teachers Salary in Dollars')
plt.xlabel('Years')
ax2 = ax.twinx()
ax2.plot(ga_grad_df_2['Georgia'].values, linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation rate')
plt.title('Average Graduation Rate Compared to Teachers Salaries in Georgia')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Teachers Salaries in Georgia')
plt.show()


# In[198]:


ratio_cleaned_df.head()


# In[199]:


# ratio_new = ratio_cleaned_df[['State', '2011_ratio', '2012_ratio', '2013_ratio', '2014_ratio', '2015_ratio', '2016_ratio']]
# ratio_new.head()
ratio_new = ratio_cleaned_df[['State', 'Ratio', 'Ratio.1', 'Ratio.2', 'Ratio.3', 'Ratio.4', 'Ratio.5']]
ratio_new.head()


# In[200]:


ratio_renamed = ratio_new.rename(columns={'2011_ratio': '2011', '2012_ratio': '2012', '2013_ratio': '2013',
                                         '2014_ratio': '2014', '2015_ratio': '2015', '2016_ratio': '2016'})
ratio_renamed.head()


# In[201]:


ga_ratio = ratio_renamed.loc[ratio_renamed['State'] == 'GEORGIA']
ga_ratio


# In[202]:


ga_ratio_set = ga_ratio.set_index('State')
ga_ratio_set


# In[203]:


ga_ratio_set_df= ga_ratio.T
ga_ratio_set_df


# In[204]:


rev_exp_df_1.head()


# In[205]:


ga_rev = rev_exp_df_1.loc[rev_exp_df_1['STATE'] == 'GEORGIA']
ga_rev


# In[206]:


ga_rev_df = ga_rev.set_index('YEAR')
ga_rev_df
ga_rev_df_1 = ga_rev_df.drop([2007, 2008, 2009], axis=0)
ga_rev_df_1


# In[207]:


# %%
pupil_spending_renamed = pupil_spending.rename(columns={"2007":"2007_PPS", "2008":"2008_PPS", "2009":"2009_PPS",
                                                       "2010":"2010_PPS", "2011":"2011_PPS", "2012":"2012_PPS", "2013":"2013_PPS",
                                                       "2014":"2014_PPS", "2015":"2015_PPS", "2016":"2016_PPS", 
                                                       "Unnamed: 2":"2007 pct_change", "Unnamed: 4":"2008 pct_change",
                                                       "Unnamed: 6":"2009 pct_change","Unnamed: 8":"2010 pct_change",
                                                       "Unnamed: 10":"2011 pct_change", "Unnamed: 12":"2012 pct_change",
                                                       "Unnamed: 14":"2013 pct_change", "Unnamed: 16":"2014 pct_change",
                                                       "Unnamed: 18":"2015 pct_change", "Unnamed: 20":"2016 pct_change"})

pupil_spending_renamed.head()


# %%
pupil_spending_df = pupil_spending_renamed.drop([0,1])
pupil_spending_df.head()


# %%
pupil_spending_df.isnull().sum()


# %%
ratio = pd.read_excel("data_files/teacher_student_ratio.xlsx")
ratio.head()


# %%
ratio_df = ratio.drop(['Unnamed: 1', 2007, 'Unnamed: 3', 'Unnamed: 4', 2008, 'Unnamed: 6', 'Unnamed: 7', 2009, 'Unnamed: 9','Unnamed: 10', 2010, 'Unnamed: 12'],axis=1)
ratio_df.head()


# %%
ratio_renamed_df = ratio_df.rename(columns={"Unnamed: 0":"State", "Unnamed: 13":"2011_staff", 2011:"2011_enrollment", 
                                     "Unnamed: 15":"2011_ratio", "Unnamed: 16":"2012_staff", 2012:"2012_enrollment", 
                                      "Unnamed: 18":"2012_ratio", "Unnamed: 19":"2013_staff", 2013:"2013_enrollment",
                                     "Unnamed: 21":"2013_ratio", "Unnamed: 22":"2014_staff", 2014:"2014_enrollment",
                                     "Unnamed: 24":"2014_ratio",  "Unnamed: 25":"2015_staff", 2015:"2015_enrollment",
                                     "Unnamed: 27":"2015_ratio", "Unnamed: 28":"2016_staff", 2016:"2016_enrollment",
                                     "Unnamed: 30":"2016_ratio"}) 
                                     
ratio_renamed_df.head()                                 
                                     


# %%
ratio_cleaned_df = ratio_renamed_df.drop([0])
ratio_cleaned_df.head()


# %%
ratio_cleaned_df.isnull().sum()


# %%
revenue.head()


# %%
revenue_grouped = revenue.groupby('YEAR')
revenue_grouped_df = pd.DataFrame(revenue_grouped['TOTAL_REVENUE'].sum()/1000000)
revenue_grouped_df


# %%
expenditure_grouped = revenue.groupby('YEAR')
expenditure_grouped_df = pd.DataFrame(revenue_grouped['TOTAL_EXPENDITURE'].sum()/1000000)
expenditure_grouped_df


# %%
x = expenditure_grouped_df.index
revenue = revenue_grouped_df['TOTAL_REVENUE']
expenditure = expenditure_grouped_df['TOTAL_EXPENDITURE']


# %%
revenue, = plt.plot(x, revenue, marker='o', color='blue', linewidth=2, label='Total Revenue')
expenditure, = plt.plot(x, expenditure, marker='+', color='red', linewidth=2, label='Total Expenditure')
plt.xlabel('Year')
plt.ylabel('Billion Dollars')
plt.title("Total Revenue and Expeditures by Year")
plt.legend(loc='best')
plt.grid()


# %%
revenue_grouped_df.plot(kind='line', label='Total_Revenue(billion)', figsize=(14,7))
plt.ylabel("Revenue in Billion")


# %%
revenue = pd.read_excel(file)
revenue


# %%
revenue_grouped_state = revenue.groupby('STATE')


# %%
revenue_grouped_state_df = pd.DataFrame(revenue_grouped_state['TOTAL_REVENUE'].mean()*10)
revenue_grouped_state_df.head()


# %%
exp_grouped_state_df = pd.DataFrame(revenue_grouped_state['TOTAL_EXPENDITURE'].mean()*10)
exp_grouped_state_df.head()


# %%
rev_exp_df = pd.merge(revenue_grouped_state_df, exp_grouped_state_df, on='STATE', how='outer')
rev_exp_df.head()


# %%
rev_exp_dif = rev_exp_df


# %%
rev_exp_dif['DIFFERENCE'] = rev_exp_df['TOTAL_REVENUE'] - rev_exp_df['TOTAL_EXPENDITURE']
rev_exp_dif.head()


# %%
rev_exp_dif_df = rev_exp_dif.drop(columns=['TOTAL_REVENUE', 'TOTAL_EXPENDITURE'], axis=1)
rev_exp_dif_df.head()


# %%
x = rev_exp_dif_df['DIFFERENCE']

# if x < 0:
#     colors = 'red'
# else:
#     colors = 'blue'
        
    
# colors[x>=0] = (0,0,1)
rev_exp_dif_df.plot(kind='barh', figsize=(25,45) )
plt.title("Revenue and Expediture with Deficit")
plt.xlabel('State')
plt.ylabel("Thousand Dollars")


# %%
rev_exp_df[['TOTAL_REVENUE', 'TOTAL_EXPENDITURE']].plot(kind='bar', figsize=(25,15))
plt.xlabel('Billion Dollars')
plt.title("Revenue and Expenditure by State")


# %%
revenue_grouped_state_df.plot(kind='barh', figsize=(10,25), color='green')
plt.xlabel('Revenue in billion')


# %%
exp_grouped_state_df.plot(kind='barh', figsize=(10,25), color='red')
plt.xlabel('Expenditures in billion')


# %%



# %%
ga_numbers = revenue.loc[revenue['STATE']=='GEORGIA']
ga_numbers_df = ga_numbers[['YEAR', 'TOTAL_REVENUE', 'TOTAL_EXPENDITURE']]
ga_numbers_df


# %%
ga_numbers_df_1 = ga_numbers_df.set_index('YEAR')
ga_numbers_df_1


# %%
ga_numbers_df_1.plot(kind='bar', figsize=(14,6))


# %%
clean_grad_rate = grad_rate_renamed #.astype(str).str.upper()


# %%
clean_grad_rate_df = clean_grad_rate
clean_grad_rate_df.head(50)


# %%
clean_grad_rate_df['Agg'] = clean_grad_rate.mean(axis=1)


# %%
grad_rate_cleaned = clean_grad_rate[['State', 'Agg']].set_index('State')
grad_rate_cleaned.head()


# %%
pupil_spending_renamed = pupil_spending_df.rename(columns={'STATE':'State'})
pupil_spending_renamed.head()


# %%
pupil_spending_df_1 = pupil_spending_renamed[['State', 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]].set_index('State')
pupil_spending_df_1.head()


# %%
pupil_spending_df_1['Avg'] = pupil_spending_df_1.mean(axis=1)
pupil_spending_df_1.head()


# %%
student_spending_df_2 = pupil_spending_df_1[['Avg']]
student_spending_df_2.head(50)
#student_spending_df_2.sort(ascending = "False")


# %%
student_spending_df_2['Avg'].plot(kind='bar', figsize=(16,8), label='Per Student Spending')
# x=student_spending_df_2['Avg']
# y=student_spending_df_2['Agg']
# ax2 = plt.twinx()
# ax2.plot(x,y)


# %%
student_spending_df_2['Agg'] = grad_rate_cleaned['Agg']
student_spending_df_2.head()


# %%
fig = plt.figure()
ax = student_spending_df_2['Avg'].plot(kind='bar', figsize=(16,8), label='Per Student Spending')
ax2 = ax.twinx()
ax2.plot(student_spending_df_2['Agg'].values, linestyle='-', marker='o', linewidth=2.0, color='red')


# %%
pupil_spending_us = round(pupil_spending_df_1.mean(),2)
student_spending_us = pd.DataFrame(pupil_spending_us)
student_spending_us_df = student_spending_us.drop(['Avg'])
student_spending_renamed = student_spending_us_df.rename(columns={0:'Avg_spent_per_student in US'})
student_spending_renamed


# %%
student_spending_renamed.plot(kind='line', marker='o', linestyle='-', linewidth=2, color='g', figsize=(10,5))
plt.title("Average Spend per Student")
plt.xlabel("Year")
plt.ylabel("Spend")
plt.grid()


# %%
clean_grad_rate_df.head()


# %%
grad_rate_df_reset = clean_grad_rate_df.set_index('State').drop(['Agg'], axis=1)
grad_rate_df_reset.head()


# %%
grad_rate_us = round(grad_rate_df_reset.mean(),2)
grad_rate_us_df = pd.DataFrame(grad_rate_us)
grad_rate_us_renamed = grad_rate_us_df.rename(columns={0:'Avg_graduation Rate in US'})
grad_rate_us_renamed


# %%
grad_rate_us_renamed.plot(kind='line', marker='^', linestyle='-', color='b', linewidth=2)
plt.title("Average Graduation Rate by Year")
plt.xlabel("Year")
plt.ylabel("Graduation Rate")
plt.grid()


# %%
clean_grad_rate_df.head(2)


# %%
ga_grad_rate = clean_grad_rate_df.loc[clean_grad_rate_df['State']=='Georgia']
ga_grad_sorted = ga_grad_rate.set_index('State')
ga_grad_sorted_df = ga_grad_sorted.drop(['Agg'], axis=1)
ga_grad_sorted_df


# %%
ga_grad_rate = clean_grad_rate_df.loc[clean_grad_rate_df['State']=='Georgia']
ga_grad_df = ga_grad_rate.drop(['Agg', 'State'], axis=1)
ga_grad_df


# %%
ga_grad_sorted_df_1 = ga_grad_sorted_df.T
ga_grad_sorted_df_1


# %%



# %%
x = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
y = [67, 70, 71.7, 70, 78.8, 79, 81]


# %%
plt.plot(x, y, marker='s', linestyle='-', color = 'red', linewidth = 2, label='Georgia_Graduation_Rate')
plt.legend(loc='best')
plt.xlabel("Year")
plt.ylabel("Graduation Rate")
plt.title("Graduation Rate in GA")
plt.grid()


# %%
ga_student_spending = pd.DataFrame(pupil_spending_df_1.loc['GEORGIA'])
ga_student_spending_rename = ga_student_spending.rename(columns={'GEORGIA':'Georgia Per Student Spending'})
ga_student_spending_df = ga_student_spending_rename.drop([2007, 2008, 2009, 'Avg'])
ga_student_spending_df


# %%
ga_student_spending_df.plot(kind='line', marker='o', linestyle='-', color = 'b', linewidth=2, label='Georgia Average Spending per Student')
plt.legend(loc='best')
plt.title("GA per Student Spending")
plt.xlabel("Year")
plt.ylabel("Spend")
plt.grid()


# %%
teacher_salaries_new.head(2)


# %%
teacher_salaries_us = round(teacher_salaries.mean(),2)
teacher_salaries_us_df = pd.DataFrame(teacher_salaries_us)
teacher_salaries_us_df.head()


# %%
teacher_salaries_renamed = teacher_salaries_us_df.rename(columns={0:'Average Teacher Salary in US'})
teacher_salaries_renamed.head(11)


# %%
teacher_salaries_renamed.plot(kind='line', marker='^', color='g', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Dollars')
plt.title('Average Teacher Salary by Year')
# plt.xlim(47000, 58000)
plt.grid()
plt.show


# %%
teacher_salaries_sorted = teacher_salaries_new.set_index('State')
teacher_salaries_sorted.head()


# %%
teacher_salaries_sorted['Avg'] = round(teacher_salaries_sorted.mean(axis=1),2)
teacher_salaries_sorted.head()


# %%
teacher_salaries_states = teacher_salaries_sorted[['Avg']]
teacher_salaries_states.head()


# %%
teacher_salaries_states.plot(kind='bar', figsize=(14,7), color='y')


# %%



# %%
teacher_salaries_ga = teacher_salaries.loc[teacher_salaries['State'] == 'Georgia']
teacher_salaries_ga_clean = teacher_salaries_ga.drop(columns=['2007', '2008', '2009'], axis=1)
teacher_salaries_ga_df = teacher_salaries_ga_clean.set_index('State')
teacher_salaries_ga_df


# %%
x_ga = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
y_ga = [53112, 52185, 52938, 52880, 52924, 53382, 54190, 55532, 56329]


# %%
plt.plot(x_ga, y_ga, label='Teachers Salaries in GA', marker='^', linestyle='-', color='b')
plt.legend(loc='best')
plt.xlabel('Years')
plt.ylabel('Dollars')
plt.title("Teacher Salaries in GA")
plt.grid()
plt.show()


# %%
revenue.head()


# %%
revenue_fed = revenue[['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenue_fed.head()


# %%
revenue_avg = pd.DataFrame(revenue_fed.mean())
revenue_avg


# %%
explode=(0.1,0,0)
revenue_avg.plot(kind='pie', explode=explode, autopct="%1.1f%%", shadow=True, subplots=True, figsize=(14,7))
plt.title("US Educational Revenue")
plt.axis('equal')


# %%
revenue_ga = revenue[['STATE', 'YEAR', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenue_ga.head()


# %%
revenue_ga_df = revenue_ga.loc[revenue_ga['STATE']=='GEORGIA']
rev_ga_df = revenue_ga_df.drop(['YEAR'], axis=1)
rev_ga_avg = rev_ga_df.mean()
rev_ga_df = pd.DataFrame(rev_ga_avg)
rev_ga_df                       


# %%
explode = (0.1,0,0)
rev_ga_df.plot(kind='pie', explode=explode, autopct = "%1.1f%%", shadow=True, subplots=True, figsize=(14,7))
plt.title("Georgia Educational Revenue")
plt.axis('equal')


# %%



# %%
rev_exp_df_1 = revenue[['STATE', 'YEAR', 'TOTAL_REVENUE', 'TOTAL_EXPENDITURE']]
rev_exp_df_1.head()


# %%
student_spending_df_2.head()


# %%
grad_rate_cleaned.head()


# %%
#x_grad = grad_rate_cleaned['Agg']
#y_spend = student_spending_df_2['Avg']
import seaborn as sns
x_spend = student_spending_df_2['Avg']
y_grad = grad_rate_cleaned['Agg']

(slope, intercept, rvalue, pvalue, stderr) = linregress(x_spend,y_grad)
regress_values = x_spend * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(x_spend,regress_values,"r-")
plt.annotate(line_eq, (20,60), fontsize=15,color="red")
plt.title("Graduation Rate vs Per Student Spend Regression")
plt.xlabel("Per Student Spend")
plt.ylabel("Graduation Percentage")
plt.scatter(x_spend,y_grad)
print(line_eq)
print(rvalue)
plt.show()


# %%
ga_student_spending_df.head()


# %%
x_grad = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
y_grad = [67, 70, 71.7, 70, 78.8, 79, 81]


# %%
x_salary = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
y_salary = [53112, 52185, 52938, 52880, 52924, 53382, 54190, 55532, 56329]


# %%
teacher_salaries_ga_df


# %%
teacher_salaries_ga_df_1 = teacher_salaries_ga_df.T
teacher_salaries_ga_df_1


# %%
x_grad


# %%
teacher_salaries_ga_df_1 = teacher_salaries_ga_df.T
teacher_salaries_ga_df_2 = teacher_salaries_ga_df_1.drop(['2017', '2018'])
teacher_salaries_ga_df_2


# %%
ga_grad_sorted_df_1['Teacher Salary'] = teacher_salaries_ga_df_2['Georgia']
ga_grad_sorted_df_1


# %%
fig = plt.figure()
ax = ga_grad_sorted_df_1['Teacher Salary'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
ax2 = ax.twinx()
ax.set_title('Georgia Teacher Salary and Graduation Rate')
ax.set_xlabel('Year')
ax.set_ylabel('Salary')
ax2.set_ylabel('Graduation Rate')
ax.grid()
ax2.plot(ga_grad_sorted_df_1['Georgia'].values, linestyle='-', marker='o', linewidth=2.0, color='red')


# %%
rev_exp_df_1.head()


# %%
ga_rev = rev_exp_df_1.loc[rev_exp_df_1['STATE'] == 'GEORGIA']
ga_rev


# %%
ga_rev_df = ga_rev.set_index('YEAR')
ga_rev_df
ga_rev_df_1 = ga_rev_df.drop([2007, 2008, 2009], axis=0)
ga_rev_df_1


# %%
ga_grad_sorted_df_1['Spending'] = teacher_salaries_ga_df_2['Georgia']
ga_grad_df_5 = ga_grad_sorted_df_1.drop(['Spending'],axis=1)
ga_grad_df_5


# In[208]:


# %%
ga_rev_df_1['Grad Rate'] = [67, 70, 71.7, 70, 78.8, 79, 81]
ga_rev_df_1


# In[209]:


# %%
ga_rev_df_2 = ga_rev_df_1.drop(['STATE', 'TOTAL_EXPENDITURE'], axis=1)
ga_rev_df_2


# In[210]:


fig = plt.figure()
ax = ga_rev_df_2['TOTAL_REVENUE'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Revenue')
plt.ylabel('Total Revenue in 10 Billion Dollars')
ax2 = ax.twinx()
ax2.plot(ga_rev_df_2['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation Rate')
plt.title('Average Graduation Rate Compared to Revenue in Georgia')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Revenue in Georgia')
plt.show()


# In[211]:


# %%
fig = plt.figure()
ax = ga_rev_df_2['TOTAL_REVENUE'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Revenue')
ax2 = ax.twinx()
ax.grid()
ax2.set_title("Georgia Total Revenue and Graduation Rate")
ax2.set_xlabel("Year")
ax2.set_ylabel("Graduation Rate")
ax.set_ylabel("Total Revenue")
ax2.plot(ga_rev_df_2['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')


# %%
ratio_cleaned_df.head()


# %%
ga_rev_df_3 = ga_rev_df_1.drop(['STATE', 'TOTAL_REVENUE'], axis=1)
ga_rev_df_3


# In[212]:


fig = plt.figure()
ax = ga_rev_df_3['TOTAL_EXPENDITURE'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Expenditure')
plt.ylabel('Total Expenditure in 10 Billion Dollars')
ax2 = ax.twinx()
ax2.plot(ga_rev_df_3['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation Rate')
plt.title('Average Graduation Rate Compared to Expenditure in Georgia')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Expenditure in Georgia')
plt.show()


# In[213]:


ga_student_spending_df


# In[214]:


# %%
fig = plt.figure()
ax = ga_rev_df_3['TOTAL_EXPENDITURE'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Expenditure')
ax2 = ax.twinx()
ax2.plot(ga_rev_df_3['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')


# %%
ga_student_spending_df


# %%
ga_student_spending_df['Grad Rate'] = [67,70,71.7, 70, 78.8, 79,81]
ga_student_spending_df


# In[215]:


fig = plt.figure()
ax = ga_student_spending_df['Georgia Per Student Spending'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Expenditure')
plt.ylabel('Average Per Student Spending in Georgia')
ax2 = ax.twinx()
ax2.plot(ga_student_spending_df['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation Rate')
plt.title('Average Graduation Rate Compared to Per Student Spending in Georgia')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Per Student Spending in Georgia')
# In[ ]:


grad_rate_renamed


# In[ ]:


# Make GA dataset for comparing to US
gaGradRate = pd.DataFrame(grad_rate_renamed.loc['Georgia'])
# Remove GA for comparison
allOtherGradRates = grad_rate_renamed.drop(index='Georgia')
anovaResult = stats.f_oneway(gaGradRate.T, allOtherGradRates)

anovaResult = pd.DataFrame(anovaResult)

anovaResult = anovaResult.rename(columns={0:"2010", 1:"2011", 2:"2012", 3:"2013", 4:"2014", 5:"2015", 6:"2016"})
anovaResult.index = ['Statistic', 'P Value']
anovaResult


# In[ ]:


# Gather regions from region data file
regionData = regionData.sort_values('Region')
statesData = regionData[['State','Region']].sort_values(['Region','State'])

# build new dataframes with states in region
gradRatesByStateRegion = statesData.join(grad_rate_renamed,on='State')
# Northeast, South, West
# Breakout data frames by region
midWestGradRates = gradRatesByStateRegion[gradRatesByStateRegion['Region'] == 'Midwest']
northEastGradRates = gradRatesByStateRegion[gradRatesByStateRegion['Region'] == 'Northeast']
westGradRates = gradRatesByStateRegion[gradRatesByStateRegion['Region'] == 'West']
southGradRates = gradRatesByStateRegion[gradRatesByStateRegion['Region'] == 'South']

midWestGradRates

# In[ ]:
yearSet = ['2010','2011','2012','2013','2014','2015','2016']

for year in yearSet:
    # get the avg value for each region that year
    ne = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'Northeast']
    s  = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'South']
    mw = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'Midwest']
    w  = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'West']


# In[ ]:
# Produce boxplots for years 2010-2016
yearSet = ['2010','2011','2012','2013','2014','2015','2016']

for year in yearSet:
    ne   = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'Northeast']
    s    = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'South']
    mw   = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'Midwest']
    w    = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] == 'West']
    USne = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] != 'Northeast']
    USs  = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] != 'South']
    USmw = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] != 'Midwest']
    USw  = gradRatesByStateRegion[year][gradRatesByStateRegion['Region'] != 'West']
    
    neTtestResults = stats.ttest_ind(ne,USne)
    sTtestResults = stats.ttest_ind(s,USs)
    mwTtestResults = stats.ttest_ind(mw,USmw)
    wTtestResults = stats.ttest_ind(w,USw)

    chartLabels = ['Northeast','South','Midwest','West']

    fig1, ax1 = plt.subplots()
    ax1.set_title('Region Graduation Rates ' + year)
    ax1.set_ylabel('Graduation %')
    ax1.set_ylim(0, 100)
    ax1.boxplot([
        # Get values for each region for current year
        ne,s,mw,w
        ],labels=chartLabels,)
    fileName = year + "RegionData.png"
    # output image
    dataFilePath = os.path.join(path,"image_files",fileName)
    plt.savefig(dataFilePath)
    # display plot
    plt.show()
    print(f"NE: {neTtestResults}")
    print(f"S: {sTtestResults}")
    print(f"MW: {mwTtestResults}")
    print(f"W: {wTtestResults}")

# In[6]:


#adding columns to add Average accross states into dataframe for United States

ratiorate=teachratio[['STATE','Ratio10','Ratio11','Ratio12','Ratio13','Ratio14','Ratio15','Ratio16']]

#getting data
r10=ratiorate['Ratio10'].mean()
r11=ratiorate['Ratio11'].mean()
r12=ratiorate['Ratio12'].mean()
r13=ratiorate['Ratio13'].mean()
r14=ratiorate['Ratio14'].mean()
r15=ratiorate['Ratio15'].mean()
r16=ratiorate['Ratio16'].mean()

#renaming columns 
ratiorate1 = pd.DataFrame({'STATE' : 'United States' , 'Ratio10': r10, 'Ratio11':r11, 'Ratio12':r12, 'Ratio13':r13, 'Ratio14':r14,'Ratio15':r15,'Ratio16':r16},index=[0])
new_df = pd.concat([ratiorate1,ratiorate]).reset_index(drop=True)


# In[7]:


#renaming columns
new_df=new_df.rename(columns={"STATE": "States"})
#dropping United States index
new_df = new_df.drop(new_df.index[0])
#exporting table to csv to be imported later.
export_csv = new_df.to_csv (r'data_files\Ratios.csv', index = None, header=True)
#dropping United States index
gradrate=gradrate.drop(gradrate.index[0])


# In[8]:


#merging tables together
gradvsrat = pd.merge(gradrate,new_df, on=["States"])


# In[9]:


#linear regression for 2010 values

rat10 = gradvsrat[2010]
r10 = gradvsrat['Ratio10']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r10,rat10)
regress_valuesR10 = r10 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r10,regress_valuesR10,"r-")
plt.annotate(line_eq,(9,65),fontsize=15,color="red")
sc = plt.scatter(r10,rat10)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2010 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[10]:


#linear regression for 2011 values

rat11 = gradvsrat[2011]
r11 = gradvsrat['Ratio11']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r11,rat11)
regress_valuesR11 = r11 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r11,regress_valuesR11,"r-")
plt.annotate(line_eq,(15,65),fontsize=15,color="red")
plt.scatter(r11,rat11)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2011 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[11]:


#linear regression for 2012 values

rat12 = gradvsrat[2012]
r12 = gradvsrat['Ratio12']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r12,rat12)
regress_valuesR12 = r12 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r12,regress_valuesR12,"r-")
plt.annotate(line_eq,(16,65),fontsize=15,color="red")
plt.scatter(r12,rat12)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2012 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[12]:


#linear regression for 2013 values

rat13 = gradvsrat[2013]
r13 = gradvsrat['Ratio13']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r13,rat13)
regress_valuesR13 = r13 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r13,regress_valuesR13,"r-")
plt.annotate(line_eq,(16,65),fontsize=15,color="red")
plt.scatter(r13,rat13)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2013 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[13]:


#linear regression for 2014 values

rat14 = gradvsrat[2014]
r14 = gradvsrat['Ratio14']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r14,rat14)
regress_valuesR14 = r14 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r14,regress_valuesR14,"r-")
plt.annotate(line_eq,(16,72),fontsize=15,color="red")
plt.scatter(r14,rat14)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2014 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[14]:


#linear regression for 2015 values

rat15 = gradvsrat[2015]
r15 = gradvsrat['Ratio15']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r15,rat15)
regress_valuesR15 = r15 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r15,regress_valuesR15,"r-")
plt.annotate(line_eq,(18,70),fontsize=15,color="red")
plt.scatter(r15,rat15)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2015 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[15]:


#linear regression for 2016 values

rat16 = gradvsrat[2016]
r16 = gradvsrat['Ratio16']
st = gradvsrat['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r16,rat16)
regress_valuesR16 = r16 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r16,regress_valuesR16,"r-")
plt.annotate(line_eq,(18,74),fontsize=15,color="red")
plt.scatter(r16,rat16)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage')
plt.title('2016 Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[16]:


#generating aggregate value
gradrate2['Agg'] = gradrate2.mean(axis=1)
teachratio2['Agg']=teachratio2.mean(axis=1)


# In[17]:


#dropping United States Value
gradrate2=gradrate2.drop(gradrate2.index[0])


# In[19]:


#linear regression for State Average across all years

ya = gradrate2['Agg']
xa = teachratio2['Agg']
(slope, intercept, rvalue, pvalue, stderr) = linregress(xa,ya)
regress_valuesRA = xa * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(xa,regress_valuesRA,"r-")
plt.annotate(line_eq,(16,65),fontsize=15,color="red")
plt.scatter(xa,ya)
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage') 
plt.title(' Average Graduation Rates vs Teacher/Student Ratio')
print(rvalue)
plt.show()


# In[21]:


#plotting all Regressions onto one plot to see trends

plt.plot(r10,regress_valuesR10,"r-",linestyle='dashed', label="2010")
plt.plot(r11,regress_valuesR11,"b-", label="2011")
plt.plot(r12,regress_valuesR12,"g-", linestyle='dashed', label="2012")
plt.plot(r13,regress_valuesR13,"y-", label="2013")
plt.plot(r14,regress_valuesR14,"p-", linestyle='dashed', label="2014")
plt.plot(r15,regress_valuesR15,"m-", label="2015")
plt.plot(r16,regress_valuesR16,"c-", linestyle='dashed', label="2016")
plt.plot(xa,regress_valuesRA,"k-", label="Aggregate")
plt.annotate(line_eq,(15,65),fontsize=25,color="red")
plt.legend()
plt.xlabel('Teacher Student Ratio')
plt.ylabel('Graduation Percentage') 
plt.title('Graduation Rate vs Teacher/Student Ratio Regressions')
plt.show()


# In[22]:


#renaming column keys for salaries
salaries = salaries.rename(columns={'State':'States'})
#dropping columns with data that isnt within range
salaries = salaries.drop(columns=['2007','2008','2009','2017','2018'])
#renaming column keys
salaries = salaries.rename(columns = {'2010':"Sal10",'2011':"Sal11",'2012':"Sal12",'2013':"Sal13",'2014':"Sal14",'2015':"Sal15",'2016':"Sal16"})
#dropping United States index
salaries= salaries.drop(salaries.index[0])
#exporting to csv for later use
export_csv = salaries.to_csv (r'data_files\gradvssal.csv', index = None, header=True)


# In[23]:


#merging dataframes on States
gradvssal = pd.merge(gradrate,salaries, on=["States"])


# In[37]:


#linear regression for 2010
rat10 = gradvssal[2010]
r10 = gradvssal['Sal10']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r10,rat10)
regress_valuesS10 = r10 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r10,regress_valuesS10,"r-")
plt.annotate(line_eq,(10,65),fontsize=15,color="red")
sc = plt.scatter(r10,rat10)
plt.title('2010 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[38]:


#linear regression for 2011
rat11 = gradvssal[2011]
r11 = gradvssal['Sal11']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r11,rat11)
regress_valuesS11 = r11 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r11,regress_valuesS11,"r-")
plt.annotate(line_eq,(60,40),fontsize=15,color="red")
plt.scatter(r11,rat11)
plt.title('2011 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[39]:


#linear regression for 2012
rat12 = gradvssal[2012]
r12 = gradvssal['Sal12']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r12,rat12)
regress_valuesS12 = r12 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r12,regress_valuesS12,"r-")
plt.annotate(line_eq,(65,45),fontsize=15,color="red")
plt.scatter(r12,rat12)
plt.title('2012 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[40]:


#linear regression for 2013
rat13 = gradvssal[2013]
r13 = gradvssal['Sal13']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r13,rat13)
regress_valuesS13 = r13 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r13,regress_valuesS13,"r-")
plt.annotate(line_eq,(65,60),fontsize=15,color="red")
plt.scatter(r13,rat13)
plt.title('2013 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[41]:


#linear regression for 2014
rat14 = gradvssal[2014]
r14 = gradvssal['Sal14']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r14,rat14)
regress_valuesS14 = r14* slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r14,regress_valuesS14,"r-")
plt.annotate(line_eq,(65,60),fontsize=15,color="red")
plt.scatter(r14,rat14)
plt.title('2014 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[42]:


#linear regression for 2015
rat15 = gradvssal[2015]
r15 = gradvssal['Sal15']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r15,rat15)
regress_valuesS15 = r15 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r15,regress_valuesS15,"r-")
plt.annotate(line_eq,(15,65),fontsize=15,color="red")
plt.scatter(r15,rat15)
plt.title('2015 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[43]:


#linear regression for 2016
rat16 = gradvssal[2016]
r16 = gradvssal['Sal16']
st = gradvssal['States']

(slope, intercept, rvalue, pvalue, stderr) = linregress(r16,rat16)
regress_valuesS16 = r16 * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(r16,regress_valuesS16,"r-")
plt.annotate(line_eq,(15,65),fontsize=15,color="red")
plt.scatter(r16,rat16)
plt.title('2016 Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
print(line_eq)
print(rvalue)
plt.show()


# In[44]:


#creating Aggregate value
salaries2['Agg']=salaries2.mean(axis=1)


# In[45]:


#Average Linear regression
ya = gradrate2['Agg']
xa = salaries2['Agg']
(slope, intercept, rvalue, pvalue, stderr) = linregress(xa,ya)
regress_valuesSA = xa * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.plot(xa,regress_valuesSA,"r-")
plt.annotate(line_eq,(79,50),fontsize=15,color="red",xycoords='axes points')
plt.scatter(xa,ya)
print(rvalue)
plt.title('Average Graduation Rate vs Teacher Salary Regression')
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage')
plt.savefig('gradvssalAVG.png')

plt.show()


# In[46]:


#plotting all Regressions onto one plot to see trends

plt.plot(r10,regress_valuesS10,"r-",linestyle='dashed', label="2010")
plt.plot(r11,regress_valuesS11,"b-", label="2011")
plt.plot(r12,regress_valuesS12,"g-", linestyle='dashed', label="2012")
plt.plot(r13,regress_valuesS13,"y-", label="2013")
plt.plot(r14,regress_valuesS14,"p-", linestyle='dashed', label="2014")
plt.plot(r15,regress_valuesS15,"m-", label="2015")
plt.plot(r16,regress_valuesS16,"c-", linestyle='dashed', label="2016")
plt.plot(xa,regress_valuesSA,"k-", label="Aggregate")
plt.annotate(line_eq,(15,65),fontsize=25,color="red")
plt.legend()
plt.xlabel('Teacher Salary')
plt.ylabel('Graduation Percentage') 
plt.title('Graduation Rate vs Teacher Salary Regressions')
plt.show()


# In[ ]:


# %%
fig = plt.figure()
ax = ga_student_spending_df['Georgia Per Student Spending'].plot(kind='line', marker='^', linestyle='-', color='b', label='Total Expenditure')
ax2 = ax.twinx()
ax.set_title("Georgia per Student Spend and Graduation Rate")
ax.grid()
ax.set_ylabel("Per Student Spending")
ax.set_xlabel("Years")
ax2.set_ylabel("Graduation Rate")
ax2.plot(ga_student_spending_df['Grad Rate'], linestyle='-', marker='o', linewidth=2.0, color='red')


# %%
ratio_new = ratio_cleaned_df[['State', '2011_ratio', '2012_ratio', '2013_ratio', '2014_ratio', '2015_ratio', '2016_ratio']]
ratio_new.head()


# %%
ratio_renamed = ratio_new.rename(columns={'2011_ratio': '2011', '2012_ratio': '2012', '2013_ratio': '2013',
                                         '2014_ratio': '2014', '2015_ratio': '2015', '2016_ratio': '2016'})
ratio_renamed.head()
ratio_renamed_set = ratio_renamed.set_index('State')
ratio_renamed_set.head()


# %%
ratio_avg = ratio_renamed_set.mean()
ratio_avg_df = pd.DataFrame(ratio_avg)
ratio_avg_df_1 = ratio_avg_df.rename(columns={0:'Ratio'})
ratio_avg_df_1


# %%
ratio_avg_df_1.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Ratio')
plt.title("Student : Teacher Ratio by Year")
plt.grid()
plt.ylim(13,16)
#plt.xlim(2010, 2016)


# %%
ratio_renamed.plot(kind='barh', figsize=(10,30))

df1 = (df.set_index(["location", "name"])
         .stack()
         .reset_index(name='Value')
         .rename(columns={'level_2':'Date'}))
# %%



# %%


