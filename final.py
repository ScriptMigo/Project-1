#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# In[2]:


pip install nbconvert


# In[3]:


file = "data_files/revenue.xlsx"


# In[4]:


revenue_df = pd.read_excel(file)
revenue_df.head()


# In[5]:


revenue_df.isnull().sum()


# In[6]:


grad_rate = pd.read_excel("data_files/grad_rate.xlsx")
grad_rate.head()


# In[7]:


grad_rate_1 = pd.read_excel("data_files/grad_rate.xlsx")
grad_rate_1.head()


# In[8]:


grad_rate.shape


# In[9]:


grad_rate_df = grad_rate.rename(columns={"Unmaned: 0":"States", "Unnamed: 1":"2011", "Unnamed: 2":"2012", "Unnamed: 3":"2013", "Unnamed: 4":"2014", "Unnamed: 5":"2015", "Unnamed: 6":"2016", "Unnamed: 7": "Average" })
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


teacher_salaries = pd.read_excel("data_files/teacher_salaries_1.xlsx").round(0)
teacher_salaries_new = teacher_salaries.drop([0])
teacher_salaries_new.head()


# In[28]:


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

grad_salary_renamed.head()


# In[33]:


grad_salary_renamed.iloc[2,6]= 80


# In[39]:


grad_salary_df = grad_salary_renamed[['State', '2011_grad_rate', '2011_salary', 
                                     '2012_grad_rate', '2012_salary', '2013_grad_rate', '2013_salary',
                                     '2014_grad_rate', '2014_salary', '2015_grad_rate', '2015_salary',
                                     '2016_grad_rate', '2016_salary']]

grad_salary_df.head()


# In[40]:


grad_salary_df.head()


# In[41]:


pupil_spending = pd.read_excel("data_files/per_pupil_spending.xlsx")
pupil_spending.head()


# In[42]:


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


pupil_spending_df = pupil_spending_renamed.drop([0,1])
pupil_spending_df.head()


# In[44]:


pupil_spending_df.isnull().sum()


# In[45]:


ratio = pd.read_excel("data_files/teacher_student_ratio.xlsx")
ratio.head()


# In[46]:


ratio_df = ratio.drop(['Unnamed: 1', 2007, 'Unnamed: 3', 'Unnamed: 4', 2008, 'Unnamed: 6', 'Unnamed: 7', 2009, 'Unnamed: 9','Unnamed: 10', 2010, 'Unnamed: 12'],axis=1)
ratio_df.head()


# In[47]:


ratio_renamed_df = ratio_df.rename(columns={"Unnamed: 0":"State", "Unnamed: 13":"2011_staff", 2011:"2011_enrollment", 
                                     "Unnamed: 15":"2011_ratio", "Unnamed: 16":"2012_staff", 2012:"2012_enrollment", 
                                      "Unnamed: 18":"2012_ratio", "Unnamed: 19":"2013_staff", 2013:"2013_enrollment",
                                     "Unnamed: 21":"2013_ratio", "Unnamed: 22":"2014_staff", 2014:"2014_enrollment",
                                     "Unnamed: 24":"2014_ratio",  "Unnamed: 25":"2015_staff", 2015:"2015_enrollment",
                                     "Unnamed: 27":"2015_ratio", "Unnamed: 28":"2016_staff", 2016:"2016_enrollment",
                                     "Unnamed: 30":"2016_ratio"}) 
                                     
ratio_renamed_df.head()                                 
                                     


# In[48]:


ratio_cleaned_df = ratio_renamed_df.drop([0])
ratio_cleaned_df.head()


# In[49]:


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


ga_grad_df_2 = ga_grad_sorted_df_1.drop(['Average'])
ga_grad_df_2


# In[196]:


ga_grad_df_2['Teacher Salary'] = teacher_salaries_ga_df_2['Georgia']
ga_grad_df_2


# In[197]:


fig = plt.figure()
ax = ga_grad_df_2['Teacher Salary'].plot(kind='line', marker='^', linestyle='-', color='b', label='Teacher Salary')
plt.ylabel('Teachers Salary in Dollars')
plt.xlabel('Years')
ax2 = ax.twinx()
ax2.plot(ga_grad_df_2['GEORGIA'].values, linestyle='-', marker='o', linewidth=2.0, color='red')
plt.ylabel('Average Graduation rate')
plt.title('Average Graduation Rate Compared to Teachers Salaries in Georgia')
plt.grid()
plt.savefig('Average Graduation Rate Compared to Teachers Salaries in Georgia')
plt.show()


# In[198]:


ratio_cleaned_df.head()


# In[199]:


ratio_new = ratio_cleaned_df[['State', '2011_ratio', '2012_ratio', '2013_ratio', '2014_ratio', '2015_ratio', '2016_ratio']]
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


ga_grad_sorted_df_1['Spending'] = teacher_salaries_ga_df_2['Georgia']
ga_grad_df_5 = ga_grad_sorted_df_1.drop(['Spending'],axis=1)
ga_grad_df_5


# In[208]:


ga_rev_df_1['Grad Rate'] = [67, 70, 71.7, 70, 78.8, 79, 81]
ga_rev_df_1


# In[209]:


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
plt.show()


# In[ ]:




