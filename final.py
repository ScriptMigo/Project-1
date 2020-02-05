# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


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


# %%
teacher_salaries_new_1 = teacher_salaries_new.drop(columns=['2007','2008','2009', '2017', '2018'], axis=1)
teacher_salaries_new_1.head()


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


# %%
grad_salary_renamed.iloc[2,6]= 80


# %%
grad_salary_df = grad_salary_renamed[['State', '2010_grad_rate', '2010_salary', '2011_grad_rate', '2011_salary', 
                                     '2012_grad_rate', '2012_salary', '2013_grad_rate', '2013_salary',
                                     '2014_grad_rate', '2014_salary', '2015_grad_rate', '2015_salary',
                                     '2016_grad_rate', '2016_salary']]

grad_salary_df.head()


# %%
grad_salary_df.head()


# %%



# %%
pupil_spending = pd.read_excel("data_files/per_pupil_spending.xlsx")
pupil_spending.head()


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


# %%
ga_rev_df_1['Grad Rate'] = [67, 70, 71.7, 70, 78.8, 79, 81]
ga_rev_df_1


# %%
ga_rev_df_2 = ga_rev_df_1.drop(['STATE', 'TOTAL_EXPENDITURE'], axis=1)
ga_rev_df_2


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


