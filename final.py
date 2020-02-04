#!/usr/bin/env python
# coding: utf-8
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


# In[ ]:


pupil_spending_df = pupil_spending_renamed.drop([0,1])
pupil_spending_df.head()


# In[ ]:


pupil_spending_df.isnull().sum()


# In[ ]:


ratio_df = ratio.drop(['Unnamed: 1', 2007, 'Unnamed: 3', 'Unnamed: 4', 2008, 'Unnamed: 6', 'Unnamed: 7', 2009, 'Unnamed: 9','Unnamed: 10', 2010, 'Unnamed: 12'],axis=1)
ratio_df.head()


# In[ ]:


ratio_renamed_df = ratio_df.rename(columns={"Unnamed: 0":"State", "Unnamed: 13":"2011_staff", 2011:"2011_enrollment", 
                                     "Unnamed: 15":"2011_ratio", "Unnamed: 16":"2012_staff", 2012:"2012_enrollment", 
                                      "Unnamed: 18":"2012_ratio", "Unnamed: 19":"2013_staff", 2013:"2013_enrollment",
                                     "Unnamed: 21":"2013_ratio", "Unnamed: 22":"2014_staff", 2014:"2014_enrollment",
                                     "Unnamed: 24":"2014_ratio",  "Unnamed: 25":"2015_staff", 2015:"2015_enrollment",
                                     "Unnamed: 27":"2015_ratio", "Unnamed: 28":"2016_staff", 2016:"2016_enrollment",
                                     "Unnamed: 30":"2016_ratio"}) 
                                     
ratio_renamed_df.head()                                 
                                     


# In[ ]:


ratio_cleaned_df = ratio_renamed_df.drop([0])
ratio_cleaned_df.head()


# In[ ]:


ratio_cleaned_df.isnull().sum()


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




