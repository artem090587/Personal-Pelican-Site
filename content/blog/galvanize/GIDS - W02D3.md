Title: Galvanize - Week 02 - Day 3
Date: 2015-06-10 10:20
Modified: 2015-06-10 10:30
Category: Galvanize
Tags: data-science, galvanize, ab testing, statistics, hypthesis testing
Slug: galvanize-data-science-02-03
Authors: Bryan Smith
Summary: The eight day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered hypthesis testing, ab test, and played with messy data.

#Galvanize Immersive Data Science
##Week 2 - Day 3

Today we had a miniquiz on making a python package that could contain an arbitrary probability mass function as a dictionary.  It had to allow new values to be set, maintain a normalized pmf, and return probabilities for values in the dictionary, None otherwise.   

##Morning

The morning lecture was on Hypthesis testing and multiple testing corrections.  We reviewed the languaged, phrasing, caveots, and particulars about the frame works.  Our sprint involved investigating some questions involving multiple testing and clickthru rates

##Multiple Testing

- A study attempted to measure the influence of patients' astrological signs on their risk for heart failure.
   12 groups of patients (1 group for each astrological sign) were reviewed and the incidence of heart failure in each group was recorded. For each of the 12 groups, the researchers performed a z-test comparing the incidence of heart failure in one group to the incidence among the patients of all the other groups (i.e. 12 tests). The group with the highest rate of heart failure was Pisces, which had a p-value of .026 when assessing the null hypothesis that it had the same heart failure rate as all the other groups. What is the the problem with concluding from this p-value that Pisces have a higher rate of heart failure at a significance level of 0.05? How might you correct this p-value?
   
   **We have 12 tests, with a 5% false positive rate.  Using the Bernoulli Distribution we can see the chance of getting x number of false positive.**


    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as sc
    from __future__ import division
    
    
    plt.bar(np.arange(12),sc.binom.pmf(np.arange(12),12,0.05),color='indianred',alpha=0.5)
    plt.xlabel("False Positive Count")
    plt.ylabel('Probability of False Positive Count')
    plt.title("Probability Of False Positives For 12 Tests")
    print "Prob of 1 false postiive with 12 tests", sc.binom.pmf(1,12,0.05)
    print "Prob of 1 or More false positives with 12 test", 1-sc.binom.pmf(0,12,0.05)

    Prob of 1 false postiive with 12 tests 0.341280055366
    Prob of 1 or More false positives with 12 test 0.459639912337



![png](http://www.bryantravissmith.com/img/GW02D2/output_1_1.png)


**There is a 34% chance that one of the 12 tests will be a false positive value.  Instead, it would make more sense to chose a false positive rate such that the chance of having false positives out of 12 tests would be less than 0.05**


    p = np.linspace(0,1,10000)
    cdf = 1-sc.binom.pmf(0,12,p)
    plt.plot(p,cdf,label="CDF of P",color='indianred')
    plt.xlabel('False Positive Rate For 12 Tests')
    plt.ylabel('Prob of 1+ False Positives')
    plt.fill_between(p, cdf, where=cdf<0.05, interpolate=True, color='red',alpha=0.2)
    plt.show()
    print "New Significant Level: ", p[cdf < 0.05].max()


![png](http://www.bryantravissmith.com/img/GW02D2/output_3_0.png)


    New Significant Level:  0.004200420042


**In this case we would want to have a false positive rate for a single test of 0.0042.  By this criteria the p-value of 0.026 is not significant for the relation between Picese and heart failure. **

**The Bonferonni correct suggests taking the significant goal of an individual test and divide it by the number of tests.  This gives a value of 0.000417.  This is very close to the above results**


    print "Significance: ", 0.05/12

    Significance:  0.00416666666667


##Click Through Rate
We will use hypothesis testing to analyze **Click Through Rate (CTR)** on the New York Times website.
CTR is defined as the number of clicks the user make per impression that is made upon the user.
We are going to determine if there is statistically significant difference between the mean CTR for
the following groups:

1. Signed in users v.s. Not signed in users
2. Male v.s. Female
3. Each of 7 age groups against each other (7 choose 2 = 21 tests)

**Because we are construction 23 hypothesis tests on this data set, we can use the Bonferroni Correction of dividing the 5% false error rate by 23 to get:**

$$\alpha_{23} = 0.00217 $$

We can now load the data.


    nyt = pd.read_csv("../ab-testing/data/nyt1.csv")
    nyt.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 458441 entries, 0 to 458440
    Data columns (total 5 columns):
    Age            458441 non-null int64
    Gender         458441 non-null int64
    Impressions    458441 non-null int64
    Clicks         458441 non-null int64
    Signed_In      458441 non-null int64
    dtypes: int64(5)
    memory usage: 21.0 MB


There are not any null values in the data set, but we need to construct a click through rate for each user.   We will do this by:

1. Removing users without Impressions
2. Dividing the Clicks by Impressions



    nyt = nyt[nyt.Impressions!=0]
    nyt['CTR'] = np.nan
    nyt.CTR = nyt.Clicks.astype(float)/nyt2.Impressions.astype(float)
    nyt.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Impressions</th>
      <th>Clicks</th>
      <th>Signed_In</th>
      <th>CTR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>73</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the click through lets look at the distributions of variables.


    nyt2.hist(figsize=(15,8),bins=30,color='indianred',alpha=0.5)




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1088e9f10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1088d9a50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x108944150>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x104598750>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10882f990>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1088cafd0>]], dtype=object)




![png](http://www.bryantravissmith.com/img/GW02D2/output_11_1.png)


The age's with zero appear to be users that are not signed in by the height of the bars in the age and signed_in graphs.   The number of total clicks is less then 50,000, and the click through rates are small are mostly zero. 

To answer question 1, we need to split the date between signed-in users and signed-out users.   


    sign = nyt[nyt['Signed_In']==1]
    nosign = nyt[nyt['Signed_In']==0]
    plt.figure()
    sign.hist(figsize=(15,8),bins=10,color='indianred',alpha=0.5)
    plt.show()
    plt.figure()
    nosign.hist(figsize=(15,8),bins=10,color='indianred',alpha=0.5)
    plt.show()


    <matplotlib.figure.Figure at 0x10ed32690>



![png](http://www.bryantravissmith.com/img/GW02D2/output_13_1.png)



    <matplotlib.figure.Figure at 0x108bbc950>



![png](http://www.bryantravissmith.com/img/GW02D2/output_13_3.png)


We can construct a hypothesis test on the data where we have the following

- H0: The mean click through rate between signed in users and signed-out users are the same
- HA: The mean click through rates between signed-in users and signed-out users aer different
- We requrire a p-value less than 0.00217 to rejec the Null Hypothesis in favor for the Alterative

Do this with a Weltch's (Non Equal Variance Assigned) t-test results in the following results:


    sc.ttest_ind(sign.CTR, nosign.CTR, equal_var=False)




    (array(-55.37611793427461), 0.0)



The results of the test states that the test statistic to measure the difference is -55, given a p-value very close to zero, which is less than 0.00217.   In this case we can see there is a mean difference in the click through rates between the two populations.  In this case the signed-out users have a higher CTR than signed-in users.

Question two has to do with the difference in click through rates between genders.   This requires that we only investigate the users that are signed in, because signed out users do not have a gender variable.   Lets look at the data and perform a Wetch's t-test.


    plt.figure()
    sign[sign.Gender==0].hist(figsize=(15,8),bins=10,color='indianred',alpha=0.5)
    plt.show()
    plt.figure()
    sign[sign.Gender==1].hist(figsize=(15,8),bins=10,color='blue',alpha=0.5)
    plt.show()


    <matplotlib.figure.Figure at 0x10ed3a550>



![png](http://www.bryantravissmith.com/img/GW02D2/output_17_1.png)



    <matplotlib.figure.Figure at 0x109a90610>



![png](http://www.bryantravissmith.com/img/GW02D2/output_17_3.png)


A visual inspect of the data shows that men and women have similar distirubtions, with the exceptions fo the impressions distributions.  Men seem to have a wider range of impressons compared to women.  

The hypthesis test we are constructing is:

H0:  Signed-in Men and Women have the same mean CTR  
HA:  Signed-in Men and Women have different mean CTR  
*significance is 0.00217*  



    sc.ttest_ind(sign[sign.Gender==0].CTR, sign[sign.Gender==1].CTR, equal_var=False)




    (array(3.2897560659373846), 0.0010028527313066396)



Our test possted a t-value of 3.29 and a p-value of 0.001, which is significant.  We reject the null in favor of the alternative, concluding that men and women have different mean click through rates.  


    print "Click through rates (SI Men, SI Women, NOT Si)",sign[sign.Gender==1].CTR.mean(),sign[sign.Gender==1].CTR.mean(),nosign.CTR.mean()

    Click through rates (SI Men, SI Women, NOT Si) 0.0139185242976 0.0139185242976 0.0283549070617


The difference in the mean CTR between men and women is significant, but not large. The difference between both groups and not-signed-in users are much more different.

Now we will construct a set of hypothesis tests comparing different age groups against each other.   


    sign['AgeGroup'] = np.nan
    sign.loc[:,'AgeGroup'] = pd.cut(sign.Age, [0,7, 18, 24, 34, 44, 54, 64, 1000])
    df = pd.DataFrame(columns=['group1','group2','meanCTR1','meanCTR2','p_value','mean_difference'])
    k = 0
    AG = sign.AgeGroup.unique()
    for i,x in enumerate(AG):
        for j,y in enumerate(AG):
            if x != '(0, 7]' and y!='(0, 7]' and i > j:
                g1 = sign[sign.AgeGroup==x].CTR
                g2 = sign[sign.AgeGroup==y].CTR
                p = sc.ttest_ind(g1,g2, equal_var=False)[1]
                diff = g1.mean()-g2.mean()
                df.loc[k] = [x,y,g1.mean(),g2.mean(),p,diff]
                k += 1
    
    df = df[df.p_value < 0.00217]
    df = df.sort('mean_difference').reset_index().drop('index',axis=1)
    df

    /Library/Python/2.7/site-packages/IPython/kernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>group1</th>
      <th>group2</th>
      <th>meanCTR1</th>
      <th>meanCTR2</th>
      <th>p_value</th>
      <th>mean_difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(18, 24]</td>
      <td>(64, 1000]</td>
      <td>0.009720</td>
      <td>0.029803</td>
      <td>2.458627e-272</td>
      <td>-0.020082</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(44, 54]</td>
      <td>(64, 1000]</td>
      <td>0.009958</td>
      <td>0.029803</td>
      <td>1.430923e-295</td>
      <td>-0.019845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(24, 34]</td>
      <td>(64, 1000]</td>
      <td>0.010146</td>
      <td>0.029803</td>
      <td>7.860398e-285</td>
      <td>-0.019656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(18, 24]</td>
      <td>(7, 18]</td>
      <td>0.009720</td>
      <td>0.026585</td>
      <td>6.900980e-144</td>
      <td>-0.016865</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(54, 64]</td>
      <td>(64, 1000]</td>
      <td>0.020307</td>
      <td>0.029803</td>
      <td>9.214903e-56</td>
      <td>-0.009496</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(54, 64]</td>
      <td>(7, 18]</td>
      <td>0.020307</td>
      <td>0.026585</td>
      <td>8.273993e-20</td>
      <td>-0.006278</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(7, 18]</td>
      <td>(64, 1000]</td>
      <td>0.026585</td>
      <td>0.029803</td>
      <td>3.563408e-05</td>
      <td>-0.003218</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(54, 64]</td>
      <td>(34, 44]</td>
      <td>0.020307</td>
      <td>0.010286</td>
      <td>7.523228e-144</td>
      <td>0.010020</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(54, 64]</td>
      <td>(24, 34]</td>
      <td>0.020307</td>
      <td>0.010146</td>
      <td>5.668132e-141</td>
      <td>0.010160</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(54, 64]</td>
      <td>(44, 54]</td>
      <td>0.020307</td>
      <td>0.009958</td>
      <td>2.525271e-151</td>
      <td>0.010349</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(54, 64]</td>
      <td>(18, 24]</td>
      <td>0.020307</td>
      <td>0.009720</td>
      <td>1.007813e-130</td>
      <td>0.010586</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(7, 18]</td>
      <td>(34, 44]</td>
      <td>0.026585</td>
      <td>0.010286</td>
      <td>4.575147e-146</td>
      <td>0.016299</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(7, 18]</td>
      <td>(24, 34]</td>
      <td>0.026585</td>
      <td>0.010146</td>
      <td>7.449266e-146</td>
      <td>0.016439</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(7, 18]</td>
      <td>(44, 54]</td>
      <td>0.026585</td>
      <td>0.009958</td>
      <td>4.014382e-151</td>
      <td>0.016628</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(64, 1000]</td>
      <td>(34, 44]</td>
      <td>0.029803</td>
      <td>0.010286</td>
      <td>5.245541e-288</td>
      <td>0.019516</td>
    </tr>
  </tbody>
</table>
</div>



We itereated through all 21 groups and found 14 pairs who's mean CTR are different from each other with a signicance less than 0.00217.   These results can be summarized in the following way:

1.  People older than 64 have a statistically significant difference in mean CTR from all other age groups
2.  People between 54 adn 64 have a statistically significant difference in mean CTR from all other age groups
3.  People between 7 and 18 have a statistically significant difference in mean CTR from all other age groups

The order of CTR seems to be older than 65, 7-18 year olds, and then 54-64 year olds.  

Its interest that the high click through rate of non-signed-in users matches those of the older groups.   


## Afternoon

In the afternoon we learned about AB testing, and our paired programming assignment had to do with analyzing the results of simulated data for [Etsy](http://etsy.com).  We were asked to analysis the data from a given Tuesday where an AB test was performed on a website between two pages.   The goal is to change the conversion rate from 10% to 10.1% with this new page, a lift of 1%.  We are told that the weekend users are different than weeday users, and most of Etsy's revenue is made on the weekend.


    df = pd.read_csv('../ab-testing/data/experiment.csv')
    df.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4040615247</td>
      <td>1356998400</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4365389205</td>
      <td>1356998400</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4256174578</td>
      <td>1356998402</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8122359922</td>
      <td>1356998402</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6077269891</td>
      <td>1356998402</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    dfa = df[(df.ab=='treatment')&(df.landing_page=='new_page')]
    dfb = df[(df.ab=='control')&(df.landing_page=='old_page')]
    dfc = df[(df.ab=='control')&(df.landing_page=='new_page')]
    dfd = df[(df.ab=='treatment')&(df.landing_page=='old_page')]
    print dfa.user_id.nunique(),dfa.user_id.count()
    print dfb.user_id.nunique(),dfb.user_id.count()
    print dfc.user_id.nunique(),dfc.user_id.count()
    print dfd.user_id.nunique(),dfd.user_id.count()

    95574 95574
    90814 90815
    0 0
    4759 4759


Right away we have a problem in the data.  We have 4759 users that are in treatment group that have seen the new page and old page.  There is also one person in the control group that appear's twice.  

In my minds, a good experiment outlines how to handle missing data, mistakes, and errors in the analysis before they a found.  This is not the case for this assignment, so we will have to decided what to do ourselves.  I am going to exam the 4759 users.  First I will get a datetime in the dataframe, and then I will try to figure out what is happening with this group.


    df['dt'] = np.nan
    df['dt'] = pd.to_datetime(df.ts,unit='s')
    dfa = df[(df.ab=='treatment')&(df.landing_page=='new_page')].sort('user_id')
    dfb = df[(df.ab=='control')&(df.landing_page=='old_page')].sort('user_id')
    dfc = df[(df.ab=='control')&(df.landing_page=='new_page')].sort('user_id')
    dfd = df[(df.ab=='treatment')&(df.landing_page=='old_page')].sort('user_id')
    
    dfd.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>189992</th>
      <td>1033628</td>
      <td>1357084275</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>0</td>
      <td>2013-01-01 23:51:15</td>
    </tr>
    <tr>
      <th>151793</th>
      <td>1891740</td>
      <td>1357066970</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>0</td>
      <td>2013-01-01 19:02:50</td>
    </tr>
    <tr>
      <th>114751</th>
      <td>4557110</td>
      <td>1357050318</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>0</td>
      <td>2013-01-01 14:25:18</td>
    </tr>
    <tr>
      <th>99066</th>
      <td>5534964</td>
      <td>1357043251</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>0</td>
      <td>2013-01-01 12:27:31</td>
    </tr>
    <tr>
      <th>104055</th>
      <td>6180378</td>
      <td>1357045528</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>0</td>
      <td>2013-01-01 13:05:28</td>
    </tr>
  </tbody>
</table>
</div>




    dfa[dfa.user_id.isin(dfd.user_id)].head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>189991</th>
      <td>1033628</td>
      <td>1357084274</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 23:51:14</td>
    </tr>
    <tr>
      <th>151790</th>
      <td>1891740</td>
      <td>1357066969</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 19:02:49</td>
    </tr>
    <tr>
      <th>114746</th>
      <td>4557110</td>
      <td>1357050317</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 14:25:17</td>
    </tr>
    <tr>
      <th>99064</th>
      <td>5534964</td>
      <td>1357043250</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 12:27:30</td>
    </tr>
    <tr>
      <th>104052</th>
      <td>6180378</td>
      <td>1357045527</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 13:05:27</td>
    </tr>
  </tbody>
</table>
</div>



In these 5 cases we see that the users saw the new landing page, then one second later saw the old landing page.  None of these users displayed converted.


    print dfd[dfd.converted==1].user_id.count()
    dfd[dfd.converted==1].head()

    501





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100082</th>
      <td>10792592</td>
      <td>1357043708</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>1</td>
      <td>2013-01-01 12:35:08</td>
    </tr>
    <tr>
      <th>67975</th>
      <td>13223933</td>
      <td>1357029144</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>1</td>
      <td>2013-01-01 08:32:24</td>
    </tr>
    <tr>
      <th>42109</th>
      <td>27727121</td>
      <td>1357017491</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>1</td>
      <td>2013-01-01 05:18:11</td>
    </tr>
    <tr>
      <th>22040</th>
      <td>34535851</td>
      <td>1357008350</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>1</td>
      <td>2013-01-01 02:45:50</td>
    </tr>
    <tr>
      <th>87355</th>
      <td>85676035</td>
      <td>1357037889</td>
      <td>treatment</td>
      <td>old_page</td>
      <td>1</td>
      <td>2013-01-01 10:58:09</td>
    </tr>
  </tbody>
</table>
</div>




    print dfa[dfa.user_id.isin(dfd[dfd.converted==1].user_id)].user_id.count()
    dfa[dfa.user_id.isin(dfd[dfd.converted==1].user_id)].head()

    501





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100079</th>
      <td>10792592</td>
      <td>1357043707</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 12:35:07</td>
    </tr>
    <tr>
      <th>67973</th>
      <td>13223933</td>
      <td>1357029143</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 08:32:23</td>
    </tr>
    <tr>
      <th>42108</th>
      <td>27727121</td>
      <td>1357017490</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 05:18:10</td>
    </tr>
    <tr>
      <th>22038</th>
      <td>34535851</td>
      <td>1357008349</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 02:45:49</td>
    </tr>
    <tr>
      <th>87352</th>
      <td>85676035</td>
      <td>1357037888</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 10:58:08</td>
    </tr>
  </tbody>
</table>
</div>



We have 501 users that saw the new page, then 1 second later saw the old page and converted.  This does not make sense interms of our project/assignment, so I think the safest thing to do for this analysis is throw out these mistakes, and only do the analysis with the untainted results.  If this was a real experiment, I would definatley investigate the details of the test.

The goal of the experiment is to have a new pages that has a conversion lift of 1 percent.   With that goal in mine we define the following test:

H0:  The lift in conversions from the new page and old page is equal to 1%
HA:  the lift if conversions from the new page to the old page is less than 1%




    
    def z_test(old_conversion, new_conversion, old_nrow, new_nrow,
               effect_size=0., two_tailed=True, alpha=.05):
        """z-test"""
        conversion = (old_conversion * old_nrow + new_conversion * new_nrow) / \
                     (old_nrow + new_nrow)
    
        se = np.sqrt(conversion * (1 - conversion) * (1 / old_nrow + 1 / new_nrow))
    
        z_score = (new_conversion - old_conversion - effect_size) / se
    
        if not two_tailed:
            p_val = 1 - sc.norm.cdf(abs(z_score))
        else:
            p_val = (1 - sc.norm.cdf(abs(z_score))) * 2
    
        reject_null = p_val < alpha
        #print 'z-score: %s, p-value: %s, reject null: %s' % (z_score, p_val, reject_null)
        return z_score, p_val, reject_null
    
    
    
    conA,cntA = dfa.converted.mean(),dfa.converted.count()
    conB,cntB = dfb.converted.mean(),dfb.converted.count()
    print conA,conB,cntA,cntB,0.01*conB
    z_test(conA,conB,cntA,cntB,two_tailed=False,effect_size=0.01*conB)

    0.0996819218616 0.0996421296041 95574 90815 0.000996421296041





    (-0.74648172622270292, 0.22768823318094589, False)



In this frame there results are not significantly different from a 1% left that we can rule them out.  But we could have also tested if there is a difference between them.  


    z_test(conA,conB,cntA,cntB,two_tailed=True,effect_size=0.0)




    (-0.028666091979081442, 0.9771308999283459, False)



This is also not signifcant difference between the groups either.  The results of our Tuesday experiment are really inconclusive.  Ultimately we are concerned with the effect on weekend users, because they are responsible for most of Etsy's revenue.   We were told this user base is different from the weekend users.  

AirBNB had a talks ([here](http://nerds.airbnb.com/experiments-airbnb/) and [here](http://nerds.airbnb.com/experiments-at-airbnb/)) about looking at the hourly change in a p-vale, and examing if and when it level's off as the 'true' p-value for an experiment.   We are going to explore this method.  


    p_values = []
    effect = 0
    last_effect=0
    for i in range(23):
        
        # Grab the hour
        df_houra = dfa[dfa.dt.dt.hour<=i]
        df_hourb = dfb[dfb.dt.dt.hour<=i]
    
        conA,cntA = df_houra.converted.mean(),df_houra.converted.count()
        conB,cntB = df_hourb.converted.mean(),df_hourb.converted.count()
        
        p_values.append( z_test(conA,conB,cntA,cntB,two_tailed=False,effect_size=0.01*conB)[1] )
    
    plt.figure()
    plt.plot(range(23),p_values,color='indianred',alpha=0.8,lw=2)
    plt.hlines(0.05,0,23,color='red',alpha=0.6,lw=2,linestyle='--')
    plt.xlabel("Hours")
    plt.ylabel("P-values")
    plt.ylim([0,1])
    plt.xlim([0,23])
    plt.show()



![png](http://www.bryantravissmith.com/img/GW02D2/output_38_0.png)


From this method we see that the p-value is still changing, making me believe that our experiment could be under-powered.

There is additional data about the country of each user.  It could be interesting to look that these results by country.


    countries = pd.read_csv("../ab-testing/data/country.csv")
    countries.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9160993935</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5879439034</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8915383273</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2917824565</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3980216975</td>
      <td>UK</td>
    </tr>
  </tbody>
</table>
</div>




    dfac = dfa.merge(countries,on='user_id')
    dfbc = dfb.merge(countries,on='user_id')
    print dfac.user_id.count(),dfac.user_id.nunique()
    print dfbc.user_id.count(),dfbc.user_id.nunique()
    dfac.head()

    97151 92554
    87927 87924





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ts</th>
      <th>ab</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>dt</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23267</td>
      <td>1357066015</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 18:46:55</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79973</td>
      <td>1357018111</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>2013-01-01 05:28:31</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>338650</td>
      <td>1357083484</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 23:38:04</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>340147</td>
      <td>1357083599</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>1</td>
      <td>2013-01-01 23:39:59</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>382429</td>
      <td>1357002072</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>2013-01-01 01:01:12</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>




    dfac.groupby('user_id').country.count().max()




    2



We see that some users have two countries listed.  Since the user country data is not time sensitive, we have to drop them.  We do not know which country they were in at the type of the experiment.


    dfac = dfa.merge(countries.drop_duplicates('user_id'),on='user_id')
    print dfac.user_id.count(),dfac.user_id.nunique()
    dfbc = dfb.merge(countries.drop_duplicates('user_id'),on='user_id')
    print dfbc.user_id.count(),dfbc.user_id.nunique()

    92554 92554
    87925 87924



    
    p_values = {'US':[],'CA':[],'UK':[]}
    effect = {'US':0,'CA':0,'UK':0}
    last_effect=0
    for i in range(23):
        for country in dfac.country.unique():
        # Grab the hour
            df_houra = dfac[(dfac.dt.dt.hour<=i)&(dfac.country==country)]
            df_hourb = dfbc[(dfbc.dt.dt.hour<=i)&(dfbc.country==country)]
    
            conA,cntA = df_houra.converted.mean(),df_houra.converted.count()
            conB,cntB = df_hourb.converted.mean(),df_hourb.converted.count()
    
            p_values[country].append( z_test(conA,conB,cntA,cntB,two_tailed=False,effect_size=0.01*effect[country])[1] )
            effect[country] = conB
    
    plt.figure()
    plt.plot(range(23),p_values['US'],color='indianred',alpha=0.8,lw=2,label='US')
    plt.plot(range(23),p_values['CA'],color='blue',alpha=0.8,lw=2,label='CA')
    plt.plot(range(23),p_values['UK'],color='green',alpha=0.8,lw=2,label='UK')
    plt.hlines(0.05,0,23,color='red',alpha=0.6,lw=2,linestyle='--')
    plt.xlabel("Hours")
    plt.ylabel("P-values")
    plt.ylim([0,1])
    plt.xlim([0,23])
    plt.legend()
    plt.show()



![png](http://www.bryantravissmith.com/img/GW02D2/output_45_0.png)


If we constructed a hypthoesis at the end of day on Tuesday, the US would look like they were responsive to the new page, but the time plot shows that this could be a cherry-picked value.  If the experiment was allowed to run for more time, or we collect more data, the value would have changed.   It is clear from these plots that the sample sizes are not large enough that a additional data does not heavily influence the result.   I would be hesitant to make any strong conclusions about the new page based on these results.  


    
