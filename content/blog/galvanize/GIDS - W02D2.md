Title: Galvanize - Week 02 - Day 2
Date: 2015-06-09 10:20
Modified: 2015-06-09 10:30
Category: Galvanize
Tags: data-science, galvanize, bootstraping, statistics, confidence interval
Slug: galvanize-data-science-02-02
Authors: Bryan Smith
Summary: The seventh day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered sampling methods, confidence intervals, and bootstrapping.

#Galvanize Immersive Data Science
##Week 2 - Day 2

Today we started with a mini-quiz, had a lecture on sampling methods, were given a talk about searching for a job, and finished the day with lecture on estimations/bootstraping and a reinforcement paired programming section.

##Mini-Quiz

The mini-quiz is interesting because it involved using pandas, which I have found to be great and flexible, while also mysterious.

We were given a salary dataset and asked to make some changes to it and answer some questions.  The first was to read in the data and convert the names to a human readiable text and transform variables to the correct type.


    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    salary = pd.read_csv("../estimation-sampling/data/salary_data.csv")
    salary.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32160 entries, 0 to 32159
    Data columns (total 5 columns):
    name          32160 non-null object
    job_title     32160 non-null object
    department    32160 non-null object
    salary        32160 non-null object
    Join Date     32160 non-null object
    dtypes: object(5)
    memory usage: 1.5+ MB



    salary.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>job_title</th>
      <th>department</th>
      <th>salary</th>
      <th>Join Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AARON,  ELVIA J</td>
      <td>WATER RATE TAKER</td>
      <td>WATER MGMNT</td>
      <td>$87228.0</td>
      <td>2000-09-27 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AARON,  JEFFERY M</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-08-04 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AARON,  KARINA</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-01-20 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AARON,  KIMBERLEI R</td>
      <td>CHIEF CONTRACT EXPEDITER</td>
      <td>GENERAL SERVICES</td>
      <td>$80916.0</td>
      <td>2000-04-27 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABAD JR,  VICENTE M</td>
      <td>CIVIL ENGINEER IV</td>
      <td>WATER MGMNT</td>
      <td>$99648.0</td>
      <td>2000-02-11 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




    salary.columns = ['Name','Position Title','Department','Employee Annual Salary','Join Date']
    salary.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Position Title</th>
      <th>Department</th>
      <th>Employee Annual Salary</th>
      <th>Join Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AARON,  ELVIA J</td>
      <td>WATER RATE TAKER</td>
      <td>WATER MGMNT</td>
      <td>$87228.0</td>
      <td>2000-09-27 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AARON,  JEFFERY M</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-08-04 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AARON,  KARINA</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-01-20 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AARON,  KIMBERLEI R</td>
      <td>CHIEF CONTRACT EXPEDITER</td>
      <td>GENERAL SERVICES</td>
      <td>$80916.0</td>
      <td>2000-04-27 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABAD JR,  VICENTE M</td>
      <td>CIVIL ENGINEER IV</td>
      <td>WATER MGMNT</td>
      <td>$99648.0</td>
      <td>2000-02-11 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



This is how I traditionally have renamed columns.   I learned a new way that involed using pandas' 'rename' function.


    salary = pd.read_csv('../estimation-sampling/data/salary_data.csv')
    salary.rename(columns={'name': 'Name',
                      'job_title': 'Position Title',
                      'department':'Department',
                      'salary':'Employee Annual Salary',
                       'join_data': 'Join Date'},
                       inplace=True)
    salary.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Position Title</th>
      <th>Department</th>
      <th>Employee Annual Salary</th>
      <th>Join Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AARON,  ELVIA J</td>
      <td>WATER RATE TAKER</td>
      <td>WATER MGMNT</td>
      <td>$87228.0</td>
      <td>2000-09-27 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AARON,  JEFFERY M</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-08-04 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AARON,  KARINA</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>$75372.0</td>
      <td>2000-01-20 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AARON,  KIMBERLEI R</td>
      <td>CHIEF CONTRACT EXPEDITER</td>
      <td>GENERAL SERVICES</td>
      <td>$80916.0</td>
      <td>2000-04-27 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABAD JR,  VICENTE M</td>
      <td>CIVIL ENGINEER IV</td>
      <td>WATER MGMNT</td>
      <td>$99648.0</td>
      <td>2000-02-11 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



I personally do not like these names for the columns because they involve spaces.   That removes the ability to us the pd.variable notation.   

I have also found multiple ways to update a variable type.  I am still not sure if there is a better method.


    salary['Employee Annual Salary'] = salary['Employee Annual Salary'].str.replace("$","").astype(float)
    salary['Join Date'] = pd.to_datetime(salary['Join Date'])
    salary.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32160 entries, 0 to 32159
    Data columns (total 5 columns):
    Name                      32160 non-null object
    Position Title            32160 non-null object
    Department                32160 non-null object
    Employee Annual Salary    32160 non-null float64
    Join Date                 32160 non-null datetime64[ns]
    dtypes: datetime64[ns](1), float64(1), object(3)
    memory usage: 1.5+ MB


Now that we have the data in the correct format, we can now answer questions about the dataset.  

1.  What are the top 5 paying job titles?
2.  How many people have "Police" in their title?
3.  What fraction of the people in 2 are a 'Police Officer'
4.  How many people were hired from July 30, 2000 to Aug 08, 2000



    salary.groupby('Position Title')['Employee Annual Salary'].mean().order(ascending=False).head()




    Position Title
    SUPERINTENDENT OF POLICE          260004
    MAYOR                             216210
    FIRE COMMISSIONER                 202728
    FIRST DEPUTY SUPERINTENDENT       188316
    FIRST DEPUTY FIRE COMMISSIONER    188316
    Name: Employee Annual Salary, dtype: float64




    print "Contains 'POLICE': ", salary[salary['Position Title'].str.contains('POLICE')]['Position Title'].count()
    salary[salary['Position Title'].str.contains('POLICE')]['Position Title'].value_counts(normalize=True)

    Contains 'POLICE':  11141





    POLICE OFFICER                                      0.847051
    POLICE OFFICER (ASSIGNED AS DETECTIVE)              0.076025
    POLICE COMMUNICATIONS OPERATOR II                   0.019747
    POLICE COMMUNICATIONS OPERATOR I                    0.012925
    POLICE OFFICER / FLD TRNG OFFICER                   0.010232
    POLICE OFFICER (ASSIGNED AS EVIDENCE TECHNICIAN)    0.006463
    POLICE OFFICER/EXPLSV DETECT K9 HNDLR               0.003590
    POLICE OFFICER (ASGND AS MARINE OFFICER)            0.002783
    POLICE CADET                                        0.002603
    ELECTRICAL MECHANIC-AUTO-POLICE MTR MNT             0.002423
    MACHINIST (AUTO) POLICE MOTOR MAINT                 0.002244
    POLICE OFFICER (ASSIGNED AS CANINE HANDLER)         0.001885
    POLICE OFFICER (ASSIGNED AS TRAFFIC SPECIALIST)     0.001795
    SUPERVISING POLICE COMMUNICATIONS OPERATOR          0.001616
    POLICE OFFICER (ASGND AS MOUNTED PATROL OFFICER)    0.001346
    POLICE OFFICER (ASSIGNED AS SECURITY SPECIALIST)    0.001346
    POLICE AGENT                                        0.001167
    POLICE FORENSIC INVESTIGATOR I                      0.001077
    POLICE OFFICER(ASGND AS LATENT PRINT EX)            0.000987
    POLICE OFFICER (PER ARBITRATION AWARD)              0.000898
    POLICE TECHNICIAN                                   0.000539
    POLICE LEGAL OFFICER II                             0.000359
    POLICE LEGAL OFFICER I                              0.000269
    DIR OF POLICE RECORDS                               0.000090
    MANAGER OF POLICE PAYROLLS                          0.000090
    POLICE OFFICER(ASGND AS SUPVG LATENT PRINT EX)      0.000090
    EXECUTIVE DIR - POLICE BOARD                        0.000090
    SUPERINTENDENT OF POLICE                            0.000090
    ASST SUPVSR OF POLICE RECORDS                       0.000090
    MANAGER OF POLICE PERSONNEL                         0.000090
    dtype: float64




    salary.set_index('Join Date',inplace=True)
    salary.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Position Title</th>
      <th>Department</th>
      <th>Employee Annual Salary</th>
    </tr>
    <tr>
      <th>Join Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-09-27</th>
      <td>AARON,  ELVIA J</td>
      <td>WATER RATE TAKER</td>
      <td>WATER MGMNT</td>
      <td>87228</td>
    </tr>
    <tr>
      <th>2000-08-04</th>
      <td>AARON,  JEFFERY M</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>75372</td>
    </tr>
    <tr>
      <th>2000-01-20</th>
      <td>AARON,  KARINA</td>
      <td>POLICE OFFICER</td>
      <td>POLICE</td>
      <td>75372</td>
    </tr>
    <tr>
      <th>2000-04-27</th>
      <td>AARON,  KIMBERLEI R</td>
      <td>CHIEF CONTRACT EXPEDITER</td>
      <td>GENERAL SERVICES</td>
      <td>80916</td>
    </tr>
    <tr>
      <th>2000-02-11</th>
      <td>ABAD JR,  VICENTE M</td>
      <td>CIVIL ENGINEER IV</td>
      <td>WATER MGMNT</td>
      <td>99648</td>
    </tr>
  </tbody>
</table>
</div>




    salary.ix['2000-07-13' : '2000-08-13'].count()




    Name                      2866
    Position Title            2866
    Department                2866
    Employee Annual Salary    2866
    dtype: int64



## Morning Sprint

The individual morning sprint covered sampling and estimation.   We were given a dataset on rain fall and attempted to use [Method of Moments](http://en.wikipedia.org/wiki/Method_of_moments_%28statistics%29) estimates on the data to approximate the distributions.  We then followed up by looking at [Maximum Likelihood Estimates](http://en.wikipedia.org/wiki/Maximum_likelihood) of the parameters.  

I first looked at the data for January rainfall over the course of several years.


    data = pd.read_csv("../estimation-sampling/data/rainfall.csv")
    print data.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 140 entries, 0 to 139
    Data columns (total 13 columns):
    Year    140 non-null int64
    Jan     140 non-null float64
    Feb     140 non-null float64
    Mar     140 non-null float64
    Apr     140 non-null float64
    May     140 non-null float64
    Jun     140 non-null float64
    Jul     140 non-null float64
    Aug     140 non-null float64
    Sep     140 non-null float64
    Oct     140 non-null float64
    Nov     140 non-null float64
    Dec     140 non-null float64
    dtypes: float64(12), int64(1)
    memory usage: 15.3 KB
    None



    data.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Jan</th>
      <th>Feb</th>
      <th>Mar</th>
      <th>Apr</th>
      <th>May</th>
      <th>Jun</th>
      <th>Jul</th>
      <th>Aug</th>
      <th>Sep</th>
      <th>Oct</th>
      <th>Nov</th>
      <th>Dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1871</td>
      <td>2.76</td>
      <td>4.58</td>
      <td>5.01</td>
      <td>4.13</td>
      <td>3.30</td>
      <td>2.98</td>
      <td>1.58</td>
      <td>2.36</td>
      <td>0.95</td>
      <td>1.31</td>
      <td>2.13</td>
      <td>1.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1872</td>
      <td>2.32</td>
      <td>2.11</td>
      <td>3.14</td>
      <td>5.91</td>
      <td>3.09</td>
      <td>5.17</td>
      <td>6.10</td>
      <td>1.65</td>
      <td>4.50</td>
      <td>1.58</td>
      <td>2.25</td>
      <td>2.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1873</td>
      <td>2.96</td>
      <td>7.14</td>
      <td>4.11</td>
      <td>3.59</td>
      <td>6.31</td>
      <td>4.20</td>
      <td>4.63</td>
      <td>2.36</td>
      <td>1.81</td>
      <td>4.28</td>
      <td>4.36</td>
      <td>5.94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1874</td>
      <td>5.22</td>
      <td>9.23</td>
      <td>5.36</td>
      <td>11.84</td>
      <td>1.49</td>
      <td>2.87</td>
      <td>2.65</td>
      <td>3.52</td>
      <td>3.12</td>
      <td>2.63</td>
      <td>6.12</td>
      <td>4.19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1875</td>
      <td>6.15</td>
      <td>3.06</td>
      <td>8.14</td>
      <td>4.22</td>
      <td>1.73</td>
      <td>5.63</td>
      <td>8.12</td>
      <td>1.60</td>
      <td>3.79</td>
      <td>1.25</td>
      <td>5.46</td>
      <td>4.30</td>
    </tr>
  </tbody>
</table>
</div>




    plt.figure()
    data.Jan.hist(bins=30,color='red',alpha=.2)
    plt.title("Rain Fall In Janary For All Years")
    plt.xlabel("Rain Values")
    plt.ylabel("Count")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D2/output_16_0.png)


To me this looks like it could be well fitted by a [Poisson distribution](http://en.wikipedia.org/wiki/Poisson_distribution) or a [Gamma distribution](http://en.wikipedia.org/wiki/Gamma_distribution).   A poisson distribution models random events occuring in a fixed time interval, and is a discreate distribution.  Gamma distributions are a continuous distribution that model how long one must way for N events to happen.   That seems like a better framework to think about rain fall.

The mean and variance for a Poisson distribution is the lambda parameter:

$$\mu = \lambda$$

So we will use this to see the fit.


    import scipy.stats as sc
    
    mean = data.Jan.mean()
    var = data.Jan.var()
    print mean,var
    x = np.arange(0,16)
    y = sc.poisson.pmf(x,mean)
    plt.figure()
    data.Jan.hist(bins=30,color='red',alpha=.2,normed=True,label='Jan Rain')
    
    plt.plot(x,y,color='red',label='Poisson Fit')
    plt.title("Rain Fall In Janary For All Years")
    plt.xlabel("Rain Values")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    4.54457142857 6.91677463515



![png](http://www.bryantravissmith.com/img/GW02D2/output_18_1.png)


This gives a fair fit to the distribution for january.   I want to compare this with the gama distribution.

The gamma function is given by:

$$X = Gamma(\alpha \ ,\beta) = \frac{\beta^\alpha \ x^{\alpha-1} \ e^{-\beta x}}{\Gamma(\alpha)}$$

The mean and variance of the gamma distribution is given by

$$\mu = \frac{\alpha}{\beta}$$

$$\sigma^2 = \frac{\alpha}{\beta^2}$$

So the estimate of alpha and beta are given by:

$$\beta = \frac{\mu}{\sigma^2}$$

$$\alpha = \frac{\mu^2}{\sigma^2}$$


    beta = mean/var
    alpha = mean**2/var
    print alpha,beta
    x1 = np.linspace(0,16,100)
    y1 = sc.gamma.pdf(x1,alpha,scale=1/beta)
    plt.figure()
    data.Jan.hist(bins=30,color='red',alpha=.2,normed=True,label='Jan Rain')
    plt.plot(x,y,color='red',linestyle='--',label='Poisson Fit')
    plt.plot(x1,y1,color='red',label='Gamma Fit',linestyle='-')
    plt.title("Rain Fall In Janary For All Years")
    plt.xlabel("Rain Values")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    2.98594801173 0.65703621533



![png](http://www.bryantravissmith.com/img/GW02D2/output_20_1.png)


The Gamma distribution fit matches the distribution's peak and tail better than the Poisson distribution fit.  There are method's to test the relative fit but I saved that for another day.

Now lets look at the Gamma fits for all months.  The reason we bin by months is that we have the prior that rain and weather is season.  


    f, axarr = plt.subplots(3, 4,figsize=(14, 8))
    
    x = np.linspace(0,16,100)
    for i,month in enumerate(data.columns[1:]):
        mean = data[month].mean()
        var = data[month].var()
        alpha = mean**2/var
        beta = mean/var
        y = sc.gamma.pdf(x,alpha,scale=1/beta)
        axarr[i/4,i%4].hist(data[month],bins=20,color='red',alpha=0.25,normed=True)
        axarr[i/4,i%4].set_xlim([0,20])
        axarr[i/4,i%4].set_ylim([0,.35])
        axarr[i/4,i%4].set_xlabel("Rain Fall")
        axarr[i/4,i%4].set_ylabel("Prob Density")
        axarr[i/4,i%4].set_title(month)
        axarr[i/4,i%4].plot(x,y,label="Gamma Fit",color='red')
        label = 'alpha = %.2f\nbeta = %.2f' % (alpha, beta)
        axarr[i/4,i%4].annotate(label, xy=(4, 0.25))
    
    plt.tight_layout()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D2/output_22_0.png)


We have a Method of Moments fit of the gamma distribution of rainfall for each month.   Now lets doo Maximum Likely Hood.  

First we need to make a funciton.  In order to test the method I will try it on a poisson generated dataset.  


    def poisson_likelihood(x, lam):
        return sc.poisson.pmf(x,lam)
    
    ##produces the probabilty of of each lambda(lam) given a value of 6
    plt.plot(range(1,20),[poisson_likelihood(6, lam) for lam in range(1,20)])
    plt.xlabel("Lambda Value")
    plt.ylabel("Probability of Lambda Value Given x=6")




    <matplotlib.text.Text at 0x10c5d0290>




![png](http://www.bryantravissmith.com/img/GW02D2/output_24_1.png)


This make sense because the maximum likelihood is 6, but there are still changes that the value is different.  Lets run this on the data now.


    p_data = pd.read_csv('../estimation-sampling/data/poisson.txt',header=None)
    p_data.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




    x_values = np.linspace(1,20,1000)
    likelihoods = np.array([np.log(poisson_likelihood(p_data.values[:,0],i)).sum() for i in x_values])
    plt.plot(x_values,likelihoods)
    plt.ylabel("Log Likelyhood For Lambda Fro Data")
    plt.xlabel("Lambda Values")




    <matplotlib.text.Text at 0x10c67f090>




![png](http://www.bryantravissmith.com/img/GW02D2/output_27_1.png)


We want to compare the maximum likelihood (argmax) of this distribution to the mean fo the data since the mean of Poisson distribution should be the lambda parameter.


    x_values[likelihoods.argmax()],p_data.mean()




    (5.0510510510510516, 0    5.0437
     dtype: float64)



The maximum likelihood estimate of the lambda parameter is very close to the sample mean, which also matches the value of lambda used to generated the data ($$\lambda = 5$$).

Scipy Stats has a fit function for each of the distributions that uses the maximum likelihood method.   I want to compare the plots the difference between the fits of the Method of Moments and the Maximum Likelihood.  


    mean = data.Jan.mean()
    var = data.Jan.var()
    beta = mean/var
    alpha = mean**2/var
    
    x1 = np.linspace(0,16,100)
    y1 = sc.gamma.pdf(x1,alpha,scale=1./beta)
    alpha_MLE,loc_MFL,one_over_beta_MLE = sc.gamma.fit(data.Jan,floc=0) #Loc = 0 - no rainfall minimum
    y2 = sc.gamma.pdf(x1,alpha_MLE,scale=one_over_beta_MLE)
    plt.figure()
    data.Jan.hist(bins=30,color='red',alpha=.2,normed=True,label='Jan Rain')
    plt.plot(x1,y1,color='red',label='Gamma Fit (MoM)',linestyle='--')
    plt.plot(x1,y2,color='red',label='Gamma Fit (MLE)',linestyle='-')
    plt.title("Rain Fall In Janary For All Years")
    plt.xlabel("Rain Values")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    print alpha, alpha_MLE
    print beta, beta_MLE

    2.98594801173 0.65703621533



![png](http://www.bryantravissmith.com/img/GW02D2/output_31_1.png)


    2.98594801173 3.25219914651
    0.65703621533 1.39738411574


We can see the two distributions are similar, but slightly different.   The MLE is more skewed right, and the MLE fits are larger then the method of moments.  I just want to finish this section with a plot of all the months.  


    plt.figure()
    f, axarr = plt.subplots(3, 4,figsize=(14, 10))
    x = np.linspace(0,16,100)
    for i,month in enumerate(data.columns[1:]):
        mean = data[month].mean()
        var = data[month].var()
        alpha = mean**2/var
        beta = mean/var
        fits = sc.gamma.fit(data[month],floc=0)
        alpha_fit = fits[0]
        beta_fit = 1./fits[2]
        y = sc.gamma.pdf(x,alpha,scale=1./beta)
        y1 = sc.gamma.pdf(x,alpha_fit,scale=1./beta_fit)
        print month, alpha, alpha_fit, beta, beta_fit
        axarr[i/4,i%4].hist(data[month],bins=20,color='red',alpha=0.25,normed=True)
        axarr[i/4,i%4].set_xlim([0,20])
        axarr[i/4,i%4].set_ylim([0,.35])
        axarr[i/4,i%4].set_xlabel("Rain Fall")
        axarr[i/4,i%4].set_ylabel("Prob Density")
        axarr[i/4,i%4].set_title(month)
        axarr[i/4,i%4].plot(x,y,label="MOD",linestyle='-',color='red',lw=3,alpha=0.5)
        axarr[i/4,i%4].plot(x,y1,label="MLE",linestyle='--',color='green',lw=3,alpha=0.5)
        axarr[i/4,i%4].legend()
    
    plt.tight_layout()
    plt.show()

    Jan 2.98594801173 3.25219914651 0.65703621533 0.715622847528
    Feb 3.0418721755 3.0803224672 0.740681272732 0.750043734186
    Mar 4.67867768543 4.64576013378 0.946813252137 0.94015180285
    Apr 4.28032831959 4.25175417646 1.01660157558 1.00981505905
    May 3.53902182175 3.82055049055 0.815644176548 0.880528551612
    Jun 2.97083704473 2.89974494499 0.765862938963 0.747535846757
    Jul 3.98358361758 3.6314371778 1.02531889482 0.934681309897
    Aug 3.02948804114 3.20185713476 0.907886646459 0.959542766645
    Sep 2.29238948802 2.14489278382 0.678766821038 0.635093671449
    Oct 2.46786112353 1.96116795936 0.945359556993 0.751261428601
    Nov 3.69539855825 3.30306402837 1.00014653216 0.893962581139
    Dec 3.23590721109 3.52396472416 0.77216125712 0.840898349041



    <matplotlib.figure.Figure at 0x10c387e10>



![png](http://www.bryantravissmith.com/img/GW02D2/output_33_2.png)


For each moths the distributions are similar, but like january the alpha and beta values are different. 

##Kernal Density Estimates

The last topic we had was to use the non-parametric method for fitting a distributions using gaussian kernal density estimates.   The idea is that each data point is fit with a gausian for some unknown variance, and the variance is shared for each such gaussian.   The variance paramenter is adjusted until an 'optimal' fit is found.

We can do this with an example by convoluting two gaussian data sets.


    data2 = [sc.norm.rvs(loc=0,scale=2) for x in range(500)]+[sc.norm.rvs(loc=4,scale=1) for x in range(400)]
    plt.figure()
    plt.hist(data,bins=30,color='red',alpha=0.2)
    plt.xlabel("Values")
    plt.ylabel("Counts")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D2/output_35_0.png)



    fit = sc.gaussian_kde(data2)
    plt.figure()
    plt.hist(data,bins=30,normed=True,alpha=0.2)
    x=np.linspace(-6,8,100)
    plt.plot(x,fit(x),color='blue',label='KDE Fit')
    plt.xlabel("Values")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D2/output_36_0.png)


The fit function can be used for a density estimate when we do not wish to model the data with a particlar model.


    plt.figure()
    f, axarr = plt.subplots(3, 4,figsize=(14, 10))
    x = np.linspace(0,16,100)
    for i,month in enumerate(data.columns[1:]):
        mean = data[month].mean()
        var = data[month].var()
        alpha = mean**2/var
        beta = mean/var
        fits = sc.gamma.fit(data[month],floc=0)
        alpha_fit = fits[0]
        beta_fit = 1./fits[2]
        y = sc.gamma.pdf(x,alpha,scale=1./beta)
        y1 = sc.gamma.pdf(x,alpha_fit,scale=1./beta_fit)
        gfit = sc.gaussian_kde(data[month])
        yg = gfit(x)
        axarr[i/4,i%4].hist(data[month],bins=20,color='red',alpha=0.25,normed=True)
        axarr[i/4,i%4].set_xlim([0,20])
        axarr[i/4,i%4].set_ylim([0,.35])
        axarr[i/4,i%4].set_xlabel("Rain Fall")
        axarr[i/4,i%4].set_ylabel("Prob Density")
        axarr[i/4,i%4].set_title(month)
        axarr[i/4,i%4].plot(x,y,label="MOD",linestyle='-',color='red',lw=3,alpha=0.5)
        axarr[i/4,i%4].plot(x,y1,label="MLE",linestyle='--',color='green',lw=3,alpha=0.5)
        axarr[i/4,i%4].plot(x,y2,label="KDE",linestyle='--',color='blue',lw=3,alpha=0.5)
        axarr[i/4,i%4].legend()
    
    plt.tight_layout()
    plt.show()


    <matplotlib.figure.Figure at 0x10c65e490>



![png](http://www.bryantravissmith.com/img/GW02D2/output_38_1.png)


In each case the non-KDE function seem tot fit the data better, but it could be used to for other estimates if we did not have a model.

##Paired Programming

In the afternoon session we had to investigate the centeral limit theorem, produce confidence intervals, and attempt some bootstrapping estimates


    def make_draws(distribution, parameters, size):
        '''
            returns distribrution or None if valid distribution is not selected
        '''    
        dist = None
        
        if distribution.lower() == 'binomial':
            n, p = parameters['n'], parameters['p']
            dist = sc.binom(n, p).rvs(size)
        
        elif distribution.lower() == 'exponential':
            l = parameters['lambda']
            dist = sc.expon(scale = l).rvs(size)
        
        elif distribution.lower() == 'poisson':
            l = parameters['lambda']
            dist = sc.poisson(mu=l).rvs(size)
    
        elif distribution.lower() == 'gamma':
            a, b = parameters['alpha'],parameters['beta']
            dist = sc.gamma(a=a,scale=1./b).rvs(size)
        
        elif distribution.lower() == 'normal':
            mean, var = parameters['mean'], parameters['var']
            dist = sc.norm(loc=mean, scale=var).rvs(size)
        
        elif distribution.lower() == 'uniform':
            low, high = parameters['low'], parameters['high']
            dist = sc.uniform(loc=low, scale=(high-low)).rvs(size)
            
        return dist
    
    def plot_means(distribution, parameters, size, repeat):
        arr = []
        for r in range(repeat):
            arr.append(make_draws(distribution, parameters, size).mean())
        plt.figure()
        plt.title("Centeral Limit Theorem: " + distribution + " for N = " + str(size))
        plt.hist(arr, normed=1,bins=100)
        plt.show()
        
    plot_means('poisson', {'lambda':10}, 10, 5000)
    plot_means('poisson', {'lambda':10}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_41_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_41_1.png)



    plot_means('binomial', {'n':10,'p':0.1}, 10, 5000)
    plot_means('binomial', {'n':10,'p':0.1}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_42_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_42_1.png)



    plot_means('exponential', {'lambda':10}, 10, 5000)
    plot_means('exponential', {'lambda':10}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_43_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_43_1.png)



    plot_means('gamma', {'alpha':10,'beta':0.1}, 10, 5000)
    plot_means('gamma', {'alpha':10,'beta':0.1}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_44_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_44_1.png)



    plot_means('normal', {'mean':10,'var':0.1}, 10, 5000)
    plot_means('normal', {'mean':10,'var':0.1}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_45_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_45_1.png)



    plot_means('uniform', {'low':10,'high':20}, 10, 5000)
    plot_means('uniform', {'low':10,'high':20}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_46_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_46_1.png)


Looking at these distirubtions we see that for N = 10, espeicaly for the discrete distriubiton, that the sampling distirubiton of the mean is not normal.   If the underlying distribution is skewed, so is the sampling distribution.  If we look at means of samples of size 200, the central limit theorm holds and the sampling distribution is normal even if the underlying distribution is skewed or discrete.  

The central limit theorm does not hold for all statistics.  We can look at the max, for instance.


    def plot_max(distribution, parameters, size, repeat):
        arr = []
        for r in range(repeat):
            arr.append(make_draws(distribution, parameters, size).max())
        plt.figure()
        plt.hist(arr, normed=1,bins=100)
        plt.show()
        
    plot_max('poisson', {'lambda':10}, 200, 5000)
    plot_max('binomial', {'n':10,'p':0.1}, 200, 5000)
    plot_max('exponential', {'lambda':10}, 200, 5000)
    plot_max('gamma', {'alpha':10,'beta':0.1}, 200, 5000)
    plot_max('normal', {'mean':10,'var':0.1}, 200, 5000)
    plot_max('uniform', {'low':5, 'high':10}, 200, 5000)


![png](http://www.bryantravissmith.com/img/GW02D2/output_48_0.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_48_1.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_48_2.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_48_3.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_48_4.png)



![png](http://www.bryantravissmith.com/img/GW02D2/output_48_5.png)


These distributions are clearly not normally distributed, and the discrete distribution results remain discrete.

###Population Inference and Confidence Interval

Our next section had to do with constructing confidence intervals on means for different situations.   We were given some lunch data, and attempted to construct the confidence interval for the mean lunch break.  




    lunch_hour = np.loadtxt('../estimation-sampling/data/lunch_hour.txt')
    plt.figure()
    plt.hist(lunch_hour)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D2/output_50_0.png)


There are 25 data points in the data.   Even though this distrubtion is not normal, the sampling distribution of the mean should be approching a normal distribution.  

The standard deviation of the sampling distirubiton is suppose to be well approximated by the standard error of the sample.

$$s = \sqrt{ \frac{\Sigma_{i}(x_i \ - \ \bar{x})^2}{N-1} }$$


    se = lunch_hour.std(ddof=1) / np.sqrt( len(lunch_hour) )
    se, sc.sem(lunch_hour) ##scipy standard error comparison




    (0.040925827625524797, 0.040925827625524797)



Using this standard error we can attempt to construct the confidence interval on the population mean.  We choose Z=1.96 for a 95% confidence interval


    lm = lunch_hour.mean()
    (ci_95_low, ci_95_hi) = (lm-1.96*se,lm+1.96*se)
    (ci_95_low, ci_95_hi)




    (2.1042853778539716, 2.264714622146029)



The 95% confidence interval interpretation is that 95% of the confidence intervals constructed through this method will contrain the population mean lunch hour.   If the sample size was smaller, the both the standard error and normal approximation would change in a way that would not allow this method to work.   

For smaller sample sizes we would want to try another method.  Bootstrapping could be effective.

Bootstrapping does not assume normality, or more any assumptions about the underlying distribution.   It is a non-parametric method of constructiong an confidence interval.   If the distribution is well approximateldy by some common distribution, the bootstrapped CI will overestimate the boundaries compared to this distribution.  

We will try this for another data set involving productivity.

###Bootstraping


    productivity = np.loadtxt('../estimation-sampling/data/productivity.txt')
    productivity




    array([-19.1, -15.2, -12.4, -15.4,  -8.7,  -6.7,  -5.9,  -3.5,  -3.1,
            -2.1,   4.2,   6.1,   7. ,   9.1,  10. ,  10.3,  13.2,  10.1,
            14.1,  14.4,  20.1,  26.3,  27.7,  22.2,  23.4])



Because the sample size is large enough, we would expect the centeral limit to hold.  Lets see if the boot strapping gives similar values.  If we do not know the population variance, we should us the t-distribution.   We will also check that.  The two results should be close


    se = sc.sem(productivity)
    mean = productivity.mean()
    ci95lo, ci95hi = mean-1.96*se, mean+1.96*se
    print ci95lo, ci95hi

    -0.330202770421 10.4182027704



    ci95lo, ci95hi = mean+sc.t.ppf(0.025,24)*se, mean+sc.t.ppf(0.975,24)*se
    print ci95lo, ci95hi

    -0.615086412127 10.7030864121


These intervals are very close.   Bootstrapping should give a similar result.


    def bootstrap(sample, B=10000):
        return np.array([ sample[np.random.randint(len(sample),size=len(sample))] for i in range(B)])
    
    def bootstrap_ci(sample, stat_function=np.mean, iterations=1000, ci=95):
        
        statistic = np.apply_along_axis(stat_function,1,bootstrap(productivity, B=iterations)) # e.g. array of means
        low = np.percentile(statistic, (100-ci)/2.)
        high = np.percentile(statistic, 100-(100-ci)/2.)
        return low, high
    
    bootstrap_ci(productivity)




    (-0.49619999999999997, 10.265000000000001)



This is in line with the previous estimates.   Lets look at the histogram of bootstrapped means.


    def bootstrap_plot_means(sample, iterations=1000):
        samples = bootstrap(sample, iterations)
        means = np.apply_along_axis(np.mean, 1, samples)
        plt.figure()
        plt.hist(means,normed=True,color='red',alpha=0.1)
        plt.xlabel('Mean Values')
        plt.ylabel('Probability Density')
        plt.show()
        
    bootstrap_plot_means(productivity)


![png](http://www.bryantravissmith.com/img/GW02D2/output_63_0.png)


Even though the results are not a statistically significant difference from zero, the results suggest that the population value is likely to be different from zero.   The uncertainty from the sample does not allow us to know that it is not zero, but we do know that it does not significantly harm productivity.

###Bootstraping Correlation

We can bootstrap other variables.   We will try it for the correlation between LSAT and GPA from law data.


    law_sample = np.loadtxt('../estimation-sampling/data/law_sample.txt')
    plt.scatter(law_sample[:,0],law_sample[:,1])
    plt.xlabel("LSAT Score")
    plt.ylabel("GPA")
    print sc.pearsonr(law_sample[:,0],law_sample[:,1])

    (0.77637449128940705, 0.00066510201110281625)



![png](http://www.bryantravissmith.com/img/GW02D2/output_65_1.png)



    data = bootstrap(law_sample,B=10000)
    corrs = np.array([sc.pearsonr(mat[:,0],mat[:,1])[0] for mat in data])
    plt.figure()
    plt.hist(corrs)
    plt.show()
    np.percentile(corrs,2.5),np.percentile(corrs,97.5)


![png](http://www.bryantravissmith.com/img/GW02D2/output_66_0.png)





    (0.45069334540504685, 0.96239529249176581)



Bootstrapping the correlation between the variables from the sampple finds that a 95% confidence interval estimates the population correlation of LSAT with GPA should be between 0.45 and 0.96.   Thankfully we have the full dataset from which this sample was pulled.   Lets compare.


    all_law = np.loadtxt('../estimation-sampling/data/law_all.txt')
    sc.pearsonr(all_law[:,0],all_law[:,1])[0]




    0.75999785550389798



This is smack in the middle of the confidence interval.   Pretty cool.   


    
