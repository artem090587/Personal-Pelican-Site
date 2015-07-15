Title: Galvanize - Week 06 - Day 5
Date: 2015-07-10 10:20
Modified: 2015-07-10 10:30
Category: Galvanize
Tags: data-science, galvanize, time series, ARMA, SARMA, Holt, Holt-Winter
Slug: galvanize-data-science-06-05
Authors: Bryan Smith
Summary: Today we covered time series.

#Galvanize Immersive Data Science

##Week 6 - Day 5

Since today was friday our quiz was our weekly survey about how we feel about our progress in the program.   Its been pretty static for me.   Instructors are good, I am learning a lot and ready to learn more, and I get the most out of working on the projects.   

Today was a lecture on time-series models.  In my opinion, it is too much to cover in a single lesson.  The presentations were to0 basic, then jump to a point that anyone who did not have time-series experience couldn't follow because they wouldn't have context.   Being able to abstractly follow a lesson is not the same as walking away with an intuition for the material.   It was one of the few days where we universally walked away without any intuition.  

##Time Series

One of my faviorite aspects of our morning sprint was learning how to use the Pandas datetime (and string) functionality in a way that perserves its underlying use of C.   There were times where I would try to do something in a very hacky sort of way, and it would take a long time.   What I have since learned is that all these tasks can be done using Pandas functionality, but you need to know how to use it in the correct way.   Read the documentation.  Serioursy!


##Exploring Monthly Birth Data   

We would given monthly birth data from 1980 to 2010.   Our goal was to try to develop a very basic model and use the timeseries functionality in pandas.  


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    birth = pd.read_csv('data/birth.txt')
    print birth.shape
    birth.head()

    (372, 1)





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>295</td>
    </tr>
    <tr>
      <th>1</th>
      <td>286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>278</td>
    </tr>
    <tr>
      <th>4</th>
      <td>272</td>
    </tr>
  </tbody>
</table>
</div>




    dates = pd.date_range(start='Jan 1980',end='Jan 2011',freq='M')
    dates




    <class 'pandas.tseries.index.DatetimeIndex'>
    [1980-01-31, ..., 2010-12-31]
    Length: 372, Freq: M, Timezone: None



3. Create a `time` variable (range: 1-372) to be used later in the regressions 
and both a `month` and `year` variable (use `pd.DatetimeIndex` to strip these 
values from your dates).


    birth['time_step'] = range(1,373)
    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>295</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>286</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>278</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>272</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




    dt_index = pd.DatetimeIndex(dates)
    birth['year'] = dt_index.year
    birth['month'] = dt_index.month
    birth['datetime'] = dt_index
    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
      <th>year</th>
      <th>month</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>295</td>
      <td>1</td>
      <td>1980</td>
      <td>1</td>
      <td>1980-01-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>286</td>
      <td>2</td>
      <td>1980</td>
      <td>2</td>
      <td>1980-02-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300</td>
      <td>3</td>
      <td>1980</td>
      <td>3</td>
      <td>1980-03-31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>278</td>
      <td>4</td>
      <td>1980</td>
      <td>4</td>
      <td>1980-04-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>272</td>
      <td>5</td>
      <td>1980</td>
      <td>5</td>
      <td>1980-05-31</td>
    </tr>
  </tbody>
</table>
</div>




    birth = birth.set_index('datetime')
    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
      <th>year</th>
      <th>month</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-31</th>
      <td>295</td>
      <td>1</td>
      <td>1980</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1980-02-29</th>
      <td>286</td>
      <td>2</td>
      <td>1980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1980-03-31</th>
      <td>300</td>
      <td>3</td>
      <td>1980</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1980-04-30</th>
      <td>278</td>
      <td>4</td>
      <td>1980</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1980-05-31</th>
      <td>272</td>
      <td>5</td>
      <td>1980</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




    plt.figure(figsize=(14,8))
    birth.num_births.plot()
    plt.xlabel('Time')
    plt.ylabel("Number of Births per Month")
    plt.title("Number of Births per Month from Jan 1980 to Dec 2010")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_7_0.png)


Now that we have our data in the correct form, we can look at different components of the behavior.  We learned that we can break time series into 4 components:  Trend, Seasonal, Cyclic, and Irreducible Error.   That will be the goal for the rest of our morning sprint.

##Trend

We are first going to look at the trend in the data.  we want to fit this as best we can with a simple model.    


    plt.figure(figsize=(14,8))
    birth.resample('AS',how='sum').num_births[1:].plot()
    plt.xlabel('Time')
    plt.ylabel("Number of Births per Year")
    plt.title("Number of Births per Year from Jan 1980 to Dec 2010")
    plt.show()



![png](http://www.bryantravissmith.com/img/GW06D5/output_9_0.png)



    def polyFit(x,y,deg=2):
        coef = np.polyfit(x,y,deg=deg)
        pred = coef[deg]*np.ones(x.shape)
        for i in range(deg-1,-1,-1):
            pred += coef[i]*x**(deg-i)
        return pred
    
    plt.figure(figsize=(14,8))
    plt.plot(birth.index,birth.num_births,alpha=0.3,label='Data')
    for i in range(2,8):
        plt.plot(birth.index,
                 polyFit(birth.time_step.values,birth.num_births.values,deg=i),
                 label='Ply Deg: {}'.format(i),
                 lw=3)
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_10_0.png)



    trend = polyFit(birth.time_step.values,birth.num_births.values,deg=5)

Our naive approach is to fit the trend with a poly nomial.  I decided to cycle through a number of polynomial fits to the 30 years of data to find a model that most smoothly matches the trend without unreal havior at the end points.   A degree 7 polynomial follows the trend well, but we can see that it projects a dramatic increase in births.   The degree 3 polynomial says that historically, births were very low.   The balance is the degree 5 (purple).

We are going to subtrack the trend out of our data and examine the seasonal component.   

##Seasonal Component    


    plt.figure(figsize=(14,8))
    plt.plot(birth.index,(birth.num_births-trend))
    plt.xlabel("Time")
    plt.ylabel("Number of Birth - Trend")




    <matplotlib.text.Text at 0x10d4c9650>




![png](http://www.bryantravissmith.com/img/GW06D5/output_13_1.png)


With the exceptional of 2000-2004, our model captures the trend.  The average deviation does look to be centered around zero, and we see a seasonal component with an amplitude of about 20 births.  Lets see what this data looks like if we aggregate.


    plt.figure(figsize=(14,8))
    birth['num_birth_minus_trend'] = birth.num_births-trend
    birth.groupby('month').num_birth_minus_trend.mean().plot(label="Average Seasonal Component")
    plt.plot(range(1,13),20*np.sin(6.28*(np.arange(1,13))/12+3.14159),label="Sin Function")
    plt.ylabel("Number of Births")
    plt.xlabel("Month")
    plt.legend()




    <matplotlib.legend.Legend at 0x10e64e610>




![png](http://www.bryantravissmith.com/img/GW06D5/output_15_1.png)


The sine function is a good approximation of the seasonal/cyclic component.   Another way we could have done this was to break up into quarterly and montly componentents, make dummy variables, and do a linear fit.  I will do this later to compare the results.


    birth['trend'] = trend
    birth['seasonal'] = 20*np.sin(6.28*(birth.month)/12+3.14159)
    birth['num_birth_minus_trend_minus_seasonal'] = birth.num_birth_minus_trend - birth.seasonal
    birth['trend_plus_seasonal'] = birth.trend + birth.seasonal
    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
      <th>year</th>
      <th>month</th>
      <th>num_birth_minus_trend</th>
      <th>seasonal</th>
      <th>trend</th>
      <th>num_birth_minus_trend_minus_seasonal</th>
      <th>trend_plus_seasonal</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-31</th>
      <td>295</td>
      <td>1</td>
      <td>1980</td>
      <td>1</td>
      <td>4.517789</td>
      <td>-9.995356</td>
      <td>290.482211</td>
      <td>14.513145</td>
      <td>280.486855</td>
    </tr>
    <tr>
      <th>1980-02-29</th>
      <td>286</td>
      <td>2</td>
      <td>1980</td>
      <td>2</td>
      <td>-4.540754</td>
      <td>-17.315170</td>
      <td>290.540754</td>
      <td>12.774416</td>
      <td>273.225584</td>
    </tr>
    <tr>
      <th>1980-03-31</th>
      <td>300</td>
      <td>3</td>
      <td>1980</td>
      <td>3</td>
      <td>9.372178</td>
      <td>-19.999994</td>
      <td>290.627822</td>
      <td>29.372171</td>
      <td>270.627829</td>
    </tr>
    <tr>
      <th>1980-04-30</th>
      <td>278</td>
      <td>4</td>
      <td>1980</td>
      <td>4</td>
      <td>-12.742668</td>
      <td>-17.331142</td>
      <td>290.742668</td>
      <td>4.588474</td>
      <td>273.411526</td>
    </tr>
    <tr>
      <th>1980-05-31</th>
      <td>272</td>
      <td>5</td>
      <td>1980</td>
      <td>5</td>
      <td>-18.884553</td>
      <td>-10.023025</td>
      <td>290.884553</td>
      <td>-8.861528</td>
      <td>280.861528</td>
    </tr>
  </tbody>
</table>
</div>




    plt.figure(figsize=(14,8))
    birth.trend_plus_seasonal.plot(label="Trend+Seasonal")
    birth.num_births.plot(label="Data")
    plt.xlabel('Time')
    plt.ylabel("Number of Births per Month")
    plt.title("Number of Births per Month from Jan 1980 to Dec 2010")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_18_0.png)



    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    birth.num_birth_minus_trend_minus_seasonal.plot()
    plt.xlabel("Time")
    plt.ylabel("Error of Fit")
    plt.subplot(1,2,2)
    birth.num_birth_minus_trend_minus_seasonal.plot(kind='hist')
    plt.ylabel("Count")
    plt.xlabel("Error of Fit")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_19_0.png)


We see that our model does not capture all of the features displayed in the birth data, but the errors are starting to look like gaussian noise.   It is at this point that we will stop, because in the afternoon we are going to be covering better models for approaching these types of problems.

##Pandas

This blog is more of a collection of notes for me to review about datascience than anything else.  In that spirit I am going to put some plots and code in here to remember for the future.


    plt.figure(figsize=(14,8))
    ax = plt.gca()
    birth.num_births.plot(ax=ax, label="Monthly",color='indianred',lw=3,alpha=0.3)
    birth.resample('Q-NOV').num_births.plot(ax=ax,label='Quarterly',color='steelblue',lw=3,alpha=0.5)
    birth.resample('AS').num_births.plot(ax=ax,label='Yearly',color='forestgreen',lw=3,alpha=0.8)
    plt.legend()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_21_0.png)


Using the series and time functionality


    nums = pd.Series(birth.num_births)
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    nums.plot()
    plt.subplot(1,2,2)
    nums['2006':'2010'].plot()




    <matplotlib.axes._subplots.AxesSubplot at 0x1139495d0>




![png](http://www.bryantravissmith.com/img/GW06D5/output_23_1.png)


##Statsmodel Package

We were encorage to do this process using the stats model package because it has a more evolved way of doing time-series analysis.  Though we were also told that R is the best way to do this kind of analysis because all the methods have been developed for years.  The statsmodel package for python are
6. Turn the `num_births` into a time series using `pd.Series()`.


    birth['time_step2'] = birth.time_step.values**2
    birth['time_step3'] = birth.time_step.values**3
    birth['time_step4'] = birth.time_step.values**4
    birth['time_step5'] = birth.time_step.values**5
    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
      <th>year</th>
      <th>month</th>
      <th>num_birth_minus_trend</th>
      <th>seasonal</th>
      <th>trend</th>
      <th>num_birth_minus_trend_minus_seasonal</th>
      <th>trend_plus_seasonal</th>
      <th>time_step2</th>
      <th>time_step3</th>
      <th>time_step4</th>
      <th>time_step5</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-31</th>
      <td>295</td>
      <td>1</td>
      <td>1980</td>
      <td>1</td>
      <td>4.517789</td>
      <td>-9.995356</td>
      <td>290.482211</td>
      <td>14.513145</td>
      <td>280.486855</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1980-02-29</th>
      <td>286</td>
      <td>2</td>
      <td>1980</td>
      <td>2</td>
      <td>-4.540754</td>
      <td>-17.315170</td>
      <td>290.540754</td>
      <td>12.774416</td>
      <td>273.225584</td>
      <td>4</td>
      <td>8</td>
      <td>16</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1980-03-31</th>
      <td>300</td>
      <td>3</td>
      <td>1980</td>
      <td>3</td>
      <td>9.372178</td>
      <td>-19.999994</td>
      <td>290.627822</td>
      <td>29.372171</td>
      <td>270.627829</td>
      <td>9</td>
      <td>27</td>
      <td>81</td>
      <td>243</td>
    </tr>
    <tr>
      <th>1980-04-30</th>
      <td>278</td>
      <td>4</td>
      <td>1980</td>
      <td>4</td>
      <td>-12.742668</td>
      <td>-17.331142</td>
      <td>290.742668</td>
      <td>4.588474</td>
      <td>273.411526</td>
      <td>16</td>
      <td>64</td>
      <td>256</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>1980-05-31</th>
      <td>272</td>
      <td>5</td>
      <td>1980</td>
      <td>5</td>
      <td>-18.884553</td>
      <td>-10.023025</td>
      <td>290.884553</td>
      <td>-8.861528</td>
      <td>280.861528</td>
      <td>25</td>
      <td>125</td>
      <td>625</td>
      <td>3125</td>
    </tr>
  </tbody>
</table>
</div>




    import statsmodels.api as sm
    model1 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step']]))
    results1 = model1.fit()
    
    model2 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step','time_step2']]))
    results2 = model2.fit()
    
    model3 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step','time_step2','time_step3']]))
    results3 = model3.fit()
    
    model4 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step','time_step2','time_step3','time_step4']]))
    results4 = model4.fit()
    
    model5 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step','time_step2','time_step3','time_step4','time_step5']]))
    results5 = model5.fit()
    
    results = [results1,results2,results3,results4,results5]
    
    plt.figure(figsize=(14,8))
    plt.plot(birth.index,birth.num_births,alpha=0.3,label='Data')
    for i,result in enumerate(results):
        print "AIC: %4.1f, BIC: %4.1f" % (result.aic, result.bic)
        plt.plot(birth.index,result.fittedvalues,label='Ploy Deg: {}'.format(i+1), lw=3)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Births Per Month")
    plt.show()

    AIC: 3593.0, BIC: 3600.8
    AIC: 3400.1, BIC: 3411.8
    AIC: 3262.1, BIC: 3277.7
    AIC: 3231.3, BIC: 3250.9
    AIC: 3225.6, BIC: 3249.1



![png](http://www.bryantravissmith.com/img/GW06D5/output_26_1.png)


The nice thing about statsmodel over numpy poly fit is that it gives a number of statistics about how well the data is being fitted.  The AIC/BIC can be used to decide which model is a better model.  In this case the biggest increase happened going to a degree 3 polynomial, after that the models are not significantly better for fitting the data.   They are, however, better and extrapolating outside of the data.

Using statsmodel, we can create categorical variables for the season, and try to fit a seasonal component that way


    birth['Wi'] = np.where(birth.month.isin([12,1,2]),1,0)
    birth['Sp'] = np.where(birth.month.isin([3,4,5]),1,0)
    birth['Su'] = np.where(birth.month.isin([6,7,8]),1,0)
    birth['Fa'] = np.where(birth.month.isin([9,10,11]),1,0)


    birth.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_births</th>
      <th>time_step</th>
      <th>year</th>
      <th>month</th>
      <th>num_birth_minus_trend</th>
      <th>seasonal</th>
      <th>trend</th>
      <th>num_birth_minus_trend_minus_seasonal</th>
      <th>trend_plus_seasonal</th>
      <th>time_step2</th>
      <th>time_step3</th>
      <th>time_step4</th>
      <th>time_step5</th>
      <th>Wi</th>
      <th>Sp</th>
      <th>Su</th>
      <th>Fa</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-31</th>
      <td>295</td>
      <td>1</td>
      <td>1980</td>
      <td>1</td>
      <td>4.517789</td>
      <td>-9.995356</td>
      <td>290.482211</td>
      <td>14.513145</td>
      <td>280.486855</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1980-02-29</th>
      <td>286</td>
      <td>2</td>
      <td>1980</td>
      <td>2</td>
      <td>-4.540754</td>
      <td>-17.315170</td>
      <td>290.540754</td>
      <td>12.774416</td>
      <td>273.225584</td>
      <td>4</td>
      <td>8</td>
      <td>16</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1980-03-31</th>
      <td>300</td>
      <td>3</td>
      <td>1980</td>
      <td>3</td>
      <td>9.372178</td>
      <td>-19.999994</td>
      <td>290.627822</td>
      <td>29.372171</td>
      <td>270.627829</td>
      <td>9</td>
      <td>27</td>
      <td>81</td>
      <td>243</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1980-04-30</th>
      <td>278</td>
      <td>4</td>
      <td>1980</td>
      <td>4</td>
      <td>-12.742668</td>
      <td>-17.331142</td>
      <td>290.742668</td>
      <td>4.588474</td>
      <td>273.411526</td>
      <td>16</td>
      <td>64</td>
      <td>256</td>
      <td>1024</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1980-05-31</th>
      <td>272</td>
      <td>5</td>
      <td>1980</td>
      <td>5</td>
      <td>-18.884553</td>
      <td>-10.023025</td>
      <td>290.884553</td>
      <td>-8.861528</td>
      <td>280.861528</td>
      <td>25</td>
      <td>125</td>
      <td>625</td>
      <td>3125</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




    model6 = sm.OLS(birth.num_births,sm.add_constant(birth[['time_step','time_step2','time_step3','time_step4','time_step5','Wi','Sp','Su','Fa']]))
    results6 = model6.fit()
    plt.figure(figsize=(14,8))
    plt.plot(birth.index,birth.num_births,alpha=0.3)
    plt.plot(birth.index,results6.fittedvalues)
    plt.xlabel("Time")
    plt.ylabel("Number of Births per Month")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_30_0.png)



    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(birth.index,results6.resid,'bo')
    plt.subplot(1,2,2)
    plt.hist(birth.num_births-results6.fittedvalues)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_31_0.png)


Using the seasonal fit gave similar results as the sine fit, but it did not assume an underlying model.


##Exponential Smooth

When fitting time series data it is sometime important to weight the most recent information as more important the older information.  This leads to a class of models that lead to time varying coefficients.  They do not explicitly depend on time, but they are fitted in a weighted way that depends on the most recent information.  For instance, a linear fit has an intercept and slope, and this model will have time varying intercept and slopes because at each point in time, the model fits in a way that biases the most recent information.   

We were introduced to a Holt Method, which is what I just described, and a Holt-Winter method that including a seasonal component to the the fit.



    class Holt:
        
        def __init__(self):
            pass
        
        def fit(self,x,y,dim=11):
            self.size = len(x)
            self.x = x
            self.y = y 
            self.l = np.zeros(self.size)
            self.b = np.zeros(self.size)
            self.s = np.zeros(self.size)
            int_fit_size = int(max(2,0.1*self.size))
            fit =  np.polyfit(x[:int_fit_size],y[:int_fit_size],deg=1)
            self.l[0] = fit[1]
            self.b[0] = fit[0]
            alphas = np.linspace(0,1,dim)
            gammas = np.linspace(0,1,dim)
            self.SE = np.zeros((dim,dim))
            for m,alpha in enumerate(alphas):
                for n,gamma in enumerate(gammas):
                    pred = np.zeros(len(x))
                    for i in range(1,len(x)):
                        #print i,alpha,gamma,alpha*self.y[i]
                        self.l[i] = alpha*self.y[i]+(1-alpha)*(self.l[i-1]+self.b[i-1])
                        self.b[i] = gamma*(self.l[i]-self.l[i-1])+(1-gamma)*self.b[i-1]
                        pred[i] = self.l[i]+self.b[i]
                    self.SE[m,n] = np.power(y[1:]-pred[:-1],2).sum()
    
            row = np.argmin(self.SE)/dim
            col = np.argmin(self.SE)-row*dim
            self.alpha = alphas[row]
            self.gamma = gammas[col]
            self.predictions = np.zeros(len(self.x))
            pred = np.zeros(len(x))
            for i in range(1,len(x)):
                self.l[i] = self.alpha*self.y[i]+(1-self.alpha)*(self.l[i-1]+self.b[i-1])
                self.b[i] = self.gamma*(self.l[i]-self.l[i-1])+(1-self.gamma)*self.b[i-1]
                self.predictions[i] = self.l[i]+self.b[i]
                
            self.s = np.sqrt(np.power(y[1:]-self.predictions[:-1],2).sum()/(len(x)-2))
            
        def predict(self,time=2,z=1.96):
            se = 1
            for i in range(time):
                se += self.alpha**2*(1+i*self.gamma)**2
            se = z*self.s*np.sqrt(se)
            
            predictions = self.l+self.b*time
            high = predictions+se
            low = predictions-se
            return low,high
        
        def forcast(self,time=10,z=1.96):
            high_forcast = []
            low_forcast = []
            forcast = []
            for i in range(time):
                se = 1
                for j in range(i):
                    se += self.alpha**2*(1+j*self.gamma)**2
                se = self.s*np.sqrt(se)
                mean = self.l[-1]+i*self.b[-1]
                forcast.append(mean)
                high_forcast.append(mean+z*se)
                low_forcast.append(mean-z*se)
            return np.array(forcast),np.array(high_forcast),np.array(low_forcast)
        
    holt = Holt()
    holt.fit(birth.time_step,birth.num_births)



    low, high = holt.predict(time=0)
    plt.figure(figsize=(14,8))
    plt.fill_between(birth.index,low,high,alpha=.2)
    plt.plot(birth.index,birth.num_births,alpha=1,)
    birth.num_births.plot(lw=4,alpha=.2)
    plt.xlabel("Time")
    plt.ylabel("Number of Births Per Month")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_34_0.png)


What is real interesting about this model is that it is strait forward to estimate the errors on the predictions.  The fit matches the actual values (interpolation) perfectly.   We can use this model to predict what will hapen in the future.



    forcast,high,low = holt.forcast(time=12)
    plt.figure(figsize=(14,8))
    plt.fill_between(range(1,13),low,high,alpha=.2)
    plt.plot(range(1,13),forcast,alpha=1,)
    plt.xlim([1,12])
    plt.xlabel("Months in the Future")
    plt.ylabel("Number of Births Per Month")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_36_0.png)


Despite fiting the data very well (because it weights the most recent information), it does not forcast seasonal variation.  At the end of the day, it is a linear model that will only extrapolate linearly into the future.  This model will not work well for this data more than a month or two in the future.  To do this we would need to use a Holt-Winter method.   That will come later

##Afternoon - SARMA 

SARMIA stands fro Seasonal AutoRegression Moving Averaging Models that including the fact that past errors can contribute to current results, and that past values can be predictive of today's results.   Our focus in the afternoon was using the statsmodel package's implementaton of these models.  Our data was a series of logins to a site over a two month period.


    logins = pd.read_json("data/logins.json")
    logins.loc[:,0] = pd.to_datetime(logins.loc[:,0])
    logins.columns = ['datetime']
    logins['count'] = np.ones(logins.shape[0])
    logins.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-03-01 00:05:55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-03-01 00:06:23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-03-01 00:06:52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-03-01 00:11:23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-03-01 00:12:47</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We are using the 1 count so that when we resample to different time periods, say by hour, we can sum all the counts in every hour to get the logins per hour.  We can also do this per day, or per week.  Next we are going to make a pandas timeseries object.


    ts = pd.Series(np.ones(logins.shape[0]),index=logins.datetime)
    plt.figure(figsize=(14,8))
    ts.resample('1D',how='sum').plot()
    plt.xlabel("Time")
    plt.ylabel("Login Counts")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW06D5/output_40_0.png)


We see here that the logins have a weekly behavior where users are loggin in on the weekends, and not using the site very much on the weekdays.  We can also see that the number of logins seems to be increasing slightly as the weeks progress.   Lets use SARMA to investigate


    import statsmodels.api as sm
    def acf_pacf(ts,lag):
        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(ts, lags=lag, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(ts, lags=lag, ax=ax2)
    
    ts1day = ts.resample('1D',how='sum')
    acf_pacf(ts1day,28)


![png](http://www.bryantravissmith.com/img/GW06D5/output_42_0.png)


The autocorrelation and partial autocorrelation show moving and seasonal variaton.   The first feature we are conserned with the the first node being out side of the statistically insignificant (purple) region.   That suggest that we need to adda  difference component.   


    ts1day.diff(1).plot()




    <matplotlib.axes._subplots.AxesSubplot at 0x1195ea0d0>




![png](http://www.bryantravissmith.com/img/GW06D5/output_44_1.png)


By ploting the difference between one day at the next, we see the average is no longer moving.  The next component is the 7 day components.   That we will do with our model fit.  We are going to use stats model to fit the data.  The (0,1,0) term is the differece we saw to deal with the moving average.  The (1,1,0,7) is the auto correclation, the difference, and finaly the seasonality of 7 days.


    model=sm.tsa.SARIMAX(ts1day, order=(0,1,0), seasonal_order=(1,1,0,7)).fit()
    model.summary()




<table class="simpletable">
<caption>Statespace Model Results</caption>
<tr>
  <th>Dep. Variable:</th>                 <td>y</td>               <th>  No. Observations:  </th>    <td>61</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 1, 0)x(1, 1, 0, 7)</td> <th>  Log Likelihood     </th> <td>-287.751</td>
</tr>
<tr>
  <th>Date:</th>                  <td>Sun, 12 Jul 2015</td>        <th>  AIC                </th>  <td>579.503</td>
</tr>
<tr>
  <th>Time:</th>                      <td>14:33:39</td>            <th>  BIC                </th>  <td>583.725</td>
</tr>
<tr>
  <th>Sample:</th>                   <td>03-01-2012</td>           <th>  HQIC               </th>  <td>581.157</td>
</tr>
<tr>
  <th></th>                         <td>- 04-30-2012</td>          <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>              <td>opg</td>              <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>ar.S.L7</th> <td>   -0.2059</td> <td>    0.135</td> <td>   -1.524</td> <td> 0.127</td> <td>   -0.471     0.059</td>
</tr>
<tr>
  <th>sigma2</th>  <td> 3025.7975</td> <td>  578.692</td> <td>    5.229</td> <td> 0.000</td> <td> 1891.582  4160.013</td>
</tr>
</table>




    plt.plot(ts1day.index[1:],model.predict(0)[0,1:])
    plt.plot(ts1day.index[1:],ts1day.values[1:])





    [<matplotlib.lines.Line2D at 0x119d6b550>]




![png](http://www.bryantravissmith.com/img/GW06D5/output_47_1.png)



    tsmodel = pd.Series(model.predict(0)[0,1:]-ts1day.values[1:],index=ts1day.index[1:])
    acf_pacf(tsmodel, 28)


![png](http://www.bryantravissmith.com/img/GW06D5/output_48_0.png)


We see that this time-series model does well on matching the historical data and removes the autocorrelation and partial correlations that appread in the previous fits.  We could try to further address the poitns at n=2.


    model=sm.tsa.SARIMAX(ts1day, order=(2,1,0), seasonal_order=(1,1,0,7)).fit()
    model.summary()




<table class="simpletable">
<caption>Statespace Model Results</caption>
<tr>
  <th>Dep. Variable:</th>                 <td>y</td>               <th>  No. Observations:  </th>    <td>61</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(2, 1, 0)x(1, 1, 0, 7)</td> <th>  Log Likelihood     </th> <td>-285.344</td>
</tr>
<tr>
  <th>Date:</th>                  <td>Sun, 12 Jul 2015</td>        <th>  AIC                </th>  <td>578.688</td>
</tr>
<tr>
  <th>Time:</th>                      <td>14:43:19</td>            <th>  BIC                </th>  <td>587.131</td>
</tr>
<tr>
  <th>Sample:</th>                   <td>03-01-2012</td>           <th>  HQIC               </th>  <td>581.997</td>
</tr>
<tr>
  <th></th>                         <td>- 04-30-2012</td>          <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>              <td>opg</td>              <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>ar.L1</th>   <td>   -0.2857</td> <td>    0.115</td> <td>   -2.479</td> <td> 0.013</td> <td>   -0.512    -0.060</td>
</tr>
<tr>
  <th>ar.L2</th>   <td>   -0.1796</td> <td>    0.132</td> <td>   -1.357</td> <td> 0.175</td> <td>   -0.439     0.080</td>
</tr>
<tr>
  <th>ar.S.L7</th> <td>   -0.2359</td> <td>    0.158</td> <td>   -1.489</td> <td> 0.136</td> <td>   -0.546     0.075</td>
</tr>
<tr>
  <th>sigma2</th>  <td> 2751.6619</td> <td>  501.665</td> <td>    5.485</td> <td> 0.000</td> <td> 1768.416  3734.907</td>
</tr>
</table>




    plt.plot(ts1day.index[1:],model.predict(0)[0,1:])
    plt.plot(ts1day.index[1:],ts1day.values[1:])




    [<matplotlib.lines.Line2D at 0x11a1bc610>]




![png](http://www.bryantravissmith.com/img/GW06D5/output_51_1.png)



    tsmodel = pd.Series(model.predict(0)[0,1:]-ts1day.values[1:],index=ts1day.index[1:])
    acf_pacf(tsmodel, 28)


![png](http://www.bryantravissmith.com/img/GW06D5/output_52_0.png)


This is a slightly better model than the previous model.  The AIC/BIC does show improvement, but not so much that I would confidently say one model is better than the other.  We are talking fractions of a percent improvment.   What is nice is that this is signficantly quicker than the methods of writing Holt and Holt-Winter classes from before.  


    
