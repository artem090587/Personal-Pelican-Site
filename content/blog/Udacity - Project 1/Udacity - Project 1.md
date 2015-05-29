Title: Udacity - Data Analysis NanoDegree - Project 1
Date: 2015-01-15 10:20
Modified: 2015-05-15 19:30
Category: Udacity
Tags: udacity, data-analysis, nanodegree, project
Slug: udacity-project-1
Authors: Bryan Smith
Summary: My passing submission project 1 for Udacity Data Analysis Nanodegree

#Udacity - Data Analysis NanoDegree

##Project 1

The goal of project one is to apply some of the concepts in Udacity.com's [Intro to Data Science](https://www.udacity.com/course/intro-to-data-science--ud359) course to find interesting patterns or trends in [New York subway data](https://www.dropbox.com/s/1lpoeh2w6px4diu/improved-dataset.zip?dl=0).  A descript of the variables in the data set can be found [here](https://s3.amazonaws.com/uploads.hipchat.com/23756/665149/05bgLZqSsMycnkg/turnstile-weather-variables.pdf).  

The question that we are investigating is:

> Does subway ridership change when it rains?

Lets begin with loading the data




    %matplotlib inline
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 12)
    
    from ggplot import *
    import scipy
    import scipy.stats
    
    import datetime
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import math
    
    from IPython.display import display
    from IPython.display import HTML
    
    filename = 'turnstile_weather_v2.csv'
    subway_data = pd.read_csv(filename, parse_dates=['datetime']) 
    
    subway_data.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNIT</th>
      <th>DATEn</th>
      <th>TIMEn</th>
      <th>ENTRIESn</th>
      <th>EXITSn</th>
      <th>ENTRIESn_hourly</th>
      <th>EXITSn_hourly</th>
      <th>datetime</th>
      <th>hour</th>
      <th>day_week</th>
      <th>...</th>
      <th>pressurei</th>
      <th>rain</th>
      <th>tempi</th>
      <th>wspdi</th>
      <th>meanprecipi</th>
      <th>meanpressurei</th>
      <th>meantempi</th>
      <th>meanwspdi</th>
      <th>weather_lat</th>
      <th>weather_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R003</td>
      <td>05-01-11</td>
      <td>00:00:00</td>
      <td>4388333</td>
      <td>2911002</td>
      <td>0</td>
      <td>0</td>
      <td>2011-05-01 00:00:00</td>
      <td>0</td>
      <td>6</td>
      <td>...</td>
      <td>30.22</td>
      <td>0</td>
      <td>55.9</td>
      <td>3.5</td>
      <td>0</td>
      <td>30.258</td>
      <td>55.98</td>
      <td>7.86</td>
      <td>40.700348</td>
      <td>-73.887177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R003</td>
      <td>05-01-11</td>
      <td>04:00:00</td>
      <td>4388333</td>
      <td>2911002</td>
      <td>0</td>
      <td>0</td>
      <td>2011-05-01 04:00:00</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>30.25</td>
      <td>0</td>
      <td>52.0</td>
      <td>3.5</td>
      <td>0</td>
      <td>30.258</td>
      <td>55.98</td>
      <td>7.86</td>
      <td>40.700348</td>
      <td>-73.887177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R003</td>
      <td>05-01-11</td>
      <td>12:00:00</td>
      <td>4388333</td>
      <td>2911002</td>
      <td>0</td>
      <td>0</td>
      <td>2011-05-01 12:00:00</td>
      <td>12</td>
      <td>6</td>
      <td>...</td>
      <td>30.28</td>
      <td>0</td>
      <td>62.1</td>
      <td>6.9</td>
      <td>0</td>
      <td>30.258</td>
      <td>55.98</td>
      <td>7.86</td>
      <td>40.700348</td>
      <td>-73.887177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R003</td>
      <td>05-01-11</td>
      <td>16:00:00</td>
      <td>4388333</td>
      <td>2911002</td>
      <td>0</td>
      <td>0</td>
      <td>2011-05-01 16:00:00</td>
      <td>16</td>
      <td>6</td>
      <td>...</td>
      <td>30.26</td>
      <td>0</td>
      <td>57.9</td>
      <td>15.0</td>
      <td>0</td>
      <td>30.258</td>
      <td>55.98</td>
      <td>7.86</td>
      <td>40.700348</td>
      <td>-73.887177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R003</td>
      <td>05-01-11</td>
      <td>20:00:00</td>
      <td>4388333</td>
      <td>2911002</td>
      <td>0</td>
      <td>0</td>
      <td>2011-05-01 20:00:00</td>
      <td>20</td>
      <td>6</td>
      <td>...</td>
      <td>30.28</td>
      <td>0</td>
      <td>52.0</td>
      <td>10.4</td>
      <td>0</td>
      <td>30.258</td>
      <td>55.98</td>
      <td>7.86</td>
      <td>40.700348</td>
      <td>-73.887177</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



###Exploration


    print "Subway Data Variable Names", subway_data.columns
    print ""
    print "Number of Subway Units", subway_data.UNIT.nunique()
    print "Number of Subway Stations", subway_data.station.nunique()


    Subway Data Variable Names Index([u'UNIT', u'DATEn', u'TIMEn', u'ENTRIESn', u'EXITSn', u'ENTRIESn_hourly', u'EXITSn_hourly', u'datetime', u'hour', u'day_week', u'weekday', u'station', u'latitude', u'longitude', u'conds', u'fog', u'precipi', u'pressurei', u'rain', u'tempi', u'wspdi', u'meanprecipi', u'meanpressurei', u'meantempi', u'meanwspdi', u'weather_lat', u'weather_lon'], dtype='object')
    
    Number of Subway Units 240
    Number of Subway Stations 207



The data consists of subway data for 240 turnstiles in 207 stations with data take every 4 hours recording the number of people entering and exiting the subway through the turnstiles, the date and time information associated with the measurement, and the corresponding weather measurements at the time of each measurement from a weather station.

Because this investigation has to do with how the weather, specifically rain, affects how the subway is used, it is important to get a feel for the data set. A google calendar is displayed below that marks the dates that rain occurred at or around New York subway stations. There are also two events in New York that happened during the time the data taken.

![Google Calendar for May 2011](http://www.bryantravissmith.com/img/udacity/project1/GoogleCalendarSubwayDates.png)

It should be noted that the 6 of the 7 days which it rained in New York of May 2011 were weekdays. Only one was during a weekend which was Saturday, May 15. It should also be noticed that May 30th is Memorial Day. It is not a weekday, but it is not a workday. 

Finally, Bin Laden was reported killed on May 2nd, 2011. This is important because there was a vigil that night at the World Trade Center, and a speech given by President Obama at the World Trade Center on May 5th, 2011. These events may have altered normal subway traffic, though there is not any obvious effect in the below graphs.


    plt.figure()
    sd_date = subway_data.groupby(['DATEn'])
    sd_date.ENTRIESn_hourly.sum().plot(legend=True)
    sd_date.EXITSn_hourly.sum().plot(legend=True)
    plt.xlabel('Date')
    plt.ylabel('Total Riders')
    plt.savefig('test1.png', bbox_inches='tight')
    
    plt.figure()
    sd_date = subway_data.groupby(['DATEn','hour'])
    sd_date.ENTRIESn_hourly.sum().plot(legend=True)
    sd_date.EXITSn_hourly.sum().plot(legend=True)
    plt.xlabel('Date')
    plt.ylabel('Total Riders')
    plt.savefig('test2.png', bbox_inches='tight')


![png](http://www.bryantravissmith.com/img/udacity/project1/output_5_0.png)



![png](http://www.bryantravissmith.com/img/udacity/project1/output_5_1.png)


The above figures shows the total number of hourly exits (blue) and entries (red) of the New York subway system during the month of May.  The top figure is summed over each day, while the bottome figure is summed over each 4 hour block.   

There is an cyclic pattern in both graphs.  The top graph shows a 7 day cycle of high ridership during the week and lower ridership during the weekends.   The bottom graph shows a 24 hour cycle that is double peaked for week days and single peaked for weekends.  The entry peaks on weekdays are approximately 2.5 times larger than the weekends, with the exception of May 30th, which has a peak consistent with weekends.   This further justifies that thinking about workdays vs non-workdays is more relavant than classifying days as weekends and non-weekends. 

It is also noteworthy that the number of exits are always less than the number of entries. If true, it implies that there is a net flow of people leaving the subways that are not being monitored. For this reason the analysis will only be done on the entry data, and not the exit data

On weekdays/workdays, there are double peaks, one in the morning and one in the afternoon. The minimum number of entries also appears to have a cycle, peaking in the middle of the week and being smallest on Sunday mornings.
One last point is that for a given day of the week, the values of hourly entries for any time/hour fluctuate from week to week but are similar in value.



    subway_data.station.unique()
    sd_station = subway_data.groupby(['station'])
    plt.figure()
    temp.plot(kind='hist',legend=True)
    plt.ylabel("Number of Stations")
    
    plt.figure()
    temp = sd_station.ENTRIESn_hourly.sum()
    temp.sort('ENTRIESn_hourly')
    temp[-15:].plot(kind='barh',legend=True)
    plt.ylabel("Station")
    plt.xlabel("Total Entries Per Month")
    
    
    plt.figure()
    temp = sd_station.ENTRIESn_hourly.sum()
    temp.sort('ENTRIESn_hourly')
    temp[:15].plot(kind='barh',legend=True)
    plt.ylabel("Station")
    plt.xlabel("Total Entries Per Month")





    <matplotlib.text.Text at 0x114773250>




![png](http://www.bryantravissmith.com/img/udacity/project1/output_7_1.png)



![png](http://www.bryantravissmith.com/img/udacity/project1/output_7_2.png)



![png](http://www.bryantravissmith.com/img/udacity/project1/output_7_3.png)


The amount of ridership also varies from station to station.  Most stations have less than 250,000 total entires per month.  The most active stations have almost 3 million total entries per month while the least active states have less than 10,000 entries per month.


    wt_data = subway_data[subway_data.station=='WORLD TRADE CTR']
    wt_data.plot(x='datetime', y=['ENTRIESn_hourly','EXITSn_hourly'],legend=False)
    plt.xticks(rotation=20)
    plt.xlabel("Date and Hour")
    plt.ylabel("WTC Total Entries per 4 hours")




    <matplotlib.text.Text at 0x1197b06d0>




![png](http://www.bryantravissmith.com/img/udacity/project1/output_9_1.png)



Finally, an inspection of the World Trade Center graph shows that there is not any obviously change in the number of people entering or exiting this station of May 2nd or May 5th. News reports from the dates do not convey the number of people that attended the events. Photos from the May 5th speech show no more than a few hundred people, and news reports estimate, on the high end, a few thousand in attendance on May 2nd vigil. Both counts are small compared to the peak ridership in excess of 20,000. It is also not clear, even if thousands attended, how many would have rode the subway specifically to attend the vigil.


###The Effect of Rain

The goal of this analysis is to determine the effect rain has on ridership, which can be measured by the numbr of people entering the subway system.  We can examine the results of 


    rain_data = subway_data.groupby('rain')
    
    plt.figure()
    rain_data.ENTRIESn_hourly.plot(kind='hist',legend=True)
    plt.legend(['No Rain','Rain'])




    <matplotlib.legend.Legend at 0x11dc2c050>




![png](http://www.bryantravissmith.com/img/udacity/project1/output_11_1.png)


The obvious feature is the distribution looks similar to a Poisson distribution, a distribution that describes the occurrence of a given number of events in a fix interval (spatial or temporal). In this case the occurrence would be the number of people entering the station. A second feature is the scale/frequency of the two distributions are different. This is largely due there being 7 rain days and 24 non­rain days. To make a comparison of the distributions this data must be normalized and a corresponding density plot be produced.


    plt.figure()
    rain_data.ENTRIESn_hourly.plot(kind='kde',legend=True)
    plt.legend(['No Rain','Rain'])
    x1,x2,y1,y2 = plt.axis()
    
    plt.axis((0,20000,y1,y2))




    (0, 20000, 0.0, 0.00050000000000000001)




![png](http://www.bryantravissmith.com/img/udacity/project1/output_13_1.png)


The above density plot shows that the two distributions have similar forms, still similar to the Poisson distribution, but the density of hourly entries when it rains is more skewed right than when it does not rain. The area under each curve is 1, so the scale is taken away. This is relevant because even though the distributions are visually different, we can not determine if it is because fewer people are riding when ridership is low, more people are riding when ridership is high, or some other effects are occurring when it rains.

One possible effect is that 6 of the 7 rainy days occur on weekdays, where the biggest peaks occur. Also 9 of the 11 days off are days where it does not rain, and these have the lowest values of the hourly entries. It is not clear whether the difference in these distributions are due to the rain, or what days the rain occurred. Statistical tests will help get to the bottom of this question.

####Statistical Test

The goal of a statistical test is to use probability to estimate whether a discrepancy between an expected set of measurements is significantly different than the actual results of those measurements. A variety of statistical tests exists, and the validity of these tests depend on conditions of the test. Some of these tests ascertain whether some summary statistic is significantly different from an expected summary statistic, and others help ascertain whether the distribution of the populations are significantly different. The two used in this analysis will be the student t-test and the Mann-Whitney U-test.

The significant level is the cutoff probability that the produced/experimental/observed distribution could be produced through randomness if our expected results were true. A typical value for this quantity is 5%, which is what will be used in this analysis.

#####Student t-test

The student t-test is a comparison of the averages of the two distributions. This is a valid test because the sample size is much larger than 30, so by the central limit theorem the distribution of averages is normally distributed even though the original distribution is similar to a Poisson distribution. The validity of the test also assumes that the observed values are independent. This is probably not a true assumption because of the cyclic nature of the events. But since the original distribution is similar to a Poisson distribution, a distribution that assumed independence, it is probably approximately true assumption for this test.

The naive comparison of dividing the data into the a set when it rained and when it does not rain leads to the following statistics:


    temp = pd.DataFrame(rain_data.ENTRIESn_hourly.count())
    temp.columns = ['count']
    temp['mean'] = rain_data.ENTRIESn_hourly.mean()
    temp['median'] = rain_data.ENTRIESn_hourly.median()
    temp['std'] = rain_data.ENTRIESn_hourly.std()
    temp




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
    </tr>
    <tr>
      <th>rain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33064</td>
      <td>1845.539439</td>
      <td>893</td>
      <td>2878.770848</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9585</td>
      <td>2028.196035</td>
      <td>939</td>
      <td>3189.433373</td>
    </tr>
  </tbody>
</table>
</div>




    rain = subway_data[subway_data.rain==1]
    norain = subway_data[subway_data.rain==0]
    
    results = scipy.stats.ttest_ind(norain.ENTRIESn_hourly,rain.ENTRIESn_hourly,equal_var=False)
    print "T Score:", results[0]
    print "p-value", results[1]

     T Score: -5.04288274762
    p-value 4.64140243163e-07


This data produces a t-value of -5.042, and a two-sided p-value of 4.64 x 10­7. This implies that these two populations have statistically significant averages. Keep in mind that a statistical test does not give a clear attribution to the underlying cause of the difference.

As pointed out in the previous section, the difference in the average number of riders could have more to do with what days it rained on rather than the fact that it rained. To separate out this effect, the data should be segmented into days that are weekends and holidays, and days that are not.


    subway_data['workday'] = 1
    subway_data.loc[(subway_data.weekday==0)|(subway_data.DATEn=='05-30-11'),'workday'] = 0
    
    rw_data = subway_data.groupby(['workday','rain'])
    temp = pd.DataFrame(rw_data.ENTRIESn_hourly.count())
    temp.columns = ['count']
    temp['mean'] = rw_data.ENTRIESn_hourly.mean()
    temp['median'] = rw_data.ENTRIESn_hourly.median()
    temp['std'] = rw_data.ENTRIESn_hourly.std()
    
    temp




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
    </tr>
    <tr>
      <th>workday</th>
      <th>rain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>11789</td>
      <td>1204.860548</td>
      <td>662.0</td>
      <td>1701.321541</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1793</td>
      <td>1065.858338</td>
      <td>596.0</td>
      <td>1539.449874</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>0</th>
      <td>21275</td>
      <td>2200.555347</td>
      <td>1070.0</td>
      <td>3304.904186</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7792</td>
      <td>2249.637449</td>
      <td>1057.5</td>
      <td>3421.444563</td>
    </tr>
  </tbody>
</table>
</div>



Before we separated the data into workdays and non-workdays, the data showed a significant increase in average hourly entires when it rained.  There was also an significant increase in the median values of hourly entries.   Now that we look at the data subsetted into workdays and non-wordays.   Non-work days see a decrease in the average hourly entries when it rains, and workdays show a slight increase in average hourly entries.  The same trends is also observed for the median values of hourly entries into the subway.




    nowork_rain = subway_data.loc[(subway_data.rain==1)&(subway_data.workday==0)]
    nowork_norain = subway_data.loc[(subway_data.rain==0)&(subway_data.workday==0)]
    
    results = scipy.stats.ttest_ind(nowork_norain.ENTRIESn_hourly,nowork_rain.ENTRIESn_hourly,equal_var=False)
    print "2-Sided T-Test for change in average number of hourly entries on non-workdays"
    print "T Score:", results[0]
    print "p-value", results[1]
    print ""
    work_rain = subway_data.loc[(subway_data.rain==1)&(subway_data.workday==1)]
    work_norain = subway_data.loc[(subway_data.rain==0)&(subway_data.workday==1)]
    
    results = scipy.stats.ttest_ind(work_norain.ENTRIESn_hourly,work_rain.ENTRIESn_hourly,equal_var=False)
    print "2-Sided T-Test for change in average number of hourly entries on workdays"
    print "T Score:", results[0]
    print "p-value", results[1]
    


    2-Sided T-Test for change in average number of hourly entries on non-workdays
    T Score: 3.51114255229
    p-value 0.000454063724831
    
    2-Sided T-Test for change in average number of hourly entries on workdays
    T Score: -1.09321652976
    p-value 0.274318319876


The non-workday data produces a t-value of 3.51, and a two-sided p-value of 4.54 x 10-3. This says that these two populations have statistically significant difference in averages. What should be noted is that previously it looked like more people ride the subway when it rained, but when the data is subsetted into workdays and not-workdays, it looks like less people ride the subway when it rains on a non-workday. It is the opposite effect observed in the naive investigation. It has a name, Simpson’s Paradox.

The workday data produces a t-value of -1.09, and a two-sided p-value of 0.274. The difference in these average is not statistically significant because the p-value is above 0.05. These difference could likely be produced by pure chance, and we would need more data to determine if rain does alter the ridership of the subway on workdays.

####Mann-Whitney U-test

The Mann-Whitney U-Test is a non-parametric test that determines the probability that a randomly chosen member of ‘larger’ set is larger than a randomly chosen member of a ‘smaller’ set. This is a valid test because the values are independent (same assumption as before) and the values have an order. Typically this test is not valid unless both sets have at least 20 points, which has been shown above in the section on the student t-test. The results for this test give a 1 sided p-value. The summary of the results can be found in the table below. This test is a better test for non-normal distributions, like the one we have. Because of the size of our data set, however, the results are very comparable. It should be noted that if the results did not match then the Mann-Whitney would be a more authoritative test.


    nowork_rain = subway_data.loc[(subway_data.rain==1)&(subway_data.workday==0)]
    nowork_norain = subway_data.loc[(subway_data.rain==0)&(subway_data.workday==0)]
    
    results = scipy.stats.mannwhitneyu(nowork_norain.ENTRIESn_hourly,nowork_rain.ENTRIESn_hourly)
    print "U-Test different distrubtions of hourly entries on non-workdays"
    print "U Score:", results[0]
    print "p-value", results[1]
    print ""
    work_rain = subway_data.loc[(subway_data.rain==1)&(subway_data.workday==1)]
    work_norain = subway_data.loc[(subway_data.rain==0)&(subway_data.workday==1)]
    
    results = scipy.stats.mannwhitneyu(work_norain.ENTRIESn_hourly,work_rain.ENTRIESn_hourly)
    print "U-Test different distrubtions of hourly entries on workdays"
    print "U Score:", results[0]
    print "p-value", results[1]

    2-Sided T-Test for change in average number of hourly entries on non-workdays
    T Score: 10160674.0
    p-value 0.00415933264978
    
    2-Sided T-Test for change in average number of hourly entries on workdays
    T Score: 82547361.5
    p-value 0.295771577293


The results for the Mann-Whitney U-Test are very similar to the those of the student t-test. When all the data is considered there is a statically significant difference between the size of the ridership when it rains compared to when it is not raining. The medians demonstrate that the ‘larger’ population for this statistical test is the population of riders when it is raining.

The other two results demonstrate the previously mentioned Simpson's Paradox in that the order of the results of the test is reversed. Days without rain have a larger ridership on both workdays and non-workdays when considered separately. And just like the student t-test, the difference is statistically significant on days off, but not on workdays.

####Summary of Statistical Tests
In summary, the statistical test allows for a way to ask questions about two populations being different. On a population as a whole, it is clear that both the student t-test and the Mann-Whitney U-test support the hypothesis that there are more riders when it rains. In the first section it was noted that weekday ridership was drastically larger than weekend ridership, and that more rainy days fell on weekdays where ridership was large. The total population being composed of a subset of two populations with different riding behavior bias this result, so the data was subsumed into the two underlying populations.
￼
The effect of rain on workday riders was not statistically different for either statistical test.
The effect of riders entering the subway on non-workdays was statistically different, and both tests showed that ridership dropped when it rained.

These tests are not conclusive about the effect of rain. The hour of the day is showed to have strong explanatory power for the number of hourly entries is explained in the next section. This is also obvious in the previous graphs. The effect on this variable, however, can not be explored with this data set because the rain is only recorded daily and not hourly. This dataset does not have accurate information for rain in 4 hour increments, so further analysis will be deferred.

This analysis focused on the total number of people entering the subway system when it rained, but there are 207 entry points into this system.   Even though we have showed there are statistically significant differences, the underlying cause is still not illuminated.   That, too, would require further analysis.

##Regression

The goal of regression is to make predictions or find functional relationships for dependent variables from independent data based on a model. Ordinary Least Squares Regressions are models where the predicted variable, y, is a linear sum of variables, x:

$y = a_0 + a_1 x_1+ a_2 x_2 + ....$

where y is the dependant variable, $x_i$ is an independent variable, and $a_i$ are the coefficients.

The biggest features that were observed in the initial analysis were that the number of riders varied cyclically over 24 hours, different stations had different amounts of ridership, and workdays had significantly more ridership than non-workdays. The goal of this analysis is to create a regression that models these features, then determine if including information about the rain significantly alters the model.

In order to facilitate the fit we will include new variables and that depend on the cyclic variation.  These variables will be

$ cos(2 \pi hour / 24), sin(2 \pi hour / 24), cos(2 \pi day_week / 7), & cos(2 \pi day_week / 7) .$

Additioanlly, we will also make a training set of the first 24 days of the month, and test the predictive ability on the last 7 days of the month.  





    totals = subway_data.groupby(['datetime','day_week','hour','workday','rain'])
    total_data = totals.ENTRIESn_hourly.sum().reset_index()
    total_data['cos_day'] = np.cos(2*math.pi*total_data.day_week/7)
    total_data['sin_day'] = np.sin(2*math.pi*total_data.day_week/7)
    total_data['cos_hour'] = np.cos(2*math.pi*total_data.hour/24)
    total_data['sin_hour'] = np.sin(2*math.pi*total_data.hour/24)
    train = total_data[total_data.datetime < '2011-05-25']
    test = total_data[total_data.datetime > '2011-05-25']
    


The graphs at the beginning of this analysis showed the number of entries into the subway system is heavily influenced by the hour of the day and if the day is a workday or not.   The most general linear combination of these variables are fitted to the first 24 days of data.


    result = smf.ols(formula="ENTRIESn_hourly ~ hour*workday*cos_hour +  hour*workday*sin_hour", data=train).fit()
    result.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>ENTRIESn_hourly</td> <th>  R-squared:         </th> <td>   0.595</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.569</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   22.61</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 20 May 2015</td> <th>  Prob (F-statistic):</th> <td>6.41e-28</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:46:20</td>     <th>  Log-Likelihood:    </th> <td> -2466.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   181</td>      <th>  AIC:               </th> <td>   4958.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   169</td>      <th>  BIC:               </th> <td>   4996.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>             <td>-1.235e+06</td> <td> 1.53e+06</td> <td>   -0.808</td> <td> 0.420</td> <td>-4.25e+06  1.78e+06</td>
</tr>
<tr>
  <th>hour</th>                  <td> 1.529e+05</td> <td> 1.52e+05</td> <td>    1.003</td> <td> 0.317</td> <td>-1.48e+05  4.54e+05</td>
</tr>
<tr>
  <th>workday</th>               <td>-2.806e+06</td> <td>  1.8e+06</td> <td>   -1.562</td> <td> 0.120</td> <td>-6.35e+06  7.41e+05</td>
</tr>
<tr>
  <th>hour:workday</th>          <td>  3.18e+05</td> <td> 1.79e+05</td> <td>    1.774</td> <td> 0.078</td> <td>-3.58e+04  6.72e+05</td>
</tr>
<tr>
  <th>cos_hour</th>              <td> 1.545e+06</td> <td> 1.52e+06</td> <td>    1.016</td> <td> 0.311</td> <td>-1.46e+06  4.55e+06</td>
</tr>
<tr>
  <th>hour:cos_hour</th>         <td>-1.037e+05</td> <td> 1.06e+05</td> <td>   -0.983</td> <td> 0.327</td> <td>-3.12e+05  1.05e+05</td>
</tr>
<tr>
  <th>workday:cos_hour</th>      <td>  2.74e+06</td> <td> 1.79e+06</td> <td>    1.534</td> <td> 0.127</td> <td>-7.86e+05  6.27e+06</td>
</tr>
<tr>
  <th>hour:workday:cos_hour</th> <td>-1.799e+05</td> <td> 1.24e+05</td> <td>   -1.451</td> <td> 0.149</td> <td>-4.24e+05  6.48e+04</td>
</tr>
<tr>
  <th>sin_hour</th>              <td>-1.673e+05</td> <td> 3.04e+05</td> <td>   -0.551</td> <td> 0.582</td> <td>-7.67e+05  4.32e+05</td>
</tr>
<tr>
  <th>hour:sin_hour</th>         <td> 7.931e+04</td> <td> 6.26e+04</td> <td>    1.266</td> <td> 0.207</td> <td>-4.43e+04  2.03e+05</td>
</tr>
<tr>
  <th>workday:sin_hour</th>      <td>-6.124e+04</td> <td> 3.54e+05</td> <td>   -0.173</td> <td> 0.863</td> <td> -7.6e+05  6.38e+05</td>
</tr>
<tr>
  <th>hour:workday:sin_hour</th> <td>  1.61e+05</td> <td> 7.39e+04</td> <td>    2.178</td> <td> 0.031</td> <td> 1.51e+04  3.07e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>80.504</td> <th>  Durbin-Watson:     </th> <td>   2.164</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 228.585</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.923</td> <th>  Prob(JB):          </th> <td>2.31e-50</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.940</td> <th>  Cond. No.          </th> <td>3.81e+03</td>
</tr>
</table>



Using this fit, we make a prediction on what the number of hourly entires for the last 7 days should look like.


    test2 = test.set_index('datetime')
    test2['predict'] = result.predict(test)
    plt.figure()
    test2.ENTRIESn_hourly.plot()
    test2.predict.plot()





    <matplotlib.axes._subplots.AxesSubplot at 0x11f33de50>




![png](http://www.bryantravissmith.com/img/udacity/project1/output_28_1.png)


This fit captures a number of features, including the double peak on workdays and the different behavior for memorial days.   We can find the $r^2$ for this fit on the test data.


    diff = test2.ENTRIESn_hourly.values-test2.ENTRIESn_hourly.mean()
    diff2 = test2.ENTRIESn_hourly.values-test2.predict.values
    
    print "R-squared on prediction:", 1-(diff2.dot(diff2)/diff.dot(diff))

    R-squared on prediction: 0.811683091424


Over 80% of the variance of this test data is accounted for by the model.  We can now see if including information abou the rain improve the performance.


    result = smf.ols(formula="ENTRIESn_hourly ~ rain*hour*workday*cos_hour + rain*hour*workday*sin_hour", data=train).fit()
    result.summary()




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>ENTRIESn_hourly</td> <th>  R-squared:         </th> <td>   0.604</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.546</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   10.40</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 20 May 2015</td> <th>  Prob (F-statistic):</th> <td>2.05e-21</td>
</tr>
<tr>
  <th>Time:</th>                 <td>07:47:25</td>     <th>  Log-Likelihood:    </th> <td> -2464.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   181</td>      <th>  AIC:               </th> <td>   4978.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   157</td>      <th>  BIC:               </th> <td>   5055.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    23</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>                  <td> -1.39e+06</td> <td> 1.81e+06</td> <td>   -0.768</td> <td> 0.443</td> <td>-4.96e+06  2.18e+06</td>
</tr>
<tr>
  <th>rain</th>                       <td> 5.582e+05</td> <td> 3.64e+06</td> <td>    0.153</td> <td> 0.878</td> <td>-6.64e+06  7.75e+06</td>
</tr>
<tr>
  <th>hour</th>                       <td> 1.719e+05</td> <td>  1.8e+05</td> <td>    0.953</td> <td> 0.342</td> <td>-1.84e+05  5.28e+05</td>
</tr>
<tr>
  <th>rain:hour</th>                  <td>-6.956e+04</td> <td> 3.63e+05</td> <td>   -0.191</td> <td> 0.848</td> <td>-7.87e+05  6.48e+05</td>
</tr>
<tr>
  <th>workday</th>                    <td>-2.621e+06</td> <td> 2.16e+06</td> <td>   -1.213</td> <td> 0.227</td> <td>-6.89e+06  1.65e+06</td>
</tr>
<tr>
  <th>rain:workday</th>               <td>-6.436e+05</td> <td> 4.19e+06</td> <td>   -0.154</td> <td> 0.878</td> <td>-8.91e+06  7.62e+06</td>
</tr>
<tr>
  <th>hour:workday</th>               <td> 2.946e+05</td> <td> 2.16e+05</td> <td>    1.367</td> <td> 0.174</td> <td>-1.31e+05   7.2e+05</td>
</tr>
<tr>
  <th>rain:hour:workday</th>          <td> 8.311e+04</td> <td> 4.18e+05</td> <td>    0.199</td> <td> 0.843</td> <td>-7.42e+05  9.08e+05</td>
</tr>
<tr>
  <th>cos_hour</th>                   <td> 1.742e+06</td> <td>  1.8e+06</td> <td>    0.968</td> <td> 0.334</td> <td>-1.81e+06  5.29e+06</td>
</tr>
<tr>
  <th>rain:cos_hour</th>              <td>-7.256e+05</td> <td> 3.62e+06</td> <td>   -0.200</td> <td> 0.841</td> <td>-7.88e+06  6.43e+06</td>
</tr>
<tr>
  <th>hour:cos_hour</th>              <td>-1.167e+05</td> <td> 1.25e+05</td> <td>   -0.934</td> <td> 0.352</td> <td>-3.63e+05   1.3e+05</td>
</tr>
<tr>
  <th>rain:hour:cos_hour</th>         <td> 4.757e+04</td> <td> 2.51e+05</td> <td>    0.189</td> <td> 0.850</td> <td>-4.48e+05  5.43e+05</td>
</tr>
<tr>
  <th>workday:cos_hour</th>           <td> 2.497e+06</td> <td> 2.15e+06</td> <td>    1.162</td> <td> 0.247</td> <td>-1.75e+06  6.74e+06</td>
</tr>
<tr>
  <th>rain:workday:cos_hour</th>      <td> 8.677e+05</td> <td> 4.16e+06</td> <td>    0.209</td> <td> 0.835</td> <td>-7.35e+06  9.09e+06</td>
</tr>
<tr>
  <th>hour:workday:cos_hour</th>      <td>-1.652e+05</td> <td> 1.49e+05</td> <td>   -1.108</td> <td> 0.270</td> <td> -4.6e+05  1.29e+05</td>
</tr>
<tr>
  <th>rain:hour:workday:cos_hour</th> <td>-5.263e+04</td> <td> 2.88e+05</td> <td>   -0.182</td> <td> 0.855</td> <td>-6.22e+05  5.17e+05</td>
</tr>
<tr>
  <th>sin_hour</th>                   <td>-1.916e+05</td> <td>  3.6e+05</td> <td>   -0.532</td> <td> 0.595</td> <td>-9.03e+05   5.2e+05</td>
</tr>
<tr>
  <th>rain:sin_hour</th>              <td> 1.001e+05</td> <td>  7.2e+05</td> <td>    0.139</td> <td> 0.890</td> <td>-1.32e+06  1.52e+06</td>
</tr>
<tr>
  <th>hour:sin_hour</th>              <td> 8.999e+04</td> <td> 7.39e+04</td> <td>    1.217</td> <td> 0.225</td> <td> -5.6e+04  2.36e+05</td>
</tr>
<tr>
  <th>rain:hour:sin_hour</th>         <td>-3.917e+04</td> <td>  1.5e+05</td> <td>   -0.261</td> <td> 0.794</td> <td>-3.36e+05  2.57e+05</td>
</tr>
<tr>
  <th>workday:sin_hour</th>           <td>-1.605e+04</td> <td> 4.25e+05</td> <td>   -0.038</td> <td> 0.970</td> <td>-8.55e+05  8.23e+05</td>
</tr>
<tr>
  <th>rain:workday:sin_hour</th>      <td> -1.69e+05</td> <td> 8.25e+05</td> <td>   -0.205</td> <td> 0.838</td> <td> -1.8e+06  1.46e+06</td>
</tr>
<tr>
  <th>hour:workday:sin_hour</th>      <td> 1.467e+05</td> <td> 8.88e+04</td> <td>    1.651</td> <td> 0.101</td> <td>-2.88e+04  3.22e+05</td>
</tr>
<tr>
  <th>rain:hour:workday:sin_hour</th> <td> 5.061e+04</td> <td> 1.73e+05</td> <td>    0.293</td> <td> 0.770</td> <td>-2.91e+05  3.92e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>81.829</td> <th>  Durbin-Watson:     </th> <td>   2.161</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 242.045</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.932</td> <th>  Prob(JB):          </th> <td>2.76e-53</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.143</td> <th>  Cond. No.          </th> <td>9.63e+03</td>
</tr>
</table>




    test3 = test.set_index('datetime')
    test3['predict'] = result.predict(test)
    test3.ENTRIESn_hourly.plot()
    test3.predict.plot()




    <matplotlib.axes._subplots.AxesSubplot at 0x11f306a90>




![png](http://www.bryantravissmith.com/img/udacity/project1/output_33_1.png)



    diff = test3.ENTRIESn_hourly.values-test2.ENTRIESn_hourly.mean()
    diff2 = test3.ENTRIESn_hourly.values-test2.predict.values
    
    print "R-squared on prediction:", 1-(diff2.dot(diff2)/diff.dot(diff))

    R-squared on prediction: 0.811683091424


This is the same $r^2$ as the fit without including information about the rain.   For this simple model, rain does not imporve its performance.   

###Summary

There is not any evidence that the information of the rain has any explanatory power for being able to estimate the number of riders entering the subway at a given station. The time of day and the day being a workday or not offer the majority of the explanatory power of the number of riders entering a given station.  Rain does not have significant explanatory power on these regression fits.

##Conclusion

The goal of this analysis is to determine if the number of people riding the subway changes when it rains. There is no evidence to support this statement based on the data.

When the data is subsetted into the two ridership patterns of workday and non-workdays, there is not a significant increase in ridership on rainy workdays. Rainy non-workdays, on the other hand, show a decrease in ridership. The Mann-Whitney analysis supports that the size of the change in ridership is a small effect.

The regression of the data show that the time of day, the station under consideration, and fact of if the day is a workday or not has great explanatory power on the data set. Over 90% of the variance can be explained by the fit shown in this analysis. Attempting to incorporate rain in the regression does not have an statistically significant improvements in the fits.


##Reflection
Obviously with any statistical test there is a possibility to confuse a statistically significant difference with an attribution of the independent variables causing the shift or change in the dependant variables. The data is not fine enough to do a set of statistical tests that subset the data by hour and workday since the categorial rain variable is set by day and not by hour.

There is a variable called ‘condition’ set hourly, but there were days where rain = 0 and the conditions = ‘Scattered Showers’. This condition variable is inconsistent with the rain variable. Since it was not in the original data set, rain was used in this analysis as the more dependable variable. It could be possible that the original rain variable was incorrect.


The data only contained 31 days of data. There were 7 rainy days and 1 was a holiday. Even though the data covered 240 turnstiles at 207 stations, there were only 7 rainy days at a given time of day, and 24 non-rainy days at the same given time of day for any single station. Because ridership varies so much with time, comparing similar times against each other when it is raining and not raining would be necessary to draw any strong conclusions.



    
