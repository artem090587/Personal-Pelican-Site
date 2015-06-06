Title: Galvanize - Week 01 - Day 5
Date: 2015-06-05 10:20
Modified: 2015-06-05 10:30
Category: Galvanize
Tags: data-science, galvanize, pandas, money ball
Slug: galvanize-data-science-01-05
Authors: Bryan Smith
Summary: The fifth day of Galvanize's Immersive Data Science program in San Francisco, CA where we got an introduction to pandas and data visualizations.  

#Galvanize Immersive Data Science
##Week 1 - Day 5

This morning we started with a reflection about the week, and completed a survey about our progress and thoughts on the program.   I think it is a very strong, hands-on program so far.

The morning lesson and sprint was on using (pandas)[http://pandas.pydata.org/].   We were given some hospital data in a CSV format, read in the data, and answered a number of questions about most common diseases, most expensive procedures, the post profitable hospitals, and various subsets of these question on different conditions.  It was a simple exercise that gave us practice making new variables, grouping, and subsetting to look massage the data into a form that allowed us to answer the questions.

During our lunch we had a presentation on learning, and the approach and attitudes that facilitate the bests learning.  It was partly motivational and partly reflective.  If you are familiar with Carol Dweck's work and the research it created, then you have a feeling for the talk.

After lunch we did a fun assignment that was to recreate the MoneyBall movie where we are trying to get a set of three players that in aggregate replace the 3 key players that were just lost.   

The data we used is hosted (here)[http://www.seanlahman.com/baseball-archive/statistics/]

##Money Ball

We first started by downloading the dataset and loading it into Pandas.  We started with the batting data:


    %matplotlib inline
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    
    batting = pd.read_csv('data/baseball-csvs/Batting.csv')
    batting.head()





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID</th>
      <th>lgID</th>
      <th>G</th>
      <th>G_batting</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>...</th>
      <th>SB</th>
      <th>CS</th>
      <th>BB</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
      <th>GIDP</th>
      <th>G_old</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>2004</td>
      <td>1</td>
      <td>SFN</td>
      <td>NL</td>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aardsda01</td>
      <td>2006</td>
      <td>1</td>
      <td>CHN</td>
      <td>NL</td>
      <td>45</td>
      <td>43</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aardsda01</td>
      <td>2007</td>
      <td>1</td>
      <td>CHA</td>
      <td>AL</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aardsda01</td>
      <td>2008</td>
      <td>1</td>
      <td>BOS</td>
      <td>AL</td>
      <td>47</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>aardsda01</td>
      <td>2009</td>
      <td>1</td>
      <td>SEA</td>
      <td>AL</td>
      <td>73</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



We then loaded the salaryd data:


    salary = pd.read_csv('data/baseball-csvs/Salaries.csv')
    salary.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearID</th>
      <th>teamID</th>
      <th>lgID</th>
      <th>playerID</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>BAL</td>
      <td>AL</td>
      <td>murraed02</td>
      <td>1472819</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>BAL</td>
      <td>AL</td>
      <td>lynnfr01</td>
      <td>1090000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>BAL</td>
      <td>AL</td>
      <td>ripkeca01</td>
      <td>800000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>BAL</td>
      <td>AL</td>
      <td>lacyle01</td>
      <td>725000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>BAL</td>
      <td>AL</td>
      <td>flanami01</td>
      <td>641667</td>
    </tr>
  </tbody>
</table>
</div>




    salary[salary.yearID==2001].salary.describe()




    count         860.000000
    mean      2279841.061628
    std       2907710.250521
    min        200000.000000
    25%        269375.000000
    50%        925000.000000
    75%       3250000.000000
    max      22000000.000000
    Name: salary, dtype: float64



In the year 2001, the year we are concerned with, the minimum salary was $200,000.

The next thing we did was merged the two dataframes and limited the data to 2001.


    mergeddf = batting.merge(salary,on=['playerID','yearID'],how='left')
    mergeddf = mergeddf[mergeddf.yearID==2001]
    mergeddf.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID_x</th>
      <th>lgID_x</th>
      <th>G</th>
      <th>G_batting</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>...</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
      <th>GIDP</th>
      <th>G_old</th>
      <th>teamID_y</th>
      <th>lgID_y</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>abadan01</td>
      <td>2001</td>
      <td>1</td>
      <td>OAK</td>
      <td>AL</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>abbotje01</td>
      <td>2001</td>
      <td>1</td>
      <td>FLO</td>
      <td>NL</td>
      <td>28</td>
      <td>28</td>
      <td>42</td>
      <td>5</td>
      <td>11</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>FLO</td>
      <td>NL</td>
      <td>300000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>abbotku01</td>
      <td>2001</td>
      <td>1</td>
      <td>ATL</td>
      <td>NL</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>ATL</td>
      <td>NL</td>
      <td>600000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>abbotpa01</td>
      <td>2001</td>
      <td>1</td>
      <td>SEA</td>
      <td>AL</td>
      <td>28</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>SEA</td>
      <td>AL</td>
      <td>1700000</td>
    </tr>
    <tr>
      <th>64</th>
      <td>abernbr01</td>
      <td>2001</td>
      <td>1</td>
      <td>TBA</td>
      <td>AL</td>
      <td>79</td>
      <td>79</td>
      <td>304</td>
      <td>43</td>
      <td>82</td>
      <td>...</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



We can see some of the salaries are missing.  There are players that can be aquired, but are not on a payroll.   If we pick them up we have to pay them 200,000.


    mergeddf.salary = mergeddf.salary.fillna(200000)
    mergeddf.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID_x</th>
      <th>lgID_x</th>
      <th>G</th>
      <th>G_batting</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>...</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
      <th>GIDP</th>
      <th>G_old</th>
      <th>teamID_y</th>
      <th>lgID_y</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>abadan01</td>
      <td>2001</td>
      <td>1</td>
      <td>OAK</td>
      <td>AL</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>abbotje01</td>
      <td>2001</td>
      <td>1</td>
      <td>FLO</td>
      <td>NL</td>
      <td>28</td>
      <td>28</td>
      <td>42</td>
      <td>5</td>
      <td>11</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>FLO</td>
      <td>NL</td>
      <td>300000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>abbotku01</td>
      <td>2001</td>
      <td>1</td>
      <td>ATL</td>
      <td>NL</td>
      <td>6</td>
      <td>6</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>ATL</td>
      <td>NL</td>
      <td>600000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>abbotpa01</td>
      <td>2001</td>
      <td>1</td>
      <td>SEA</td>
      <td>AL</td>
      <td>28</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>SEA</td>
      <td>AL</td>
      <td>1700000</td>
    </tr>
    <tr>
      <th>64</th>
      <td>abernbr01</td>
      <td>2001</td>
      <td>1</td>
      <td>TBA</td>
      <td>AL</td>
      <td>79</td>
      <td>79</td>
      <td>304</td>
      <td>43</td>
      <td>82</td>
      <td>...</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



Now we need to make some new variables.   The number of times they were on First Base,the Batting Average (BA), the On Base Percentage (OBP), and the Slugg (SLG)


    mergeddf['BA'] = mergeddf['H']/mergeddf['AB']
    mergeddf.BA.describe()




    count    1044.000000
    mean        0.202532
    std         0.140697
    min         0.000000
    25%         0.117647
    50%         0.235227
    75%         0.274705
    max         1.000000
    Name: BA, dtype: float64




    mergeddf['1B'] = mergeddf['H']-mergeddf['2B']-mergeddf['3B']-mergeddf['HR']
    mergeddf['1B'].describe()




    count    1237.000000
    mean       23.185125
    std        34.327716
    min         0.000000
    25%         0.000000
    50%         4.000000
    75%        33.000000
    max       192.000000
    Name: 1B, dtype: float64




    mergeddf['SLG']=(mergeddf['1B']+2*mergeddf['2B']+3*mergeddf['3B'] \
                     +4*mergeddf['HR'])/mergeddf['AB']
    mergeddf['SLG'].describe()




    count    1044.000000
    mean        0.303628
    std         0.214569
    min         0.000000
    25%         0.142857
    50%         0.337722
    75%         0.436874
    max         2.000000
    Name: SLG, dtype: float64




    mergeddf['OBP']=(mergeddf['H']+mergeddf['BB']+mergeddf['HBP']) \
    /(mergeddf['AB']+mergeddf['BB']+mergeddf['HBP']+mergeddf['SF'])
    mergeddf['OBP'].describe()




    count    1047.000000
    mean        0.254084
    std         0.159932
    min         0.000000
    25%         0.162162
    50%         0.293103
    75%         0.338235
    max         1.000000
    Name: OBP, dtype: float64



The A's lost Jason Giambi (`giambja01`), Johnny Damon (`damonjo01`), Jason Isringhausen (`isrinja01`), and Rainer Gustavo "Ray" Olmedo (``'saenzol01'``).

These player need to replaced with similar player that bat, in total, as much as these guys, get on base as often as these guys, and can be payed less than these guys.  



    my_mask = mergeddf['playerID'].isin(['giambja01','damonjo01','isrinja01','saenzol01'])
    lostboysdf = mergeddf[my_mask]
    imp_var = ['playerID', 'teamID_x','AB','HR', 'OBP', 'SLG', 'salary']
    lostboysdf[imp_var]




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>teamID_x</th>
      <th>AB</th>
      <th>HR</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7065</th>
      <td>damonjo01</td>
      <td>OAK</td>
      <td>644</td>
      <td>9</td>
      <td>0.323529</td>
      <td>0.363354</td>
      <td>7100000</td>
    </tr>
    <tr>
      <th>10836</th>
      <td>giambja01</td>
      <td>OAK</td>
      <td>520</td>
      <td>38</td>
      <td>0.476900</td>
      <td>0.659615</td>
      <td>4103333</td>
    </tr>
    <tr>
      <th>14911</th>
      <td>isrinja01</td>
      <td>OAK</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3300000</td>
    </tr>
    <tr>
      <th>27408</th>
      <td>saenzol01</td>
      <td>OAK</td>
      <td>305</td>
      <td>9</td>
      <td>0.291176</td>
      <td>0.383607</td>
      <td>290000</td>
    </tr>
  </tbody>
</table>
</div>




    print "Avg OBP Needed to be replaced:", 3*lostboysdf[imp_var].OBP.mean()
    print "Total Bats needed to be replaced:", lostboysdf[imp_var].AB.sum()

    Avg OBP Needed to be replaced: 1.09160603138
    Total Bats needed to be replaced: 1469.0


We would ideally like to get every combination of 3 players that are available.  That would be:


    mergeddf = mergeddf[~mergeddf.playerID.isin(['giambja01','damonjo01','isrinja01','saenzol01'])]
    size = len(mergeddf)
    print size
    import math
    
    def nCr(n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    print nCr(size,3)

    1335
    395654395


That is almost 400,000,000 combinations to search through.   Less make some reasonable assumptions about about the minimum At Bats and On Base Percentages


    mergeddf.plot(kind='scatter',x='AB',y='OBP')




    <matplotlib.axes._subplots.AxesSubplot at 0x1063f3f10>




![png](http://www.bryantravissmith.com/img/gw1d5_1.png)


It looks like the variance does not change much above 200 AB.  We probably want them to bat about 500 times a season, however.  We also want the average to be above 0.33 for the OPB, so that seems like a reasonable initial cutoff.  We also want there average salary to be less than 5000000.


    size = len(mergeddf[(mergeddf.AB > 400) & (mergeddf.OBP > 0.33)])
    nCr(size,3)




    246905L



That gives us 11,480 combinations to search through.  Lets do it.


    #mdf = mergeddf[(mergeddf.yearID==2001)&(mergeddf.AB>50)] #.salary.describe()
    #mdf.salary = mdf.salary.fillna(200000)
    #mdf[imp_var].head()
    mdf = mergeddf[(mergeddf.AB > 400) & (mergeddf.OBP > 0.33)]
    
    from itertools import combinations
    good_combinations = []
    mdf = mdf[~mdf.playerID.isin( ['giambja01','damonjo01','isrinja01','saenzol01'])]
    gc = pd.DataFrame(columns=['player1','player2','player3','total_AB','total_OBP','total_salary'])
    i = 0
    j = 0
    for x in combinations(mdf[mdf.OBP>0.4].playerID.tolist(),3):
        
        total_salary = mdf[mdf.playerID==x[0]].salary.values[0]
        total_salary += mdf[mdf.playerID==x[1]].salary.values[0]
        total_salary += mdf[mdf.playerID==x[2]].salary.values[0]
        
        total_AB = mdf[mdf.playerID==x[0]].AB.values[0]
        total_AB += mdf[mdf.playerID==x[1]].AB.values[0]
        total_AB += mdf[mdf.playerID==x[2]].AB.values[0]
        
        total_obp = mdf[mdf.playerID==x[0]].OBP.values[0]
        total_obp += mdf[mdf.playerID==x[1]].OBP.values[0]
        total_obp += mdf[mdf.playerID==x[2]].OBP.values[0]
            
        if (total_salary < 15000000) & (total_obp > 1.0961):
            gc.loc[i] = [x[0],x[1],x[2],total_AB,total_obp,total_salary]
            i += 1
    
    gc = gc.sort(['total_salary'])
    gc.head(10)




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player1</th>
      <th>player2</th>
      <th>player3</th>
      <th>total_AB</th>
      <th>total_OBP</th>
      <th>total_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>berkmla01</td>
      <td>gonzalu01</td>
      <td>pujolal01</td>
      <td>1776</td>
      <td>1.261767</td>
      <td>5338333</td>
    </tr>
    <tr>
      <th>32</th>
      <td>berkmla01</td>
      <td>heltoto01</td>
      <td>pujolal01</td>
      <td>1754</td>
      <td>1.264850</td>
      <td>5455000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>berkmla01</td>
      <td>martied01</td>
      <td>pujolal01</td>
      <td>1637</td>
      <td>1.256603</td>
      <td>6005000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>berkmla01</td>
      <td>edmonji01</td>
      <td>pujolal01</td>
      <td>1667</td>
      <td>1.243410</td>
      <td>6838333</td>
    </tr>
    <tr>
      <th>38</th>
      <td>berkmla01</td>
      <td>olerujo01</td>
      <td>pujolal01</td>
      <td>1739</td>
      <td>1.234375</td>
      <td>7205000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>berkmla01</td>
      <td>gilesbr02</td>
      <td>pujolal01</td>
      <td>1743</td>
      <td>1.236756</td>
      <td>7838333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alomaro01</td>
      <td>berkmla01</td>
      <td>pujolal01</td>
      <td>1742</td>
      <td>1.247866</td>
      <td>8255000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>berkmla01</td>
      <td>pujolal01</td>
      <td>thomeji01</td>
      <td>1693</td>
      <td>1.249345</td>
      <td>8380000</td>
    </tr>
    <tr>
      <th>55</th>
      <td>gonzalu01</td>
      <td>heltoto01</td>
      <td>pujolal01</td>
      <td>1786</td>
      <td>1.263189</td>
      <td>9983333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>berkmla01</td>
      <td>gonzalu01</td>
      <td>heltoto01</td>
      <td>1773</td>
      <td>1.290459</td>
      <td>10088333</td>
    </tr>
  </tbody>
</table>
</div>




    print len(gc)

    66


This allowed us to find 66 combination of players that would, in aggregate, have better statistics that the players that were lost.  It also turns out to be cheaper to do that.   This is the story of money ball.  It was a fun project.  


    
