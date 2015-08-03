Title: Galvanize - Week 08 - Day 2 
Date: 2015-07-22 10:20
Modified: 2015-07-22 10:30
Category: Galvanize
Tags: data-science, galvanize, seaborn, bokeh
Slug: galvanize-data-science-08-02
Authors: Bryan Smith
Summary: Today we covered data visualizations

#Galvanize Immersive Data Science

##Week 8 Day 2

Today we covered data visualizations in python, and some of the newer visualization including seaborn and bokeh.  Our quiz was to watch a set videos from Udacities Data Visualization course.  

We are going to visualization crime rate data for different classes of crime.


    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from bokeh.plotting import figure, output_file, show
    import seaborn as sns
    %matplotlib inline
    df = pd.read_csv("data/crime.csv")
    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Murder</th>
      <th>Rape</th>
      <th>Robbery</th>
      <th>Aggravated Assault</th>
      <th>Burglary</th>
      <th>Larceny Theft</th>
      <th>Motor Vehicle Theft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>8.2</td>
      <td>34.3</td>
      <td>141.4</td>
      <td>247.8</td>
      <td>953.8</td>
      <td>2650.0</td>
      <td>288.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>4.8</td>
      <td>81.1</td>
      <td>80.9</td>
      <td>465.1</td>
      <td>622.5</td>
      <td>2599.1</td>
      <td>391.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>7.5</td>
      <td>33.8</td>
      <td>144.4</td>
      <td>327.4</td>
      <td>948.4</td>
      <td>2965.2</td>
      <td>924.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>6.7</td>
      <td>42.9</td>
      <td>91.1</td>
      <td>386.8</td>
      <td>1084.6</td>
      <td>2711.2</td>
      <td>262.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>6.9</td>
      <td>26.0</td>
      <td>176.1</td>
      <td>317.3</td>
      <td>693.3</td>
      <td>1916.5</td>
      <td>712.8</td>
    </tr>
  </tbody>
</table>
</div>



These numbers are per/100,000 people.   


    df.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 51 entries, 0 to 50
    Data columns (total 8 columns):
    State                  51 non-null object
    Murder                 51 non-null float64
    Rape                   51 non-null float64
    Robbery                51 non-null float64
    Aggravated Assault     51 non-null float64
    Burglary               51 non-null float64
    Larceny Theft          51 non-null float64
    Motor Vehicle Theft    51 non-null float64
    dtypes: float64(7), object(1)
    memory usage: 3.6+ KB


We are not missing any data in this data set, so we are likely good for visualizations.   Lets find the states that are the lowest and the highest rate states for each class of crime:


    def get_state_min(cat):
        return df[df[cat]==df[cat].min()].State.values[0]
    
    def get_state_max(cat):
        return df[df[cat]==df[cat].max()].State.values[0]
    
    for c in df.columns:
        if c != "State":
            print c + " "*(25-len(c)), get_state_min(c)," "*(25-len(get_state_min(c))),get_state_max(c)

    Murder                    North Dakota               District of Columbia
    Rape                      New Jersey                 Alaska
    Robbery                   North Dakota               District of Columbia
    Aggravated Assault        Maine                      District of Columbia
    Burglary                  North Dakota               North Carolina
    Larceny Theft             South Dakota               Hawaii
    Motor Vehicle Theft       Maine                      District of Columbia


So North Dakota and Maine are commonly in  the lowest, while DC is the highest in multiple categories.

A correlation matrix will show if there are some crimes that are associated with each other accorse states


    df.drop('State',axis=1).corr()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Murder</th>
      <th>Rape</th>
      <th>Robbery</th>
      <th>Aggravated Assault</th>
      <th>Burglary</th>
      <th>Larceny Theft</th>
      <th>Motor Vehicle Theft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Murder</th>
      <td>1.000000</td>
      <td>0.019347</td>
      <td>0.925005</td>
      <td>0.673629</td>
      <td>0.276312</td>
      <td>0.223320</td>
      <td>0.674955</td>
    </tr>
    <tr>
      <th>Rape</th>
      <td>0.019347</td>
      <td>1.000000</td>
      <td>-0.056155</td>
      <td>0.379253</td>
      <td>0.356518</td>
      <td>0.318834</td>
      <td>0.139953</td>
    </tr>
    <tr>
      <th>Robbery</th>
      <td>0.925005</td>
      <td>-0.056155</td>
      <td>1.000000</td>
      <td>0.653702</td>
      <td>0.229434</td>
      <td>0.181704</td>
      <td>0.712368</td>
    </tr>
    <tr>
      <th>Aggravated Assault</th>
      <td>0.673629</td>
      <td>0.379253</td>
      <td>0.653702</td>
      <td>1.000000</td>
      <td>0.547263</td>
      <td>0.438468</td>
      <td>0.526181</td>
    </tr>
    <tr>
      <th>Burglary</th>
      <td>0.276312</td>
      <td>0.356518</td>
      <td>0.229434</td>
      <td>0.547263</td>
      <td>1.000000</td>
      <td>0.679892</td>
      <td>0.411503</td>
    </tr>
    <tr>
      <th>Larceny Theft</th>
      <td>0.223320</td>
      <td>0.318834</td>
      <td>0.181704</td>
      <td>0.438468</td>
      <td>0.679892</td>
      <td>1.000000</td>
      <td>0.473116</td>
    </tr>
    <tr>
      <th>Motor Vehicle Theft</th>
      <td>0.674955</td>
      <td>0.139953</td>
      <td>0.712368</td>
      <td>0.526181</td>
      <td>0.411503</td>
      <td>0.473116</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We see that the  Murder and Robbery seem to occur at related rates across states.   Burglar is most correlated with larceny theft. 

This is difficult to decode.  A visualization, per our lesson, would be a better way to show this information.


    import seaborn as sb
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.title("Pearson Correlation")
    sns.heatmap(df.drop('State',axis=1).corr())
    plt.subplot(1,2,2)
    
    plt.title("Spearman Correlation")
    sb.heatmap(df.drop('State',axis=1).corr(method='spearman'))
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_10_0.png)


The pearson correlation makes the assumption there is a linear relationship between the varaibles, while the spearman tracks increasing and decreasing together without an underlying model.  The two relationships that stand out is that murder and robery are less correlated by spearman, but rape and burglar and more correlated.   We also see stronger cluster with Burglary, Larceny, and Motor Vehicle Theft.  

Seaborn is built on Matplotlib, and pandas integrates matplotlib for plotting dataframes.   The nice thing about seaborn is that you can make more complex plots more simply. 

I am going to make two density plots for each type of crame.  One with pandas plotting, and the other with seaborn.  


    from scipy.stats import gaussian_kde
    i = 1
    plt.figure(figsize=(16,8))
    for c in df.columns:
        if c != 'State':
            c_min = df[c].min()
            c_max = df[c].max()
            x = np.linspace(c_min,c_max,100)
            density = gaussian_kde(df[c].values)
            plt.subplot(2,4,i)
            plt.title(c)
            plt.plot(x,density(x))
            i += 1
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_12_0.png)



    plt.figure(figsize=(16,8))
    i=1
    for c in df.columns:
        if c != 'State':
            plt.subplot(2,4,i)
            plt.title(c)
            sb.distplot(df[c])
            i += 1
    plt.show()



![png](http://www.bryantravissmith.com/img/GW08D2/output_13_0.png)


Each of these plots are interestesting, but they do not allow one to quickly compare the scales and ranges for each plot because they have difference scales.  To address that we can do a boxplot.   Pandas makes this very easy, as does seaborn.


    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.title("Matplotlib")
    df.drop('State',axis=1).boxplot()
    plt.xticks(rotation=30)
    plt.subplot(1,2,2)
    plt.title("Seaborn")
    sb.boxplot(df.drop('State',axis=1))
    plt.xticks(rotation=30)
    plt.show()

    /Library/Python/2.7/site-packages/pandas/tools/plotting.py:2633: FutureWarning: 
    The default value for 'return_type' will change to 'axes' in a future release.
     To use the future behavior now, set return_type='axes'.
     To keep the previous behavior and silence this warning, set return_type='dict'.
      warnings.warn(msg, FutureWarning)



![png](http://www.bryantravissmith.com/img/GW08D2/output_15_1.png)


Another related plot is the violin plot.  It is a mixture between the boxplot and the kernal density plot.  Though it is integrated in matplotlib, it is not directly accessible in pandas.  Seaborn, however, does make in easy.


    plt.figure(figsize=(14,8))
    sb.violinplot(df.drop('State',axis=1))
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_17_0.png)


I am thankful that Murder and Rape are must smaller than the other types of crime.  Visually, it is not informative, so lets look that just these two categories of crime.


    plt.figure(figsize=(14,8))
    sb.violinplot(df[['Murder','Rape']])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_19_0.png)


The boxplot easily shows the macro view and sumary of resistant statistics.   It violine plot shows a more detail view than the 5 number summary, but it losses the ease of processing that data.  

## Bar charts

We can look at the total crime rate for each state by adding up the crimerates.  A bar chart is a way to represent categorical variables.  


    df['Total'] = df.drop('State',axis=1).values.sum(axis=1)
    df.head()
    df = df.sort("Total")


    plt.figure(figsize=(14,8))
    ax = plt.gca()
    df.plot(kind='bar',x='State',y='Total',ax=ax)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_23_0.png)


DC has the highest crime rate.  It is almost an outlier.   New Hampshire and the Dakotas, on the other hand, have the least crime rate.  We can use a stacked bar chart to show how much of the total is made out of the each type of crme.  To make it more readiable I will rotate the chart to horizontial.  


    plt.figure(figsize=(14,14))
    ax = plt.gca()
    df.drop('Total',axis=1).plot(kind='barh',x="State", stacked=True,ax=ax);
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_25_0.png)


This does not allow us to compare the compositions of crimes accrose states.  To do that we need to normalize the results.   


    df2 = df.copy()
    for c in df2.columns:
        if c != 'State':
            df2[c] = df[c]/df2.Total
    df2.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Murder</th>
      <th>Rape</th>
      <th>Robbery</th>
      <th>Aggravated Assault</th>
      <th>Burglary</th>
      <th>Larceny Theft</th>
      <th>Motor Vehicle Theft</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>0.000726</td>
      <td>0.016024</td>
      <td>0.014209</td>
      <td>0.037492</td>
      <td>0.164385</td>
      <td>0.714219</td>
      <td>0.052945</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Dakota</td>
      <td>0.001178</td>
      <td>0.023922</td>
      <td>0.009528</td>
      <td>0.055373</td>
      <td>0.166171</td>
      <td>0.688300</td>
      <td>0.055527</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>0.000530</td>
      <td>0.011655</td>
      <td>0.003564</td>
      <td>0.031545</td>
      <td>0.150212</td>
      <td>0.722549</td>
      <td>0.079946</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Vermont</td>
      <td>0.000542</td>
      <td>0.009706</td>
      <td>0.004874</td>
      <td>0.034783</td>
      <td>0.204865</td>
      <td>0.702366</td>
      <td>0.042864</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>0.000554</td>
      <td>0.009781</td>
      <td>0.009662</td>
      <td>0.024433</td>
      <td>0.189482</td>
      <td>0.725696</td>
      <td>0.040391</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




    plt.figure(figsize=(10,14))
    ax = plt.gca()
    df2.drop('Total',axis=1).plot(kind='barh',x='State', stacked=True,ax=ax);
    plt.legend( bbox_to_anchor=(1.2,1))
    plt.show()


![png](http://www.bryantravissmith.com/img/GW08D2/output_28_0.png)


This is interesting, but it is difficult to compare porotions for the middle values because they are not lined up.   This is one of the benefits of javascript based visualizations, because tool tipping allows a direct display of data.

##Bokeh

Bokeh is a new pythong library that outputs interactive plots that are based on web technologies.  


    from collections import OrderedDict
    from bokeh.charts import Bar, output_file, show
    from bokeh.sampledata.olympics2014 import data
    from bokeh.io import output_notebook
    from bokeh.models import HoverTool
    #output_notebook()


    df3 = df.copy()
    df3.columns = [c.replace(" ","_") for c in df3.columns]
    df3.columns




    Index([u'State', u'Murder', u'Rape', u'Robbery', u'Aggravated_Assault',
           u'Burglary', u'Larceny_Theft', u'Motor_Vehicle_Theft', u'Total'],
          dtype='object')




    df3 = df.drop('Total',axis=1).set_index("State")
    df3.columns = [c.replace(" ","_") for c in df3.columns]
    output_file("crime_rate_stacked_bar.html")
    bar = Bar(df3,stacked=True,tools="resize,hover,save,pan,box_zoom,wheel_zoom")
    hover = bar.select(dict(type=HoverTool))
    hover.tooltips = [ (c.replace("_"," "),"@"+c) for c in df3.columns]
    show(bar)

10. To better communicate the breakdown per state, transform the data to normalize each state's data to give a percentage.  This will make all of the bars the full width of the plot.


    df2.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Murder</th>
      <th>Rape</th>
      <th>Robbery</th>
      <th>Aggravated Assault</th>
      <th>Burglary</th>
      <th>Larceny Theft</th>
      <th>Motor Vehicle Theft</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>0.000726</td>
      <td>0.016024</td>
      <td>0.014209</td>
      <td>0.037492</td>
      <td>0.164385</td>
      <td>0.714219</td>
      <td>0.052945</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>South Dakota</td>
      <td>0.001178</td>
      <td>0.023922</td>
      <td>0.009528</td>
      <td>0.055373</td>
      <td>0.166171</td>
      <td>0.688300</td>
      <td>0.055527</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>0.000530</td>
      <td>0.011655</td>
      <td>0.003564</td>
      <td>0.031545</td>
      <td>0.150212</td>
      <td>0.722549</td>
      <td>0.079946</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Vermont</td>
      <td>0.000542</td>
      <td>0.009706</td>
      <td>0.004874</td>
      <td>0.034783</td>
      <td>0.204865</td>
      <td>0.702366</td>
      <td>0.042864</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>0.000554</td>
      <td>0.009781</td>
      <td>0.009662</td>
      <td>0.024433</td>
      <td>0.189482</td>
      <td>0.725696</td>
      <td>0.040391</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




    df4 = df2.drop('Total',axis=1).set_index("State")
    df4.columns = [c.replace(" ","_") for c in df3.columns]
    output_file("crime_rate_stacked_bar_norm.html")
    bar = Bar(df4,stacked=True,tools='resize,hover,save,pan,box_zoom,wheel_zoom')
    hover = bar.select(dict(type=HoverTool))
    hover.tooltips = [ (c.replace("_"," "),"@"+c) for c in df3.columns]
    show(bar)

##Bokeh Maps

We are going to use Bokeh's map system to make some visualization about crime.  To do this we are going to use the US python package to get state information, and work between the bokeh use of state appreviations and our data's list of state names.

For reference there are two map systems that are exceptional.

1.  [CartoDB](https://cartodb.com/)   
2.  [Mapbox](https://www.mapbox.com/) 

I might come back to CartoDB for the afternoon sprint.


    from bokeh.sampledata import us_states
    from bokeh.plotting import *
    
    us_states = us_states.data.copy()
    
    ##The map looks odd with these included - needs updates or scaling
    del us_states["HI"]
    del us_states["AK"]
    
    state_xs = [us_states[code]["lons"] for code in us_states]
    state_ys = [us_states[code]["lats"] for code in us_states]
    
    output_file("states.html")
    p = figure(title="State Crime Rates", toolbar_location="left",
    plot_width=800, plot_height=500)
    
    p.patches(state_xs, state_ys, fill_alpha=0.5,
    line_color="#884444", fill_color="#118833", line_width=2)
    
    show(p)

We are going to make plots for the difference crime rates.   First we need to make a scaler for the color scheme.  We will use matplotlib.  Then we need to convert the scale to hex colors.   After that, we will set the Bokeh fill colors for each state.


    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import us
    
    ##get larceny theft values for each state by name
    lt = [df3.loc[us.states.lookup(x).name,'Larceny_Theft'] for x in us_states.keys()]
    ##make color scalar using jet color scheme
    csm = ScalarMappable(norm = Normalize(0,max(lt)),cmap='YlOrRd')
    lt_fill_colors = [csm.to_rgba(x) for x in lt]
    lt_fill_colors = ['#%02x%02x%02x' % (255*r, 255*g, 255*b) for (r,g,b,a) in lt_fill_colors]


    state_xs = [us_states[code]["lons"] for code in us_states]
    state_ys = [us_states[code]["lats"] for code in us_states]
    
    output_file("larceny_theft_states.html")
    p = figure(title="State Larceny Rates", toolbar_location="left",
    plot_width=800, plot_height=500, tools='resize,hover,save,pan,box_zoom,wheel_zoom')
    
    source = ColumnDataSource(
        data = dict(
            x=state_xs,
            y=state_ys,
            color=lt_fill_colors,
            name=us_states.keys(),
            larceny=lt,
        )
    )
    
    p.patches('x','y',fill_color='color',fill_alpha=0.5,
    line_color="#884444",line_width=2,source=source)
    
    hover = p.select(dict(type=HoverTool))
    hover.point_policy = "follow_mouse"
    hover.tooltips = OrderedDict([
        ("State", "@name"),
        ("Larceny Rate", "@larceny"),
    ])
    
    show(p)


    def make_state_graph(crime_type):
        ##get larceny theft values for each state by name
        crime = [df3.loc[us.states.lookup(x).name,crime_type] for x in us_states.keys()]
        ##make color scalar using jet color scheme
        csm = ScalarMappable(norm = Normalize(0,max(crime)),cmap='YlOrRd')
        lt_fill_colors = [csm.to_rgba(x) for x in crime]
        lt_fill_colors = ['#%02x%02x%02x' % (255*r, 255*g, 255*b) for (r,g,b,a) in lt_fill_colors]
        
        state_xs = [us_states[code]["lons"] for code in us_states]
        state_ys = [us_states[code]["lats"] for code in us_states]
    
        output_file(crime_type.lower()+"_states.html")
        p = figure(title="State " + " ".join(crime_type.split("_")) + " Rates", toolbar_location="left",
        plot_width=800, plot_height=500, tools='resize,hover,save,pan,box_zoom,wheel_zoom')
    
        source = ColumnDataSource(
            data = dict(
                x=state_xs,
                y=state_ys,
                color=lt_fill_colors,
                name=us_states.keys(),
                amount=crime,
            )
        )
    
        p.patches('x','y',fill_color='color',fill_alpha=0.5,
        line_color="#884444",line_width=2,source=source)
    
        hover = p.select(dict(type=HoverTool))
        hover.point_policy = "follow_mouse"
        hover.tooltips = OrderedDict([
            ("State", "@name"),
            ("Rate", "@amount"),
        ])
    
        show(p)


    make_state_graph(df3.columns[0])


    make_state_graph(df3.columns[1])


    make_state_graph(df3.columns[2])


    make_state_graph(df3.columns[3])


    make_state_graph(df3.columns[4])


    make_state_graph(df3.columns[6])


    
