Title: Galvanize - Week 02 - Day 1
Date: 2015-06-08 10:20
Modified: 2015-06-08 10:30
Category: Galvanize
Tags: data-science, galvanize, pandas, money ball
Slug: galvanize-data-science-02-01
Authors: Bryan Smith
Summary: The sixth day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered probability.

#Galvanize Immersive Data Science
##Week 2 - Day 1

Since this is the first day of the week we started with an hour long assessment on what we did last week.  The assessment was straight forward, and very doable with a modest understanding of the previous material.   

After the test, we started a lecture on probability, which we finished in the afternoon.  There was the individual sprint after the morning lecture, and a paired spring as the afternoon lecture

##Conditional Probabilities

We started with some simple questions like the following:

* Suppose two cards are drawn from a standard 52 card deck.  What's the probability that the first is a queen and the second is a king?

    $$P\left(Q\right) = \frac{4}{52}$$

    $$P\left(K,Q\right) = P\left(K|Q\right)*P\left(Q\right) = \frac{4}{51} * \frac{4}{52} = \frac{16}{2652}$$



    print "Answer: ", 16./2652

    Answer:  0.00603318250377


* What's the probability that both cards are queens?

    $$P\left(Q\right) = \frac{4}{52}$$

    $$P\left(Q,Q\right) = P\left(Q|Q\right)*P\left(Q\right) = \frac{3}{51} * \frac{4}{52} = \frac{12}{2652}$$
  


    print "Answer: ", 12./2652

    Answer:  0.00452488687783


  * Suppose that before the second card was drawn, the first was inserted back into the deck and the deck reshuffled. What's the probability that both cards are queens?

$$P\left(Q\right) = \frac{4}{52}$$

$$P\left(Q,Q\right) = P\left(Q\right)*P\left(Q\right) = \frac{4}{52} * \frac{4}{52} = \frac{16}{2705}$$
 


    print "Answer: ", 16./2705

    Answer:  0.00591497227357


We had similar questions about tables of data:

A Store Manager wants to understand how his customers use different payment methods, and suspects that the size of the purchase is a major deciding factor. He organizes the table below.


   |           | Cash | Debit | Credit |
   |-----------|-----:|------:|-------:|
   | Under 20 |  400 |   150 | 150    |
   | 20 - 50 |  200 |  1200 | 800    |
   | Over 50  |  100 |   600 | 1400   |
   

* Given that a customer spent over $50, what's the probability that the customer used a credit card?
    
    $$P\left(C|S>$50\right) = \frac{1400}{100+600+1400}$$ 
   
   


    print "Answer: ", 1400./(100+600+1400)

    Answer:  0.666666666667


* Given that a customer paid in cash, what's the probability that the customer spent less than $20?

    $$P\left(S<20|Cash\right) = \frac{400}{400+200+100}$$
            


    print "Answer: ", 400./(400+200+100)

    Answer:  0.571428571429


* What's the probability that a customer spent under $20 using cash?

	$$P\left(S < $20,Cash\right) = \frac{400}{400+150+150+200+1200+800+100+600+1400}$$


    print "Answer: ", 400./(400+150+150+200+1200+800+100+600+1400)

    Answer:  0.08


We also had a question about job offers - something near and dear to our hearts:

* A gSchool grad is looking for her first job!  Given that she is freaked out, her chances of not getting an offer are 70%.  Given that she isn't freaked out, her chances of not getting an offer are 30%.  Suppose that the probability that she's freaked out is 80%. What's the probability that she gets an offer?

    $$P\left(Offer|Freak Out\right) = 0.7$$  
    
	$$P\left(Offer|No Freak Oout\right) = 0.3$$
	
    $$P\left(Freak Out\right) = 0.8$$

$$ P\left(A\right) = \Sigma_{B} P\left(A|B\right) $$
    
$$ P\left(Offer\right) = P\left(Offer|Freak Out\right) * P\left(Freak Out\right) + P\left(Offer|No Freak Out\right) * P\left(No Freak Out\right)$$

$$P\left(Offer\right) = 0.7 * 0.8 + 0.3 * 0.2$$ 


    print "Answer: ", 0.7 * 0.8 + 0.3 * 0.2

    Answer:  0.62


We also tackled the deep issue go heroin use at Google: 

*. Google decides to do random drug tests for heroin on their employees.
   They know that 3% of their population uses heroin. The drug test has the
   following accuracy: The test correctly identifies 95% of the
   heroin users (sensitivity) and 90% of the non-users (specificity).

   |     Test Results   | Uses heroin | Doesn't use heroin |
   | -------------- | ----------: | -----------------: |
   | Tests positive |        0.95 |               0.10 |
   | Tests negative |        0.05 |               0.90 |

   Alice gets tested and the test comes back positive. What is the probability
   that she uses heroin?
   
$$P(Heroin \ | \ Positive Test) = \frac{P(Positive Test|Heroin) P(Heroin)}{P(Positive Test)}$$

$$P(Heroin \ | \ Positive Test) = \frac{P(Positive Test|Heroin) P(Heroin)}{P(Positive Test|Heroin) P(Heroin) + P(Positive Test|No Heroin) P(No Heroin)}$$ 
        
    
$$ = \frac{0.95 \ 0.03}{0.95 \ 0.03 + 0.1 \ 0.97}$$


    print "Answer: ", 0.95*0.03/(0.95*0.03 + 0.1*0.97)

    Answer:  0.227091633466


Finally we had the manditory birthday problem:   

*The Birthday Problem.  Suppose there are 23 people in a data science class, lined up in a single file line.  
Let A_i be the probability that the i'th person doesn't have the same birthday as the j'th person for any j < i.  
Use the chain rule from probability to calculate the probability that at least 2 people share the same birthday. 

$$P(1,2,3,...,23) = \mbox{Probability that 23 people do not have the same birthday}$$
		
$$P( \ 1, \ 2, \ 3, \ ..., \ 23) = P(1) \ P(2|1) \ P(3 \ | \ 2 \ , \ 1) \ ... \ P(23|22 \ , \ ... \ , \ 2, \ 1)$$

		
Given that 2 people don't have the same birthday, there are 363 days that are not taken that a new person could have:
	
$$P(3|2,1) = \frac{363}{365}$$


Similarly, given that 3 people don't have the same birthday, then there are 362 days not occupied:
	
$$P(4|3,2,1) = \frac{362}{365}$$
	

Extending this to the problem we have the probability of no matching birthdays being 
    
$$P( \ 1, \ 2,\ 3, \ ..., \ 23) = \frac{1 \ 364 \ 363 \ ... \ (365-23)}{365^{23}}$$


    def bday(N=23):
        prob = 1
        for i in range(1,N+1):
            prob = prob*(365.0-i+1)/365.
        return prob
    
    print bday()

    0.492702765676


The probability of having 2 or more matches is us the inverse of this
	
$$P(Matches >= 2| \ 23) = 1 - P(No Maches|23)$$

$$P(Matches >= 2| \ 23) = 1-0.4927$$


    print "Answer: ", round(100-100*bday(),1)

    Answer:  50.7


##Distributions

The afternoon paired programming assignment involved developing an intuition for and using various distributions.

###Discrete:

- Bernoulli
    * Model one instance of a success or failure trial (p)

- Binomial
    * Number of successes out of a number of trials (n), each with probability of success (p)

- Poisson
    * Model the number of events occurring in a fixed interval
    * Events occur at an average rate (lambda) independently of the last event

- Geometric
    * Sequence of Bernoulli trials until first success (p)


###Continuous:

- Uniform
    * Any of the values in the interval of a to b are equally likely

- Gaussian
    * Commonly occurring distribution shaped like a bell curve
    * Often comes up because of the Central Limit Theorem (to be discussed later)

- Exponential
    * Model time between Poisson events
    * Events occur continuously and independently


Some of our questions involved being given examples and identify the distribution that describes it.

Often we have to identify what distribution we should use to model a real-life
situation. This exercise is designed to equip you with the ability to do so.


*. A typist makes on average 2 mistakes per page.  What is the probability of a particular page having no errors on it?

   $$X = Poisson(\lambda = 2 \frac{mistakes}{page})$$
   


    %matplotlib inline
    import matplotlib.pyplot as plt
    import scipy.stats as sc
    import numpy as np
    
    
    x = np.array(range(10))
    y = sc.poisson.pmf(x,2)
    plt.bar(x,y)
    plt.xlabel("Number of Mistakes on a Page")
    plt.ylabel("Probability")
    print "Prob of No Mistakes: ", sc.poisson.pmf(0,2)

    Prob of No Mistakes:  0.135335283237
    0.135335283237



![png](http://www.bryantravissmith.com/img/GW2D1/output_24_1.png)


*. Components are packed in boxes of 20. The probability of a component being
   defective is 0.1.  What is the probability of a box containing 2 defective components?

$$ X = Binomial(p=0.1,k=2,n=20) $$


    x = np.arange(20)
    y = sc.binom.pmf(x,20,.1)
    plt.bar(x,y,color='red',alpha=.2)
    plt.xlabel("Number of Defective Components")
    plt.ylabel("Probability")
    print "Prob of 2 Defects: ", sc.binom.pmf(2,20,.1)

    Prob of 2 Defects:  0.285179807064



![png](http://www.bryantravissmith.com/img/GW2D1/output_26_1.png)


*. Patrons arrive at a local bar at a mean rate of 30 per hour.  What is the probability that the bouncer has to wait more than 3 minutes to card the next patron?
   
   $$X = Exponential(\lambda = .5 \frac{cards}{minute})$$

   $$ P(t \ > \ 3min) = \int_{t=3}^{t=\infty} Exponential(\lambda = .5 \frac{cards}{minute}) $$
   


    print "Answer: ", sc.expon.cdf(1e10,scale=0.5)-sc.expon.cdf(3,scale=0.5)
                      

     Anser:  0.00247875217667



    x = np.linspace(0,5,1000)
    y = sc.expon.pdf(x,scale = 0.5)
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("Minutes")
    plt.ylabel("Probability Density")
    plt.ylim([0,0.1])
    d = np.zeros(len(y))
    plt.fill_between(x, y, where=x>=3, interpolate=True, color='blue',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW2D1/output_29_0.png)


*. A variable is normally distributed with a mean of 120 and a standard
   deviation of 5. One score is randomly sampled. What is the probability the score is above 127?
   
   $$Z = (127-120)/5 = 7/5 = 1.4$$
   
   $$P(Z>1.4) = 1 - .91924 \ \mbox{(area to left)}$$


    x = np.linspace(100,140,1000)
    y = sc.norm.pdf(x,loc=120,scale=5)
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("Variable Value")
    plt.ylabel("Probability Density")
    d = np.zeros(len(y))
    plt.fill_between(x, y, where=x>=127, interpolate=True, color='blue',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW2D1/output_31_0.png)


*. You need to find a tall person, at least 6 feet tall, to help you reach
   a cookie jar. 8% of the population is 6 feet or taller.  If you wait on the sidewalk, how many people would you expect to have passed you by before you'd have a candidate to reach the jar?
   
   $$X = Geometric(p=0.08)$$
   
   $$average = \frac{1}{p} = \frac{1}{0.08} = 12.5$$

We round up - 13th person is expected to be it - 12 people pass.


    x = np.arange(40)
    y = sc.geom.pmf(x,.08)
    plt.bar(x,y,color='red',alpha=.2)
    plt.xlabel("Number of People")
    plt.ylabel("Probability Person is Above 6ft")




    <matplotlib.text.Text at 0x106020790>




![png](http://www.bryantravissmith.com/img/GW2D1/output_33_1.png)



    x = np.arange(40)
    y = sc.geom.cdf(x,.08)
    plt.bar(x,y,color='red',alpha=.2)
    plt.xlabel("Number of People")
    plt.ylabel("Cumlative Probability Person is Above 6ft")




    <matplotlib.text.Text at 0x106451c50>




![png](http://www.bryantravissmith.com/img/GW2D1/output_34_1.png)


6. A harried passenger will be several minutes late for a scheduled 10 A.M.
   flight to NYC. Nevertheless, he might still make the flight, since boarding
   is always allowed until 10:10 A.M., and boarding is sometimes
   permitted up to 10:30 AM.

   Assuming the extended boarding time is **uniformly distributed** over the above
   limits, find the probability that the passenger will make his flight,
   assuming he arrives at the boarding gate at 10:25.
   
   $$X = Uniform(0,30)$$
   
   $$P( x > 25 ) = \int_{25}^{30} \frac{dx}{30}$$
   


    print "Answer: ", 5./30

    Answer:  0.166666666667



    x = np.linspace(0,30,1000)
    y = sc.uniform.pdf(x,loc=0,scale=30)
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("Minutes Late")
    plt.ylabel("Probability Density")
    d = np.zeros(len(y))
    plt.fill_between(x, y, where=x>=25, interpolate=True, color='blue',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW2D1/output_37_0.png)


##Covariance and Joint Distribution
Suppose a university wants to look for factors that are correlated with the GPA of the students that they
are going to admit. 


    import pandas as pd
    admissions = pd.read_csv('../probability/data/admissions.csv')
    addmissions.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family_income</th>
      <th>gpa</th>
      <th>parent_avg_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31402</td>
      <td>3.18</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32247</td>
      <td>2.98</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34732</td>
      <td>2.85</td>
      <td>61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53759</td>
      <td>3.39</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50952</td>
      <td>3.10</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



- Implement a `covariance` function and compute the covariance matrix of the dataframe. Check your results 
   with `df.cov()`. Make sure you understand what each of the numbers in the matrix represents


    def make_cov(df):
        N = len(df)
        cols = df.columns
        return [[(df[x]*df[y]).sum()/(N)-(df[x].sum()*df[y].sum())/(N**2) for y in cols] for x in cols]
    
    from pprint import pprint
    pprint (make_cov(addmissions))

    [[332910756.59847927, 4014.9337921708066, -1226.2147143883631],
     [4014.9337921708066, 0.087883196618951942, -0.028782641179958546],
     [-1226.2147143883631, -0.028782641179958546, 112]]



    addmissions.cov()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family_income</th>
      <th>gpa</th>
      <th>parent_avg_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>family_income</th>
      <td>3.329410e+08</td>
      <td>4015.299085</td>
      <td>-1226.326280</td>
    </tr>
    <tr>
      <th>gpa</th>
      <td>4.015299e+03</td>
      <td>0.087891</td>
      <td>-0.028785</td>
    </tr>
    <tr>
      <th>parent_avg_age</th>
      <td>-1.226326e+03</td>
      <td>-0.028785</td>
      <td>112.977442</td>
    </tr>
  </tbody>
</table>
</div>



- Implement a `normalize` function that would compute the correlation matrix from the covariance matrix.
   Check your results with `df.corr()`


    def make_corr(df):
        N = len(df)
        cols = df.columns
        return [[((df[x]*df[y]).sum()/(N)-(df[x].sum()*df[y].sum())/(N**2))/(df[x].std() * df[y].std()) for y in cols] for x in cols]
    
    pprint (make_corr(addmissions))

    [[0.99990902474526921, 0.74220186205662952, -0.0063224730309758576],
     [0.74220186205662952, 0.99990902474528243, -0.0091340229188836969],
     [-0.0063224730309758576, -0.0091340229188836969, 0.99134834685640671]]



    addmissions.corr()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family_income</th>
      <th>gpa</th>
      <th>parent_avg_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>family_income</th>
      <td>1.000000</td>
      <td>0.742269</td>
      <td>-0.006323</td>
    </tr>
    <tr>
      <th>gpa</th>
      <td>0.742269</td>
      <td>1.000000</td>
      <td>-0.009135</td>
    </tr>
    <tr>
      <th>parent_avg_age</th>
      <td>-0.006323</td>
      <td>-0.009135</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- You should identify `family_income` as being the most correlated with GPA. The university wants to make
   an effort to make sure people of all family income are being fairly represented in the admissions process.
   In order to achieve that, different GPA thresholds will be set according to family income. 

   The low, medium and high family income groups are `0 to 26832`, `26833 to 37510` and `37511 to 51112` respectively. 
   Implement a function that would plot the distribution of GPA scores for each family income category. These are the 
   conditional probability distributions of `gpa` given certain levels of `family_income`.


    
    def make_hist(df):
        low = df[df.family_income <= 26832]
        med = df[(df.family_income > 26832) & (df.family_income <= 37519)]
        high = df[(df.family_income > 37519) & (df.family_income <= 51112)]
        low.gpa.plot(kind="kde", color="blue",label='Low Income')
        med.gpa.plot(kind="kde", color="green",label='Medium Income')
        high.gpa.plot(kind="kde", color="red",label='High Income')
        plt.xlim([2.0, 4.0])
        plt.legend()
        plt.show()
        
    make_hist(addmissions)


![png](http://www.bryantravissmith.com/img/GW2D1/output_47_0.png)


- If the university decides to accept students with GPA above the 90th percentile within the respective family 
   income categories, what are the GPA thresholds for each of the categories?
   


    low = addmissions[addmissions.family_income <= 26832]
    med = addmissions[(addmissions.family_income > 26832) & (addmissions.family_income <= 37519)]
    high = addmissions[(addmissions.family_income > 37519) & (addmissions.family_income <= 51112)]
    print "Low 90th Percentile", low.gpa.quantile(.9)
    print "Medium 90th Percentile", med.gpa.quantile(.9)
    print "High 90th Percentile", high.gpa.quantile(.9)

    Low 90th Percentile 3.01
    Medium 90th Percentile 3.26
    High 90th Percentile 3.36


##Pearson Correlation vs Spearman Correlation

The Pearson correlation evaluates the linear relationship between two continuous 
variables. The Spearman correlation evaluates the monotonic relationship between two continuous or ordinal variables
without assuming linearity of the variables. Spearman correlation is often more robust in capturing non-linear relationship
between variables.

- In addition to the `family_income` and `parent_avg_age`, you are also given data about the number of hours the 
   students studied. Load the new data in from `data/admissions_with_study_hrs_and_sports.csv`.
   


    studydf = pd.read_csv('../probability/data/admissions_with_study_hrs_and_sports.csv')
    studydf.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family_income</th>
      <th>gpa</th>
      <th>family_income_cat</th>
      <th>parent_avg_age</th>
      <th>hrs_studied</th>
      <th>sport_performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31402</td>
      <td>3.18</td>
      <td>medium</td>
      <td>32</td>
      <td>49.463745</td>
      <td>0.033196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32247</td>
      <td>2.98</td>
      <td>medium</td>
      <td>48</td>
      <td>16.414467</td>
      <td>0.000317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34732</td>
      <td>2.85</td>
      <td>medium</td>
      <td>61</td>
      <td>4.937079</td>
      <td>0.021845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53759</td>
      <td>3.39</td>
      <td>high</td>
      <td>62</td>
      <td>160.210286</td>
      <td>0.153819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50952</td>
      <td>3.10</td>
      <td>medium</td>
      <td>45</td>
      <td>36.417860</td>
      <td>0.010444</td>
    </tr>
  </tbody>
</table>
</div>



- Make a scatter plot of the `gpa` against `hrs_studied`. Make the points more transperant so you can see the density
   of the points. Use the following command get the slope and intercept of a straight line to fit the data.


    slope, intercept, r_value, p_value, std_err = sc.linregress(studydf.gpa,studydf.hrs_studied)
    print slope, intercept, r_value, p_value
    x = np.linspace(studydf.gpa.min(),studydf.gpa.max(),100)
    y = slope*x+intercept
    
    studydf.plot(kind='scatter',x='gpa',y='hrs_studied',alpha=0.01)
    plt.plot(x,y,color='red')
    plt.xlabel("GPA")
    plt.ylabel("Hours Studied")
    plt.show()

    494.329335528 -1400.63719543 0.475940264662 0.0



![png](http://www.bryantravissmith.com/img/GW2D1/output_53_1.png)


- Use the functions `scipy.stats.pearsonr` and `scipy.stats.spearmanr` to compute the Pearson and Spearman correlation


    print sc.pearsonr(studydf.gpa,studydf.hrs_studied)
    print sc.spearmanr(studydf.gpa,studydf.hrs_studied)

    (0.47594026466220946, 0.0)
    (0.98495916559333341, 0.0)


Repeat step `2` and `3` for `gpa` and `sport_performance`. Is there a strong relationship between the two variables?


    slope, intercept, r_value, p_value, std_err = sc.linregress(studydf.gpa,studydf.sport_performance)
    print slope, intercept, r_value, p_value
    x = np.linspace(studydf.gpa.min(),studydf.gpa.max(),100)
    y = slope*x+intercept
    
    studydf.plot(kind='scatter',x='gpa',y='sport_performance',alpha=0.1)
    plt.plot(x,y,color='black')
    plt.show()

    0.00979813693421 0.0585103217504 0.0238485969548 0.0124044928587



![png](http://www.bryantravissmith.com/img/GW2D1/output_57_1.png)



    print sc.pearsonr(studydf.gpa,studydf.sport_performance)
    print sc.spearmanr(studydf.gpa,studydf.sport_performance)

    (0.023848596954761905, 0.012404492858691094)
    (0.0022881402736224248, 0.81043264616449484)



    temp = studydf[studydf.gpa > 3.0]
    slope, intercept, r_value, p_value, std_err = sc.linregress(temp.gpa,temp.sport_performance)
    print slope, intercept, r_value, p_value
    x = np.linspace(temp.gpa.min(),temp.gpa.max(),100)
    y = slope*x+intercept
    
    temp.plot(kind='scatter',x='gpa',y='sport_performance',alpha=0.1)
    plt.plot(x,y,color='black')
    plt.show()
    print sc.pearsonr(temp.gpa,temp.sport_performance)
    print sc.spearmanr(temp.gpa,temp.sport_performance)

    0.65608660543 -2.03523616554 0.945140987506 0.0



![png](http://www.bryantravissmith.com/img/GW2D1/output_59_1.png)


    (0.94514098750633013, 0.0)
    (1.0, 0.0)


##Part 4: Distribution Simulation

Often times in real life applications, we can specify the values of a variable to be of a particular distribution,
for example the number of sales made in the next month can be modeled as a uniform distribution over the range of
5000 and 6000.
 
In this scenario, we are modeling `profit` as a product of `number of views`, `conversion` and `profit per sale`,
where `number of views`, `conversion` and `profit per sale` can be modeled as probabilistic distributions.
By randomly drawing values from these distributions, we are able to get a distribution of the range of `profit` 
based on the uncertainties in the other variables.

`Profit = Number of views * Conversion * Profit per sale`

Assumptions:
- `Number of views` is a uniform distribution over the range of 5000 and 6000
- `Conversion is a binomial distribution where the probability of success is `0.12` for each sale among the `Number on views made 
- `Profit per sale` has `0.2` probability of taking the value `50` (for wholesale) and `0.8` of 
  taking the value `60` (non-wholesale)
  
- Given the distributions of each of variables, use scipy to write a function that would draw random values from each of the distributions to simulate a distribution for `profit`
  


    def get_profit():
        num_views = np.round(sc.uniform.rvs(loc=5000,scale=1000,size=1),0)
        conversions = sc.binom.rvs(num_views,0.12)
        wholesale = sc.binom.rvs(conversions,0.2)
        return wholesale*50+(conversions-wholesale)*60
    
    profits = np.array([get_profit() for i in xrange(100000)])
    plt.hist(profits)
    plt.show()
    print "Low: ",np.percentile(profits,2.5)
    print "High: ",np.percentile(profits,97.5)


![png](http://www.bryantravissmith.com/img/GW2D1/output_61_0.png)


    Low:  33800.0
    High:  42920.0



    
