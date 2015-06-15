Title: Galvanize - Week 02 - Day 4
Date: 2015-06-11 10:20
Modified: 2015-06-11 10:30
Category: Galvanize
Tags: data-science, galvanize, ab testing, statistics, hypothesis testing
Slug: galvanize-data-science-02-04
Authors: Bryan Smith
Summary: The ninth day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered power and bayesian analysis of tests.

#Galvanize Immersive Data Science

##Week 2 - Day 4

Our morning quiz was fun.  It was the first time that we built off a prevous quiz.  We were required to build a random variable class that used our probability mass function from the day before.   The end result was being able to simulate a random value from any distrubiton able to be defined by a PMF.  

Our morning lecture was about power and sample size, as was our individual sprint.  The afternoon lecture was on Bayesian Inference, and the afternoon paired sprint was investigating evolving likelihood functions as we gained more information/data.  



##Power

-Suppose you are interested in testing if on average a bottle of coke weighs 20.4 ounces. You have collected
simple random samples of 130 bottles of coke and weighed them.


    %matplotlib inline
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats as sc
    from __future__ import division
    
    data = np.loadtxt('data/coke_weights.txt')
    len(data)




    130



2. State your null and alternative hypothesis.

    1.**H0: The mean weight of coke bottles is 20.4 oz**  
    2.**HA: The mean weight of coke bottles is different form 20.4 oz**     
    
      
      

3. Compute the mean and standard error of the sample. State why you are able to apply the Central
   Limit Theorem here to approximate the sample distribution to be normal.


    mean = data.mean()
    std = data.std()
    se = sc.sem(data)
    print "Sample Mean: ", mean
    print "Sample STD", std
    print "Sample Standard Error",se
    print "Sample Size: ", len(data)

    Sample Mean:  20.519861441
    Sample STD 0.957682215104
    Sample Standard Error 0.084319217426
    Sample Size:  130


**We can use the CLT on this problem because the sample size is 130, much greater than 30.**

We can make a simulation of the sampling distribution of the null hypthosis, and we can make a sampling distirubiton for another believe, say the true value is the sample man.   If this is true, we can ask questions about how powerful our test is at discovering the mean coke bottle weight is not 20.4, but 20.52.
   


    def power_graph(mu1,std1,n1,mu2,std2,n2,alpha=0.05,two_sided=True):
        x = np.array([sc.norm.rvs(loc=mu1,scale=std1,size=n1).mean() for i in range(10000)])
        y = np.array([sc.norm.rvs(loc=mu2,scale=std2,size=n2).mean() for i in range(10000)])
        plt.figure()
        plt.hist(x,normed=True,color='lightseagreen',edgecolor='lightseagreen',alpha=0.4,bins=30,label="Null")
        plt.hist(y,normed=True,color='lightsalmon',edgecolor='lightsalmon',alpha=0.4,bins=30,label="W=20.52")
        if two_sided:
            x95 = np.percentile(x,100-100*alpha/2.)
        else:
            x95 = np.percentile(x,100-100*alpha)
            
        plt.axvline(x95,0,10,color='gray',lw=2,linestyle='--',alpha=0.8)
        plt.legend()
        plt.show()
        print "Value Threshold For Significance: ", x95
        power = len(y[y>x95])/len(y)
        print "Power of Finding True Positive: ",power
        return power
    
    power_graph(20.4,data.std(),130,data.mean(),data.std(),130)
    power_graph(20.4,data.std(),130,data.mean(),data.std(),130,two_sided=False)


![png](http://www.bryantravissmith.com/img/GW02D4/output_6_0.png)


    Value Threshold For Significance:  20.5657024198
    Power of Finding True Positive:  0.2899



![png](http://www.bryantravissmith.com/img/GW02D4/output_6_2.png)


    Value Threshold For Significance:  20.5387014434
    Power of Finding True Positive:  0.4037





    0.4037



These powers are 30% for the two sided, and 42% for the onsided.   It deends if our originaly hypothesis test is a not equal or greater than.   

We can also do this analytically since we are using the centeral limit.  For a two sided test, we find the following:


    x = np.linspace(20.4-4*se,20.4+4*se,100)
    
    y1 = sc.norm.pdf(x,loc=20.4,scale=se)
    cumy1 = sc.norm.cdf(x,loc=20.4,scale=se)
    y2 = sc.norm.pdf(x,loc=data.mean(),scale=se)
    x975 = 20.4+1.96*se
    x025 = 20.4-1.95*se
    print x025,x975
    plt.plot(x,y1,color='indianred',label='Null Hypthesis')
    plt.plot(x,y2,color='steelblue',label='Data')
    plt.fill_between(x[x>=x975],y2[x>=x975],color='steelblue',alpha=0.4)
    plt.fill_between(x[x>=x975],y1[x>=x975],color='indianred',alpha=0.4)
    plt.fill_between(x[x<=x025],y1[x<=x025],color='indianred',alpha=0.4)
    plt.axvline(x975,0,10,color='black',lw=3,linestyle='--')
    plt.axvline(x025,0,10,color='black',lw=3,linestyle='--')
    plt.show()

    20.235577526 20.5652656662



![png](http://www.bryantravissmith.com/img/GW02D4/output_8_1.png)


**We can see that we would not reject the null hypothesis in favor of the alternative because the peak of the blue curve is to the left of the bounding of significance.  The area under the red cuver outside of the black boundaries is 0.05, the signifcance level.   The power of detecing a signficant difference assuming the data's mean is the true value is the blue area.   In this case it is ~ 30%**
   

The probability of making a type II error (false negative) is called beta.
   


    beta = sc.norm.cdf(x975,loc=data.mean(),scale=se)-sc.norm.cdf(x025,loc=data.mean(),scale=se)
    print "Beta (Prob of Type II Error) Assuming data value is the true value", beta

    Beta (Prob of Type II Error) Assuming data value is the true value 0.704503425135


The power is always 1 minus the false negative rate:

$$\mbox{Power} = 1 - \beta$$


    print "Power: ",1-beta

    Power:  0.295496574865


Statistical power is affected by a number of factors, including the **sample size**, the **effect size (difference
between the sample statistic and the statistic formulated under the null)**, and the **significance level**. Here
we are going to explore the effect of these factors on power.

If we assuming that we have a different null hypothesis, we can find the power of detecting anther effect size.  Lets stick with the sample data


    def explore_power(null_mu,sample_size,effect_size,null_standard_dev,significance_level=0.95,two_sided=True):
        if two_sided:
            critical_z = sc.norm.isf((1-significance_level)/2.)
            se = np.sqrt(null_standard_dev**2/(sample_size-1))
            x025 = null_mu-critical_z*se
            x975 = null_mu+critical_z*se
            alt_mean = null_mu+effect_size
            return (1-np.abs(sc.norm.cdf(x975,loc=alt_mean,scale=se)-sc.norm.cdf(x025,loc=alt_mean,scale=se)))*100
        else:
            critical_z = sc.norm.isf(significance_level)
            se = np.sqrt(null_standard_dev**2/(sample_size-1))
            x95 = null_mu-critical_z*se
            alt_mean = null_mu+effect_size
            return 100*(np.abs(sc.norm.cdf(x95,loc=alt_mean,scale=se)))
    
    print explore_power(20.4,130,data.mean()-20.4,data.std())
    explore_power(20.2,130,data.mean()-20.2,data.std())

    29.5495708063





    96.663546292685481



The power increased when we assumed the distributions were more different.  This makes sense because the two values are father appart, and the sampling distributions have less overlap.  We can see that by calling power_graph, or calling the analyical functions because we have access to the central limit theorem.


    power_graph(20.2,data.std(),130,data.mean(),data.std(),130)


![png](http://www.bryantravissmith.com/img/GW02D4/output_17_0.png)


    Value Threshold For Significance:  20.364830609
    Power of Finding True Positive:  0.9698





    0.9698




    x = np.linspace(20.2-5*se,20.2+6*se,100)
    
    y1 = sc.norm.pdf(x,loc=20.2,scale=se)
    cumy1 = sc.norm.cdf(x,loc=20.2,scale=se)
    y2 = sc.norm.pdf(x,loc=data.mean(),scale=se)
    x975 = 20.2+1.96*se
    x025 = 20.2-1.95*se
    plt.plot(x,y1,color='indianred',label='Null Hypthesis')
    plt.plot(x,y2,color='steelblue',label='Data')
    plt.axvline(x975,0,10,color='black',lw=3,linestyle='--')
    plt.axvline(x025,0,10,color='black',lw=3,linestyle='--')
    plt.fill_between(x[x>=x975],y2[x>=x975],color='steelblue',alpha=0.4)
    plt.fill_between(x[x>=x975],y1[x>=x975],color='indianred',alpha=0.4)
    plt.fill_between(x[x<=x025],y1[x<=x025],color='indianred',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_18_0.png)


This makes me wonder how the power changes with the effect size.  I image that bigger effects are easier to detect.  It makes good sense


    plt.plot(np.linspace(0.1,1.2,20),[explore_power(20.2,130,x,data.std()) for x in np.linspace(0.0,1.2,20)],'bo--',
            color='lightseagreen',alpha=0.8,label="Power")
    plt.ylim([0,110])
    plt.xlabel("Effect Size")
    plt.ylabel("Power")
    plt.legend(loc=4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_20_0.png)


The power of a study also depends on the sample size.  As the sample size increases, the standard error decreases by the central limit theorem.   This will make two different distributions overlap less. 



    power_graph(20.4,data.std(),130,data.mean(),data.std(),130)
    power_graph(20.4,data.std(),1000,data.mean(),data.std(),1000)


![png](http://www.bryantravissmith.com/img/GW02D4/output_22_0.png)


    Value Threshold For Significance:  20.5657528314
    Power of Finding True Positive:  0.2857



![png](http://www.bryantravissmith.com/img/GW02D4/output_22_2.png)


    Value Threshold For Significance:  20.4590858498
    Power of Finding True Positive:  0.9802





    0.9802




    data2 = np.loadtxt('data/coke_weights_1000.txt')
    mean = data2.mean()
    std = data2.std()
    se = sc.sem(data2)
    x = np.linspace(20.4-5*se,20.4+6*se,100)
    
    y1 = sc.norm.pdf(x,loc=20.4,scale=se)
    cumy1 = sc.norm.cdf(x,loc=20.4,scale=se)
    y2 = sc.norm.pdf(x,loc=data2.mean(),scale=se)
    x975 = 20.4+1.96*se
    x025 = 20.4-1.95*se
    
    plt.plot(x,y1,color='indianred',label='Null Hypthesis')
    plt.plot(x,y2,color='steelblue',label='Data')
    plt.axvline(x975,0,10,color='black',lw=3,linestyle='--')
    plt.axvline(x025,0,10,color='black',lw=3,linestyle='--')
    plt.fill_between(x[x>=x975],y2[x>=x975],color='steelblue',alpha=0.4)
    plt.fill_between(x[x>=x975],y1[x>=x975],color='indianred',alpha=0.4)
    plt.fill_between(x[x<=x025],y1[x<=x025],color='indianred',alpha=0.4)
    plt.show()
    beta = sc.norm.cdf(x975,loc=data2.mean(),scale=se)-sc.norm.cdf(x025,loc=data2.mean(),scale=se)
    print "Beta (Prob of Type II Error) Assuming data value is the true value", beta
    print "Power:",1-beta


![png](http://www.bryantravissmith.com/img/GW02D4/output_23_0.png)


    Beta (Prob of Type II Error) Assuming data value is the true value 0.271504963539
    Power: 0.728495036461


We can also see how chaning the significance level affects the power of a study.  
 


    plt.plot(np.linspace(0.01,0.3,40),[explore_power(20.4,130,0.01,data.std(),significance_level=x) for x in np.linspace(0.01,0.3,40)],'bo--',
            color='lightseagreen',alpha=0.6,label="Power")
    plt.ylim([0,110])
    plt.xlabel("Significance Level")
    plt.ylabel("Power")
    plt.legend(loc=4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_25_0.png)


##Power Calculations for A/B testing  

We continued yesterday's case study with Esty to find the power needed.  It looked like our Etsy Tuesday Landing Page experiment was under-powered.


    etsy = pd.read_csv('data/experiment.csv')
    old_data = etsy[etsy['landing_page'] == 'old_page']['converted']
    new_data = etsy[etsy['landing_page'] == 'new_page']['converted']

We will set up the following hypthesis test.

X ~ p_new - p_old

H0: X = 0.001
H1: X > 0.001

We need to set up the proportions of conversions and find the standard error for this experiment.


    po = old_data.mean()
    pn = new_data.mean()
    p = (old_data.mean()*len(old_data)+new_data.mean()*len(new_data))/(len(old_data)+len(new_data))
    se = np.sqrt(p*(1-p)/len(old_data)+p*(1-p)/len(new_data))


We can make the same plots as before and compared null to our data.
  


    x = np.linspace(-5*se,5*se,100)
    
    y1 = sc.norm.pdf(x,loc=0.001,scale=se)
    cumy1 = sc.norm.cdf(x,loc=0.001,scale=se)
    y2 = sc.norm.pdf(x,loc=(pn-po),scale=se)
    x95 = 0.001+1.68*se
    plt.plot(x,y1,color='indianred',label='Null Hypthesis')
    plt.plot(x,y2,color='steelblue',label='New Data')
    plt.axvline(x95,0,10,color='black',lw=3,linestyle='--')
    plt.fill_between(x[x>=x95],y2[x>=x95],color='steelblue',alpha=0.4)
    plt.fill_between(x[x>=x95],y1[x>=x95],color='indianred',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_31_0.png)


We can see our data is not statistically different of 0.001, but if we assume the data represents the true mean, we have a very weak test.


    critical_z = sc.norm.ppf(0.95)
    x95 = 0.001+critical_z*se
    1-sc.norm.cdf(x95,loc=(pn-po),scale=se)




    0.005391084734824525



We have a power of less than 1/2% from our data.   

Increasing the sample size will weaken the power in this case, because our data is less than the null hypthesis, and we are constructing a one sided t-test.  In this case we accept the results and fail to reject the null hypthesis. 
   

We were told that Etsy decided the pilot is a plausible enough representation of the company's daily  traffic. As a result, Esty decided on a two-tailed test instead, which is as follows:

   ```
   X ~ p_new - p_old

   H0: X = 0.001
   H1: X != 0.001
   ```




    x = np.linspace(-5*se,5*se,100)
    
    y1 = sc.norm.pdf(x,loc=0.001,scale=se)
    cumy1 = sc.norm.cdf(x,loc=0.001,scale=se)
    y2 = sc.norm.pdf(x,loc=(pn-po),scale=se)
    x025 = 0.001-1.96*se
    x975 = 0.001+1.96*se
    plt.plot(x,y1,color='indianred',label='Null Hypthesis')
    plt.plot(x,y2,color='steelblue',label='New Data')
    plt.axvline(x025,0,10,color='black',lw=3,linestyle='--')
    plt.axvline(x975,0,10,color='black',lw=3,linestyle='--')
    plt.fill_between(x[x>=x975],y2[x>=x975],color='steelblue',alpha=0.4)
    plt.fill_between(x[x<=x025],y2[x<=x025],color='steelblue',alpha=0.4)
    plt.fill_between(x[x>=x975],y1[x>=x975],color='indianred',alpha=0.4)
    plt.fill_between(x[x<=x025],y1[x<=x025],color='indianred',alpha=0.4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_36_0.png)



    print "2-Sided Power: ", sc.norm.cdf(x025,loc=(pn-po),scale=se)+(1-sc.norm.cdf(x975,loc=(pn-po),scale=se))

    2-Sided Power:  0.14775925242


### Minimum Sample Size

When desigining an experiment, or conducting a test, it is important to get enough data to measure what we are looking for.  In the case of Etsy, its a 1% lift in conversions.  To try to figure out the sample size we might considered our desired false postive and false negative rates, and the Z values associated with them.

$$\alpha = \mbox{False Positive Rate}, \ Z_{\alpha}$$

$$\beta = \mbox{False Negative Rate}, \ Z_{\beta}$$

In an experiment we will have one values, and the Z's will be related to it by the following:

$$\mu_{exp} = \mbox{Experimental Result}$$

$$s_{sample} = \mbox{Sampling Erroring}$$

$$Z_{\alpha} = \frac{\mu_{exp} \ - \ \mu_{Null}}{s_{sample}}$$

$$Z_{\beta} = \frac{\mu_{exp} \ - \ \mu_{Alternative}}{s_{sample}}$$

It is worth noting that one of these Z's can be positive, while the other can be negative.  You will find equations that differ from what is done here because of that.  Taking the difference between these two equations give:

$$\mbox{Effect Size} = \mu_{Alternative} - \mu_{Null}$$

$$Z_{\alpha} \ - \ Z_{\beta} = \frac{\mbox{Effect Size}}{s_{sample}}$$

We can use the eqauation for the sampling distribution for two sample proproption test of equal size for the sampling error.

$$s_{sample} = \sqrt{\frac{ p \ (1 - p) \ 2}{n}}$$

Where p is the weighted proportion of the two groups.  This gives the previous equation as.

$$Z_{\alpha} \ - \ Z_{\beta} = \frac{\mbox{Effect Size}}{\sqrt{\frac{ p \ (1 - p) \ 2}{n}}}$$

Squaring both sides we get:

$$(Z_{\alpha} \ - \ Z_{\beta})^2 = \frac{\mbox{Effect Size}^2 \ n}{ p \ (1 - p) \ 2}$$

So the needed sample size should be

$$n = \frac{2 \ (Z_{\alpha} \ - \ Z_{\beta})^2 \ p \ (1-p)}{\mbox{Effect Size}^2}$$


    def calc_min_sample_size(old,new,effect_size,sig=0.05,pow=0.8,one_tail=False):
        pn = new.mean()
        po = old.mean()
        no = len(old)
        nn = len(new)
        dp = pn-po
        p = ( po*no+pn*nn ) / (nn+no)
        se = np.sqrt(p*(1-p)*(1/nn+1/no))
        if one_tail:
            z_null = sc.norm.ppf((1-sig))
        else:
            z_null = sc.norm.ppf((1-sig/2))
        z_pow = sc.norm.ppf(1-pow)
        
            
        return (z_null-z_pow)**2*2*p*(1-p)/effect_size**2
        
        
    calc_min_sample_size(old_data,new_data,0.001,one_tail=False)




    1410314.3210247809



So if we want a power of 80% for our Etsy experiment, meaning its likely for us to detect results that are different from a lift of 1%, we need to have 1.4 million users in each sample.  Our results are definately underpowered.

##Afternoon - Bayes

We covered Bayes' Theorem and updating priors with likelihoods to produce a posterior distribution of sample paramters given data.   We started with a created class, and explore the results of these distributions given data.

$$ P(Parameters \ | \ Data) = \frac{P(Data \ | \ Parameters) * P(Paramters)}{P(Data)} $$

$$ \mbox{Posterior} = \frac{\mbox{Likelihood} \ \times \ \mbox{Prior}}{\mbox{Normalization}} $$


    class Bayes(object):
        '''
        INPUT:
            prior (dict): key is the value (e.g. 4-sided die),
                          value is the probability
    
            likelihood_func (function): takes a new piece of data and the value and
                                        outputs the likelihood of getting that data
        '''
        def __init__(self, prior, likelihood_func):
            self.prior = prior
            self.likelihood_func = likelihood_func
    
    
        def normalize(self):
            '''
            INPUT: None
            OUTPUT: None
    
            Makes the sum of the probabilities equal 1.
            '''
            total = sum(self.prior.values())
            for key in self.prior:
                self.prior[key] = self.prior[key]/total
        
        def update(self, data):
            '''
            INPUT:
                data (int or str): A single observation (data point)
    
            OUTPUT: None
            
            Conduct a bayesian update. Multiply the prior by the likelihood and
            make this the new prior.
            '''
            for k in self.prior:
                self.prior[k] = self.prior[k]*self.likelihood_func(data,k)
            self.normalize()
    
        def print_distribution(self):
            '''
            Print the current posterior probability.
            '''
            for k in sorted(self.prior.keys()):
                print k, self.prior[k]
        
        def plot(self, color=None, title=None, label=None):
            '''
            Plot the current prior.
            '''
            if color == None:
                c = 'blue'
            else:
                c=color
            k = sorted(self.prior.keys())
            v = [self.prior[key] for key in k]
            plt.bar(np.arange(len(self.prior)),v,color=c,alpha=0.2,label=label)
            plt.xticks(np.arange(len(self.prior))+0.1,k)
    
    def likelihood_dice(data,value):
        if data > int(value):
            return 0
        else:
            return 1./int(value)
            
    def likelihood_coin(data,value):
        if data=='H':
            return float(value)
        else:
            return 1-float(value)

### The problem
A box contains a 4-sided die, a 6-sided die, an 8-sided die,a 12-sided die, and a 20-sided die. A die is selected at random, and the rest are destroyed.  

What is the prior?  
**All Dice Equally Likely**


    b = Bayes({'04':0.2,'06':0.2,'08':0.2,'12':0.2,'20':0.2},likelihood_dice)
    b.plot()


![png](http://www.bryantravissmith.com/img/GW02D4/output_44_0.png)


Say I roll an 8. After one bayesian update, what is the probability that I chose each of the dice?


    b.update(8)
    b.plot()


![png](http://www.bryantravissmith.com/img/GW02D4/output_46_0.png)


We know that we do not have a 4 or 6 sided dice, and the 8 is most likly because the 8 has a 1 in 8 chance of getting an 8, the 12 has a 1 in 12 chance of rolling and 8, and the 20 sided dice has a 1 in 20 chance of rolling an 8.

Comment on the difference in the posteriors if I had rolled the die 50 times instead of 1.


    [b.update(8) for i in range(49)]
    b.plot()


![png](http://www.bryantravissmith.com/img/GW02D4/output_49_0.png)


However unlikely it is, the posterier suggest that the post likely culperite of 50 roles all coming to 8 is an 8 sided dice.

Which one of these two sets of data gives you a more certain posterior and why?
`[1, 1, 1, 3, 1, 2]` or `[10, 10, 10, 10, 8, 8]`


    b = Bayes({'04':0.2,'06':0.2,'08':0.2,'12':0.2,'20':0.2},likelihood_dice)
    [b.update(x) for x in [1,1,1,3,1,2]]
    b.plot()
    plt.show()
    b = Bayes({'04':0.2,'06':0.2,'08':0.2,'12':0.2,'20':0.2},likelihood_dice)
    [b.update(x) for x in [10,10,10,10,8,8]]
    b.plot()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_52_0.png)



![png](http://www.bryantravissmith.com/img/GW02D4/output_52_1.png)


We are most certain in the second case becasue 3 of the dice have been ruled out, and a 1/12 per roll is much bigger than a 1/20.

Say the prior of the dice is:

    ```
    4-sided die: 8%
    6-sided die: 12%
    8-sided die: 16%
    12-sided die: 24%
    20-sided die: 40%
    ```

What are posteriors for each die after rolling the 8?


    b = Bayes({'04':0.08,'06':0.12,'08':0.16,'12':0.24,'20':0.40},likelihood_dice)
    b.update(8)
    b.plot()


![png](http://www.bryantravissmith.com/img/GW02D4/output_55_0.png)


The postior makes 8 sided, 12 sided, and 20 sided dice equally likely after 1 roll of an 8.  

Say you keep the same prior and you roll the die 50 times and get values 1-8 every time. What would you expect of the posterior? How different do you think it would be if you'd used the uniform prior?


    b = Bayes({'04':0.2,'06':0.2,'08':0.2,'12':0.2,'20':0.2},likelihood_dice)
    [b.update(x) for x in np.random.randint(1,8,size=50)]
    b.plot()
    plt.show()
    b = Bayes({'04':0.08,'06':0.12,'08':0.16,'12':0.24,'20':0.40},likelihood_dice)
    [b.update(x) for x in np.random.randint(1,8,size=50)]
    b.plot()
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_58_0.png)



![png](http://www.bryantravissmith.com/img/GW02D4/output_58_1.png)


With enough data, the difference in priors do not matter.   They converge to the same value.  

##Bayes and Coin Flips

We can consider a random flip of a coin and ask what is the believe about our belief of the probability the coin resulting in heads.   Before any flip we would assume a unifor distrubiton.   


    p = np.linspace(0,0.99,100)
    prior = dict()
    for v in p:
        prior[str(v)] = 0.01
    flips = [['H'],['T'],['H','H'],['H','T'],['H','H','H'],['T','H','T'],['H','H','H','H'],['T','H','T','H']]
      
    plt.figure(figsize=(15,10))
    for i,f in enumerate(flips):
        plt.subplot(4,2,i+1)
        b = Bayes(prior.copy(),likelihood_coin)
        for x in f:
             b.update(x)
        b.plot(label=str(f))
        plt.legend(loc='best')
        plt.xticks(rotation=70,fontsize=4)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_61_0.png)


We were given a cone class, with a unknown probability.  I am going to plot the update in my posteriers after fixed number of flips to see if a Bayesian would believe this is a biased coin


    from coin import Coin
    c = Coin()
    plt.figure(figsize=(18,10))
    col = {'1':'indianred','2':'steelblue','10':'coral','50':'lightseagreen','250':'skyblue','500':'purple','1000':'indianred'}
    b = Bayes(prior.copy(),likelihood_coin)
    for i in range(1000):
        b.update(c.flip())
        if i in [0,1,9,49,249,499,999]:
            b.plot(color=col[str(i+1)],label=(str(i+1)+' Flips'))
    plt.legend()
    plt.xticks(rotation=70)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D4/output_63_0.png)


As we update our believes, upto 1000 flips, we see that 50 looks to be in the edge of our posterior space, but the distribution is centered around 0.53, making this the maximum likelihood value after 1000 flips.


    
