Title: Galvanize - Week 02 - Day 5
Date: 2015-06-12 10:20
Modified: 2015-06-12 10:30
Category: Galvanize
Tags: data-science, galvanize, ab testing, multi-arm bandit
Slug: galvanize-data-science-02-05
Authors: Bryan Smith
Summary: The tenth day of Galvanize's Immersive Data Science program in San Francisco, CA where we covered bayesian analysis and multi-arm bandits.

#Galvanize Immersive Data Science

##Week 2 - Day 5

Our more quiz was a survey about our progress in the program.  The morning lecture was about the beta distribution, and its relation to A/B testing from a Bayesian perspective, and a brief introduction to the multi-arm bandit method.   

##Individual Morning Sprint

This morning we had some simulated click through data for two different sites and attempted to answer question about the outcome of the two pages.  We want to find the probability of a click through on the two pages and be able to quantify how much better one site is over the other. }


    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as sc
    A = np.loadtxt('data/siteA.txt')
    B = np.loadtxt('data/siteB.txt')


##Bayesian Analysis

We are going to start with a prior that the click through rate of both sites can be anything between 0 and 1.  This will look like the following:


    x=np.arange(0,1.01,0.01)
    y = sc.beta(a=1.,b=1.).pdf(x)
    
    def plot_with_fill(x,y,label,color):
        lines = plt.plot(x,y,label=label,lw=2,color=color)
        plt.fill_between(x,0,y,alpha=0.2,color=color)
        plt.ylim([0,1.2*y.max()])
        plt.legend()
        
    
    plot_with_fill(x,y,'Beta a=1,b=1','seagreen')
    plt.xlabel("Probability of Click Through")
    plt.ylabel("Probability Density")
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_3_0.png)


If we look at 1 value from the click through where the user does not click through, we update our believe:

$$\mbox{Posterior} = \mbox{Prior} * \mbox{Likelihood}$$

$$\mbox{Beta}(\alpha=1, \ \beta=2) = \mbox{Beta}(\alpha=1, \ \beta=1) \times (1 - p)$$


    b1,a1 = 1,0
    y1 = sc.beta(a=1,b=1).pdf(x)
    y2 = sc.beta(a=a1+1,b=b1+1).pdf(x)
    plot_with_fill(x,y1,'Prior','seagreen')
    plot_with_fill(x,y2,'1 Views','lightsalmon')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_5_0.png)


If we have 50 views, the equation would be:


$$\mbox{Beta}(\alpha=1+n_{clicks}, \ \beta=1+n_{no clicks}) = \mbox{Beta}(\alpha=1, \ \beta=1) \times (1 - p)^{n_{no clicks}} \times p^{n_{clicks}}$$


    b50,a50 = np.bincount(A[:50].astype(int))
    y1 = sc.beta(a=1,b=1).pdf(x)
    y2 = sc.beta(a=a50+1,b=b50+1).pdf(x)
    plot_with_fill(x,y1,'Prior','seagreen')
    plot_with_fill(x,y2,'50 Views','lightsalmon')


![png](http://www.bryantravissmith.com/img/GW02D5/output_7_0.png)


We can imagine that we continue getting data, and our believe about the click through rate would evolve:


    x=np.arange(0,1.001,0.001)
    plt.figure(figsize=(15,10))
    y1 = sc.beta(a=1.,b=1.).pdf(x)
    plot_with_fill(x,y1,'Prior','seagreen')
    color_s = {'50':'lightsalmon','100':'aquamarine','200':'turquoise','400':'NavajoWhite','800':'forestgreen'}
    for count in [50,100,200,400,800]:
        bc,ac = np.bincount(A[:count].astype(int))
        y = sc.beta(a=ac,b=bc).pdf(x)
        plot_with_fill(x,y,str(count)+' Views',color_s[str(count)])
    plt.xlim([0,.2])




    (0, 0.2)




![png](http://www.bryantravissmith.com/img/GW02D5/output_9_1.png)


We can see that as we increase the amount of data we have, we have a more specific believe about the click through rate of site A.  **This is different from a hypthesis test.  It does not required a fix sample size or fixed amount of time**  

We can look at the two sites for all the data.  


    x=np.arange(0,1.001,0.001)
    plt.figure(figsize=(15,10))
    bA,aA = np.bincount(A.astype(int))
    bB,aB = np.bincount(B.astype(int))
    y1 = sc.beta(a=float(aA),b=float(bA)).pdf(x)
    y2 = sc.beta(a=float(aB),b=float(bB)).pdf(x)
    plot_with_fill(x,y1,'A Site','seagreen')
    plot_with_fill(x,y2,'B Site','aquamarine')
    plt.xlim([0,.2])
    plt.ylim([0,50])
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_11_0.png)


We now want to determine, given these distributions, what is the probability that site B is better than site A.  We can take random variables from both distributions and count the number of times that the random value from B is greater than the random variable from A.   We can do this 10,000 times.


    rA = sc.beta(a=aA,b=bA).rvs(size=10000)
    rB = sc.beta(a=aB,b=bB).rvs(size=10000)
    nLess,nMore = np.bincount((rA<rB).astype(int))
    
    plt.figure()
    ps = []
    for i in range(1000):
        rA = sc.beta(a=aA,b=bA).rvs(size=10000)
        rB = sc.beta(a=aB,b=bB).rvs(size=10000)
        nLess,nMore = np.bincount((rA<rB).astype(int))
        ps.append(nMore/10000.)
    
    plt.hist(ps,bins=15,color='aquamarine',alpha=0.5)
    plt.show()
    print "Mean Prob: ", nMore/10000.


![png](http://www.bryantravissmith.com/img/GW02D5/output_13_0.png)


    Mean Prob:  0.9971


Based on our current belief based on the data, the chance that sight B is better than site A is 99.7%.  

We can also estimate the Bayesian equivalant of a confidence interval.

This is the centeral credible region - range of the 2.5 precentile and the 97.5 percentile:


    rA = sc.beta(a=aA,b=bA).rvs(size=10000)
    HDI_A = (np.percentile(rA,2.5),np.percentile(rA,97.5))
    rB = sc.beta(a=aB,b=bB).rvs(size=10000)
    HDI_B = (np.percentile(rB,2.5),np.percentile(rB,97.5))
    print "Site A: ", HDI_A
    print "Site B: ", HDI_B

    Site A:  (0.05002097381711422, 0.084804921163330979)
    Site B:  (0.082330727908278764, 0.12425696702783418)


An 95% highest density interval (HDI) is the most dense interval of a posterior distribution containing X% of its mass. It is analagous to frequentist analysis's confidence intervals.


    def hdi_beta(data,percent=0.95):
        bD,aD = np.bincount(data.astype(int))
        x_max = sc.beta(a=aD,b=bD).ppf(1 - percent - 1e-6)
        x_high_max = sc.beta(a=aD,b=bD).ppf(percent - 1e-6)
        vals = np.linspace(0,x_max,1000)
        p_vals = 0.95+sc.beta(a=aD,b=bD).cdf(vals)
        x_high = sc.beta(a=aD,b=bD).ppf(p_vals)
        width = x_high-vals
        low = vals[np.argmin(width)]
        high = low+width.min()
        return (low,high)
    
    print "HDI A:", hdi_beta(A)
    print "HDI B:", hdi_beta(B)


    HDI A: (0.049391799061910255, 0.083668677250890555)
    HDI B: (0.081861282294831139, 0.12374716027178045)


These HDI are close to the central credibility region, but they are systematically lower.

What is nicse about Baysina inference is we can ask what is the probability that site B is 2 percentage points better than A.
    


    nLess2,nMore2 = np.bincount((rB>(rA+0.02)).astype(int))
    print "Probabilty B > A + 0.02:  ", nMore2/10000.
    
    plt.figure()
    plt.hist((rB-rA),bins=30,color='steelblue',alpha=0.4)
    plt.show()

    Probabilty B > A + 0.02:   0.881



![png](http://www.bryantravissmith.com/img/GW02D5/output_19_1.png)


In this case we can see that the difference is near zero (really 0.02), but is likely to be higher.   The probabiliy given by the simulation is 88.1%. 

For a sanity check we can do a hypothesis test.  We would perform a 1 sided Hypthesiss test:

H0: The mean clickthrough rate is the same.  
HA: The mean clicktrhough rate for site B is greater than site A.  


    mDiff = rB.mean()-rA.mean()
    se = np.sqrt(rB.var()/(len(rB)-1)+rA.var()/(len(rA)-1))
    Z = mDiff/se
    print mDiff,se,Z
    print "p-value: ",1-sc.norm.cdf(Z)
    print "The mean difference is statistically significant"


    0.0363429336139 0.000138288570828 262.805041634
    p-value:  0.0
    The mean difference is statistically significant


We are now told that there is a business model for this website, and it should inform our decision to implement the new site:  

    * the average click on site A yields $1.00 in profit  
* the average click on site B yields $1.05 in profit  



    gain_per_click = (rB.sum()*1.05-rA.sum())/10000.
    gain_per_click




    0.041471285947814275



There is an average gain of 4 cents per click increase in revenume.  Roughly 3 cents from the increase in click through rate and 1.4 cent from the increase in yield.  

We are not told over what time the experiment we analysed took place over.  A site with 10 Million users per month will see an increase revenue of 400,000 dollars per month, while a site of 10,000 users per month will only see an increase of of 400 dollars per month.  The business decision of investing money and time depends on the context.  This could be a great boom for the company, or a waste of time, depending on the situation.  



##Multi-arm Bandit

The multi-arm bandit approach is a method of balancing the exploration of strategies against the explortation of the best strategy.  We impleted a multi-arm bandit class with a number of strategies:  Random Choice, Max Mean, Epsilon Greedy, Soft Max, UCB1, Bayesian, and Annealing.  

We have two ways to measure the success of a search strategy for now.  The regret:

$$\mbox{Regret} = N \ p_{optimal} - \Sigma_{i=1}^{N} p_{i} $$

And the ratio of optimal decision:

$$ \frac{N_{optimal}}{N}$$

The bandit class takes an array of deciion options where each value is the probability of success.  The spirit of this is examining multiple metrics at the same time: like conversion rates across multiple simulatenous tests.

##Max Mean


    from bandits import Bandits
    from banditstrategy import *
    import numpy as np
    import scipy.stats as sc
    import matplotlib.pyplot as plt
    import prettyplotlib as ppl
    %matplotlib inline
    
    plt.figure(figsize=(14,10))
    for i in range(10):
        bandits = Bandits([0.05, 0.03, 0.06])
        strat = BanditStrategy(bandits, random_choice)
        strat.sample_bandits(1000)
        ppl.plot(range(strat.N),strat.regrets,label='Random Choice',color='steelblue',alpha=0.4,lw=3)
    
        bandits = Bandits([0.05, 0.03, 0.06])
        strat = BanditStrategy(bandits, max_mean)
        strat.sample_bandits(1000)
        ppl.plot(range(strat.N),strat.regrets,label='Max Mean'+str(i),alpha=0.4,color='indianred',lw=3)
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_26_0.png)


The max mean, on average, performs better than a random choice.  The issue with max mean is that its very exploitive.   When there is a statisitcal fluxtuation in makes it look like a worse option is the best option, it will stick with it.


    plt.figure(figsize=(14,10))
    for i in range(10):
        bandits = Bandits([0.05, 0.03, 0.06])
        strat = BanditStrategy(bandits, random_choice)
        strat.sample_bandits(1000)
        ppl.plot(range(strat.N),strat.percentOpt,label='Random Choice',color='steelblue',alpha=0.4,lw=3)
    
        bandits = Bandits([0.05, 0.03, 0.06])
        strat = BanditStrategy(bandits, max_mean)
        strat.sample_bandits(1000)
        ppl.plot(range(strat.N),strat.percentOpt,label='Max Mean'+str(i),alpha=0.4,color='indianred',lw=3)
    
    plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_28_0.png)


This is best illustrated if we look at the percent of the time that the optimal solution is chosen.   The max-mean strategy will often find one value and stay with it.

We can explore this method on three different array options:

    '''
    One Stand Out Best = [0.1, 0.1, 0.1, 0.1, 0.9]
    One Small Best = [0.1, 0.1, 0.1, 0.1, 0.12]
    A clear rank = [0.1, 0.2, 0.3, 0.4, 0.5]
    '''
    
Lets look at one 


    
    def make_multi_plots(size,func):
        values = [[0.1, 0.1, 0.1, 0.1, 0.9],[0.1, 0.1, 0.1, 0.1, 0.12],[0.1, 0.2, 0.3, 0.4, 0.5]]
        titles = ['One Stand Out Best','One Slightly Best', "A clear rank"]
        for v,t in zip(values,titles): 
            plt.figure(figsize=(14,5))
            plt.subplot(1,2,1)
            for i in range(10):
                bandits = Bandits(v[:])
                strat = BanditStrategy(bandits, random_choice)
                strat.sample_bandits(size)
                ppl.plot(range(strat.N),strat.regrets,label='Random Choice',color='steelblue',alpha=0.4,lw=3)
    
                bandits = Bandits(v[:])
                strat = BanditStrategy(bandits, func)
                strat.sample_bandits(size)
                ppl.plot(range(strat.N),strat.regrets,label='Max Mean'+str(i),alpha=0.4,color='indianred',lw=3)
            plt.title('Regret - ' + t)
    
            plt.subplot(1,2,2)
            for i in range(10):
                bandits = Bandits(v[:])
                strat = BanditStrategy(bandits, random_choice)
                strat.sample_bandits(size)
                ppl.plot(range(strat.N),strat.percentOpt,label='Random Choice',color='steelblue',alpha=0.4,lw=3)
    
                bandits = Bandits(v[:])
                strat = BanditStrategy(bandits, func)
                strat.sample_bandits(size)
                ppl.plot(range(strat.N),strat.percentOpt,label='Max Mean'+str(i),alpha=0.4,color='indianred',lw=3)
            plt.title('Percent Optimal - ' + t)
            plt.show()
        
    make_multi_plots(1000,max_mean)


![png](http://www.bryantravissmith.com/img/GW02D5/output_30_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_30_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_30_2.png)


The Max-Mean appear to work well in cases where there is a single clear best option, but does not consistently find optimal results in the other options.

##Epsilon Greedy

The epsilon greed strategy is a alternative to the max mean where some fraction of the time, say 10%, it will explore instead of exploit.   Well look at it for the same cases as the max mean method.  


    make_multi_plots(1000,epsilon_greedy)


![png](http://www.bryantravissmith.com/img/GW02D5/output_32_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_32_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_32_2.png)


This strategy has the benefit that if it does start to exploit a sub-optimal strategy, it will eventually find its way out.   This is clear in the "Slightly Best" plots where as the number of trials increase, bandits are leaving the sub-optimal strategy.  

##Softmax

The soft max is another strategy that balances exploitation vs exploration by use of a parameter.   It has an anology to temperature is method of statistical mechanics.


    func = lambda x: softmax(x,tau=0.5)
    make_multi_plots(1000,func)


![png](http://www.bryantravissmith.com/img/GW02D5/output_34_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_34_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_34_2.png)


The tau parameter of 0.5, in this example, seems to provide random solutions.   If we reduce the size of tau to 0.005, we should get solutions close to max mean.


    func = lambda x: softmax(x,tau=0.005)
    make_multi_plots(1000,func)


![png](http://www.bryantravissmith.com/img/GW02D5/output_36_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_36_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_36_2.png)


Inbetween, the soft max is to reduce the initial effect of outliers, but on these arrays we get less then optimal soltuions.

##UCB1

The UCB1 picks the strategy based on the number of trials and its success rate.


    make_multi_plots(1000,ucb1)


![png](http://www.bryantravissmith.com/img/GW02D5/output_38_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_38_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_38_2.png)


This method worked great on one standout best, but behaved randomly on the other options

##Bayesian Bandit

This method uses the ration of success and failures in a trial to pick options based on the beta distribution.


    
    make_multi_plots(1000,bayesian_bandit)


![png](http://www.bryantravissmith.com/img/GW02D5/output_40_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_40_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_40_2.png)


This method wored well on a clear best and eventually found the optimal solution for a ranked system.  The slight best behaved randomly.

##Annealing

This method is working with a softmax that starts with a 'hot' system that explores randomly and cools to a 'cold' system that exploits.  




    func = lambda x:annealing(x,discount=0.90,tau=0.5)
    make_multi_plots(1000,annealing)


![png](http://www.bryantravissmith.com/img/GW02D5/output_42_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_42_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_42_2.png)


We can see evidence of the cooling because the random values are starting to move away from random, but we would need to see more trials to see the evenatual discovery of optimal solutions.


    color_scheme = [(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),(188, 189, 34),(23, 190, 207)] # (219, 219, 141)]#, (]
    color_scheme = [ (x/255.,y/255.,z/255.) for x,y,z in color_scheme]
    
    func1 = lambda x: softmax(x,tau=0.5)
    strategies = [max_mean,random_choice,epsilon_greedy,func1,ucb1,bayesian_bandit]
    
    size=10000
    values = [[0.1, 0.1, 0.1, 0.1, 0.9],[0.1, 0.1, 0.1, 0.1, 0.12],[0.1, 0.2, 0.3, 0.4, 0.5]]
    titles = ['One Stand Out Best','One Slightly Best', "A clear rank"]
    for v,t in zip(values,titles): 
        plt.figure(figsize=(14,5))
        for color,func in zip(color_scheme,strategies):
            plt.subplot(1,2,1)
            bandits = Bandits(v[:])
            strat = BanditStrategy(bandits, func)
            strat.sample_bandits(size)
            ppl.plot(range(strat.N),strat.regrets,label=func.__name__,alpha=1,color=color,lw=4)
            plt.title('Regret - ' + t)
            plt.legend(loc=2)
    
    
            plt.subplot(1,2,2)
            bandits = Bandits(v[:])
            strat = BanditStrategy(bandits, func)
            strat.sample_bandits(size)
            ppl.plot(range(strat.N),strat.percentOpt,label=func.__name__,alpha=1,color=color,lw=4)
            plt.title('Percent Optimal - ' + t)
    
        plt.show()


![png](http://www.bryantravissmith.com/img/GW02D5/output_44_0.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_44_1.png)



![png](http://www.bryantravissmith.com/img/GW02D5/output_44_2.png)


It seems that the epsilon greed strategy and bayesian bandit algorithms have the best overall performance and avoid getting stuck in non-optimal solutions.   The soft max and annealing both require turning, so they are not fairly compared.   It is nice to see that both epsilon greed and bayesian bandit solutions offer easy to implment and robust use.  


    
