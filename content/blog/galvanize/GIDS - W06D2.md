
#Galvanize Immersive Data Science

##Week 6 - Day 2

Today we had a weekly assessment.  It was on natural language processing, profit curves, dimentional reduction, and clustering.   We were given an hour, but it was more like a 40 minute assessment.   After that we broke out into groups to discuss capstone project ideas.  We each gave 5 rough ideas, and made suggesting for refining and improving them.  Right now I working with 3 personal projects, and 2 prospective projects for companies I have talked with.   

##Non-Negative Matrix Facorization

Today we covered NMF methods for topic modeling.  I was shocked witht the results we found when we used this method on the NY Times dataset we have been playing with for the last week.   

The ideas is that we break a matrix, like the TfIdf vectors relating articles to words for the New York Times.  This matrix can be decomposed into two matrixes.   One that relates Articles to Topics, and another non-negative matrix that relates Topics to words.  The first task we had was writing our own NMF algorithm class.  We use the method proposed by [Daniel D. Lee and H. Sebastian Seung](http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf).  

We are going to first load in our NY Times data, show our class, then analysis the topics


    import numpy as np
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    %matplotlib inline
    ny = pd.read_pickle('data/articles.pkl')
    ny.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document_type</th>
      <th>web_url</th>
      <th>lead_paragraph</th>
      <th>abstract</th>
      <th>snippet</th>
      <th>news_desk</th>
      <th>word_count</th>
      <th>source</th>
      <th>section_name</th>
      <th>subsection_name</th>
      <th>_id</th>
      <th>pub_date</th>
      <th>print_page</th>
      <th>headline</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>article</td>
      <td>http://www.nytimes.com/2013/10/03/sports/footb...</td>
      <td>You would think that in a symmetric zero-sum s...</td>
      <td>None</td>
      <td>You would think that in a symmetric zero-sum s...</td>
      <td>Sports</td>
      <td>347</td>
      <td>The New York Times</td>
      <td>Sports</td>
      <td>Pro Football</td>
      <td>524d4e3a38f0d8198974001f</td>
      <td>2013-10-03T00:00:00Z</td>
      <td>None</td>
      <td>Week 5 Probabilities: Why Offense Is More Impo...</td>
      <td>the original goal building model football fore...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>article</td>
      <td>http://www.nytimes.com/2013/10/03/us/new-immig...</td>
      <td>House Democrats on Wednesday unveiled an immig...</td>
      <td>House Democrats unveil immigration bill that p...</td>
      <td>House Democrats on Wednesday unveiled an immig...</td>
      <td>National</td>
      <td>83</td>
      <td>The New York Times</td>
      <td>U.S.</td>
      <td>None</td>
      <td>524cf71338f0d8198973ff7b</td>
      <td>2013-10-03T00:00:00Z</td>
      <td>21</td>
      <td>New Immigration Bill Put Forward</td>
      <td>house unveiled immigration bill provides path ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>article</td>
      <td>http://www.nytimes.com/2013/10/03/us/arizona-j...</td>
      <td>A federal judge on Wednesday ordered the appoi...</td>
      <td>Federal Judge Murray Snow orders the appointme...</td>
      <td>A federal judge on Wednesday ordered the appoi...</td>
      <td>National</td>
      <td>160</td>
      <td>The New York Times</td>
      <td>U.S.</td>
      <td>None</td>
      <td>524cf50e38f0d8198973ff79</td>
      <td>2013-10-03T00:00:00Z</td>
      <td>21</td>
      <td>Arizona: Judge Orders Monitor to Oversee Maric...</td>
      <td>federal judge wednesday ordered appointment in...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>article</td>
      <td>http://www.nytimes.com/2013/10/03/us/texas-sta...</td>
      <td>Texas has turned to a compounding pharmacy to ...</td>
      <td>Documents show that Texas, nation's most activ...</td>
      <td>Texas has turned to a compounding pharmacy to ...</td>
      <td>National</td>
      <td>112</td>
      <td>The New York Times</td>
      <td>U.S.</td>
      <td>None</td>
      <td>524cf39a38f0d8198973ff78</td>
      <td>2013-10-03T00:00:00Z</td>
      <td>21</td>
      <td>Texas: State Bought Execution Drugs From a Com...</td>
      <td>texas nation’s active death-penalty state turn...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>article</td>
      <td>http://www.nytimes.com/2013/10/03/sports/tenni...</td>
      <td>Rafael Nadal, aiming to end Novak Djokovic’s r...</td>
      <td>None</td>
      <td>Rafael Nadal, aiming to end Novak Djokovic’s r...</td>
      <td>Sports</td>
      <td>49</td>
      <td>The New York Times</td>
      <td>Sports</td>
      <td>Tennis</td>
      <td>524cf28b38f0d8198973ff73</td>
      <td>2013-10-03T00:00:00Z</td>
      <td>14</td>
      <td>Nadal on Track for No. 1 Spot</td>
      <td>rafael nadal aiming end novak djokovic’s run 1...</td>
    </tr>
  </tbody>
</table>
</div>




    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    tfidf = TfidfVectorizer(max_features=5000,stop_words='english')
    tfidf_articles = tfidf.fit_transform(ny.content.values)
    
    tfidf_df = pd.DataFrame(tfidf_articles.todense(),columns=tfidf.get_feature_names())
    print tfidf_df.shape
    tfidf_df.head()

    (1405, 5000)





<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>000</th>
      <th>10</th>
      <th>100</th>
      <th>10th</th>
      <th>11</th>
      <th>11th</th>
      <th>12</th>
      <th>120</th>
      <th>13</th>
      <th>130</th>
      <th>...</th>
      <th>youtube</th>
      <th>zarif</th>
      <th>zealand</th>
      <th>zen</th>
      <th>zero</th>
      <th>zhang</th>
      <th>zhou</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zorn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.063525</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.103347</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.243745</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5000 columns</p>
</div>



We have a matrix V, and our goal is to factor it into the product of two matrixes.

$$V = W \ H$$

The algorithm we implemented randomly initates W and H with positive, non-zero values.   It then solves for new W's an d H's using the least squared minimization.   We then continue this updateing until we have reached the desired tolerance or reached the max number of iterations.  For illustration purposed I have printed the iteration, cost, and change in cost as we fit.


    class NMF:
        
        def __init__(self,V,k,max_iter=100,tol=1e-4):
            self.data_frame = V
            self.V = V.values
            self.latent_topics = int(k)
            self.max_iter = max_iter
            self.W = np.random.randint(1,100,size=(V.shape[0],k))/100.
            self.H = np.random.randint(1,100,size=(k,V.shape[1]))/100.
            self.tol = tol
            
        def fit(self):
            dcost = self._cost()
            cost = dcost.copy()
            iter = 0
            while( (dcost > self.tol) and (iter < self.max_iter) ):
                self._update()
                temp = self._cost()
                dcost = np.abs(temp-cost)
                cost = temp
                
                iter += 1
                print iter,cost,dcost
            return self.W, self.H
                
            
        def _cost(self):
            return np.sum(np.power(self.V-self.W.dot(self.H),2))
        
        def mse(self):
            return self._cost()/(self.V.shape[0]*self.V.shape[1])
        
        def _update(self):
            H,resid,rank,S = np.linalg.lstsq(self.W,self.V)
            a,b = np.where(H<=0)
            H[a,b]=1e-6
            self.H = H
            
            WT,resid,rank,S = np.linalg.lstsq(self.H.T,self.V.T)
            W = WT.T
            a,b = np.where(W<=0)
            W[a,b]=1e-6
            self.W = W
        
        def top10headlines(self):
            top10_indexes = np.argsort(self.W,axis=0)[-11:-1,:]
            for i,indexes in enumerate(top10_indexes.T):
                print "New Topic: {}".format(i+1)
                for index in indexes:
                    print ny.loc[index,'headline']
                print ""
                
        def top3topics(self,doc):
            return np.argsort(doc.dot(self.H.T))[::-1][0,0:3].tolist()[0]


    nmf = NMF(tfidf_df,20,max_iter=100)
    nmf.fit()

    1 1358.53775257 183045559.295
    2 1300.48172616 58.0560264073
    3 1268.32077125 32.1609549088
    4 1248.0232425 20.2975287559
    5 1229.45407174 18.5691707553
    6 1215.00022839 14.4538433585
    7 1204.01153417 10.9886942162
    8 1196.65905672 7.35247744427
    9 1192.26204639 4.39701033659
    10 1189.20603662 3.05600976726
    11 1186.97772507 2.22831154859
    12 1185.18812772 1.7895973489
    13 1183.65092545 1.53720227374
    14 1182.66629927 0.984626175047
    15 1182.18979125 0.47650802397
    16 1181.95567812 0.234113130044
    17 1181.81545439 0.140223734529
    18 1181.71161609 0.103838298331
    19 1181.62183591 0.0897801769704
    20 1181.53770034 0.0841355706787
    21 1181.4564612 0.0812391435775
    22 1181.38118262 0.0752785771172
    23 1181.31682066 0.0643619614168
    24 1181.2649946 0.0518260621986
    25 1181.22347343 0.0415211619361
    26 1181.19324932 0.0302241145484
    27 1181.17174525 0.0215040704031
    28 1181.15728225 0.0144629971055
    29 1181.14809469 0.00918756042415
    30 1181.14236544 0.00572925622464
    31 1181.13921747 0.0031479679767
    32 1181.13736328 0.00185418691763
    33 1181.13635232 0.00101096086269
    34 1181.1358406 0.000511715468292
    35 1181.13573402 0.000106588699282
    36 1181.13608484 0.000350827998545
    37 1181.13675405 0.00066920801919
    38 1181.13754313 0.000789074088061
    39 1181.1385491 0.0010059775077
    40 1181.13956705 0.00101795043247
    41 1181.14056188 0.000994825212956
    42 1181.14148084 0.000918961011848
    43 1181.14239117 0.000910335122171
    44 1181.14328144 0.000890265901035
    45 1181.14410508 0.000823640306407
    46 1181.14485309 0.000748005865034
    47 1181.14551725 0.00066415951801
    48 1181.14612006 0.000602811966701
    49 1181.14665547 0.000535414914339
    50 1181.14714023 0.000484758575794
    51 1181.14758135 0.000441115262447
    52 1181.14797781 0.000396463954758
    53 1181.14832584 0.000348033931004
    54 1181.14864366 0.000317815084145
    55 1181.14892223 0.000278567937812
    56 1181.14917425 0.000252021858387
    57 1181.14941041 0.000236161348994
    58 1181.14962429 0.000213882947492
    59 1181.14981494 0.000190645713246
    60 1181.14999555 0.000180610762527
    61 1181.15015622 0.000160667058935
    62 1181.15029924 0.000143026843944
    63 1181.15043527 0.000136026186738
    64 1181.15055687 0.000121601296541
    65 1181.15066677 0.000109900271582
    66 1181.15076618 9.94073752736e-05





    (array([[  1.00119406e-01,   5.68400697e-01,   1.00000000e-06, ...,
               1.64106596e+00,   5.05033775e-02,   7.65333772e-04],
            [  1.53327329e-01,   1.00000000e-06,   6.22590411e-02, ...,
               1.00000000e-06,   1.00000000e-06,   1.00000000e-06],
            [  1.00000000e-06,   1.00000000e-06,   1.00000000e-06, ...,
               3.75267776e-02,   1.00000000e-06,   1.00000000e-06],
            ..., 
            [  1.00000000e-06,   1.00000000e-06,   1.00000000e-06, ...,
               1.00000000e-06,   1.09754697e-01,   1.00000000e-06],
            [  3.89527959e+00,   1.43767310e+00,   5.01533481e-01, ...,
               1.00000000e-06,   1.21685726e+00,   1.00000000e-06],
            [  3.68718629e+00,   1.00000000e-06,   2.72451285e-01, ...,
               1.43511299e-01,   4.36065874e-01,   1.00000000e-06]]),
     array([[  5.40116149e-04,   8.76588753e-04,   1.63700931e-04, ...,
               1.00000000e-06,   1.00000000e-06,   1.00000000e-06],
            [  2.03678399e-03,   2.11042158e-03,   2.47977461e-04, ...,
               2.68599462e-03,   4.31713278e-05,   1.00000000e-06],
            [  3.06087950e-03,   2.04330832e-03,   7.32513575e-04, ...,
               1.00000000e-06,   7.69852261e-04,   2.37192788e-03],
            ..., 
            [  1.00000000e-06,   4.41264815e-03,   4.46157256e-04, ...,
               1.77517615e-03,   1.00000000e-06,   1.00000000e-06],
            [  1.00000000e-06,   1.36233670e-03,   1.00000000e-06, ...,
               1.00000000e-06,   2.48418509e-04,   2.45724743e-03],
            [  1.00000000e-06,   1.00000000e-06,   1.00000000e-06, ...,
               1.00000000e-06,   1.00000000e-06,   6.87259348e-05]]))



For our NY Times example with 1405 articles, we see the fit took 66 iterations and a few seconds to complete.  We can estimate the mean squared error of our results from the matrix we were trying to fit.   


    nmf.mse()




    0.00016813534038140441



Which is close to the tolerance we were shooting for.  Our results do a solid approximation of the Tfidf matrix we were trying to approximate as a product of two matrixes.   Now that we have these matrixes, lets look at the results.

##Words for Topics

We did an initial fit of 20 topics in our NY Times corpus.   For each topic, we can go through and find the top words associted with the topic.


    for i,row in enumerate(tfidf_df.columns[np.argsort(nmf.H,axis=1)[:,-15:-1]]):
        print "Topic {}:".format(i+1)
        print " ".join(row.tolist())
        print ""

    Topic 1:
    college life poverty people writer student gun york teacher new child school 2013 editor
    
    Topic 2:
    shutdown economist growth price stock index bond investor government economy debt rate market bank
    
    Topic 3:
    org photograph city drawing like curator street collection work exhibition artist painting gallery museum
    
    Topic 4:
    fan stadium baseball cano yankees mariano season inning rodriguez game jeter girardi pettitte rivera
    
    Topic 5:
    crime sentence prosecutor trial prison law federal lawyer justice state case said judge court
    
    Topic 6:
    berlin vote chancellor coalition election social democrat euro european europe german party ms germany
    
    Topic 7:
    medical cost pay people affordable employee coverage plan federal law company exchange care insurance
    
    Topic 8:
    red indian wright league homer season win host mets card wild hit game run
    
    Topic 9:
    care debt law cruz senator vote democrat party mr obama shutdown government senate house
    
    Topic 10:
    fan played sport year goal football giant play coach league said player season game
    
    Topic 11:
    comedy breaking hbo million television fox emmy drama series network abc rating cbs nbc
    
    Topic 12:
    war al international rebel security russia council nation assad resolution united syrian weapon chemical
    
    Topic 13:
    state beijing kong hong nuclear energy gas said korea north company plant chinese oil
    
    Topic 14:
    wind yacht club regatta francisco san racing spithill america boat zealand team oracle race
    
    Topic 15:
    killing soldier iraq city taliban official baghdad bomb afghan pakistan people killed police attack
    
    Topic 16:
    arrested right immigrant athens member political murder parliament greece government greek police golden dawn
    
    Topic 17:
    terrorist group attacker westgate american official militant somali attack nairobi somalia kenyan mall shabab
    
    Topic 18:
    pas manning tennessee turnover jet giant quarter game interception smith pass threw quarterback touchdown
    
    Topic 19:
    piece concert album dancer performance work band orchestra ms song ballet opera dance mr
    
    Topic 20:
    meeting speech nation israeli sanction president united netanyahu obama israel mr iranian nuclear rouhani
    


These words for each topic are very distinctive.  The first has to do with schools, the second with the economy, and the third with life style.   The forth is baseball, and the firth is legal.   If we had a huge corpus and did not know how to make subsections, thes topics would do.  Infact they line up pretty well with the 19 sections in the NY Times website. 

Instead of looking at the words for each topic, we can look at the headlines assoicated with the articles that are characterist of the topics.  


    nmf.top10headlines()

    New Topic: 1
    A New (Harder?) Admissions Option at Bard College
    Children Killed by Guns: Stopping the Scourge
    Measuring Poverty and the Income Gap
    Preparing Teachers for the Urban Classroom
    Private School Admission
    Trophies for All, or Just the Deserving?
    G.O.P. vs. Health Law and Food Stamps
    Two Approaches to Homelessness
    Investing in Early Childhood Now, for a Payoff Later
    When Evangelicals Adopt Children Abroad
    
    New Topic: 2
    Orders for Durable Goods Increased Slightly in August
    Consumer Spending Rose Slightly in August
    European Unemployment Steady, Hinting at Progress in Crisis
    Jobs Data Helps Wall St. Halt a 5-Day Slide
    Markets Fall as Investors Rehash the Fed&#8217;s Decision
    Housing Recovery Seems Still on Track 
    Shadow of Shutdown Looms Over Markets
    Wall Street Closes Higher Despite U.S. Shutdown
    Fight Over U.S. Budget Weighs on Shares
    S.&P. Falls for a 5th Day As Federal Shutdown Looms
    
    New Topic: 3
    Spare Times for Sept. 27-Oct. 3
    A Monet for Show and Tell
    Midnight at the Museum, Breakfast at the Vatican
    A New Survey Finds a Drop in Arts Attendance
    Passion, Principle or Both? Deciphering Art Vandalism
    Newcomb Pottery Will Go on View at Tulane
    The Agony of Suspense in Detroit
    From Beirut to Bogotá: Art Cities to Watch?
    Museum and Gallery Listings for Sept. 27-Oct. 3
    Hey ‘Starry Night,’ Say ‘Cheese!’
    
    New Topic: 4
    For Yankees, Emotional Conclusion Isn’t End
    The Yankees’ Farewell Sundays of 2008 and 2013
    Memorable End for Two; Forgettable Year for Yanks
    Young Catcher Can Help Tell Ending to Rivera’s Story
    Mariano Rivera: A Zen Master With a Mean Cutter
    A Final Bow for Rivera
    When Rivera Started, and Pettitte Relieved
    Mariano Rivera’s Saving Grace
    Yankees’ Costly Loss Puts a Damper on Rivera’s Party
    Closing Scene: Hugs and Tears in Rivera’s Last Home Game
    
    New Topic: 5
    In Supreme Court Opinions, Web Links to Nowhere
    Former F.B.I. Agent to Plead Guilty in Press Leak
    The Hague: Warrant Issued for Kenyan
    Intriguing Tip From a Source Who’s Suspect
    Louisiana: New Trial for Man Convicted of Murder in '74
    Why Judges Are Scowling at Banks
    A Rare Plea to the Court
    Citing New Evidence, Urging a Posthumous Pardon in 1992 Case
    Rubio Withdraws Support for Gay Black Judge’s Nomination to the Federal Bench
    50-Year Sentence Upheld for Ex-President of Liberia
    
    New Topic: 6
    German Politician Faces Plagiarism Accusations
    A Challenge to European Political Elite
    Anti-Euro Party Gaining Steam in Germany
    Merkel the Great
    After Rout in German Elections, Social Democrats Consider Coalition With Merkel
    German Campaign, Amid Fiery Debate Abroad, Shuffles Toward Consensus
    Angela Merkel’s Next Challenge
    Vote for Merkel Seen as Victory for Austerity
    Merkel Re-elected in Show of Strong Support for Party
    After Rewarding Merkel, Germans Seek Focus at Home
    
    New Topic: 7
    Health Insurance Exchanges Scramble to Be Ready as Opening Day Nears
    Opening Rush to Insurance Markets Runs Into Snags
    Health Care Choices in the New Era
    Lacking Rules, Insurers Balk at Paying for Intensive Psychiatric Care
    U.S. Plans to Unveil New Insurance Options
    As Some Companies Turn to Health Exchanges, G.E. Seeks a New Path
    Day 1: The New World of Health Care
    Dawn of a Revolution in Health Care
    On the Threshold of Obamacare, Warily
    The Landscape of Small-Business Health Insurance
    
    New Topic: 8
    Wright Homers Again to Help Mets Past Phillies
    Indians Get Wild-Card Slot as Rays and Rangers Go to a Tiebreaker
    Scherzer Wins 21st as Tigers Clinch
    Cardinals Prevail Behind a One-Hitter
    Rays and Rangers Tied as A.L. Race Nears End
    Game Ends on Wild Pitch, Giving Marlin a No-Hitter
    Wild-Card Race Tightens as Rays Lose to Blue Jays
    Red Sox Clinch East; Indians Win, As Do Royals
    21-Year Wait Is Over for Playoff-Bound Pirates 
    Rays, Royals and Indians Win in A.L. Playoff Race
    
    New Topic: 9
    House Bill Links Health Care Law and Budget Plan
    Government Shuts Down in Budget Impasse
    Senator Persists Battling Health Law, Irking Even Many in His Own Party
    Staunch Group of Republicans Outflanks House Leaders
    Those Banana Republicans
    U.S. Shutdown
    Nears as House
    Votes to Delay
    Health Law
    Senate Action on Health Law Moves to Brink of Shutdown
    Obama Sets Conditions for Talks: Pass Funding and Raise Debt Ceiling
    Conservatives With a Cause: ‘We’re Right’
    Shutdown Looms as Senate Passes Budget Bill
    
    New Topic: 10
    Coughlin Leads Calls for Giants to Show More Character
    Giants’ Battered Line Faces a Tough Test
    Favorites? Broncos, by a Wide Margin 
    Knicks and Nets, Growing Rivals, Will Work Together to Host All-Star Events
    Petke Sobered by Club’s Checkered Past
    No Quick-Fix Recipe for the Giants
    Hype Machine in Overdrive, N.F.L. Pops Back to London
    Cosmos-Rowdies: That ’70s Rivalry, Updated
    Where Mets Go Deep
    Goalkeeper Makes a Case for Team M.V.P.
    
    New Topic: 11
    Race to End for ‘Breaking Bad’ Fans Who Got Behind
    In Ratings War, ‘G.M.A.’ Beats ‘Today’ for Full Season
    Head of NBC Entertainment Extends Contract to 2017
    Networks Go to Extremes to Promote New Shows for the Fall Season
    2013 Emmy Award Winners and Nominees
    Emmys Highlight a Changing TV Industry
    Emmys Draw 17 Million Viewers, Up From Recent Shows
    ‘Big Bang’ Reinforces Its Status as Biggest Hit on Network TV
    ‘Agents of S.H.I.E.L.D.’ Gains a Big Audience While ‘The Voice’ Keeps Rolling
    ‘The Voice’ Propels NBC to a Big Ratings Win on Monday Night
    
    New Topic: 12
    Stance on Peace Talks Suggests Syria and West Differ on Tactics
    Missed Opportunity in Syria Haunts U.N. Official
    For the U.N., Syria Is Both Promise and Peril
    Text of Draft United Nations Resolution on Syrian Chemical Weapons
    Invoking Sept. 11, Syrian Accuses U.S. of Hypocrisy
    Some Progress on Syria
    Syria Meets First Test of Accord on Weapons
    U.N. Investigates More Alleged Chemical Attacks in Syria
    Swift Movement Is Seen on Syria After U.N. Action
    U.N. Deal on Syrian Arms Is Milestone After Years of Inertia
    
    New Topic: 13
    Japan’s Leader Gives No Ground in Islands Dispute
    U.S. Gears Up to Be a Prime Gas Exporter
    As Stability Eludes Region, Western Oil Giants Hesitate
    Chinese Titan Takes Aim at Hollywood
    Hacking U.S. Secrets, China Pushes for Drones
    China Is Said to Be Holding a Professor Based in Japan
    Volkswagen Expanding Production in China
    Volkswagen Expanding Production in China
    China Bans Items for Export to North Korea, Fearing Their Use in Weapons
    China Ban on Items for Nuclear Use to North Korea May Stall Arms Bid
    
    New Topic: 14
    Australia Primed for First Cup Challenge Since 2000
    In New Zealand, Jitters Yield to Cheers, Then Sighs
    Extended America’s Cup Leaves Some High, Dry and Homeless
    New Zealand Is Kept Waiting Again
    Oracle Team USA Now Has the Wind at Its Back
    Oracle Completes Voyage to History, Winning America’s Cup
    Saved by Light Breeze, Oracle Will Race Again
    The Cup May Stay, but There’s No Going Back on Speed
    Oracle Sweeps Two Races to Tie America’s Cup
    After Comeback for the Ages, a Last Dash for America’s Cup
    
    New Topic: 15
    Once-Calm Area of Iraq Is Shaken by Bombings
    Pakistan Christians Issue Call for Protection
    Iraq: Multiple Bomb Attacks at Markets
    11 Officers Killed as Taliban Strike Afghan Border Post
    Bus Bombing in Pakistan Kills at Least 17 Government Employees
    Scores Are Killed by Suicide Bomb Attack at Historic Church in Pakistan
    Iraq: Bombings Kill at Least 55
    A Deadly Week in Northwestern Pakistan Ends With a Car Bomb Blast
    Attacks Kill Scores in Iraq as Violence Surges
    Bomber Hits Sunni Funeral as Attacks Mount in Iraq
    
    New Topic: 16
    A Challenge to European Political Elite
    Norway: New Coalition Tilts Right
    Smaller Parties Gain in Austrian National Elections
    British Party Leader Suspended After Crude Remark and Swipe at Reporter
    Greek Civil Servants Start 2-Day Strike
    Greece: A Vow to Erase a ‘Shame’
    Greece: Police Under Scrutiny
    Greece, in Anti-Fascist Crackdown, Investigates Police
    Case Against Greek Far-Right Party Draws Critics
    Greece Arrests Senior Members of Far-Right Party
    
    New Topic: 17
    Why Nairobi
    A Shaken Kenya Is Hit Again in 2 Deadly Attacks by Militants
    Kenya Forces Said to Be Securing Mall After Long Standoff
    Gunmen Kill Dozens in Terror Attack at Kenyan Mall
    Somali Militants Mixing Business and Terror
    U.S. Sees Direct Threat in Attack at Kenya Mall
    Kenya Presses Assault Against Militants in Mall Siege
    Attention Switches to Investigation of Kenyan Mall Siege
    Before Kenya Attack, Rehearsals and Planting of Machine Guns
    Kenya’s Brutal Coming of Age
    
    New Topic: 18
    Fantasy Football: Week 4 Matchup Breakdown
    Niners Rebound With a Rout Against St. Louis
    College Football Around the Country
    Saints Hand Dolphins Their First Defeat
    College Football Around the Country
     Ohio State Streak Is at 17 With Win Over Wisconsin
    Rare Rushing Feat by a Princeton Quarterback Helps the Tigers Crush Georgetown
    Nova Throws for 3 Touchdowns as Rutgers Rallies Past Arkansas
    Michigan Hangs On After Scare, Again
    In Game of Turnovers, Bengals Get Last One, and a Win
    
    New Topic: 19
    Dance Listings for Sept. 27-Oct. 3
    Experimenting Begins as the Music Starts to Play
    The Dawn of a World, Dreamlike Yet Chaotic
    A Sampling of Old and New, Side by Side
    It’s Not What You Wear, It’s How You Dance in It
    Giddy Freedom (a Little Mambo!), as Well as Pianistic Elegance and Wit
    A Swing and a Jaunt Around the Globe
    An Impish Faun, Swans, and a Lively Card Deck
    Movers and Shapers
    A Giant of Dance, Seizing on Musicality to Weave His Spell
    
    New Topic: 20
    President Rouhani Comes to Town
    Israel and Others in Mideast View Overtures of U.S. and Iran With Suspicion
    Israel and Others in Mideast View Overtures of U.S. and Iran With Suspicion
    Hassan Does Manhattan
    Now, the Hard Part
    Iran’s President Responds to Netanyahu
    Netanyahu Pushes Back on Iran
    Discussing Iran, Obama and Netanyahu Display Unity
    Iran’s New President Preaches Tolerance in First U.N. Appearance
    Israeli Leader Excoriates New President of Iran
    


These headline reinforce the words results we saw above.  The NMF techniques are finding topics in this corpus.   The question, I am wondering, is how do you know when you have enough topics.  We know from yesterday that 90% of the energy is in the first 500 topics (if SVD can carry over).   But we only have 1400 articles.  This does not see useful.    I will think on this topic.

##Sklearn NMF

As usual, we want to compare our results with the results of Sklearn's implementation.   In this case, eventhough we are using the same algorithm, sklearn is much smarter about the setup.  Looking at thier code, I see that they use the SVD decomposition to make a smart first guess as to the matrixes W and H.  I am sure this leads to quicker convergence.   They are also taking advantage of thier c optimization libraries.  


    from sklearn.decomposition import NMF
    nmf2 = NMF(n_components=20).fit(tfidf_df.values)


    for i,row in enumerate(tfidf_df.columns[np.argsort(nmf2.components_,axis=1)[:,-15:-1]]):
        print "Topic {}:".format(i+1)
        print " ".join(row.tolist())
        print ""

    Topic 1:
    obama president people navy prime john mayor alexis political maduro year minister official said
    
    Topic 2:
    rodriguez jeter pitch league baseball girardi said hit run season pettitte inning game rivera
    
    Topic 3:
    care shutdown obamacare conservative government obama law vote party cruz senator democrat senate house
    
    Topic 4:
    hassan speech nation israeli sanction mr president united netanyahu obama israel iranian nuclear rouhani
    
    Topic 5:
    group somali people somalia nairobi official police killed militant said kenyan shabab kenya mall
    
    Topic 6:
    war international al security rebel russia assad council nation resolution united syrian weapon chemical
    
    Topic 7:
    government political social coalition democrat parliament election euro european ms europe german germany merkel
    
    Topic 8:
    said regatta match francisco san racing spithill america club boat zealand oracle team race
    
    Topic 9:
    people act cost insurer medical state federal affordable plan coverage law exchange care insurance
    
    Topic 10:
    woman york writer college life gun new people student teacher child 2013 school editor
    
    Topic 11:
    investor house washington limit economy obama federal bond congress treasury default ceiling shutdown government
    
    Topic 12:
    korean japanese said state asia south japan nuclear kong beijing hong north korea chinese
    
    Topic 13:
    company drilling putin said carbon coal russia russian plant energy gas greenpeace ship arctic
    
    Topic 14:
    photograph magritte like city drawing curator street collection work exhibition artist painting gallery museum
    
    Topic 15:
    rose year said august stock economist month economy index growth price market rate bank
    
    Topic 16:
    like employee creative vice group advertising brand york chief business new agency said executive
    
    Topic 17:
    league sunday smith manning said player play coach giant quarterback team season touchdown yard
    
    Topic 18:
    season breaking million hbo fox television emmy drama network series abc rating cbs nbc
    
    Topic 19:
    criminal ruling trial supreme department prison sex lawyer federal justice law state case judge
    
    Topic 20:
    piece festival concert album performance dancer work orchestra band song ms opera ballet dance
    


The sklearn implmentation was faster, but the results are similar.  They are not identical.   Because we are hill climbing, I am sure that we can get stuck in different local minima than sklearn, and we are not ordering topics.  It seems sklearn is.   

The topics classifications, however, are almost identical.  That is really cool that there are letent features that help or organize and thinking about these articles.  It is also cool that they match our intuition about what they are.   
