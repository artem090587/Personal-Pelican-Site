Title: Udacity - Data Analysis NanoDegree - Project 4
Date: 2015-03-21 10:20
Modified: 2015-05-16 19:30
Category: Udacity
Tags: udacity, data-analysis, nanodegree, project
Slug: udacity-project-4
Authors: Bryan Smith
Summary: My passing submission project 4 for Udacity Data Analysis Nanodegree

#Udacity - NanoDegree - Project 4 - Identifying POI

##Project Goal
The goal of this project is to develop and tune a supervised classification algorithm to identify persons of interest (POI) in the Enron scandal based on a combination of publically available Enron financial data and email records. The modest goals are to have recall and precision scores above 0.3.  

The compiled data set contains information for 144 people employed at Enron, a ‘Travel Agency in the Park’, and the total compensations for all of these sources. Additionally, the Udacity.com course designer created email features that give the total number of e-mails sent and received for each user, and the total number of emails sent to and receieved from a POI. Of the 144 people, 18 of them are labeled as POIs.

Lets start by reading in the data;



    import sys
    import pickle
    import numpy as np
    import pandas as pd
    
    #Load Udacity Tools for Testing Results
    sys.path.append("../tools/")
    from feature_format import featureFormat, targetFeatureSplit
    from tester import test_classifier, dump_classifier_and_data
    
    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    data = pd.DataFrame.from_dict(data_dict)
    #Remove Invalide Rows
    data = data.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK'],axis=1)
    data = data.transpose()
    #Give Each Person their own row
    data




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bonus</th>
      <th>deferral_payments</th>
      <th>deferred_income</th>
      <th>director_fees</th>
      <th>email_address</th>
      <th>exercised_stock_options</th>
      <th>expenses</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>...</th>
      <th>long_term_incentive</th>
      <th>other</th>
      <th>poi</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>salary</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
      <th>total_payments</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>4175000</td>
      <td>2869717</td>
      <td>-3081055</td>
      <td>NaN</td>
      <td>phillip.allen@enron.com</td>
      <td>1729541</td>
      <td>13868</td>
      <td>2195</td>
      <td>47</td>
      <td>65</td>
      <td>...</td>
      <td>304805</td>
      <td>152</td>
      <td>False</td>
      <td>126027</td>
      <td>-126027</td>
      <td>201955</td>
      <td>1407</td>
      <td>2902</td>
      <td>4484442</td>
      <td>1729541</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>178980</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>3486</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>182466</td>
      <td>257817</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5104</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>4046157</td>
      <td>56301</td>
      <td>29</td>
      <td>39</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>864523</td>
      <td>False</td>
      <td>1757552</td>
      <td>-560222</td>
      <td>477</td>
      <td>465</td>
      <td>566</td>
      <td>916197</td>
      <td>5243487</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>1200000</td>
      <td>1295738</td>
      <td>-1386055</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>11200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1586055</td>
      <td>2660303</td>
      <td>False</td>
      <td>3942714</td>
      <td>NaN</td>
      <td>267102</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5634343</td>
      <td>10623258</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>400000</td>
      <td>260455</td>
      <td>-201641</td>
      <td>NaN</td>
      <td>frank.bay@enron.com</td>
      <td>NaN</td>
      <td>129142</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>69</td>
      <td>False</td>
      <td>145796</td>
      <td>-82782</td>
      <td>239671</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>827696</td>
      <td>63014</td>
    </tr>
    <tr>
      <th>BAZELIDES PHILIP J</th>
      <td>NaN</td>
      <td>684694</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1599641</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>93750</td>
      <td>874</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80818</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>860136</td>
      <td>1599641</td>
    </tr>
    <tr>
      <th>BECK SALLY W</th>
      <td>700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sally.beck@enron.com</td>
      <td>NaN</td>
      <td>37172</td>
      <td>4343</td>
      <td>144</td>
      <td>386</td>
      <td>...</td>
      <td>NaN</td>
      <td>566</td>
      <td>False</td>
      <td>126027</td>
      <td>NaN</td>
      <td>231330</td>
      <td>2639</td>
      <td>7315</td>
      <td>969068</td>
      <td>126027</td>
    </tr>
    <tr>
      <th>BELDEN TIMOTHY N</th>
      <td>5249999</td>
      <td>2144013</td>
      <td>-2334434</td>
      <td>NaN</td>
      <td>tim.belden@enron.com</td>
      <td>953136</td>
      <td>17355</td>
      <td>484</td>
      <td>228</td>
      <td>108</td>
      <td>...</td>
      <td>NaN</td>
      <td>210698</td>
      <td>True</td>
      <td>157569</td>
      <td>NaN</td>
      <td>213999</td>
      <td>5521</td>
      <td>7991</td>
      <td>5501630</td>
      <td>1110705</td>
    </tr>
    <tr>
      <th>BELFER ROBERT</th>
      <td>NaN</td>
      <td>-102500</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>44093</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102500</td>
      <td>-44093</td>
    </tr>
    <tr>
      <th>BERBERIAN DAVID</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>david.berberian@enron.com</td>
      <td>1624396</td>
      <td>11892</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>869220</td>
      <td>NaN</td>
      <td>216582</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228474</td>
      <td>2493616</td>
    </tr>
    <tr>
      <th>BERGSIEKER RICHARD P</th>
      <td>250000</td>
      <td>NaN</td>
      <td>-485813</td>
      <td>NaN</td>
      <td>rick.bergsieker@enron.com</td>
      <td>NaN</td>
      <td>59175</td>
      <td>59</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>180250</td>
      <td>427316</td>
      <td>False</td>
      <td>659249</td>
      <td>NaN</td>
      <td>187922</td>
      <td>233</td>
      <td>383</td>
      <td>618850</td>
      <td>659249</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>137864</td>
      <td>sanjay.bhatnagar@enron.com</td>
      <td>2604490</td>
      <td>NaN</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>137864</td>
      <td>False</td>
      <td>-2604490</td>
      <td>15456290</td>
      <td>NaN</td>
      <td>463</td>
      <td>523</td>
      <td>15456290</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BIBI PHILIPPE A</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>philippe.bibi@enron.com</td>
      <td>1465734</td>
      <td>38559</td>
      <td>40</td>
      <td>23</td>
      <td>8</td>
      <td>...</td>
      <td>369721</td>
      <td>425688</td>
      <td>False</td>
      <td>378082</td>
      <td>NaN</td>
      <td>213625</td>
      <td>1336</td>
      <td>1607</td>
      <td>2047593</td>
      <td>1843816</td>
    </tr>
    <tr>
      <th>BLACHMAN JEREMY M</th>
      <td>850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeremy.blachman@enron.com</td>
      <td>765313</td>
      <td>84208</td>
      <td>14</td>
      <td>25</td>
      <td>2</td>
      <td>...</td>
      <td>831809</td>
      <td>272</td>
      <td>False</td>
      <td>189041</td>
      <td>NaN</td>
      <td>248546</td>
      <td>2326</td>
      <td>2475</td>
      <td>2014835</td>
      <td>954354</td>
    </tr>
    <tr>
      <th>BLAKE JR. NORMAN P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-113784</td>
      <td>113784</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1279</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1279</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BOWEN JR RAYMOND M</th>
      <td>1350000</td>
      <td>NaN</td>
      <td>-833</td>
      <td>NaN</td>
      <td>raymond.bowen@enron.com</td>
      <td>NaN</td>
      <td>65907</td>
      <td>27</td>
      <td>140</td>
      <td>15</td>
      <td>...</td>
      <td>974293</td>
      <td>1621</td>
      <td>True</td>
      <td>252055</td>
      <td>NaN</td>
      <td>278601</td>
      <td>1593</td>
      <td>1858</td>
      <td>2669589</td>
      <td>252055</td>
    </tr>
    <tr>
      <th>BROWN MICHAEL</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>michael.brown@enron.com</td>
      <td>NaN</td>
      <td>49288</td>
      <td>41</td>
      <td>13</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>761</td>
      <td>1486</td>
      <td>49288</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BUCHANAN HAROLD G</th>
      <td>500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>john.buchanan@enron.com</td>
      <td>825464</td>
      <td>600</td>
      <td>125</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>304805</td>
      <td>1215</td>
      <td>False</td>
      <td>189041</td>
      <td>NaN</td>
      <td>248017</td>
      <td>23</td>
      <td>1088</td>
      <td>1054637</td>
      <td>1014505</td>
    </tr>
    <tr>
      <th>BUTTS ROBERT H</th>
      <td>750000</td>
      <td>NaN</td>
      <td>-75000</td>
      <td>NaN</td>
      <td>bob.butts@enron.com</td>
      <td>NaN</td>
      <td>9410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>175000</td>
      <td>150656</td>
      <td>False</td>
      <td>417619</td>
      <td>NaN</td>
      <td>261516</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1271582</td>
      <td>417619</td>
    </tr>
    <tr>
      <th>BUY RICHARD B</th>
      <td>900000</td>
      <td>649584</td>
      <td>-694862</td>
      <td>NaN</td>
      <td>rick.buy@enron.com</td>
      <td>2542813</td>
      <td>NaN</td>
      <td>1053</td>
      <td>156</td>
      <td>71</td>
      <td>...</td>
      <td>769862</td>
      <td>400572</td>
      <td>False</td>
      <td>901657</td>
      <td>NaN</td>
      <td>330546</td>
      <td>2333</td>
      <td>3523</td>
      <td>2355702</td>
      <td>3444470</td>
    </tr>
    <tr>
      <th>CALGER CHRISTOPHER F</th>
      <td>1250000</td>
      <td>NaN</td>
      <td>-262500</td>
      <td>NaN</td>
      <td>christopher.calger@enron.com</td>
      <td>NaN</td>
      <td>35818</td>
      <td>144</td>
      <td>199</td>
      <td>25</td>
      <td>...</td>
      <td>375304</td>
      <td>486</td>
      <td>True</td>
      <td>126027</td>
      <td>NaN</td>
      <td>240189</td>
      <td>2188</td>
      <td>2598</td>
      <td>1639297</td>
      <td>126027</td>
    </tr>
    <tr>
      <th>CARTER REBECCA C</th>
      <td>300000</td>
      <td>NaN</td>
      <td>-159792</td>
      <td>NaN</td>
      <td>rebecca.carter@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>29</td>
      <td>7</td>
      <td>...</td>
      <td>75000</td>
      <td>540</td>
      <td>False</td>
      <td>307301</td>
      <td>-307301</td>
      <td>261809</td>
      <td>196</td>
      <td>312</td>
      <td>477557</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAUSEY RICHARD A</th>
      <td>1000000</td>
      <td>NaN</td>
      <td>-235000</td>
      <td>NaN</td>
      <td>richard.causey@enron.com</td>
      <td>NaN</td>
      <td>30674</td>
      <td>49</td>
      <td>58</td>
      <td>12</td>
      <td>...</td>
      <td>350000</td>
      <td>307895</td>
      <td>True</td>
      <td>2502063</td>
      <td>NaN</td>
      <td>415189</td>
      <td>1585</td>
      <td>1892</td>
      <td>1868758</td>
      <td>2502063</td>
    </tr>
    <tr>
      <th>CHAN RONNIE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-98784</td>
      <td>98784</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>32460</td>
      <td>-32460</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CHRISTODOULOU DIOMEDES</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>diomedes.christodoulou@enron.com</td>
      <td>5127155</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>950730</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6077885</td>
    </tr>
    <tr>
      <th>CLINE KENNETH W</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>662086</td>
      <td>-472568</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>189518</td>
    </tr>
    <tr>
      <th>COLWELL WESLEY</th>
      <td>1200000</td>
      <td>27610</td>
      <td>-144062</td>
      <td>NaN</td>
      <td>wes.colwell@enron.com</td>
      <td>NaN</td>
      <td>16514</td>
      <td>40</td>
      <td>240</td>
      <td>11</td>
      <td>...</td>
      <td>NaN</td>
      <td>101740</td>
      <td>True</td>
      <td>698242</td>
      <td>NaN</td>
      <td>288542</td>
      <td>1132</td>
      <td>1758</td>
      <td>1490344</td>
      <td>698242</td>
    </tr>
    <tr>
      <th>CORDES WILLIAM R</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>bill.cordes@enron.com</td>
      <td>651850</td>
      <td>NaN</td>
      <td>12</td>
      <td>10</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>386335</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>58</td>
      <td>764</td>
      <td>NaN</td>
      <td>1038185</td>
    </tr>
    <tr>
      <th>COX DAVID</th>
      <td>800000</td>
      <td>NaN</td>
      <td>-41250</td>
      <td>NaN</td>
      <td>chip.cox@enron.com</td>
      <td>117551</td>
      <td>27861</td>
      <td>33</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>NaN</td>
      <td>494</td>
      <td>False</td>
      <td>378082</td>
      <td>NaN</td>
      <td>314288</td>
      <td>71</td>
      <td>102</td>
      <td>1101393</td>
      <td>495633</td>
    </tr>
    <tr>
      <th>CUMBERLAND MICHAEL S</th>
      <td>325000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22344</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>275000</td>
      <td>713</td>
      <td>False</td>
      <td>207940</td>
      <td>NaN</td>
      <td>184899</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>807956</td>
      <td>207940</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SAVAGE FRANK</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-121284</td>
      <td>125034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3750</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SCRIMSHAW MATTHEW</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>matthew.scrimshaw@enron.com</td>
      <td>759557</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>759557</td>
    </tr>
    <tr>
      <th>SHANKMAN JEFFREY A</th>
      <td>2000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeffrey.shankman@enron.com</td>
      <td>1441898</td>
      <td>178979</td>
      <td>2681</td>
      <td>94</td>
      <td>83</td>
      <td>...</td>
      <td>554422</td>
      <td>1191</td>
      <td>False</td>
      <td>630137</td>
      <td>NaN</td>
      <td>304110</td>
      <td>1730</td>
      <td>3221</td>
      <td>3038702</td>
      <td>2072035</td>
    </tr>
    <tr>
      <th>SHAPIRO RICHARD S</th>
      <td>650000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>richard.shapiro@enron.com</td>
      <td>607837</td>
      <td>137767</td>
      <td>1215</td>
      <td>74</td>
      <td>65</td>
      <td>...</td>
      <td>NaN</td>
      <td>705</td>
      <td>False</td>
      <td>379164</td>
      <td>NaN</td>
      <td>269076</td>
      <td>4527</td>
      <td>15149</td>
      <td>1057548</td>
      <td>987001</td>
    </tr>
    <tr>
      <th>SHARP VICTORIA T</th>
      <td>600000</td>
      <td>187469</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>vicki.sharp@enron.com</td>
      <td>281073</td>
      <td>116337</td>
      <td>136</td>
      <td>24</td>
      <td>6</td>
      <td>...</td>
      <td>422158</td>
      <td>2401</td>
      <td>False</td>
      <td>213063</td>
      <td>NaN</td>
      <td>248146</td>
      <td>2477</td>
      <td>3136</td>
      <td>1576511</td>
      <td>494136</td>
    </tr>
    <tr>
      <th>SHELBY REX</th>
      <td>200000</td>
      <td>NaN</td>
      <td>-4167</td>
      <td>NaN</td>
      <td>rex.shelby@enron.com</td>
      <td>1624396</td>
      <td>22884</td>
      <td>39</td>
      <td>13</td>
      <td>14</td>
      <td>...</td>
      <td>NaN</td>
      <td>1573324</td>
      <td>True</td>
      <td>869220</td>
      <td>NaN</td>
      <td>211844</td>
      <td>91</td>
      <td>225</td>
      <td>2003885</td>
      <td>2493616</td>
    </tr>
    <tr>
      <th>SHERRICK JEFFREY B</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeffrey.sherrick@enron.com</td>
      <td>1426469</td>
      <td>NaN</td>
      <td>25</td>
      <td>39</td>
      <td>18</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>405999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>583</td>
      <td>613</td>
      <td>NaN</td>
      <td>1832468</td>
    </tr>
    <tr>
      <th>SHERRIFF JOHN R</th>
      <td>1500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>john.sherriff@enron.com</td>
      <td>1835558</td>
      <td>NaN</td>
      <td>92</td>
      <td>28</td>
      <td>23</td>
      <td>...</td>
      <td>554422</td>
      <td>1852186</td>
      <td>False</td>
      <td>1293424</td>
      <td>NaN</td>
      <td>428780</td>
      <td>2103</td>
      <td>3187</td>
      <td>4335388</td>
      <td>3128982</td>
    </tr>
    <tr>
      <th>SKILLING JEFFREY K</th>
      <td>5600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>jeff.skilling@enron.com</td>
      <td>19250000</td>
      <td>29336</td>
      <td>108</td>
      <td>88</td>
      <td>30</td>
      <td>...</td>
      <td>1920000</td>
      <td>22122</td>
      <td>True</td>
      <td>6843672</td>
      <td>NaN</td>
      <td>1111258</td>
      <td>2042</td>
      <td>3627</td>
      <td>8682716</td>
      <td>26093672</td>
    </tr>
    <tr>
      <th>STABLER FRANK</th>
      <td>500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>frank.stabler@enron.com</td>
      <td>NaN</td>
      <td>16514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>356071</td>
      <td>False</td>
      <td>511734</td>
      <td>NaN</td>
      <td>239502</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1112087</td>
      <td>511734</td>
    </tr>
    <tr>
      <th>SULLIVAN-SHAKLOVITZ COLLEEN</th>
      <td>100000</td>
      <td>181993</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1362375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>554422</td>
      <td>162</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>162779</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>999356</td>
      <td>1362375</td>
    </tr>
    <tr>
      <th>SUNDE MARTIN</th>
      <td>700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>marty.sunde@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>37</td>
      <td>13</td>
      <td>...</td>
      <td>476451</td>
      <td>111122</td>
      <td>False</td>
      <td>698920</td>
      <td>NaN</td>
      <td>257486</td>
      <td>2565</td>
      <td>2647</td>
      <td>1545059</td>
      <td>698920</td>
    </tr>
    <tr>
      <th>TAYLOR MITCHELL S</th>
      <td>600000</td>
      <td>227449</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>mitchell.taylor@enron.com</td>
      <td>3181250</td>
      <td>NaN</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>563798</td>
      <td>NaN</td>
      <td>265214</td>
      <td>300</td>
      <td>533</td>
      <td>1092663</td>
      <td>3745048</td>
    </tr>
    <tr>
      <th>THORN TERENCE H</th>
      <td>NaN</td>
      <td>16586</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>terence.thorn@enron.com</td>
      <td>4452476</td>
      <td>46145</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>200000</td>
      <td>426629</td>
      <td>False</td>
      <td>365320</td>
      <td>NaN</td>
      <td>222093</td>
      <td>73</td>
      <td>266</td>
      <td>911453</td>
      <td>4817796</td>
    </tr>
    <tr>
      <th>TILNEY ELIZABETH A</th>
      <td>300000</td>
      <td>NaN</td>
      <td>-575000</td>
      <td>NaN</td>
      <td>elizabeth.tilney@enron.com</td>
      <td>591250</td>
      <td>NaN</td>
      <td>19</td>
      <td>10</td>
      <td>11</td>
      <td>...</td>
      <td>275000</td>
      <td>152055</td>
      <td>False</td>
      <td>576792</td>
      <td>NaN</td>
      <td>247338</td>
      <td>379</td>
      <td>460</td>
      <td>399393</td>
      <td>1168042</td>
    </tr>
    <tr>
      <th>UMANOFF ADAM S</th>
      <td>788750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>adam.umanoff@enron.com</td>
      <td>NaN</td>
      <td>53122</td>
      <td>18</td>
      <td>12</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>288589</td>
      <td>41</td>
      <td>111</td>
      <td>1130461</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>URQUHART JOHN A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-36666</td>
      <td>36666</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228656</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228656</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WAKEHAM JOHN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>109298</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>103773</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>213071</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WALLS JR ROBERT H</th>
      <td>850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>rob.walls@enron.com</td>
      <td>4346544</td>
      <td>50936</td>
      <td>146</td>
      <td>17</td>
      <td>0</td>
      <td>...</td>
      <td>540751</td>
      <td>2</td>
      <td>False</td>
      <td>1552453</td>
      <td>NaN</td>
      <td>357091</td>
      <td>215</td>
      <td>671</td>
      <td>1798780</td>
      <td>5898997</td>
    </tr>
    <tr>
      <th>WALTERS GARETH W</th>
      <td>NaN</td>
      <td>53625</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1030329</td>
      <td>33785</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>87410</td>
      <td>1030329</td>
    </tr>
    <tr>
      <th>WASAFF GEORGE</th>
      <td>325000</td>
      <td>831299</td>
      <td>-583325</td>
      <td>NaN</td>
      <td>george.wasaff@enron.com</td>
      <td>1668260</td>
      <td>NaN</td>
      <td>30</td>
      <td>22</td>
      <td>7</td>
      <td>...</td>
      <td>200000</td>
      <td>1425</td>
      <td>False</td>
      <td>388167</td>
      <td>NaN</td>
      <td>259996</td>
      <td>337</td>
      <td>400</td>
      <td>1034395</td>
      <td>2056427</td>
    </tr>
    <tr>
      <th>WESTFAHL RICHARD K</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-10800</td>
      <td>NaN</td>
      <td>dick.westfahl@enron.com</td>
      <td>NaN</td>
      <td>51870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>256191</td>
      <td>401130</td>
      <td>False</td>
      <td>384930</td>
      <td>NaN</td>
      <td>63744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>762135</td>
      <td>384930</td>
    </tr>
    <tr>
      <th>WHALEY DAVID A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98718</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98718</td>
    </tr>
    <tr>
      <th>WHALLEY LAWRENCE G</th>
      <td>3000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>greg.whalley@enron.com</td>
      <td>3282960</td>
      <td>57838</td>
      <td>556</td>
      <td>186</td>
      <td>24</td>
      <td>...</td>
      <td>808346</td>
      <td>301026</td>
      <td>False</td>
      <td>2796177</td>
      <td>NaN</td>
      <td>510364</td>
      <td>3920</td>
      <td>6019</td>
      <td>4677574</td>
      <td>6079137</td>
    </tr>
    <tr>
      <th>WHITE JR THOMAS E</th>
      <td>450000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>thomas.white@enron.com</td>
      <td>1297049</td>
      <td>81353</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1085463</td>
      <td>False</td>
      <td>13847074</td>
      <td>NaN</td>
      <td>317543</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1934359</td>
      <td>15144123</td>
    </tr>
    <tr>
      <th>WINOKUR JR. HERBERT S</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-25000</td>
      <td>108579</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84992</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WODRASKA JOHN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>john.wodraska@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>189583</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>189583</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WROBEL BRUCE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>139130</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>139130</td>
    </tr>
    <tr>
      <th>YEAGER F SCOTT</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>scott.yeager@enron.com</td>
      <td>8308552</td>
      <td>53947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>147950</td>
      <td>True</td>
      <td>3576206</td>
      <td>NaN</td>
      <td>158403</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>360300</td>
      <td>11884758</td>
    </tr>
    <tr>
      <th>YEAP SOON</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>192758</td>
      <td>55097</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55097</td>
      <td>192758</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 21 columns</p>
</div>



In regards to the financial information, I defined outliers as having values that are more than 3 standard deviations from the mean value for the group. This is not the traditional definition or criteria for an outlier which is 1.5 times the interquartile range below the first quartile or Above the third quartile. I used my definition after I have replaced missing values with zero. Using this definition of outliers there are 25 people in the data set that are financial outliers:


    finance = ['salary',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'other',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive']
    
    from scipy import stats
    #Use unique because some people are financial outliers in multiple variables
    data[np.abs(stats.zscore(data[finance].replace('NaN',0.0))) > 3].index.unique()




    array(['ALLEN PHILLIP K', 'BELDEN TIMOTHY N', 'BHATNAGAR SANJAY',
           'BLAKE JR. NORMAN P', 'FREVERT MARK A', 'GRAMM WENDY L',
           'HANNON KEVIN P', 'HIRKO JOSEPH', 'HORTON STANLEY C',
           'HUMPHREY GENE E', 'JAEDICKE ROBERT', 'LAVORATO JOHN J',
           'LAY KENNETH L', 'LEMAISTRE CHARLES', 'MARTIN AMANDA K',
           'MCCLELLAN GEORGE', 'MENDELSOHN JOHN', 'PAI LOU L',
           'RICE KENNETH D', 'SAVAGE FRANK', 'SHANKMAN JEFFREY A',
           'SKILLING JEFFREY K', 'URQUHART JOHN A', 'WAKEHAM JOHN',
           'WHITE JR THOMAS E', 'WINOKUR JR. HERBERT S'], dtype=object)



This accounts for 17% of the data being considered an outlier and also contains 33% of the POI. Ultimately I decided that financial outliers were relevant information, and decided not to remove them from the data for this analysis.

##New Feature

I decided to create two features I thought were be informative to the data: the ratio of emails received from a POI to the total number of emails received and the ratio of emails sent to a POI to the total number of emails sent.
I thought these features would be more informative than the total number of emails sent or received from a POI because it normalizes by how active or how popular a person is.

If a person sends 10 emails, and 5 of them are to a POI, that seems more relevant than if a person sends 1000 emails and 5 of them are to a POI. A person who sends 50% of their emails to a person of interest seems more suspect than a person who only sends 0.5% of their emails to a POI.

The inverse is not true, however. If a person received 10 emails from a POI, that is as relevant regardless if the person receives 20 emails or 2000 emails. The important idea is that how often is a POI is contacting this person, and how does that affect the likelihood that a person is also a POI. This ratio does not capture that idea, but I created it because I was willing to be proven wrong.


    data.from_this_person_to_poi = data.from_this_person_to_poi.astype(float)
    data.from_poi_to_this_person = data.from_poi_to_this_person.astype(float)
    data.to_messages = data.to_messages.astype(float)
    data.from_messages = data.from_messages.astype(float)
    data['from_this_person_to_poi_ratio'] = data.from_this_person_to_poi/data.to_messages
    data['from_poi_to_this_person_ratio'] = data.from_poi_to_this_person/data.from_messages
    data = data.replace('NaN',0.0)

After the creation of these features I used sklearn's Pipeline, FeatureUnion, and GridSearch to search through a combintation of variable to find the set of features that gave the best performance for a Decision Tree Classifier.  I search between 1 and 9 of the best features using a 5 folds stratified cross validation to gauge performance for each set.  I am using a f1 score to maximize the weight combintation of recall and precision. 


    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.grid_search import GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve
    from sklearn.learning_curve import learning_curve
    
    tru = data.poi.values
    trn = data.drop(['poi','email_address'],axis=1).values
    clf = DecisionTreeClassifier()
    
    #pca = PCA(n_components=2)
    selection = SelectKBest(k=1)
    #combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    combined_features = FeatureUnion([("univ_select", selection)])
    
    X_features = combined_features.fit(trn, tru).transform(trn)
    
    pipeline = Pipeline([("features", combined_features), ("clf", clf)])
    
    #param_grid = dict(features__pca__n_components=range(0,15),
    #                  features__univ_select__k=range(1,10))
    
    param_grid = dict(features__univ_select__k=range(1,10))
    
    skf = StratifiedKFold(tru, n_folds=5)
    grid_search = GridSearchCV(pipeline, cv=skf, param_grid=param_grid, scoring='f1')
    grid_search.fit(trn, tru)
    print grid_search.best_score_
    print grid_search.best_params_
    var_scores = grid_search.best_estimator_.steps[0][1].get_params()['univ_select'].scores_
    print var_scores
    num_features = grid_search.best_params_['features__univ_select__k']
    best_features = data.drop(['poi','email_address'],axis=1).columns[np.argsort(var_scores)[::-1][:num_features]]
    best_features

    0.36813973064
    {'features__univ_select__k': 3}
    [ 21.06000171   0.21705893  11.59554766   2.10765594  25.09754153
       6.23420114   0.1641645    5.34494152   2.42650813   7.2427304
      10.07245453   4.24615354   9.34670079   0.06498431  18.57570327
       8.74648553   1.69882435   8.87383526  24.46765405   4.16908382
       5.20965022]





    Index([u'exercised_stock_options', u'total_stock_value', u'bonus'], dtype='object')



##Result of Search

The search is finished and the best f1 score of 0.368 with 3 'best features'.  Because of the random shuffling involved in the scoring, it is possible to get a variety in number and combintations of best features.  

The best features for predicting if a person is a person of interest are financal features: Exercised stock options, total stock value, and bonus.  Even when the number of best features fluxuate, these 3 are always among them.   

##Turning Model
Using the paramemters from the above search we will now tune the classifier to optimize its performance. I am searching through the spliting and depth criteria for the Decision Tree Classifier.  



    tru = data.poi.values
    trn = data[best_features].values
    clf = DecisionTreeClassifier()
    pipeline = Pipeline([("clf", clf)])
    
    param_grid = dict(clf__criterion=("gini","entropy"),
                      clf__min_samples_split=[1,2,4,8,16,32],
                       clf__min_samples_leaf=[1,2,4,8,16,32],
                       clf__max_depth=[None,1,2,4,8,16,32])
    
    skf = StratifiedKFold(tru, n_folds=5)
    grid_search = GridSearchCV(pipeline, cv=skf, param_grid=param_grid, scoring='f1')
    grid_search.fit(trn, tru)
    print grid_search.best_params_
    print grid_search.best_score_
    best_clf = grid_search.best_estimator_
    best_clf

    {'clf__criterion': 'gini', 'clf__max_depth': 32, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 1}
    0.47075617284





    Pipeline(steps=[('clf', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=1, min_weight_fraction_leaf=0.0,
                random_state=None, splitter='best'))])



The best parameters for the Decision Tree Classifier has a max depth of 32, a min sample leaf size of 1, and requires at least 1 values to split the tree when fitting.  These results produced the best averge f1 score of 0.47.   

##Importance of Validation

Validation is an attempt to confirm that a model will give reasonable or consistent results on new, untrained data. A classic mistake is to test the results of a model on the data used to train the model. This is no doubt give the best possible score, but can over fit the data leading to less than desired results on new data. Validation protects against this mistake by training the model on one set of data and testing on yet another.

I used 5-Fold Statfied Cross Validation for investigating and comparing algorithms in this analysis. This is where there the algorithm is trained on the on the data 5 times using 80% of the data as a training set and 20% of the data as a testing set.  Each set has approxiamately the same ratio of positive and negative examples of POI.  Each time this is done, the training and testing set are shuffled to create a new 80/20 split on the data. I then used the average performance as an estimate of its performance on new data.

## Final Performance

Using the Udacity 'test_classifier' function I evaluate my classifer to see if the recall and precision scores are both greater than 0.3.



    #tt = pd.DataFrame(pca.fit_transform(trn),index=data.index)
    #data2 = data.join(tt)
    #print data2.columns
    my_data = data.transpose().to_dict()
    features =  best_features.tolist()
    features = ['poi']+features
    test_classifier(best_clf, my_data, features)

    Pipeline(steps=[('clf', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=32,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=1, min_weight_fraction_leaf=0.0,
                random_state=None, splitter='best'))])
    	Accuracy: 0.80492	Precision: 0.37066	Recall: 0.38400	F1: 0.37721	F2: 0.38125
    	Total predictions: 13000	True positives:  768	False positives: 1304	False negatives: 1232	True negatives: 9696
    


The results of the tuned classifier are shown above.   The average precision of the 13000 prediction is 37% and the average recall is 38%.   The precision value is that out of all predictions of people being a POI, 38% of them are actually POI.  The recall value is that out of all of the POI, 37% are actually predicted to be POI by my tuned classifier.  


    
