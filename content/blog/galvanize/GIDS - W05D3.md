
#Galvanize Immersive Data Science

##Week 5 - Day 3

Today we covered natural language processing and and NaiveBayes methods of machine learning.   We were introduced to the NLTK package of python, and the sklearn text processing packagages.

##Yelp
We started of with a daily quiz, per usual.   We were suppose to signup for the Yelp! API and find out how many gastropubs are in San Francisco.  

Looking at the Yelp! API I see there is a 'category_filter' term and one of the valid inputs is 'gastropubs'.   

Yelp, unlike any service we used before, requires oAuth.   This was the most difficult part of the quiz for me.  Not difficult in hard, but in that I had zero experience and it was a tall order to get it figured out in a short time.  One convoluting factor was that Yelp! gives you your tokens and keys, but they do not work for the first 10 minutes if you have a new account.   Once I refreshed them, everything worked beautifully



Today is my first day attending Galvanize's Immersive Data Science Program in San Francisco, CA.   The program is a 12 week program that is approximately 10 hours a day of learning and activities to reinforce and refine the learning.   I am very excited to be a part of this program.



    import json
    import urllib2
    import oauth2
    
    
    url_params = {'category_filter' : 'gastropubs', 'location':'San Francisco'}
    url = 'http://api.yelp.com/v2/search/?'
    consumer = oauth2.Consumer(CONSUMER_KEY, CONSUMER_SECRET)
    oauth_request = oauth2.Request(method="GET", url=url, parameters=url_params)
    
    oauth_request.update(
        {
            'oauth_nonce': oauth2.generate_nonce(),
            'oauth_timestamp': oauth2.generate_timestamp(),
            'oauth_token': TOKEN,
            'oauth_consumer_key': CONSUMER_KEY
        }
    )
    token = oauth2.Token(TOKEN, TOKEN_SECRET)
    oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
    signed_url = oauth_request.to_url()
    
    conn = urllib2.urlopen(signed_url, None)
    try:
        response = json.loads(conn.read())
        print response['total']
        print len(response['businesses'])
    finally:
        conn.close()

The YELP! API limits you to twenty detailed responses, but it does let you know that there are 48 Businesses in San Francisco listed in the gastropub category.

##Natural Language Processing

The goal of today's morning sprint is to use a subset of New York Times articles and attempt to find related articles.   We are also starting out by going through the process ourselves, then using sklearn's packages for processing text.

First we will need to pull articles from the database.


    from pymongo import MongoClient
    client = MongoClient()
    db = client.nyt_dump
    coll = db.articles
    docs = []
    labels = []
    
    for i,d in enumerate(coll.find()):
        doc = ''.join(d['content'])
        
        doc = doc.encode('utf8', 'replace').decode('utf8')
        doc = doc.strip().lower()
        if len(doc) > 0:
            docs.append(doc)
            labels.append(d['section_name'])
    print len(docs),len(labels)

    984 984


Of the 999 documents that are in our subset, a few are empty of text.  The web scraper must not have found the text when pulling it from the ny times.   We will drop these words from our corpus.  

For our purpose we have going to take the words in the NY Times articles and lemmatize them, then stem them.   The lemmatize them will take words that are in difference contexts but have similar meaning to a signular word.  The stemmmer will remove other 'decorative' aspect of the word to capture just the word.   

The reason I am lemmatize this is that NY times is documented of having one of the highest vocabularities online, which means that there will be a pleathor of word choice for similar ideas.   If we want to capture related articles, then we will need to attempt to captures words that are similar.

We also removed the common english stopwords.  Because of the unicode characters, I will be using a regular expression tokenizer.  


    from nltk import word_tokenize, wordpunct_tokenize,RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk import Text
    import re
    
    sw = set(stopwords.words('english'))
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    reg = RegexpTokenizer(r'\w+',flags=re.UNICODE)
    stemmer_words = []
    doc_tokens = []
    for doc in docs:
        tokens = []
        for t in reg.tokenize(doc):
            if t not in sw:
                l = wordnet.lemmatize(t)
                s = snowball.stem(l)
                stemmer_words.append([porter.stem(t),snowball.stem(t),wordnet.lemmatize(t),snowball.stem(wordnet.lemmatize(t))])
                tokens.append(s)      
        doc_tokens.append(tokens) 

If you inspect the above code you noticed that I make a list of each word and the results of the different stemmers.  We can see for a random selection of 20 words that the the stemming all lead to the same outcome.   The lemmatizer gave different results, but once that result was fed to the stemmer it was identical.   


    import numpy as np
    for i in range(20):
        print stemmer_words[np.random.randint(len(stemmer_words))]

    [u'fashion', u'fashion', u'fashion', u'fashion']
    [u'mr', u'mr', u'mr', u'mr']
    [u'could', u'could', u'could', u'could']
    [u'u', u'u', u'u', u'u']
    [u'3', u'3', u'3', u'3']
    [u'day', u'day', u'day', u'day']
    [u'flagship\xe2', u'flagship\xe2', u'flagship\xe2', u'flagship\xe2']
    [u'germani', u'germani', u'germany', u'germani']
    [u'bleak', u'bleak', u'bleak', u'bleak']
    [u'one', u'one', u'one', u'one']
    [u'would', u'would', u'would', u'would']
    [u'make', u'make', u'making', u'make']
    [u'point', u'point', u'point', u'point']
    [u'also', u'also', u'also', u'also']
    [u'spi', u'spi', u'spy', u'spi']
    [u'sergeant', u'sergeant', u'sergeant', u'sergeant']
    [u'divorc', u'divorc', u'divorced', u'divorc']
    [u'deni', u'deni', u'denied', u'deni']
    [u'don\xe2', u'don\xe2', u'don\xe2', u'don\xe2']
    [u'battl', u'battl', u'battle', u'battl']


##Bag of Words

Now that we have stemmed all the words in our document, we likely have duplicates.  We want to get a vocabulary of our corpus, the 984 new york times articles.   We can do this with a simple list comprehension to flatten the list, perform a set operation, then turn it back into a list.  I will sort it just because.

Because we will be cycling through this list, it will be able to look up the location of a particlar word.  We will also make a dictionary of each word and its index in the list.  


    bag = [item for row in doc_tokens for item in row]
    bag = set(bag)
    bag = sorted(list(bag))
    
    bag_dict = dict()
    for i,word in enumerate(bag):
        bag_dict[word] = i
        
    print len(bag),len(bag_dict)

    26505 26505


The next step in finding similarity between documents is finding the frequency of each word in each document.   For our purposes, each document has a potential of having some number of the 26505 words in our bag, and we have 984 documents.   That means we need a matrix that has a shape of (984,26505)



    #make word frequency
    word_freq = np.zeros((len(doc_tokens),len(bag_dict)))
    
    #iterate through each doc
    for i, tokens in enumerate(doc_tokens):
        #iterate through the word tokens of the doc
        for token in tokens:
            #us the lookup dictionary and add one to that word position
            word_freq[i,bag_dict[token]] +=1

Now that we have the word frequency, we need a way to find if this is a relevant similarity between two documents.  If a word is popular in all documents, then it is not particularly special.  If a word is only freqnency in a 2 or 3 docs, there is an increased likelihood that they are related.

Unlike the word frequency, the document frequency does not care how many times the word appears in the document.  It is a boolean measure.   Either 'USS' appears in the document or not.  It does not mater that it apears 10 times in an article about a naval yard.  


    #Convert the word frequency to a bool, then an int.  Sum along the documents.
    #Use the sum along the documents to identify how many documents the word apears in
    #Divide by the number of documents to get the document frequency for each word
    doc_freq = np.sum((word_freq>0).astype(int),axis=0).astype(float)/word_freq.shape[0]
    doc_freq.shape




    (26505,)



One problem we have at the moment is that we have word frequencies, but articles have different length.  They also have a different number of words.   Two articles on the same topic many have different lengths, thus different word counts.  The relative frequency is not necessarily similary, but if you just copied and pasted the same document it should also not be different.  To address this we need to normalize the rows in our word freqency matrix.


    #calculate the row norms
    row_norms = np.linalg.norm(word_freq,axis=1)
    #if we have empty documents we need this line (but we dont!)
    row_norms[row_norms==0] = 1
    #reshape so that we can use broadcast division in numpy
    row_norms = row_norms.reshape(row_norms.shape[0],1)
    #have a row normalized row frequency
    norm_word_freq = word_freq/row_norms

The final feature vector we are constructing is the tf-idf vector that is the product of the normalized term frequency times the log of one over the document frquency of the term.   This operation changes the scale of the vector, so they must be renormalized after.   

There are theoretical reason for the log, but I honestly have not understood them yet.  



    #make tfidf vector
    tfidf = norm_word_freq*np.log(1/doc_freq)
    #find norms of each row
    tfidf_norms = np.linalg.norm(tfidf,axis=1)
    #needs this line for empty documents
    tfidf_norms[tfidf_norms==0] = 1
    #reshape for numpy broadcast division
    tfidf_norms = tfidf_norms.reshape(tfidf_norms.shape[0],1)
    #normlize tfidf
    norm_tfidf = tfidf/tfidf_norms

Now that we have these tfidf vectors, we can find document vectors that are similar to each other.   Similarity in a normalized vector space is similarity in direction.  This is known as cosign similarity.

I am going to cycle through the 984x984 similarity calculations of similarity by the linear_kernel function of sklearn.   If the vectors are less than 30 degrees appart in this word-vector space, I will find the second most similar article.   I skip the first most because that will be its self.   They will have a similarity of 1.  


    from sklearn.metrics.pairwise import linear_kernel
    cosine_similarities = linear_kernel(norm_tfidf, norm_tfidf)
    for i, row in enumerate(cosine_similarities):
        if row[np.argsort(row)[::-1][1:2]][0] > 0.707:
            print i,np.argsort(row)[::-1][1:2][0]

    7 158
    113 712
    158 7
    174 521
    187 960
    240 251
    251 251
    260 425
    289 474
    335 372
    372 372
    425 260
    431 810
    439 615
    444 445
    445 444
    463 700
    474 289
    508 508
    509 508
    521 174
    615 439
    700 463
    710 756
    712 920
    756 710
    767 767
    768 767
    810 431
    919 756
    920 712
    939 956
    956 939
    960 187


##We have a match

We can see that document 7 and 158 should be similar to each other.  I will print them out.  

As you read through them you will find that they are two different articles on the same story about crashed cruise ship, both writen by the same author.   That is awesome that it found this.    


    print docs[7]

    giglio, italy â after a costly, painstaking and potentially perilous operation to raise the battered hull of the cruise ship costa concordia, engineers said early tuesday that they had succeeded in righting the ship, removing it from two granite reefs where it ran aground last year, killing 32 people.        
    
    
    the 19-hour, highly complicated salvage operation had managed to completely rotate the ship, leaning it on an underwater platform built underneath, the engineers said.        
    âthis was an important, visible step,â franco gabrielli, head of italyâs civil protection agency, told reporters at 4 a.m., accompanied by applause from a few residents who had stayed up all night to follow the operation.        
    âthe rotation happened in the way we thought and hoped it would happen,â echoed franco porcellacchia, project manager for costa cruises, the shipâs operator. âthere is no evidence so far of any impact to the environment. if there are debris to be removed, we will do it tomorrow.â        
    as parts of the vessel emerged in the later afternoon on monday, discolored and rusting, from the waters where the concordia had languished, listing on its side, engineers said the operation would most likely take longer than initially planned.        
    salvage experts have said the dimensions of the stricken 951-foot vessel made the operation unparalleled in the annals of marine salvage, as more than 500 divers, technicians, engineers and biologists prepared the ship for what is known as âparbucklingâ to bring it upright and minimize environmental risks to giglio island, a marine sanctuary.        
    using huge jacks, cables, pulleys and specialized equipment, the salvage effort had been set to begin at first light, but a sudden storm prevented workers from moving a barge and rubber booms close to the ship.        
    three hours after work started, engineers said the first phase of the operation â easing the vessel away from its rocky perch â was going according to plan. âthese hours were the most uncertain, as we could not establish how much the hull was wedged,â said sergio girotto, project manager with micoperi, the projectâs underwater construction and offshore contractor. ânow we have to guide it into the desired position.â        
    the next phase of the salvage, engineers said, involved settling the wreck on an artificial seabed made of bags of cement next to underwater steel platforms. to achieve that, the cruise liner needed to be rotated about 65 degrees, they said. if it all goes well, the ship will be towed away and broken up for scrap by spring.        
    the operation was broadcast live on television and the internet. the italian news media portrayed the salvage as a chance for italy to revamp its image after the wreck, in which the captain fled the damaged ship and the evacuation was chaotic.        
    the leading national daily, corriere della sera, called the shipwreck âa monument to human stupidityâ and a âhumiliationâ for italy. it said it hoped that the salvage effort would provide a ânew and different storyâ for the country.        
    the shipâs captain, francesco schettino, is scheduled to go on trial this fall on charges of multiple manslaughter, causing a shipwreck and abandoning â the vessel before everyone was safe. he has denied wrongdoing. a company official and four crew members have already pleaded guilty to reduced charges.        
    preparations for the salvage operation took 14 months, and the cost has increased to $799 million from $300 million and could rise further, according to costa cruises. the costa concordia has been stabilized with anchors and cement bags, and underwater platforms have been built on the port side. salvage crews used pulleys, strand jacks and steel cables placed on nine caissons attached to the left side of the ship to slowly dislodge it on monday from the two rocks where it had been resting.        
    the operation was monitored by engineers and remotely operated vehicle pilots from a control room on a barge close to the bow of the ship. if images or sonar showed dangerous twisting, the technicians could adjust the process. at a command center onshore, engineers could intervene if the ship did not rotate, or did not rotate properly.        
    salvage masters and the italian authorities had prepared for complications. most of the fuel was siphoned off within months of the wreck. but the vessel that once transported and entertained 4,229 people still contains chemicals and diesel fuel that could leak into the pristine waters for which giglio, a popular tourist spot, is known.        
    during the rotation process, the regionâs environmental agency took samples to monitor water quality.        
    âdetaching the ship from the rocks was the most complicated phase, which is probably why they decided to do it very cautiously,â said emilio campana, the director of the research office for naval and maritime engineering at italyâs national research council. âwe have to keep in mind that the structure is heavily damaged, and see if and how it holds together from now on.â        
    
    
    
     
    gaia pianigiani reported from giglio, and alan cowell from london.



    print docs[158]

    giglio, italy â salvage workers righted the scarred and discolored hull of the cruise ship costa concordia early tuesday after coaxing it from two granite reefs it ran aground on just off this tiny tourist island 20 months ago, killing 32 people.        
    
    
    as the vast hull slowly emerged during the complex, 19-hour salvage operation, the full extent of damage to the vessel became apparent. it looked as if a giant fist had driven into the shipâs flank.        
    shipsâ horns blared over giglioâs tiny port to celebrate the moment, and some of the islandâs 1,500 residents hugged salvage workers as they came ashore from what is likely to be seen as a bold step toward redressing some of italyâs anguish after the costa concordia, the length of three football fields, careened into the reefs on a wintry night in january 2012.        
    âthis was an important, visible step,â franco gabrielli, head of italyâs civil protection agency, told reporters at 4 a.m., accompanied by applause from a few residents who had stayed up all night to follow the operation.        
    he was echoed by franco porcellacchia, project manager for costa cruises, the shipâs operator. âthere is no evidence so far of any impact to the environment,â he said. âif there are debris to be removed, we will do it tomorrow.â        
    on tuesday morning, at a crowded news conference on giglio port, italian officials seemed almost surprised by how precisely their calculations had worked, but expressed caution about future steps to secure the vessel before it can be towed away and scrapped, probably in the spring. âthe phases to come will be just as complicated,â mr. porcellacchia said.        
    nick sloane, the salvage master, said the operation exceeded his expectations. âit was nice to see that at 4 a.m.,â he told reporters tuesday afternoon.        
    mr. sloane explained that a full survey of the damage, which he called substantial, would be possible only after italian authorities carried out their inspections and searched for the bodies of two people aboard the ship who are still missing.        
    the operation left the 951-foot ship resting on an artificial platform 90 feet below the surface, with only about a third of its once-sleek white lines visible above water. engineers said the badly damaged starboard side would need to be welded and reinforced, so that other steel chambers, known as caissons and crucial to the operation to right the ship, can be attached. the vessel will also need to be further secured to withstand winter weather, engineers said.        
    âwe will consider the operation concluded once the ship leaves giglio island,â mr. gabrielli told reporters, acknowledging that risks remained while the wreck was at sea. âweâll carry out all the needed interventions to mitigate it and allow the ship to face the next winter in secure conditions,â he said.        
    mr. sloane said he could hear workers jumping around with relief and delight as the ship was gently laid on the platform. âit was like a roller coaster,â he said.        
    the righting of the vessel did not draw universal applause.        
    âi donât necessarily think that this is a great victory for italy, maybe for the italian and american companies involved,â said suzanne kmetyko, 50, a tourist from austria who has visited the island for the past seven years.        
    âand the real success for the island will come only once people around the world will stop remembering it for the shipwreck rather than for its natural beauty,â she said. âa long way to go.â        
    italian news media, by contrast, portrayed the salvage, broadcast live on television and the internet, as a chance for the country to revamp its image after the wreck, in which the captain fled the damaged ship and the evacuation was chaotic. the leading national daily, corriere della sera, called the shipwreck âa monument to human stupidityâ and a âhumiliationâ for italy. it said it hoped that the salvage effort would provide a ânew and different storyâ for the country.        
    it was not always clear what that story would be.        
    as parts of the concordia emerged in the late afternoon on monday, stained and rusting, from the waters where the vessel had languished, engineers had said the operation would most likely take longer than initially planned.        
    the sheer size of the concordia had created what salvage specialists called unparalleled challenges not only to right the ship but also to protect giglio island, a marine sanctuary, from environmental hazard.        
    salvage workers used huge jacks, cables, pulleys and specialized equipment, first to ease the vessel off its rocky perch and then to right it. the first few hours âwere the most uncertain, as we could not establish how much the hull was wedged,â sergio girotto, project manager with micoperi, the projectâs underwater construction and offshore contractor, said on monday.        
    the shipâs captain, francesco schettino, is scheduled to go on trial this fall on charges of multiple manslaughter, causing a shipwreck and abandoning the vessel before everyone was safe. he has denied wrongdoing. a company official and four crew members have already pleaded guilty to reduced charges.        
    salvage masters and the italian authorities had prepared for complications. most of the fuel was siphoned off within months of the wreck. but the vessel that once transported and entertained 4,229 people still contains chemicals and diesel fuel that could leak into the pristine mediterranean waters for which giglio, a popular tourist spot, is known.        
    
    
    
     
    gaia pianigiani reported from giglio, and alan cowell from london.


##Sklearn

In this section I will redo the procedure we did above using sklearn's methods.   To get similar results, I have to use the same tokenizer I used above.   Sklearn makes that relatively easy.  I just need to defien it and pass it into the constructor of the text processing object I am using.  I am going to include the sklearn stopwords the strip unicode.  This will lead to a slightly smaller word-matrix than in the previous part. 


    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    def tokenize(doc):
        '''
        INPUT: string
        OUTPUT: list of strings
    
        Tokenize and stem/lemmatize the document.
        '''
        sw = set(stopwords.words('english'))
        snowball = SnowballStemmer('english')
        wordnet = WordNetLemmatizer()
        reg = RegexpTokenizer(r'\w+',flags=re.UNICODE)
        doc_tokens = []
        for t in reg.tokenize(doc):
            if t not in sw:
                l = wordnet.lemmatize(t)
                s = snowball.stem(l)
                doc_tokens.append(s)    
        return doc_tokens
    
    vect = CountVectorizer(stop_words='english',tokenizer=tokenize,strip_accents='unicode')
    word_counts = vect.fit_transform(docs)
    word_counts




    <984x26280 sparse matrix of type '<type 'numpy.int64'>'
    	with 220702 stored elements in Compressed Sparse Row format>



We do not need construct the wordcount, however, to get the tfidf.  Sklearn can construct it directly.  When can then do the same calculation and find the most related documents by the measure of cosign similarity.   


    from sklearn.feature_extraction.text import TfidfVectorizer
    results = TfidfVectorizer(tokenizer=tokenize,strip_accents='unicode',stop_words='english')
    sp = results.fit_transform(docs)
    sp




    <984x26280 sparse matrix of type '<type 'numpy.float64'>'
    	with 220702 stored elements in Compressed Sparse Row format>




    from sklearn.metrics.pairwise import linear_kernel
    cosine_similarities = linear_kernel(sp, sp)
    for i, row in enumerate(cosine_similarities):
        if row[np.argsort(row)[::-1][1:2]][0] > 0.707:
            print i,np.argsort(row)[::-1][1:2][0]

    7 158
    113 712
    158 7
    174 521
    187 960
    240 251
    251 251
    253 563
    260 425
    289 474
    335 372
    372 372
    425 260
    439 615
    444 445
    445 444
    463 700
    474 289
    508 509
    509 509
    521 174
    563 253
    615 439
    700 463
    710 756
    712 920
    756 710
    767 767
    768 767
    919 756
    920 712
    939 956
    956 939
    960 187


Shockingly they produce the same pairs!   That is so cool.   Maybe I should be less surprised than I am, but I think this is cool.   We just made word vector, found vectors pointing in similar directions, and the documents are similar.   Abstractly it makes complete and simple sense.  On an emotional level, this is fracking cool!

##Part of speech tagging

I did not end up pursuing this today because we had to move onto the afternoon assignment, but I am leaving this here as a reminder to go back an explore it.   It is realted to a topic in AI called 'Incremental Concept Learning' that allow an agent to learn concepts.   When I learned it at the time I thought it was a cool concept, but not practical.  Now I see some direct applications.


    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    tokens = word_tokenize(docs[7][18:304])
    pos_tag(tokens)




    [(u'after', 'IN'),
     (u'a', 'DT'),
     (u'costly', 'JJ'),
     (u',', ','),
     (u'painstaking', 'VBG'),
     (u'and', 'CC'),
     (u'potentially', 'RB'),
     (u'perilous', 'JJ'),
     (u'operation', 'NN'),
     (u'to', 'TO'),
     (u'raise', 'VB'),
     (u'the', 'DT'),
     (u'battered', 'VBN'),
     (u'hull', 'NN'),
     (u'of', 'IN'),
     (u'the', 'DT'),
     (u'cruise', 'NN'),
     (u'ship', 'NN'),
     (u'costa', 'NN'),
     (u'concordia', 'NN'),
     (u',', ','),
     (u'engineers', 'NNS'),
     (u'said', 'VBD'),
     (u'early', 'RB'),
     (u'tuesday', 'NN'),
     (u'that', 'IN'),
     (u'they', 'PRP'),
     (u'had', 'VBD'),
     (u'succeeded', 'VBN'),
     (u'in', 'IN'),
     (u'righting', 'NN'),
     (u'the', 'DT'),
     (u'ship', 'NN'),
     (u',', ','),
     (u'removing', 'VBG'),
     (u'it', 'PRP'),
     (u'from', 'IN'),
     (u'two', 'CD'),
     (u'granite', 'JJ'),
     (u'reefs', 'NNS'),
     (u'where', 'WRB'),
     (u'it', 'PRP'),
     (u'ran', 'VBD'),
     (u'aground', 'NN'),
     (u'last', 'JJ'),
     (u'year', 'NN'),
     (u',', ','),
     (u'killing', 'VBG'),
     (u'32', 'CD'),
     (u'people', 'NNS'),
     (u'.', '.')]



##Naive Bayes

The afternoon assignment was on Naive Bayes, and its use with text classification.   Our first goal was to implement our own Naive Bayes algorithm, then to use in on a text problem.

The Naive part of Naive bases is the assumption that all the features are independant of each other.  Bayes Rule states:

$$P(C|X) = \mbox{Probability of Label C given data X} = \frac{P(X|C) \ P(C)}{P(X)}$$

The naive assumption of Naive Bayes assumed the data is independant.  This is expressed using the following formula.

$$P(C|X) = \mbox{Probability of Label C given data X} = \frac{P(x_1|C) \ P(x_2|C) \ ... \ P(x_n|C) \ P(C)}{P(X)}$$

So given the data, we choose the c with the maximium probability or likely hood.

$$\mbox{argmax}_c \left[ \ P(C|X) \ \right]  = \mbox{argmax}_c \left[ \  P(C) \ P(x_1|C) \ P(x_2|C) \ ... \ P(x_n|C)  \ \right]$$

We can ignore the Probability of getting the data X becuase that is independant of the C.  For a host of reason, but mostly floating point limitations, we often calculated the log maximum and solve this equation.

$$\mbox{argmax}_c \left[ \ logP(C|X) \ \right]  = \mbox{argmax}_c \left[ \ log(P(C)) + \sum_i log(P(x_i|C))  \ \right]$$

Another limitation is dealing with data/featuers that are zero or missing.  We can not compute the log of zero, so we have to do a smoothing on the probability to allow it to be small, but not zero.  The standard is to perform a Laplace smoothing:

$$P(x_i|C) = \frac{\sum x_{iC} + \alpha}{\sum X + \alpha \ p}$$

Where $p$ is the number of features in the dataset and $\alpha$ is a smoothing parameter.   It is commonly chosen to be 1.   This is also assuming we are dealing with count data.   If we are not, then we use a normal distrubtion for the probability.

Below is my implementation of the NaiveBayes function


    from collections import Counter
    import numpy as np
    
    
    class NaiveBayes(object):
    
        def __init__(self, alpha=1):
            '''
            INPUT:
            - alpha: float, laplace smoothing constant
            '''
    
            self.class_totals = None
            self.class_feature_totals = None
            self.class_counts = None
            self.alpha = alpha
            self.num_features = None
            self.num_trains = None
    
        def _compute_likelihood(self, X, y):
            '''
            INPUT:
            - X: 2d numpy array, feature matrix
            - y: numpy array, labels
    
            Compute the totals for each class and the totals for each feature
            and class.
            '''
            self.num_features = X.shape[1]
            self.num_trains = float(X.shape[0])
            self.class_totals = Counter()
            
            for yp in np.unique(y):
                self.class_totals[yp] = np.sum(X[y==yp,:])
           
       
            self.class_feature_totals = dict()
            
            for yp in np.unique(y):
                self.class_feature_totals[yp] = np.sum(X[y==yp,:],axis=0)[0,:]
    
    
        def fit(self, X, y):
            '''
            INPUT:
            - X: 2d numpy array, feature matrix
            - y: numpy array, labels
    
            OUTPUT: None
            '''
    
            # compute priors
            self.class_counts = Counter(y)  
            
            #print Counter(y)
    
            # compute likelihoods
            self._compute_likelihood(X, y)
    
        def predict(self, X):
            '''
            INPUT:
            - X: 2d numpy array, feature matrix
    
            OUTPUT:
            - predictions: numpy array
            '''
    
            predictions = np.zeros(X.shape[0]).astype(str)
            ys = self.class_counts.keys()
            for i,row in enumerate(X):
                best_y = None
                logs = []
                for yp in ys:
                    loglike = (row*np.log(self.p_feature(yp)).T)[0,0]
                    loglike += np.log(self.class_counts[yp]/self.num_trains)
                    logs.append(loglike)
                predictions[i] = ys[np.argmax(logs)]
    
            return predictions
    
        def p_feature(self,yp):
            result = self.class_feature_totals[yp]+self.alpha
            result = result/(self.class_totals[yp]+self.alpha*self.num_features)
            return result
                       
        def score(self, X, y):
            '''
            INPUT:
            - X: 2d numpy array, feature matrix
            - y: numpy array, labels
    
            OUTPUT:
            - accuracy: float between 0 and 1
    
            Calculate the accuracy, the percent predicted correctly.
            '''
    
            return np.sum(self.predict(X) == y) / float(len(y))

We are going to test this classifier against the sklearn implementation of Naive Bayese by taking a secion of Spors and Fasion articles from our new york times MongoDB database.  


    import numpy as np
    from pymongo import MongoClient
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from nltk.stem.snowball import SnowballStemmer
    from sklearn import preprocessing
    from sklearn.cross_validation import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    #from naive_bayes import NaiveBayes
    
    
    def tokenize(doc):
            '''
            INPUT: string
            OUTPUT: list of strings
    
            Tokenize and stem/lemmatize the document.
            '''
            snowball = SnowballStemmer('english')
            return [snowball.stem(word) for word in word_tokenize(doc.lower())]
    
    def load_data(sections):
        client = MongoClient()
        db = client.nyt_dump
        coll = db.articles
        y = []
        docs = []
        for article in coll.find({'section_name':{'$in':sections}}):
            y.append(article['section_name'])
            doc = ''.join(article['content'])
            doc = doc.encode('utf8', 'replace').decode('utf8')
            doc = doc.strip().lower()
            docs.append(doc)
    
        results = TfidfVectorizer(tokenizer=tokenize,strip_accents='unicode',stop_words='english')
        tfidf_vectorized = results.fit_transform(docs)
        sections = np.array(y)
        return tfidf_vectorized, sections
    
    def run_trial(sections=['Sports','Fashion & Style']):
        X, y = load_data(sections)
        X = X.todense()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        print 'Train shape:', X_train.shape
        print 'Test shape:', X_test.shape
    
        print
    
        print "My Implementation:"
        my_nb = NaiveBayes()
        my_nb.fit(X_train, y_train)
        print 'Accuracy:', my_nb.score(X_test, y_test)
        my_predictions =  my_nb.predict(X_test)
    
        print my_predictions
    
        print "sklearn's Implementation"
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        print 'Accuracy:', mnb.score(X_test, y_test)
        sklearn_predictions = mnb.predict(X_test)
        print sklearn_predictions
    
        # Assert I get the same results as sklearn
        # (will give an error if different)
        assert np.all(sklearn_predictions == my_predictions)
        
    run_trial()

    Train shape: (134, 11470)
    Test shape: (45, 11470)
    
    My Implementation:
    Accuracy: 0.755555555556
    ['Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'Sports' 'Sports']
    sklearn's Implementation
    Accuracy: 0.755555555556
    [u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'Sports' u'Sports' u'Sports']


The lack of an error of the last line says they're prediction match.  Now this is not a well defined measure of consistency between the two classes both prediction all articles are sports pages.   What we would like to see is there to be same predictions when there are different precictions for each article.   We can see in going through the counts of articles we have text for that the "World" section and the "Sports" section have a similar count.   We will re-run this for those two sections.  


    client = MongoClient()
    db = client.nyt_dump
    coll = db.articles
    coll.distinct('section_name')
    for x in coll.aggregate([{'$group':{'_id':'$section_name','count':{"$sum":1}}}]):
        print x

    {u'count': 5, u'_id': u'Automobiles'}
    {u'count': 2, u'_id': u'Crosswords & Games'}
    {u'count': 5, u'_id': u'Great Homes and Destinations'}
    {u'count': 11, u'_id': u'Paid Death Notices'}
    {u'count': 9, u'_id': u'Travel'}
    {u'count': 7, u'_id': u'Booming'}
    {u'count': 11, u'_id': u'Magazine'}
    {u'count': 10, u'_id': u'Corrections'}
    {u'count': 16, u'_id': u'Theater'}
    {u'count': 133, u'_id': u'Sports'}
    {u'count': 91, u'_id': u'Arts'}
    {u'count': 88, u'_id': u'U.S.'}
    {u'count': 46, u'_id': u'Fashion & Style'}
    {u'count': 92, u'_id': u'N.Y. / Region'}
    {u'count': 100, u'_id': u'Business Day'}
    {u'count': 28, u'_id': u'Movies'}
    {u'count': 18, u'_id': u'Science'}
    {u'count': 13, u'_id': u'Technology'}
    {u'count': 10, u'_id': u'Home & Garden'}
    {u'count': 84, u'_id': u'Opinion'}
    {u'count': 131, u'_id': u'World'}
    {u'count': 6, u'_id': u'Your Money'}
    {u'count': 19, u'_id': u'Dining & Wine'}
    {u'count': 10, u'_id': u'Health'}
    {u'count': 4, u'_id': u'Education'}
    {u'count': 13, u'_id': u'Real Estate'}
    {u'count': 37, u'_id': u'Books'}



    run_trial(['Sports','World'])

    Train shape: (198, 12872)
    Test shape: (66, 12872)
    
    My Implementation:
    Accuracy: 0.984848484848
    ['World' 'Sports' 'Sports' 'Sports' 'Sports' 'World' 'Sports' 'Sports'
     'World' 'World' 'Sports' 'World' 'Sports' 'Sports' 'World' 'World' 'World'
     'World' 'Sports' 'Sports' 'Sports' 'World' 'Sports' 'Sports' 'Sports'
     'Sports' 'World' 'Sports' 'Sports' 'World' 'World' 'World' 'World'
     'Sports' 'Sports' 'World' 'World' 'Sports' 'World' 'Sports' 'Sports'
     'Sports' 'Sports' 'Sports' 'World' 'Sports' 'World' 'Sports' 'Sports'
     'Sports' 'World' 'World' 'Sports' 'World' 'World' 'Sports' 'World' 'World'
     'Sports' 'World' 'World' 'Sports' 'World' 'Sports' 'World' 'World']
    sklearn's Implementation
    Accuracy: 0.984848484848
    [u'World' u'Sports' u'Sports' u'Sports' u'Sports' u'World' u'Sports'
     u'Sports' u'World' u'World' u'Sports' u'World' u'Sports' u'Sports'
     u'World' u'World' u'World' u'World' u'Sports' u'Sports' u'Sports' u'World'
     u'Sports' u'Sports' u'Sports' u'Sports' u'World' u'Sports' u'Sports'
     u'World' u'World' u'World' u'World' u'Sports' u'Sports' u'World' u'World'
     u'Sports' u'World' u'Sports' u'Sports' u'Sports' u'Sports' u'Sports'
     u'World' u'Sports' u'World' u'Sports' u'Sports' u'Sports' u'World'
     u'World' u'Sports' u'World' u'World' u'Sports' u'World' u'World' u'Sports'
     u'World' u'World' u'Sports' u'World' u'Sports' u'World' u'World']


Now we have a more likely comparision between the classifieres.  Here there is high accuracy and diversity in choices.   The two classifieres still match, so I am more conviced they are implementing the same algorithm.

##News Groups

A common natural language processing task is to explore and find related posts among the 20 news groupds datasets that come with a number of packages including sklearn and nltk.  Another task is to find important topics among the news groups.   We will be exploring both in our afternoon assignment.   I will start off by taking 4 groups from the newgroups and transforming their text into a text frequuency, inverse document frequency vector for each post.


    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train',categories=['sci.crypt',
                                         'sci.electronics',
                                         'sci.med',
                                         'sci.space'])
    vectorizor = TfidfVectorizer(stop_words='english',strip_accents='unicode')
    tfidf = vectorizor.fit_transform(newsgroups_train.data)
    tfidf




    <2373x38375 sparse matrix of type '<type 'numpy.float64'>'
    	with 283932 stored elements in Compressed Sparse Row format>




    for top10index in np.argsort(vectorizor.vocabulary_.values())[::-1][0:10]:
        print vectorizor.vocabulary_.keys()[top10index],vectorizor.vocabulary_.values()[top10index]

     zzz 38374
    zzi776 38373
    zzcrm 38372
    zz 38371
    zysv 38370
    zy 38369
    zxgxrggwf6wp2edst 38368
    zwp4q 38367
    zwl76 38366
    zwarte 38365


These are the most frequently used words in the corpus, and to someone who did not use news groups, it looks like garbage.  We can do this another way by using the tfidf vector, taking the words with the greatest score.  We can do this by summing or averaging over the tfidf values over the documents for each word.   When we average, we will remove zero values.  


    words = np.array(vectorizor.vocabulary_.keys())
    tfdif_scores_by_wordl = np.array(np.sum(tfidf.todense(),axis=0))[0]
    words[np.argsort(tfdif_scores_by_wordl)[::-1][:10]]




    array([u'iv', u'contestents', u'jams', u'noah', u'aloud', u'taf', u'chen',
           u'19600', u'sensitive', u'pjs269'], 
          dtype='<U180')




    tfdif_avg_scores_by_word = np.array(np.average(tfidf.todense(),weights=tfidf.todense().astype(bool),axis=0).tolist()[0])
    words[np.argsort(tfdif_avg_scores_by_word)[::-1][:10]]




    array([u'mjzzs', u'7394', u'casserole', u'1r47l1inn8gq', u'perihelions',
           u'transistors', u'afoxx', u'92621', u'curt', u'necisa'], 
          dtype='<U180')



In all of these methods, we are left with words that do not given human insight into the problem. We are left with words that are not anchor words.   What I mean by that is that if a new's article mentions President Obama then we know it is highly likely for it be in some sections, like World, and not others, like Fashion.  If we break the newsgroups up by category we might see some anchor words for that section. 


    for c in ['sci.crypt','sci.electronics','sci.med','sci.space']:
        newsgroups_train = fetch_20newsgroups(subset='train',categories=[c])
        vectorizor = TfidfVectorizer(stop_words='english',strip_accents='unicode')
        tfidf = vectorizor.fit_transform(newsgroups_train.data)
        print c
        print np.array(vectorizor.vocabulary_.keys())[np.argsort(np.array(np.sum(tfidf.todense(),axis=0).tolist()[0]))[::-1][:10]]
        print np.array(vectorizor.vocabulary_.keys())[np.argsort(np.array(np.average(tfidf.todense(),weights=tfidf.todense().astype(bool),axis=0).tolist()[0]))[::-1][:10]]
        print np.array(vectorizor.vocabulary_.keys())[np.argsort(np.array(vectorizor.vocabulary_.values()))[::-1][:10]]
        print ""

    sci.crypt
    [u'93apr21095141' u'kennedys' u'kronos' u'reasonable' u'roomful'
     u'adventurers' u'decwrl' u'altran' u'patterns' u'usage']
    [u'sacrifice' u'incrimination' u'archived' u'enemy' u'figure' u'sudden'
     u'invent' u'doen' u'inversions' u'value']
    [u'zzi776' u'zzcrm' u'zz' u'zysv' u'zy' u'zxgxrggwf6wp2edst' u'zwp4q'
     u'zwl76' u'zvt' u'zusman']
    
    sci.electronics
    [u'dissertation' u'w1gsl' u'advance' u'facetious' u'amplified' u'dealt'
     u'gerg' u'site' u'govern' u'finite']
    [u'illuminators' u'constructs' u'lc' u'laserjet' u'drove' u'floors'
     u'sincerely' u'136' u'9995' u'confirms']
    [u'zucchini' u'ztimer' u'zstewart' u'zoology' u'zoo' u'zone' u'zlau'
     u'zl1ttg' u'zklf0b' u'zjoc01']
    
    sci.med
    [u'fermi' u'merkle' u'weinreigh' u'steve' u'scallop' u'jmeritt' u'chairs'
     u'wg' u'reduce' u'britain']
    [u'gw' u'iastate' u'126645' u'donnell' u'l988' u'mini' u'pot' u'2423'
     u'smoky' u'foxxjac']
    [u'zzz' u'zz' u'zurich' u'zubkoff' u'zooid' u'zonker' u'zone' u'zonal'
     u'zoloft' u'zolft']
    
    sci.space
    [u'206265' u'507' u'mmc' u'convenient' u'gap' u'maneuvers' u'3rds'
     u'extension' u'gateway' u'winner']
    [u'wesley' u'lonely' u'flexibility' u'replacement' u'curry' u'robert'
     u'barium' u'restricted' u'dia' u'164655']
    [u'zwarte' u'zware' u'zwakke' u'zwak' u'zwaartepunten' u'zurbrin' u'zulu'
     u'zullen' u'zowie' u'zoology']
    


We do not see clear anchor words in the top 10 rankings, but we do ahve some likely words.   Seeing 'patterns' and 'inversions' in the crypto section is suggestive.  As is 'amplified' and 'illuminators' in the electronics section.   I do not know enough abou thte medical words to know if some of those are distincitive, but I do know the space ones are not.   I would argue that 'maneuvers' is consistent with 'space', but 'space' is not the MLE from 'maneuvers'.   

The most frequent words do not tells us anything useful in these corpuses, but the sum and average of the TFIDF vector along documents for each word does give us some interesting words.  

##Searching Newsgroups

After exploring the most important words in the corpus, we were asked to use a text file with search terms to search the documents and find the top 3 results for each search.  Our search is going to do something naive.  Since each tfidf vector is a normalized, we can find the difference in direction of the two vectors.  If they are small, we assume they are related documents.   If they are in very different directions, we will assume they are very different documents.   


    import pandas as pd
    from sklearn.metrics.pairwise import pairwise_distances
    
    search_terms = pd.read_csv('data/queries.txt',header=False).values[:,0]
    newsgroups_train = fetch_20newsgroups(subset='train')
    vectorizor = TfidfVectorizer(stop_words='english')
    tfidf = vectorizor.fit_transform(newsgroups_train.data)
    search = vectorizor.transform(search_terms)
    cos_sim = pairwise_distances(search,tfidf,metric='cosine')
    top_3 = np.argsort(cos_sim,axis=1)[:,:3]
    results = zip(search_terms,top_3)
    for x in results:
        print x[0], x[1]

    budget rental cars   [ 5769 10771  4253]
    children who have died from moms postpartum depression   [ 197  798 2240]
    compaq presario notebook v5005us [2561 7748 8640]
    boxed set of fruits basket  [4644 8917 7505]
    sun sentinal news paper [3110 5378 3719]
    puerto rico economy  [ 4179 10372  4811]
    wireless networking  [10486  7180   695]
    hidden valley ranch commercials [1410 1228 4285]
    jimmy carter the panama canal [10580  3322   147]



    print newsgroups_train.data[results[0][1][0]][:1000]

    From: joes@telxon.mis.telxon.com (Joe Staudt)
    Subject: Re: Renting from Alamo	
    Organization: TELXON Corporation
    Lines: 45
    
    In article <1993Apr20.142818.14969@ericsson.se> etxmst@sta.ericsson.se writes:
    >Hello netters!
    >
    >I'm visiting the US (I'm from Sweden) in August. I will probably rent a Chevy
    >Beretta from Alamo. I've been quoted $225 for a week/ $54 for additional days.
    >This would include free driving distance, but not local taxes (Baltimore). 
    >They also told me all insurance thats necessary is included, but I doubt that,
    > 'cause a friend rented a car last year and it turned out he needed a lot more
    >insurance than what's included in the base price. But on the other hand he 
    >didn't rent it from Alamo.
    >
    >Does anyone have some info on this?
    >
    >Is $225 a rip-off? 
    No, that sounds pretty reasonable for that car and that city.
    
    >Probability that I'll be needing more insurance?
    Unless you have an accident, you won't need more.  If you plan on
    paying for the car with a credit card,



    print newsgroups_train.data[results[0][1][1]][:1000]

    From: cds7k@Virginia.EDU (Christopher Douglas Saady)
    Subject: Re: Looking for MOVIES w/ BIKES
    Organization: University of Virginia
    Lines: 4
    
    There's also Billy Jack, The Wild One, Smokey and the Bandit
    (Where Jerry Reed runs his truck over Motorcycle Gangs Bikes),
    and a video tape documentary on the Hell's Angels I
    found in a rental store once
    



    print newsgroups_train.data[results[0][1][2]][:1000]

    From: Clinton-HQ@Campaign92.Org (Clinton/Gore '92)
    Subject: CLINTON: President's Radio Interview in Pittsburgh 4.17.93
    Organization: MIT Artificial Intelligence Lab
    Lines: 212
    NNTP-Posting-Host: life.ai.mit.edu
    
    
    
    
    
                             THE WHITE HOUSE
    
                      Office of the Press Secretary
                        (Pittsburgh, Pennsylvania)
    ______________________________________________________________
    For Immediate Release                         April 17, 1993     
    
    	     
                        INTERVIEW OF THE PRESIDENT
                          BY MICHAEL WHITELY OF
                        KDKA-AM RADIO, PITTSBURGH
    	     
                     Pittsburgh International Airport
                         Pittsburgh, Pennsylvania    
    
    
    
    10:40 A.M. EDT
    	     
    	     
    	     Q	  For everyone listening on KDKA Radio, I'm Mike 
    Whitely, KDKA Radio News.  We're here at the Pittsburgh 
    International Airport and with me is the President of the United 
    States Bill Clinton.
    	     
    	     And I'd like to wel


We can see the first response is spot on.  The second two values are not similar in the way of topics, but the preseidential document says budget a lot.   From the TFIDF perspective, the search term has a high ration of budget, a word with a low frequency over the documents.   The same is true for this presidential interview.   Obviously this can help find related documents, but there are obvious limitations for this methodology.   I am impressed that it does as well as it does, but as we get more documents there will be more overlap.   We need an additional filter in finding related documents before we measure similarity.   

##NLP with SQL

The goal of this section is to perform natural language processing using SQL for our New York times article data.  To do this we will need to make a table of documents, a table of words, and a connecting table between words and documents.   I am going to be doing this in a local PostgreSQL database where I created a new database called articles.

The structure of the database is there are 3 tables.

1. article_id => url, category  
2. word_id => word  
3. id => article_id, word_id, location  



    import psycopg2
    
    conn = psycopg2.connect(dbname='articles', 
                            user='postgres',
                            password='password', 
                            host='/tmp')
    cur = conn.cursor()


    client = MongoClient()
    db = client.nyt_dump
    coll = db.articles
    article_dict = dict()
    for i, article in enumerate(coll.find()):
        article_dict[article['_id']] = i
        cur.execute("INSERT INTO url VALUES (%s,%s,%s);", [i,article['web_url'], article['section_name']])
        conn.commit()


    from nltk import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk import Text
    import re
    
    sw = set(stopwords.words('english'))
    snowball = SnowballStemmer('english')
    reg = RegexpTokenizer(r'\w+',flags=re.UNICODE)
    words = []
    doc_tokens = []
    for doc in coll.find():
        doc = "".join(doc['content']).strip()
        tokens = []
        for t in reg.tokenize(doc):
            if t not in sw:
                t = t.encode('ascii','ignore')
                if str(t) != str():
                    s = snowball.stem(str(t))
                    tokens.append(s)
                    words.append(s)
        doc_tokens.append(tokens)
    words = set(words)


    len(words)




    23917




    word_dict = dict()
    for i, word in enumerate(words):
        word_dict[word] = i
        cur.execute("INSERT INTO wordlist VALUES (%s,%s);", [i,word])
        conn.commit()


    cur.execute("SELECT COUNT(*) FROM url;")
    print "Article Count: ", cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM wordlist;")
    print "Stemmed Word Count: ", cur.fetchone()

    Article Count:  (999L,)
    Stemmed Word Count:  (23917L,)



    for i, tokens in enumerate(doc_tokens):
        for j, token in enumerate(tokens):
            cur.execute("INSERT INTO wordlocation VALUES (%s, %s, %s,%s);", [100000*i+j,i,word_dict[token],j])
            conn.commit()


    cur.execute("SELECT COUNT(*) FROM wordlocation;")
    cur.fetchone()




    (374759L,)



So far we have set up the SQL tables, tokenized the words, recorded the 374759 locations of the 23917 stemmed words in 99 articles.

The next step we are going to engage in is the bag of words method.   Previous, we used sparse matrixes to represent the words in each article.  That is not a realisitic option for an sql table.  We do not want wide tables.  Instead we are going to make a new table and record the counts appropriately.

## Bag of words

The bag of words model counts the number of occurence of each word in each article.   We are going to create a new table that uses the url id (article id) the word id, and the count of occurance.


    cur.execute("""
                CREATE TABLE bagofwords AS
                  SELECT a.id, b.word_id, COUNT(*) 
                  FROM url a 
                  JOIN wordlocation b
                  ON a.id = b.url_id
                  GROUP BY a.id, b.word_id;
                """)
    conn.commit()

From the bagofwords table we created, we can construct the term frequency and inverse document frequency.   There are many definitions of [term frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), and we will use the double normalized 0.5 defintion:

$$tf(t,d) = 0.5 + \frac{0.5 \ f(t,d)}{ max(f(w,d): \ w \in d) } $$


    cur.execute("""
                SELECT a.id, a.word_id, (0.5 + 0.5*a.count/b.max) as tf 
                FROM bagofwords a JOIN
                (SELECT id, MAX(count) as max 
                    FROM bagofwords 
                    GROUP BY id) b
                ON a.id=b.id
                ORDER BY a.id, a.word_id;
                """)
    cur.fetchmany(10)




    [(0, 75, Decimal('0.55555555555555555556')),
     (0, 124, Decimal('0.55555555555555555556')),
     (0, 247, Decimal('0.66666666666666666667')),
     (0, 315, Decimal('0.61111111111111111111')),
     (0, 516, Decimal('0.55555555555555555556')),
     (0, 590, Decimal('0.55555555555555555556')),
     (0, 728, Decimal('0.55555555555555555556')),
     (0, 885, Decimal('0.55555555555555555556')),
     (0, 930, Decimal('0.55555555555555555556')),
     (0, 993, Decimal('0.55555555555555555556'))]



The inverse document frequency also has many definitions, but we will use the base definition:

$$idf(t,D) = log_{10}(\frac{N_{D}}{N_{D,t}})$$


    cur.execute("""
                SELECT word_id, 
                       LOG( (SELECT COUNT(*) FROM url) / doc_count ) as df 
                FROM (SELECT word_id, COUNT(1) as doc_count 
                      FROM bagofwords 
                      GROUP BY word_id) a;
                """)
    cur.fetchmany(10)




    [(21370, 1.43136376415899),
     (2848, 2.99956548822598),
     (2026, 1.53147891704226),
     (10295, 2.99956548822598),
     (11890, 2.99956548822598),
     (17928, 2.52244423350632),
     (22262, 2.99956548822598),
     (16703, 2.09342168516224),
     (9545, 2.39619934709574),
     (14724, 2.99956548822598)]




    cur.execute(""" CREATE TABLE tfidf AS
                SELECT tf.id, tf.word_id, tf.tf*idf.idf as tfidf
                FROM (SELECT a.id, a.word_id, (0.5 + 0.5*a.count/b.max) as tf 
                      FROM bagofwords a JOIN
                        (SELECT id, MAX(count) as max 
                         FROM bagofwords 
                         GROUP BY id) b
                      ON a.id=b.id
                      ) tf 
                JOIN (SELECT word_id, 
                             LOG( (SELECT COUNT(*) FROM url) / doc_count ) as idf 
                      FROM (SELECT word_id, COUNT(1) as doc_count 
                            FROM bagofwords 
                            GROUP BY word_id) a 
                      ) idf 
                ON tf.word_id=idf.word_id;
                """
               )
    conn.commit()

##SQL NYT Ranking

We are going to write a query function that will take a search term and return the top 3 articles that 'match' the query by summing the tfidf scores.


    def query(string):
        query_string = """SELECT a.id, SUM(a.tfidf) as total 
                          FROM tfidf a 
                          JOIN (SELECT id FROM wordlist WHERE word in ({})) b 
                          ON a.word_id = b.id GROUP BY a.id ORDER BY total DESC limit 3;"""
        string = " ".join([snowball.stem(word) for word in string.split()])
        cur.execute(query_string.format("'"+"','".join(string.split())+"'"))
        print string
        return [x[0] for x in cur.fetchall()]


    def get_headlines(query_string):
        results = query(query_string)
        article_ids = [article_dict.keys()[article_dict.values().index(x)] for x in results]
        print""
        for art in coll.find({"_id":{"$in":article_ids}}):
            print art['headline']['print_headline']
            print ""


    get_headlines("Obama upsets Congress")

    obama upset congress
    
    5 Years After Financial Collapse, Obama Says House G.O.P. Could Reverse Gains
    
    As Budget Fight Looms, Obama Sees Defiance in His Own Party
    
    Obama Highlights Fiscal Risks in Addressing Business Group
    



    get_headlines("Cowboys win")

    cowboy win
    
    The Good and the Bad Of the Saints, Reversed
    
    Giants Hope for U-Turn On Familiar Road Trip
    
    Overhaul Of Red Sox Is Beyond Their Chins
    


We see that the search for terms related to Obama, a common topic in the NYT, return relavant results.  Searching for something the NYTimes does not normally cover returns unrelated results.   

Another method we could use to filter would be to select based on word location.


    def query_by_location(query):
        query_string = """
        SELECT a.url_id, 1./SUM(a.min) as loc 
        FROM (SELECT a.url_id, a.word_id, MIN(a.location) 
              FROM wordlocation a 
              WHERE word_id IN
                 (SELECT id FROM wordlist 
                  WHERE word IN ({})) 
                  GROUP BY a.url_id, a.word_id) a 
        GROUP BY a.url_id ORDER BY loc DESC LIMIT 3;
        """
        query = " ".join([snowball.stem(word) for word in query.split()])
        cur.execute(query_string.format("'"+"','".join(query.split())+"'"))
        print query
        return [x[0] for x in cur.fetchall()]
        
    query_by_location("Obama upsets Congress")

    obama upset congress





    [314, 193, 17]




    def get_headlines_by_location(query_string):
        results = query_by_location(query_string)
        article_ids = [article_dict.keys()[article_dict.values().index(x)] for x in results]
        print""
        for art in coll.find({"_id":{"$in":article_ids}}):
            print art['headline']['print_headline']
            print ""


    get_headlines_by_location("Obama upsets Congress")

    obama upset congress
    
    How Old a Democracy?
    
    Wage Law Will Cover Home Aides
    
    New Chief Nominated For Justice Dept. Division
    



    def get_content_by_location(query_string):
        results = query_by_location(query_string)
        article_ids = [article_dict.keys()[article_dict.values().index(x)] for x in results]
        print""
        for art in coll.find({"_id":{"$in":article_ids}}):
            print "".join(art['content']).strip()[:500]
            print ""
    get_content_by_location("Obama upsets Congress")

    obama upset congress
    
    President Obama recently declared that the United States is âthe worldâs oldest constitutional democracy,â and he is echoed by Timothy Egan (âA brilliant mess,â Sept. 14), without challenge. It is, nonetheless, a historically dubious claim. That honor might belong to either Iceland or Switzerland, though the details are open to debate. But since the United States did not allow equal voting rights for all its citizens until 1965, its democracy, by that standard, must be counted young. A
    
    The Obama administration announced on Tuesday that it was extending minimum wage and overtime protections to the nationâs nearly two million home care workers.         
    Advocates for low-wage workers have pushed for this change, asserting that home care workers, who care for elderly and disabled Americans, were wrongly classified into the same âcompanionship servicesâ category as baby sitters â a group that is exempt from minimum wage and overtime coverage. Under the new rule, home care 
    
    President Obama on Tuesday nominated Leslie R. Caldwell, a defense lawyer specializing in white-collar cases, to be assistant attorney general for the Justice Departmentâs criminal division. From 2002 to 2004, Ms. Caldwell, a former federal prosecutor, was the director of the Justice Departmentâs task force that handled prosecutions related to the 2001 collapse of Enron. Ms. Caldwell is a graduate of Penn State and George Washington University Law School and has worked in the United States a
    


These results give different values.   We could potentially combine these two metrics in a way to give the most relavant results using a weighting sceme.  That way we can use both word location and word uniqueness to determine which articles are most important to show.

##Tuning and Model Comparison

We are going through the New York Times articles and attempt to identify which section they are apprt of using different supervised learning techniques.  First we need to encode the section names to variables, then we need to make a training and testing set.

We will make a tfidf on the training set, train our algorithm on the training set, then convert the test set and predict.  We will be using accuracy as our metrics.


    from sklearn.preprocessing import LabelEncoder
    le_section_name = LabelEncoder()
    le_section_name.fit(coll.distinct('section_name'))
    y = le_section_name.transform(labels)
    len(docs), len(y)




    (984, 984)




    from sklearn.cross_validation import train_test_split
    doc_trn, doc_tst, y_trn,y_tst = train_test_split(docs, y,test_size=0.3)


    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize,strip_accents='unicode',stop_words='english')
    tfidf_trn = vectorizer.fit_transform(doc_trn)
    tfidf_tst = vectorizer.transform(doc_tst)

Now we are going to compare the different algorithms.  Today I will stick with the methods that have built-in multi-classification.  I could us One vs. One or One vs All, but I will save that for another time.  Today's work has already taken a significant amount of time!


    clf = MultinomialNB()
    clf.__class__.__name__




    'MultinomialNB'




    import time
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    def supervise_time(clf):
        print ""
        print clf.__class__.__name__
        start = time.time()
        clf.fit(tfidf_trn,y_trn)
        end = time.time()
        print "Time to Fit:", end - start
        start = time.time()
        pred = clf.predict(tfidf_tst)
        end = time.time()
        print "Time to Predict:", end - start
        print "Accuracy: ", accuracy_score(y_tst,pred)
        
    supervise_time(MultinomialNB())
    supervise_time(DecisionTreeClassifier())
    supervise_time(RandomForestClassifier(n_estimators=1000))
    supervise_time(AdaBoostClassifier())
    supervise_time(KNeighborsClassifier(15))
    supervise_time(AdaBoostClassifier(base_estimator=MultinomialNB(),n_estimators=5000))

    
    MultinomialNB
    Time to Fit: 0.0335011482239
    Time to Predict: 0.00691485404968
    Accuracy:  0.493243243243
    
    DecisionTreeClassifier
    Time to Fit: 0.796107053757
    Time to Predict: 0.00192499160767
    Accuracy:  0.371621621622
    
    RandomForestClassifier
    Time to Fit: 10.8660159111
    Time to Predict: 0.547387123108
    Accuracy:  0.614864864865
    
    AdaBoostClassifier
    Time to Fit: 3.87863016129
    Time to Predict: 0.0303628444672
    Accuracy:  0.222972972973
    
    KNeighborsClassifier
    Time to Fit: 0.000900983810425
    Time to Predict: 0.037672996521
    Accuracy:  0.652027027027
    
    AdaBoostClassifier
    Time to Fit: 233.7222929
    Time to Predict: 276.877113819
    Accuracy:  0.652027027027


We see that the classification of text (sparse) data is not well done by untuned classifiers.   The KNN method does well, and I am sure that if we used a cosine similarity metric, it would do even better.

The most interest point of this exercise was that I Adaboosted a NaiveBayes classifier to the point that it gave the exact results of the nearest neighbor method.   I am wondering if the sparse data lead to this fact, and if this can be generalized in other contexts


    
