Title: Galvanize - Week 05 - Day 1
Date: 2015-06-29 10:20
Modified: 2015-06-29 10:30
Category: Galvanize
Tags: data-science, galvanize, web scraping
Slug: galvanize-data-science-05-01
Authors: Bryan Smith
Summary: Today we covered web scraping.

#Galvanize Immersive Data Science

##Week 5 - Day 1

Today we had an introduction to webscraping, web apis, and [MongoDB](http://www.mongodb.org/).  

The morning quiz was on answer questions using data stored in a 5 table PostgreSQL database that were suppose to simulate [Hithc](http://hitchapp.com).  An example question is: Find the number of unique users that used the service over the last 10 days that were driven by drivers who started driving between DATE1 and DATE2.  Are not important.  


##MongoDB

We started off by downloading and installing a local copy of MongoDB from [here](http://www.mongodb.org/downloads?_ga=1.2370361.886345798.1422741448).  Some of the issues my cohort ran into was the database locking because proper permissions were to given to the 'data/db/' directory on the mac.   Other then that, it was a very easy processes.   

Though it was not part of the exercise, I also installed pymongo and used that.   I would then do what was asked directly in the database terminal, then replicate it with pymongo in an IPython Notebook.  We were given an option of downloading some GUI interfaces, but I opted not to use them.   Just incase I change my mind in the future I will list them here:

- [Robomongo (Multiplatform)](http://robomongo.org/)
- [MongoHub (Mac OSX)](https://github.com/fotonauts/MongoHub-Mac) 
   with down-loadable [binary](https://mongohub.s3.amazonaws.com/MongoHub.zip)
- [Humongous (web based)](https://github.com/bagwanpankaj/humongous)

## Mongo Queries 

Our first task was to load in some messy click data into our MongoDB database.  I used the command line for this.

>mongoimport --db clicks --collection log < click_log.json

I then used the mongo db to findOne() and find().limit(3).  There are over 2000 click results of different lengths.

>db.log.findOne()
>>{ "_id" : ObjectId("559198d72c8e29706d471d8e"), "_heartbeat_" : 1368774601 }

>db.log.find().limit(3)
>>{ "_id" : ObjectId("559198d72c8e29706d471d8e"), "_heartbeat_" : 1368774601 }
{ "_id" : ObjectId("559198d72c8e29706d471d8f"), "a" : "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)", "c" : "NL", "nk" : 0, "tz" : "Europe/Amsterdam", "gr" : "06", "g" : "15r91", "h" : "10OBm3W", "l" : "pontifier", "al" : "en-GB", "hh" : "j.mp", "r" : "direct", "u" : "http://www.nsa.gov/", "t" : ISODate("2013-05-17T07:09:59Z"), "hc" : 1365701422, "cy" : "Oss", "ll" : [ 5.5333, 51.766701 ] }
{ "_id" : ObjectId("559198d72c8e29706d471d90"), "a" : "Mozilla/5.0 (iPhone; CPU iPhone OS 6_1_3 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Mobile/10B329", "c" : "US", "nk" : 0, "tz" : "America/Los_Angeles", "gr" : "CA", "g" : "1084Psg", "h" : "19Cztuz", "l" : "tweetdeckapi", "al" : "en-us", "hh" : "1.usa.gov", "r" : "http://t.co/btKvKFBaF5", "u" : "http://science.nasa.gov/science-news/science-at-nasa/2013/16may_lunarimpact/", "t" : ISODate("2013-05-17T07:09:59Z"), "hc" : 1368774179, "cy" : "Palm Desert", "ll" : [ -116.345802, 33.7724 ] }
 
   


    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db_nyt = client.test_database
    ny = db_nyt.ny_times
    db = client.clicks
    log = db.log
    print log.find_one()
    print
    print "Limit 3"
    print 
    for r in log.find(limit=3):
        print str(r) + "\n"


    {u'_id': ObjectId('559198d72c8e29706d471d8e'), u'_heartbeat_': 1368774601}
    
    Limit 3
    
    {u'_id': ObjectId('559198d72c8e29706d471d8e'), u'_heartbeat_': 1368774601}
    
    {u'a': u'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)', u'c': u'NL', u'nk': 0, u'tz': u'Europe/Amsterdam', u'gr': u'06', u'g': u'15r91', u'h': u'10OBm3W', u'cy': u'Oss', u'l': u'pontifier', u'al': u'en-GB', u'hh': u'j.mp', u'r': u'direct', u'u': u'http://www.nsa.gov/', u't': datetime.datetime(2013, 5, 17, 7, 9, 59), u'hc': 1365701422, u'_id': ObjectId('559198d72c8e29706d471d8f'), u'll': [5.5333, 51.766701]}
    
    {u'a': u'Mozilla/5.0 (iPhone; CPU iPhone OS 6_1_3 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Mobile/10B329', u'c': u'US', u'nk': 0, u'tz': u'America/Los_Angeles', u'gr': u'CA', u'g': u'1084Psg', u'h': u'19Cztuz', u'cy': u'Palm Desert', u'l': u'tweetdeckapi', u'al': u'en-us', u'hh': u'1.usa.gov', u'r': u'http://t.co/btKvKFBaF5', u'u': u'http://science.nasa.gov/science-news/science-at-nasa/2013/16may_lunarimpact/', u't': datetime.datetime(2013, 5, 17, 7, 9, 59), u'hc': 1368774179, u'_id': ObjectId('559198d72c8e29706d471d90'), u'll': [-116.345802, 33.7724]}
    


We can see the pymongo results are identical to the commandline results.   What I like about pymongo, as we will see later, is that I can pull data from different sources and use that information to make queries to the MongoDB database.

Our next task was to find out how many clicks were in San Francisco.  This is relatively easy becasue we can see that the records that are clearly user click have an element 'cy' which looks to be the city they are in.  It seems the two users above are in Palm Desert and Oss.

>db.log.find({'cy':'San Francisco'}).count()
>>11


    log.find({'cy':'San Francisco'}).count()




    11



Back in my "I want to be a front end web developer" days, I was very much aware that there are different browsers, and they can have different features avaialbe or represent css in slightly different ways.   I had no idea how many different browers there were.  We can see in the query results there is an element 'a' that is the webbrowser.

>db.log.distinct('a').length
>>559

There are over 559 different browers types/versions in this dataset.   I do not miss front end development in the least! 


    len(log.distinct('a'))




    559



We learned that one of the strong use cases for MongoDB is text data.   We did an afternoon project that will be covered later scaping the New York Times.  More on that to come.

MongoDB, like almost all other databases, support regular expressions.  We can do a simple query and find out how many user use Mozilla, Opera, or both.

> db.log.find({'a': {'\$regex':'Mozilla|Opera'} }).count()
>>2830

> db.log.find({'a': {'\$regex':'Mozilla'} }).count()
>>2723

> db.log.find({'a': {'\$regex':'Opera'} }).count()
>>107


    print log.find({'a': {'$regex':'Mozilla|Opera'} }).count()
    print log.find({'a': {'$regex':'Mozilla'} }).count()
    print log.find({'a': {'$regex':'Opera'} }).count()

    2830
    2723
    107


A careful inspect will show that there is a variable 't' that is a DateTime object.   Originally it was the unix timestamp in seconds.   We had ot convert it to the datetime object, that was was doing with the following code:

       db.log.find({'t': {'\$exists': true}}).forEach(function(entry){ 
           entry.t = new Data(entry.t*1000); 
           db.log.update({'_id':entry._id},{'$set':entry});
       })

This is important because we were next asked to findout how many clicks were in the first hours.  I just happened to notice this:

> db.log.find({'t':{\$exists:1}}).sort({'t':-1})[0].t
>>ISODate("2013-05-17T08:09:56Z")

>db.log.find({'t':{\$exists:1}}).sort({'t':1})[0].t
>>ISODate("2013-05-17T07:09:57Z")

We were were given only 1 hours worth of data!  So all the records should be there when we do the correct query.  The query to do this if we did not notice this fact would look like:

>t1 = db.log.find({'t':{'\$exists':'true'}}).sort({'t':1})[0].t

>t2 = new Date(t1.getTime()+3600000)

>db.log.find({'t':{\$gte:t1,\$lte:t2}}).count()
>>2949

And if were just to count rectors where the 't' variable exists:

> db.log.find({'t':{'$exists':'true'}}).count()
>>2949




    from datetime import datetime, timedelta
    t1 = log.find({'t':{'$exists':'true'}}).sort('t',1).limit(1)[0]['t']
    t2 = t1+timedelta(hours=1)
    log.find({'t':{'$gte':t1,'$lte':t2}}).count()




    2949



The last question we had to answer was about what links the users clicked on the most.   This was an introduction to MongoDB's [aggregation](http://docs.mongodb.org/manual/reference/sql-aggregation-comparison/) functionality.

My first idea worked.  In terms of SQL I would group by the link address, count the number of records for each link, and sort by the link results.  In MongoDB this looks like the following:

>db.log.aggregate([{\$group:{_id:'\$u',count:{\$sum:1}}},{\$sort:{count:-1}},{\$limit:1}])
>>{ "_id" : "http://www.nsa.gov/", "count" : 478 }


    pipeline = [{"$group": {'_id':'$u','count':{'$sum':1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1}]
    for r in log.aggregate(pipeline):
        print str(r) + "\n"

    {u'count': 478, u'_id': u'http://www.nsa.gov/'}
    


##Geospacial

We were given some extra-credit MongoDB tasks involving [geospatial](http://docs.mongodb.org/manual/administration/indexes-geo/) data.  MongoDB requires that the data be longitude, latitude, but our data is stored as latitude, longitude.  We need to go through the data base and fix this.

        db.log.find({'ll':{\$exists:'true'}}).forEach(function(entry){ 
            entry.ll = [entry.ll[1],entry.ll[0]]; 
            db.log.update({'_id':entry._id},{$set:entry}); 
        })

Now that our database is in the correct format, we need to create an 2D Spacial indexo n the data.

>db.log.createIndex( { ll : "2d" } )

Now we can ask questions about how many clicks are within 50 miles of San Francisco:

>db.log.find( { 'll' : { \$geoWithin :{ \$centerSphere : [ [  -122.4167, 37.7833 ] , 50 / 3963.2 ] } } } ).count()
>>226


    log.find({'ll':
              {'$geoWithin' :
               {'$centerSphere' : [ [ -122.4167, 37.7833 ] , 50 / 3963.2 ] } #Lat,Lng SF 50 Miles/3964.2 Miles/Degree
              } 
             }).count()




    226



We could ask the questions:  "How many users are in Maine?" 

>db.log.find({'gr':'ME'}).count()
>>2


    log.find({'gr':'ME'}).count()




    2



That was easy, but if we did not have the state information, we would use the location data.   I found a site that hosts maps of the state boundries as polylines of gps coordinates.   This is where pymongo comes in handy over the command line.  I read in the data from a webrequest, then constructed a query from the GPS data using MongoDB's geo-features.


    import xml.etree.ElementTree as ET
    import urllib2
    request = urllib2.urlopen('http://econym.org.uk/gmap/states.xml')
    contents = request.read()
    ll = []
    lng = []
    lat = []
    root = ET.fromstring(contents.strip())
    for child in root.getchildren():
        if child.attrib['name'] == 'Maine':
            for c in child.getchildren():
                ll.append([float(c.attrib['lng']),float(c.attrib['lat'])])
                lng.append(c.attrib['lng'])
                lat.append(c.attrib['lat'])
    ll.append(ll[0])
    log.find({"ll": {"$geoWithin": {"$geometry": { 'type': "Polygon", "coordinates": [ll] } }}}).count()




    2



We also get 2, without using any pre-identified cate information.   This allows us to check for consistency incase we need to clean data.  This is awesome.   


    import matplotlib.pyplot as plt
    %matplotlib inline
    
    plt.plot(lng,lat,lw=3,color='seagreen')
    for r in log.find({'gr':'ME'}):
        plt.plot(r['ll'][0],r['ll'][1],'go')
    plt.xlim([-72,-62])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D1/output_19_0.png)


We can see where our two Maine users in the map.  This is a feature of using pymongo that is not avaialble in just the termial application of MongoDB.

Just for reference I wanted to list the geospacial tools listed for us if we wanted to explore them.  I did not, but I am using this post as a reference.  

-[CartoDB](http://cartodb.com/)

-[torque map](http://blog.cartodb.com/post/66687861735/torque-is-live-try-it-on-your-cartodb-maps-today).

##Web Scraping:  Ebay

In the afternoon we were asked to engage in a couple of web scraping projects.  One was to find a topic on ebay, and retreive all the images on the search result page.   I, of course, searched for [Teslas](http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR5.TRC0.A0.H0.Xtesla.TRS0&_nkw=tesla&_sacat=0)

The first thing we need is a place to store our images:


    %mkdir tesla
    %ls tesla/

We can see I just made the directory and that it is empty.  Now its type to scrape the webpage and pull out the images features.  The process we will do is request the webpage, use Beautiful soup to get the image sources, then down load each image source and store it in the tesla directory


    import requests
    from bs4 import BeautifulSoup


    tesla_html = requests.get('http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR5.TRC0.A0.H0.Xtesla.TRS0&_nkw=tesla&_sacat=0')
    sup = BeautifulSoup(tesla_html.content)
    imgs = [el['src'] for el in sup.select('div.lvpicinner a img') ]
    
    for img_str in imgs:
        img = requests.get(img_str).content
        to_img = open('tesla/'+img_str.split('/')[-1], 'w')
        to_img.write(img)
    
    %ls tesla/

    m0Q3jSK6NQ9_Hu_YIl38T9g.jpg  mbbybBkxozI9joXb1CgjfKQ.jpg
    m45xVWpc86gaOwXzaWdXXOQ.jpg  mgTnEkhYXi4kepjp3aNCx7Q.jpg
    m52qwfleCJ-encMaAJO9Taw.jpg  mgoIrvmOD-scAVdqXD1pirA.jpg
    mDbg8VxsScOpYeQ6VH1TBEg.jpg  mjMTLKMNaSsJNsRzn-zhQPg.jpg
    mEDv_smHtanyCtScA1bCnpA.jpg  ml5LUVDTC8ysTyvl70jOZMQ.jpg
    mEEQdnmeKYAH32KJpZElmfw.jpg  mmstVHfStN09f4KjbYZpQyA.jpg
    mE_xrXWO7xW9JF0uKYi6YZA.jpg  mpSnWCeMxvy3kVTjq3Ij44A.jpg
    mV4lwZBpH6-wbB6xoZhRB1A.jpg  mu5fgrRMhaOCwRhDu_eI1sw.jpg
    mVP1Wc7_ekfxObQrj4DaqtQ.jpg  mwdcx7MTbRMNuGi_4k7FiSw.jpg
    mWrzlrW8xau7TE3WtJJtE9Q.jpg  mzqPumoh3YFjZ8kvXLi2oMw.jpg
    mXyFRK9gbpV9-5f3-4QVuzA.jpg  s_1x2.gif


We see that we have downlaoded 22


    img = plt.imread(open('tesla/m0Q3jSK6NQ9_Hu_YIl38T9g.jpg'))
    plt.imshow(img)
    plt.show()


![png](http://www.bryantravissmith.com/img/GW05D1/output_26_0.png)


I'm in love!!!!!   Only 55K, what a steal!

This was fun, but my partner got burned out.  Ebay is full of iframes when looking at an individual description of a car, and I had us jumping to thhose pages to also scrape descriptions and prices.   Since we were not asked to do this, we moved on with our afternoon sprint. 

#New York Times Scraping

The NY times has, as we learned, over 15 Million articles posted on this website, some going back to 1851.   These are images/pdfs that you can view, but text you can easily scrape.  Reguardless, they are there.  

Probably because of people like us, the New York Times also has an API to access aspects of their data base.  This makes it easier for us to get data and allows them to manage/mitigate the affect these requests have on their servers.

The api requires an api.  You will see reference to it as a variable, but for obvious reasons I will not post my actual api key.  We have a simple function that allows us to send and process the response from the NY Times API.  


    import requests
    def single_query(link, payload):
        response = requests.get(link, params=payload)
        if response.status_code != 200:
            print 'WARNING', response.status_code
        else:
            return response.json()
        
    pay = {'sort':'oldest', 'api-key':ny_key}
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    
    ny_articles = single_query(link,pay)
    print ny_articles.keys()

    [u'status', u'response', u'copyright']


The response is a JSON document, and the first level has data about the query.   The next level has information about the responses, and then we have meta data about the article documents.


    ny_articles['response'].keys()




    [u'docs', u'meta']




    ny_articles['response']['meta']




    {u'hits': 15569986, u'offset': 0, u'time': 179}



We see that our response has 15,569,986 matches, taking 179ms to response.  Our offset is 0.


    ny_articles['response']['docs'][0]




    {u'_id': u'4fbfd23e45c1498b0d004db6',
     u'abstract': None,
     u'blog': [],
     u'byline': None,
     u'document_type': u'article',
     u'headline': {u'kicker': u'1',
      u'main': u'??itive Salve Case in Philadelphia'},
     u'keywords': [],
     u'lead_paragraph': None,
     u'multimedia': [],
     u'news_desk': None,
     u'print_page': u'4',
     u'pub_date': u'1851-09-18T00:03:58Z',
     u'section_name': None,
     u'snippet': None,
     u'source': u'The New York Times',
     u'subsection_name': None,
     u'type_of_material': u'Article',
     u'web_url': u'http://query.nytimes.com/gst/abstract.html?res=9904E7DE1430EF33A2575BC1A96F9C946092D7CF',
     u'word_count': 74}



The docs are a list of 10 document's JSON formated meta data.  We originally sorted by oldest, but we can look at the meta data for current documents.


    pay = {'sort':'newest', 'api-key':ny_key}
    ny_articles = single_query(link,pay)
    print ny_articles['response']['docs'][0]

    {u'type_of_material': u'Op-Ed', u'blog': [], u'news_desk': u'OpEd', u'lead_paragraph': u'The Greek debt crisis taxes the best minds of Europe.', u'headline': {u'main': u'Do We Have a Plan?', u'content_kicker': u'Op-Ed Columnist'}, u'abstract': None, u'print_page': None, u'word_count': u'10', u'_id': u'557ec00038f0d86829e412aa', u'snippet': u'The Greek debt crisis taxes the best minds of Europe.', u'source': u'The New York Times', u'web_url': u'http://www.nytimes.com/2015/07/12/opinion/do-we-have-a-plan.html', u'multimedia': [{u'subtype': u'wide', u'url': u'images/2015/06/15/opinion/15edchapart/15edchapart-thumbWide.jpg', u'height': 126, u'width': 190, u'legacy': {u'wide': u'images/2015/06/15/opinion/15edchapart/15edchapart-thumbWide.jpg', u'wideheight': u'126', u'widewidth': u'190'}, u'type': u'image'}, {u'subtype': u'xlarge', u'url': u'images/2015/06/15/opinion/15edchapart/15edchapart-articleLarge.jpg', u'height': 443, u'width': 600, u'legacy': {u'xlargewidth': u'600', u'xlarge': u'images/2015/06/15/opinion/15edchapart/15edchapart-articleLarge.jpg', u'xlargeheight': u'443'}, u'type': u'image'}, {u'subtype': u'thumbnail', u'url': u'images/2015/06/15/opinion/15edchapart/15edchapart-thumbStandard.jpg', u'height': 75, u'width': 75, u'legacy': {u'thumbnailheight': u'75', u'thumbnail': u'images/2015/06/15/opinion/15edchapart/15edchapart-thumbStandard.jpg', u'thumbnailwidth': u'75'}, u'type': u'image'}], u'subsection_name': None, u'keywords': [{u'value': u'European Sovereign Debt Crisis (2010- )', u'is_major': u'N', u'rank': u'1', u'name': u'subject'}, {u'value': u'Greece', u'is_major': u'N', u'rank': u'2', u'name': u'glocations'}, {u'value': u'Europe', u'is_major': u'N', u'rank': u'3', u'name': u'glocations'}, {u'value': u'European Union', u'is_major': u'N', u'rank': u'4', u'name': u'organizations'}, {u'value': u'European Central Bank', u'is_major': u'N', u'rank': u'5', u'name': u'organizations'}, {u'value': u'Eurozone', u'is_major': u'N', u'rank': u'6', u'name': u'organizations'}], u'byline': {u'person': [{u'organization': u'', u'role': u'reported', u'rank': 1, u'firstname': u'Patrick', u'lastname': u'CHAPPATTE'}], u'original': u'By PATRICK CHAPPATTE'}, u'document_type': u'article', u'pub_date': u'2015-07-12T00:00:00Z', u'section_name': u'Opinion'}


The NYT has different information for different articles.  We decided we would see how many we would download before we reached our daily limit.  It turns out that it was over 30,000 articles.   Since we were encoraged to see how many articles we could pull, we did not do it thoughtfully.  The NY times is fill of videos and receipes and other content that is not text based.   In retrospect I wish we focused on only 'News' articles.

A number of our cohort had trouble getting past 1000 articles.  This is because the NY Times limits the pagation of their response to 99.  We got around this by cycling through endates as well as pages in the response.   We also found, that because of the type of material that they post, some days have more than 1000 articles.  We limited our selves to 1000 articles from a given day, then moved on to the previous day.  


    import time
    
    #day we did the assignment
    end_date = '20150622'
    
    i = 0
    while i < 100:
        pay = {'sort':'newest','end_date':end_date, 'api-key':ny_key,'page':i}
        articles = single_query(link,pay)
        for doc in articles['response']['docs']:
            ##We upsert into the database because we when change the 
            ##enddate we get some redundant articles
            ny.update({"_id":doc['_id']},doc,upsert=True)
        
        if i%49==0:
            print i
        i = i + 1 
        if i == 99:
            end_date = articles['response']['docs'][-1]['pub_date'][:10].replace('-','')
            end_date = str(int(end_date)-1)
            i = 0
        
        ##Avoid being blocked by a too high query rate
        time.sleep(.15)

I am not going to run this code for this post, but it allowed us to pull 30,000+ articles and store them in our MongoDB database.  But we can look at the results.


    ny.find().count()




    31140



This is only meta-data however, and not the actual articles from the website.   The final project was to download all the article text for each article.  It is clear that we were not expected to be able to download 30,000+ articles for this assignment.   We started the project by only downloading the articles a few news articles.

We we also structured it in a way that allowed us to look for articles that have not yet been scraped and stored in our database.   We wanted to avoid double scraping at all cost.  


    import time
    def pull_articles(n):
        for x in ny.find({'type_of_material': 'News', 'HTML_content': {'$exists':0}, 'web_url':{'$exists':'true'}}).limit(n).sort('pub_date',-1):
            sup = BeautifulSoup(requests.get(x['web_url']).content)
            art = '\n'.join([text.text for text in sup.select('.story-content')])
            print art[:75]
            x['HTML_content'] = art
            ny.update({'_id': x['_id']}, x)
            time.sleep(.5)
    
    pull_articles(1)

    The movable feast of fashion weeks continues. After Valentino decided to ho


I did not print out the entire articles because I am affid of being in violation of copyright with the New York Times.  We did scrape a modest amount of articles.  After pulling the meta data for 30,000 articles we did not want to continue to scrape the web articles as well


    ny.find({'HTML_content': {'$exists':1}}).count()




    65


