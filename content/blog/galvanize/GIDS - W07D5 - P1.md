Title: Galvanize - Week 07 - Day 5 - Part 1
Date: 2015-07-17 10:20
Modified: 2015-07-17 10:30
Category: Galvanize
Tags: data-science, galvanize, spark, aws
Slug: galvanize-data-science-07-05-P1
Authors: Bryan Smith
Summary: Today we covered spark on aws

#Galvanize Immersive Data Science

##Week 7 - Day 5 - Part 1 (Local Spark)

Today we explore using sparkSQL and use Spark on AWS with multiple cores.  The analysis we are doing are going to be on two different sources.  The first part will be run on a local spark server.  The second part will be on AWS.

**Note:  This notebook is using Pysark with a running Spark Server and two 

This morning our quiz was to build a function that would predict the closest bart station for a given IP address.  The site <http://freegeoip.net> will return an object (XML,json, ...) that contains the (lat,lng) of the IP address submitted.

First we need to download the Bart Data.


    sc




    <pyspark.context.SparkContext at 0x104e81fd0>



We can see we have the pyspark context loaded already. 


    import urllib2
    text = urllib2.urlopen('https://raw.githubusercontent.com/enjalot/bart/master/data/bart_stations.csv').read().splitlines()
    bart = sc.parallelize(text)
    bart_loc = bart.filter(lambda x: x[:4] != 'name').map(lambda x: x.split(',')).map(lambda x: (x[0],(float(x[3]),float(x[4]))))
    bar_loc_data = bart_loc.collect()
    for x in bar_loc_data[:5]:
        print x

    ('12th St. Oakland City Center', (37.803664, -122.271604))
    ('16th St. Mission', (37.765062, -122.419694))
    ('19th St. Oakland', (37.80787, -122.269029))
    ('24th St. Mission', (37.752254, -122.418466))
    ('Ashby', (37.853024, -122.26978))


We have the location of the Bart locations.  Now we need to get the location of an IP address or url.


    import json
    def get_lat_long(url):
        data = json.loads(urllib2.urlopen('https://freegeoip.net/json/'+url).read())
        return (data['latitude'],data['longitude'])
    twitter_location = get_lat_long("twitter.com")
    twitter_location




    (37.77, -122.394)



We are dealing with (lat,lng) coordinates, which are sphereical coordinates.   Since we are dealing with a small location, the eucidean distance between the locations will be appropriate.  We should attempt another method if we were dealing with air travel.


    import math
    def distance(pt1,pt2):
        return math.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    
    bart_to_twitter = map(lambda x: (x[0],distance(twitter_location,x[1])), bar_loc_data)
    sorted(bart_to_twitter,cmp = lambda x,y: cmp(x[1],y[1]))[0]




    ('Powell St.', 0.019749241251242125)



Power St Bart Station is the closest to the twitter head quarters.  It is less than a quarter of a mile from the building.  For the final function, and quiz solution, we can do the following.



    def closest_bart(ip):
        location = get_lat_long(ip)
        bart_distances = map(lambda x: (x[0],distance(location,x[1])), bar_loc_data)
        return sorted(bart_distances,cmp = lambda x,y: cmp(x[1],y[1]))[0][0]
    
    closest_bart('twitter.com')




    'Powell St.'




    closest_bart('4.15.120.11')




    'Dublin/Pleasanton'



That is not the closest bart to my true location, despite me putting in my public ip.  The problem with the usuability of this quiz is that IP's can get bounced around.  Still a pretty cool project

##SparkSQL

Spark allows us to interact with geneteric data, but if the data is structures we can use SparkSQL.   The RDD objects we are using cna be converted into a SchemaRDD that lets up use SparkSQL.   We will be using the hivecontext to read in structured json yelp data.


    


    hive_contxt = HiveContext(sc)
    yelp_business_schema_rdd = hive_contxt.read.json('s3n://'+awskey+'@sparkdatasets/yelp_academic_dataset_business.json')


    yelp_business_schema_rdd.printSchema()

    root
     |-- attributes: struct (nullable = true)
     |    |-- Accepts Credit Cards: string (nullable = true)
     |    |-- Accepts Insurance: boolean (nullable = true)
     |    |-- Ages Allowed: string (nullable = true)
     |    |-- Alcohol: string (nullable = true)
     |    |-- Ambience: struct (nullable = true)
     |    |    |-- casual: boolean (nullable = true)
     |    |    |-- classy: boolean (nullable = true)
     |    |    |-- divey: boolean (nullable = true)
     |    |    |-- hipster: boolean (nullable = true)
     |    |    |-- intimate: boolean (nullable = true)
     |    |    |-- romantic: boolean (nullable = true)
     |    |    |-- touristy: boolean (nullable = true)
     |    |    |-- trendy: boolean (nullable = true)
     |    |    |-- upscale: boolean (nullable = true)
     |    |-- Attire: string (nullable = true)
     |    |-- BYOB: boolean (nullable = true)
     |    |-- BYOB/Corkage: string (nullable = true)
     |    |-- By Appointment Only: boolean (nullable = true)
     |    |-- Caters: boolean (nullable = true)
     |    |-- Coat Check: boolean (nullable = true)
     |    |-- Corkage: boolean (nullable = true)
     |    |-- Delivery: boolean (nullable = true)
     |    |-- Dietary Restrictions: struct (nullable = true)
     |    |    |-- dairy-free: boolean (nullable = true)
     |    |    |-- gluten-free: boolean (nullable = true)
     |    |    |-- halal: boolean (nullable = true)
     |    |    |-- kosher: boolean (nullable = true)
     |    |    |-- soy-free: boolean (nullable = true)
     |    |    |-- vegan: boolean (nullable = true)
     |    |    |-- vegetarian: boolean (nullable = true)
     |    |-- Dogs Allowed: boolean (nullable = true)
     |    |-- Drive-Thru: boolean (nullable = true)
     |    |-- Good For: struct (nullable = true)
     |    |    |-- breakfast: boolean (nullable = true)
     |    |    |-- brunch: boolean (nullable = true)
     |    |    |-- dessert: boolean (nullable = true)
     |    |    |-- dinner: boolean (nullable = true)
     |    |    |-- latenight: boolean (nullable = true)
     |    |    |-- lunch: boolean (nullable = true)
     |    |-- Good For Dancing: boolean (nullable = true)
     |    |-- Good For Groups: boolean (nullable = true)
     |    |-- Good For Kids: boolean (nullable = true)
     |    |-- Good for Kids: boolean (nullable = true)
     |    |-- Hair Types Specialized In: struct (nullable = true)
     |    |    |-- africanamerican: boolean (nullable = true)
     |    |    |-- asian: boolean (nullable = true)
     |    |    |-- coloring: boolean (nullable = true)
     |    |    |-- curly: boolean (nullable = true)
     |    |    |-- extensions: boolean (nullable = true)
     |    |    |-- kids: boolean (nullable = true)
     |    |    |-- perms: boolean (nullable = true)
     |    |    |-- straightperms: boolean (nullable = true)
     |    |-- Happy Hour: boolean (nullable = true)
     |    |-- Has TV: boolean (nullable = true)
     |    |-- Music: struct (nullable = true)
     |    |    |-- background_music: boolean (nullable = true)
     |    |    |-- dj: boolean (nullable = true)
     |    |    |-- jukebox: boolean (nullable = true)
     |    |    |-- karaoke: boolean (nullable = true)
     |    |    |-- live: boolean (nullable = true)
     |    |    |-- playlist: boolean (nullable = true)
     |    |    |-- video: boolean (nullable = true)
     |    |-- Noise Level: string (nullable = true)
     |    |-- Open 24 Hours: boolean (nullable = true)
     |    |-- Order at Counter: boolean (nullable = true)
     |    |-- Outdoor Seating: boolean (nullable = true)
     |    |-- Parking: struct (nullable = true)
     |    |    |-- garage: boolean (nullable = true)
     |    |    |-- lot: boolean (nullable = true)
     |    |    |-- street: boolean (nullable = true)
     |    |    |-- valet: boolean (nullable = true)
     |    |    |-- validated: boolean (nullable = true)
     |    |-- Payment Types: struct (nullable = true)
     |    |    |-- amex: boolean (nullable = true)
     |    |    |-- cash_only: boolean (nullable = true)
     |    |    |-- discover: boolean (nullable = true)
     |    |    |-- mastercard: boolean (nullable = true)
     |    |    |-- visa: boolean (nullable = true)
     |    |-- Price Range: long (nullable = true)
     |    |-- Smoking: string (nullable = true)
     |    |-- Take-out: boolean (nullable = true)
     |    |-- Takes Reservations: boolean (nullable = true)
     |    |-- Waiter Service: boolean (nullable = true)
     |    |-- Wheelchair Accessible: boolean (nullable = true)
     |    |-- Wi-Fi: string (nullable = true)
     |-- business_id: string (nullable = true)
     |-- categories: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- city: string (nullable = true)
     |-- full_address: string (nullable = true)
     |-- hours: struct (nullable = true)
     |    |-- Friday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Monday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Saturday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Sunday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Thursday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Tuesday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |    |-- Wednesday: struct (nullable = true)
     |    |    |-- close: string (nullable = true)
     |    |    |-- open: string (nullable = true)
     |-- latitude: double (nullable = true)
     |-- longitude: double (nullable = true)
     |-- name: string (nullable = true)
     |-- neighborhoods: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- open: boolean (nullable = true)
     |-- review_count: long (nullable = true)
     |-- stars: double (nullable = true)
     |-- state: string (nullable = true)
     |-- type: string (nullable = true)
    


To make SQL queries on this data we need to register the RDD as a temporary table to query.


    yelp_business_schema_rdd.registerTempTable('yelp_business')

Lets say we are traveling to Phoenix and we want to go to a good restaurant that accepts credit cards.  Yahoo for reimbursement of travel expenses!   We can perform an sql query on this data that looks like the following.


    hive_contxt.sql("""
                    SELECT name FROM yelp_business 
                    LATERAL VIEW explode(categories) c as category
                    WHERE
                        stars = 5 AND
                        city = 'Phoenix' AND 
                        attributes.`Accepts Credit Cards` = 'true' AND
                        category = 'Restaurants'
                    """).collect()




    [Row(name=u'Auslers Grill'),
     Row(name=u"Mulligan's Restaurant"),
     Row(name=u'Sunfare'),
     Row(name=u'Subway'),
     Row(name=u"Lil Cal's"),
     Row(name=u"Ed's"),
     Row(name=u'Frenchys Caribbean Dogs'),
     Row(name=u'WY Market'),
     Row(name=u'Pollo Sabroso'),
     Row(name=u'Queen Creek Olive Mill Oils & Olives Biltmore Fashion Park'),
     Row(name=u'Gluten Free Creations Bakery'),
     Row(name=u'Panini Bread and Grill'),
     Row(name=u'One Eighty Q'),
     Row(name=u'Saffron JAK Original Stonebread Pizzas'),
     Row(name=u'Los Primos Carniceria'),
     Row(name=u"Bertie's Of Arcadia"),
     Row(name=u'Little Miss BBQ'),
     Row(name=u'Las Jicaras Mexican Grill'),
     Row(name=u'Santos Lucha Libre'),
     Row(name=u'Taqueria El Chino'),
     Row(name=u"Filiberto's Mexican Food"),
     Row(name=u'Helpings Cafe, Market and Catering'),
     Row(name=u'Altamimi Restutant'),
     Row(name=u'Tacos Huicho'),
     Row(name=u"Jimmy John's"),
     Row(name=u'Ten Handcrafted American Fare & Spirits'),
     Row(name=u'The Brown Bag'),
     Row(name=u'Coe Casa'),
     Row(name=u"Adela's Italian"),
     Row(name=u'The Loaded Potato'),
     Row(name=u'Banh Mi Bistro Vietnamese Eatery'),
     Row(name=u'Couscous Express')]



##SparkSQL in Practice 

In practice we are given data that is not structured in a way that we want, but we would like to query it.  To do this we need to provide the data some structure.  In this case we are going to take user and transcation data, and try to find the 10 users with the most transactions amounts.  

First we will load in the data:


    user_rdd = sc.textFile('s3n://'+awskey+'@sparkdatasets/users.txt')
    transaction_rdd = sc.textFile('s3n://'+awskey+'@sparkdatasets/transactions.txt')


    user_rdd.take(2)




    [u'1106214172;Prometheus Barwis;prometheus.barwis@me.com;(533) 072-2779',
     u'527133132;Ashraf Bainbridge;ashraf.bainbridge@gmail.com;']



The structure for the user data is a user id, a name, an email, and finally a phone number.   Lets make this structured.  First we will split on the ';', then we will map it to a dictionary.  Finally we will dump the dictionary into a json format to be processed by hive_context.


    user_json = hive_contxt.jsonRDD(user_rdd.map(lambda x: x.split(";")) \
            .map(lambda x: json.dumps({'user_id':x[0],'name':x[1],'email':x[2],'phone':x[3]})))


    user_json.take(2)




    [Row(email=u'prometheus.barwis@me.com', name=u'Prometheus Barwis', phone=u'(533) 072-2779', user_id=u'1106214172'),
     Row(email=u'ashraf.bainbridge@gmail.com', name=u'Ashraf Bainbridge', phone=u'', user_id=u'527133132')]




    user_json.printSchema()

    root
     |-- email: string (nullable = true)
     |-- name: string (nullable = true)
     |-- phone: string (nullable = true)
     |-- user_id: string (nullable = true)
    



    user_json.registerTempTable('users')

Now we need to explore the transaction data:


    transaction_rdd.take(2)




    [u'815581247;$144.82;2015-09-05', u'1534673027;$140.93;2014-03-11']



This is similar to the previous data table.  We have a user id, an amount, and a date.   We will split on the ';', make a dict, then dump into json.   The only difference is we want to remove the '$' so we can add up the amounts.


    transaction_json = hive_contxt.jsonRDD(transaction_rdd.map(lambda x: x.replace("$","").split(";")) \
                   .map(lambda x: json.dumps({"user_id":x[0],"amount_paid":float(x[1]),"date":x[2]})))
    transaction_json.printSchema()

    root
     |-- amount_paid: double (nullable = true)
     |-- date: string (nullable = true)
     |-- user_id: string (nullable = true)
    



    transaction_json.registerTempTable('transactions')

##Top 10 Users with total transaction amount

This is a simple query using a join in sql.


    hive_contxt.sql("""
                    SELECT a.name, t.total FROM users a JOIN 
                    (SELECT user_id, SUM(amount_paid) as total FROM transactions
                    GROUP BY user_id) t ON
                    (a.user_id = t.user_id)
                    ORDER BY t.total DESC
                    LIMIT 10
                    """).collect()




    [Row(name=u'Kashawn Macpherson', total=21945.300000000003),
     Row(name=u'Brysten Jeffs', total=21773.510000000002),
     Row(name=u'Martez Carlyle', total=21120.550000000003),
     Row(name=u'Jaivyn Hoks', total=20641.109999999997),
     Row(name=u'Bryanne Stopp', total=20380.16),
     Row(name=u'Leanthony Waldegrave', total=20322.11),
     Row(name=u'Roosevelt Gooderham', total=20230.059999999998),
     Row(name=u'Demont Howell', total=20172.17),
     Row(name=u'Nasteha Bister', total=20163.909999999996),
     Row(name=u'Analaura Beetham', total=19998.19)]



We see that Kashawn Macpherson has the highest transaction amount out of all the customers.  We do not know if this is due to a long history of transaction, large purchases, or something inbetween.   That would now involve SQL type queries on the data, then possible exporting the results.  
