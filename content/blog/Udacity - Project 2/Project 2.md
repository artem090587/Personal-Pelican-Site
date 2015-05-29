Title: Udacity - Data Analysis NanoDegree - Project 2
Date: 2015-01-31 10:20
Modified: 2015-05-15 19:30
Category: Udacity
Tags: udacity, data-analysis, nanodegree, project
Slug: udacity-project-2
Authors: Bryan Smith
Summary: My passing submission project 1 for Udacity Data Analysis Nanodegree

#Udacity - Data Analysis NanoDegree

##Project 2 - Data Mugging 

The goal of this project is to read in data from openstreetmap.com stored in an XML format, reformat the data into JSON, load it into a MongoDB database, and audit the data for cleanliness and consistency.  The dataset I used is of Colorado Springs CO, a place I have visited several times to see family and friends.   The data set can be found [here](https://s3.amazonaws.com/metro-extracts.mapzen.com/colorado-springs_colorado.osm.bz2).   A summary of the OpenStreetMap format and structur can be found on their [documentation page](http://wiki.openstreetmap.org/wiki/OSM_XML)

The openstreetmap.com OSM XML file consists of three types of primitive elements: nodes, ways, and relations.   Nodes are gps coordinates on a map, and they are used to identify features in a city or, in collections, use to construct ways.  Nodes could have features, and these features are stored in subtags called “tag”.  Ways, as just suggested, are a collection of nodes that create a path.   Ways also have descriptions that are stored in “tag” subtags.  Relations are a collection of ways, nodes, and relations that describe features of the map.  They can be routes, restrictions, and multipolygon boundaries.  

The Colorado Springs data is contained in a 103 MB file that has the following numbers of each primitive element.


<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
	<thead>
		<tr>
			<th> Element Type</th>
			<th> No Sub-Elements </th>
			<th> Total Elements </th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td> Node </td>
			<td> 468,637 </td>
			<td> 480,216 </td>
		</tr>
		<tr>
			<td> Ways </td>
			<td> 0 </td>
			<td> 55,926 </td>
		</tr>
		<tr>
			<td> Relations </td>
			<td> 0 </td>
			<td> 153 </td>
		</tr>
</table>
</div>


Most of the nodes being empty signify that most nodes are likely to be used solely for defining ways.   This is something that can be checked for consistency.  There are two following up questions that one could explore here:

Does the node for each way exist in this dataset?
Is each node referenced in this data set?   

These questions are computational expensive, but could be necessary depending on the ultimate purpose of this data set.

Since both ways and relationship are collections of subelements, it would not make sense that these elements are empty.  Finding that any of them are empty would be a sign that there is something about the data the needs to be fixed.  The documentation for these primitives states that ways and relationships are collections of 2 or more elements, something that will be checked later.

###Nodes

Of the 11,624 nodes with subelement tags, 10,206 of them have a single description tag.  Most of them are minimal descriptions with keys of “highway” (8185) and “power” (1204).  There are some that are clearly incomplete.    There are 157 nodes with the sole tag descriptions is “addr:house number.”   Since an address usually consists of a number, street name, and a zip code, it seems clear that these 157 nodes are incomplete descriptions of a feature with an address.  

There is also one node with a key value “FIXME” and the value of “Denver area needs work.” The google maps of the position coordinates associated with this node shows that this is St. Francis Hospital in Colorado Springs.   It is not clear why user “AMY-jin” put this information here because this hospital is 65 miles from Denver, Co.  This is 1 of 4 contributions this user made  

There are only 1418 nodes with multi-tag descriptions.

###Ways

Of the 35,928 ways, 24 of them only contain a single “nd” tag.  The documentation defines a way of a polyline between 2 and 2000 nodes.  These ways only contain 1 node.   Thus there are 24 ways that are needed to be either updated or removed from the data set. 

###Relations

Of the 153 relations, 10 of them contain a single “member” tag.   The documentation defines a relation between 2 or more members consisting of ways, nodes, or relationships.   A relation with only 1 member is ill defined.   Therefor there are 10 relationship that need to be updated or removed from the dataset.

 
###Problem Character

Problems characters are defined as characters that have issues associated with storage in a database.  These characters consists of the following:

> = + / & < > ; ' " ? % # $ @ , . \t \r \n

There were no problem characters with any of the elements’ key-values.   Of all the element values associated with the previously mentioned keys, there were 1,716 values that had problem characters associated with them.  These values will be dropped before the data is inserted into the MongoDB Database

##MongoDB

The nodes and ways from the dataset was formatted into the following JSON form,

	{
	 "id": "2406124091",
	 "type": "node",
	 "visible":"true",
	 "created": 
		{
	    	"version":"2", 
		 	"changeset":"17206049", 		 
		 	"timestamp":"2013-08-03T16:43:42Z",    
		 	"user":"linuxUser16", 
		 	"uid":"1219059"
		},
	  "pos": [41.9757030, -87.6921867],
	  "address": 
	  {
	  		"housenumber": "5157", 
	  		"postcode": "60625", "street": "North Lincoln Ave"
	  	},
		"key1":"val1", 
		"key2":"val2", 
	}

and inserted into a local MongoDB database.   As measured in the previous section, key/value pairs with problem characters were dropped from database.

The following commands using the pymongo library:

	c = MongoClient()
	db = c['test-database']
	osm = db.osm
	osm.count()
	osm.find({'type':'node'}).count()
	osm.find({'type':'way'}).count()

Of the 536,142 nodes and ways listed in the original xml files, all 536,142 appeared in the database.  The count of elements with type “ways” is 480216, and the count of elements with type “ways” is 55926.  These values are consistent with the original examination of the XML file. 

###Address - Street

Looking at the address information we can find with the following commands that 5689 entries contain an “address” field and 5453 of these elements contain a subfield of “street”.

	pipeline = [{"$match":{"address":{"$exists":1}}}]
	print len(osm.aggregate(pipeline)['result'])

	pipeline = [{"$match":{"address.street":{"$exists":1}}}]
	print len(osm.aggregate(pipeline)['result'])

An examination of the street fields shows that the data is mostly consistently formatted and clean.  The street names do not generally have abbreviations, though there are some exceptions.   Only 10 of the 5689 data points have values that are not just street names:

	pipeline = [{"$match":{"address.street":{"$exists":1}}},
	        {"$group":{"_id":"$address.street","count":{"$sum":1}}},
	        {"$sort":{"_id":-1}}]

	street_descriptions = set()
	for x in osm.aggregate(pipeline)['result']:
	  print x['_id']
	  street_split = x['_id'].split(" ")
	  length = len(street_split)
	  street_descriptions.add(street_split[length-1])

	for y in street_descriptions:
	  print y

The common inconsistencies in the street names are abbreviations.  Some examples are that “Dr.” is used for “Drive”, “E.” is used for “East”, and “Ave” is used for “Avenue”.  These values are exceptions, surprisingly. 

###Users

The dataset was produced by 300 users, the top user named “FrozenFlame22” contributing 424,585 of the 536,142 data entries (79.1%).  The top users are:

<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
	<thead>
		<tr>
			<th> Username</th>
			<th> Number of Entries Associated with User </th>
			<th> Percent of Dataset </th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td> FrozenFlame22 </td>
			<td> 424,585 </td>
			<td> 79.8% </td>
		</tr>
		<tr>
			<td> CS_Mur </td>
			<td> 20185 </td>
			<td> 3.79% </td>
		</tr>
		<tr>
			<td> Jason R Surrat </td>
			<td> 15930 </td>
			<td> 3.0% </td>
		</tr>
		<tr>
			<td> Your Village Maps </td>
			<td> 15628 </td>
			<td> 2.94% </td>
		</tr>
		<tr>
			<td> GPS_dr </td>
			<td> 12643 </td>
			<td> 2.38% </td>
		</tr>
		<tr>
			<td> Chris CA</td>
			<td> 7194 </td>
			<td> 1.35% </td>
		</tr>
		<tr>
			<td> Mark Newnham </td>
			<td> 3108 </td>
			<td> 0.58% </td>
		</tr>
		<tr>
			<td> SpenSmith </td>
			<td> 2738 </td>
			<td> 0.51% </td>
		</tr>
		<tr>
			<td> woodpeck_fixbo </td>
			<td> 2566 </td>
			<td> 0.48% </td>
		</tr>
</table>
</div>

This information was generated with the following code:

	pipeline = [{"$group":{"_id":"$created.user","count":{"$sum":1}}},
	        {"$sort":{"count":-1}}]
	result = osm.aggregate(pipeline)['result']
	sum = 0
	for x in result:
	  print x, x['count']/536142.0
	  sum += x['count']

	print sum/536142.0
	print len(result)

It is possible that a majority of these users are cleaning data, and not generating data, for openstreetmap.  Whenever a new data point is entered the server automatically generates it with version of “1”.   Looking at the subset of entries with version equal to 1 shows the following top 10 users.

<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
	<thead>
		<tr>
			<th> Username</th>
			<th> New Entries </th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td> FrozenFlame22 </td>
			<td> 343854 </td>
		</tr>
		<tr>
			<td> Jason R Surrat </td>
			<td> 14780 </td>
		</tr>
		<tr>
			<td> CS_Mur </td>
			<td> 12646 </td>
		</tr>
		
		<tr>
			<td> Your Village Maps </td>
			<td> 10684 </td>
		</tr>
		<tr>
			<td> GPS_dr </td>
			<td> 9486 </td>
		</tr>
		<tr>
			<td> Chris CA</td>
			<td> 4036 </td>
		</tr>
		<tr>
			<td> Mark Newnham </td>
			<td> 3034 </td>
		</tr>
		<tr>
			<td> SpenSmith </td>
			<td> 2430 </td>
		</tr>
		<tr>
			<td> gps_pilot</td>
			<td> 2187 </td>
		</tr>
		<tr>
			<td> Rub21 </td>
			<td> 1615 </td>
		</tr>
</table>
</div>

The 9 of the 10 top-ten contributes to the osm data are also the top-ten original generators of edits of Colorado Springs dataset.  

	pipeline = [{"$match":{"created.version":"1"}},
	        {"$group":{"_id":"$created.user","count":{"$sum":1}}},
	        {"$sort":{"count":-1}},
	        {"$limit":10}]
	result = osm.aggregate(pipeline)['result']

	for x in result:
	  print x, x['count']


###Fences

Fences are way to enforce a boundary around a given area.  Given there is a military base in Colorado springs, one would expect to find a number of barbed wire fences.   

	pipeline = [{"$match":{"fence_type":{"$exists":1}}},
	        {"$group":{"_id":"$fence_type","count":{"$sum":1}}}]
	result = osm.aggregate(pipeline)['result']

	fences = []
	counts = []
	for x in result:
	  print x
	  fences.append(x['_id'])
	  counts.append(x['count'])

	import numpy as np
	import matplotlib.pyplot as plt
	ind = np.arange(len(fences))    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence

	p1 = plt.bar(ind, np.array(counts),width, color='r')
	plt.ylabel('Fence Count')
	plt.title('Count of Fences in Colorado Springs, CO')
	plt.xticks(ind+width/2., fences, rotation=10 )
	plt.show()

The previous code generate the following plot.   

![Fence Count by Category](http://www.bryantravissmith.com/img/udacity/udacity-project2.png)

The most cataloged fence in the dataset is a pole fence.   There are only 2 barbed wire fences.  I know for fact that this is far from a full catalog of the fences in Colorado Spring.  My visit to the military base there showed me there is infact more than 2 fences with barbed wire.   This is one clear aspect of the map in need of improvement.   

The fact that fences are ways, and not nodes, is consistent with the definition of ways being paths and a collection of nodes.     

	pipeline = [{"$match":{"fence_type":"barbed_wire"}}]
	result = osm.aggregate(pipeline)['result']

	for nd in result[0]['node_refs']:
	  for y in osm.find({"id":nd}):
	     print y['pos']

	print 'next'
	for nd in result[1]['node_refs']:
	  for y in osm.find({"id":nd}):
	     print y['pos']

The first barbed wire fence in the dataset is around the Zebulon Pike Detention Center, and the second barbed wire fence is also around the same Zebulon Pike Detention Center.  These two fences could be the same fence around the complex, or two sub fences around buildings.   The google street view of the area shows that there appears to be a single short barbed wire fence around the property.   

This highlights one of the major issues I see with auditing the openstreetmap data.   Nodes are points, but ways are collections of points.   Two ways may have different number of nodes, likes the barbed wire fences, use a difference collection of nodes, like the two barbed wire fences, but are attempting to describe the same feature.  

An examination of the user that created these two separate features shows that it was the same user, Jason R Surratt.

If the same user is capable of double describing the same feature for a map, then it highlights an issues with openstreetmap’s ability to uniquely describe features in an area or on a map that are a collection of points.  

##Conclusion

Raw data is messy.  Even with a dataset that has been ‘cleaned’ by dozens of users, like the dataset used in this paper, there are hundreds of issues and inconsistency.  Before the data could be loaded into MongoDB there are over 1700 key/values with invalid characters.   Additionally there are dozens of ways and relationships with only a single subelement.

A review of the address data showed that there was inconsistencies with the street names.   This data was obviously previously cleaned, and there are still aspects that escaped being cleaned.   It is important to realizing that any cleaning process will not catch all the inconsistencies in a dataset.   

Additionally the dataset has relationships, and with relationship produce an additional layer of complexity.   Relationships need to be correctly and consistently formatted.  The examination of barbed wire fences in the Colorado Spring dataset showed that even in the case of data entered/edited by the same user can lead to the same fence being imputed twice with complete different sets of nodes.  There are doubtlessly a number of features in this data yet undiscovered that will have the same problem.