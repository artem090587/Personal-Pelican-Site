Title: Galvanize - Week 01 - Day 3
Date: 2015-06-03 10:20
Modified: 2015-06-03 10:30
Category: Galvanize
Tags: data-science, galvanize, sql, postgresql
Slug: galvanize-data-science-01-03
Authors: Bryan Smith
Summary: The third day of Galvanize's Immersive Data Science program in San Francisco, CA where we got an in depth introduction to SQL and PostgreSQL.

#Galvanize Immersive Data Science
##Week 1 - Day 3

Today was an 'introduction' to SQL and PostgreSQL.  I put introduction in quotes because it does not properly describe what we did.  The pre-reading was to complete all 9 (1-9) tutorials on [SQLZoo](http://sqlzoo.net/).  This took me about 5 hours.   During lecture we have a review of the order of operation of SQL queries, as well as a detailed explanation of joins.    

The sprint for the day involved install [PostgreSQL](http://www.postgresql.org/) locally, loading a database into it, then completing ~25 basic and 10 advance (extra credit) queries.   Our database had 3 tables with 300k, 500k, and 5k entries respectively.

These were a great set of assignment because of how they 'leveled-up'.  Even the few among us that were sophisticated with SQL had difficulty with the advance problems.

Now that I have completed ~10 hours of SQL queries today, I am going to end this post now.

    
    SELECT * 
    FROM bryan JOIN bed 
    ON bryan.location=bed.location 
    AND bryan.state='sleep' 
	AND bed.state='comfy'



    
    



    
