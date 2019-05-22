/* Welcome to the SQL mini project. For this project, you will use
Springboard' online SQL platform, which you can log into through the
following link:

https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */



/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */

SELECT name,
       membercost
  FROM country_club.Facilities 
  WHERE membercost > 0
  ORDER BY 2 DESC

/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT(CASE WHEN membercost = 0 THEN 1 ELSE NULL END) AS no_cost_facilities
  FROM country_club.Facilities

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid,
       name,
       membercost,
       monthlymaintenance,
       (membercost / monthlymaintenance) AS cost_pct_main
   FROM country_club.Facilities 
   WHERE (membercost / monthlymaintenance) < 0.20 
   ORDER BY cost_pct_main DESC

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */

SELECT *
  FROM country_club.Facilities
  WHERE facid IN (1, 5)
  ORDER BY facid

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT name,
       monthlymaintenance,
       CASE WHEN monthlymaintenance <= 100 THEN 'cheap'
     	    ELSE 'expensive' END AS cost_scale
  FROM country_club.Facilities 
  ORDER BY monthlymaintenance DESC

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT surname,
       firstname,
       joindate
  FROM country_club.Members
  WHERE memid != 0
  HAVING MAX(joindate)

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT name,
       surname, 
       firstname,
       CONCAT(surname,', ', firstname,' - ',name) AS memb_court
  FROM country_club.Bookings
  JOIN country_club.Facilities
  ON Bookings.facid = Facilities.facid 
  JOIN country_club.Members
  ON Bookings.memid = Members.memid
  WHERE Bookings.memid != 0 AND Facilities.facid IN (0,1)
  GROUP BY Bookings.memid
  ORDER BY Members.surname

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT name,
       CASE WHEN Bookings.memid = 0 THEN surname 
            ELSE CONCAT(surname,', ',firstname) END AS memb_name,
       CASE WHEN Bookings.memid = 0 THEN guestcost
			ELSE membercost END AS cost
  FROM country_club.Bookings
  JOIN country_club.Facilities
  ON Bookings.facid = Facilities.facid 
  JOIN country_club.Members
  ON Bookings.memid = Members.memid
  WHERE Bookings.starttime >= '2012-09-14' AND Bookings.starttime < '2012-09-15'
  HAVING cost > 30
  ORDER BY cost DESC


/* Q9: This time, produce the same result as in Q8, but using a subquery. */

SELECT name,
       CASE WHEN bookings.memid = 0 THEN surname 
            ELSE CONCAT(surname,', ',firstname) END AS memb_name,
       CASE WHEN bookings.memid = 0 THEN fac.guestcost * slots
	    ELSE fac.membercost * bookings.slots END AS cost
  FROM (
        SELECT facid,
               memid,
      		   slots,
               starttime
         FROM country_club.Bookings
         WHERE Bookings.starttime >= '2012-09-14' AND Bookings.starttime < '2012-09-15' 
       ) bookings
  JOIN 
       (
        SELECT facid,
               name,
               guestcost,
               membercost
         FROM country_club.Facilities
       ) fac
  ON bookings.facid = fac.facid 
  JOIN 
       (
        SELECT surname,
               firstname,
               memid
         FROM country_club.Members
       ) memb
  ON memb.memid = bookings.memid
  HAVING cost > 30
  ORDER BY cost DESC

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT name,
       CASE WHEN memid = 0 THEN tot_slots * guestcost 
            ELSE tot_slots * membercost END AS total_rev
  FROM (
       SELECT facid,
              memid,
      		  COUNT(slots) AS tot_slots
         FROM country_club.Bookings
         GROUP BY memid
       ) bookings
  JOIN 
       (
        SELECT facid,
               name,
               guestcost,
               membercost
         FROM country_club.Facilities
       ) fac
  ON bookings.facid = fac.facid 
  GROUP BY bookings.facid
  HAVING total_rev < 1000
  ORDER BY total_rev DESC