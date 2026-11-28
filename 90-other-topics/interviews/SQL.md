## SQLBolt
1. Exercise 1 of [page](https://sqlbolt.com/lesson/select_queries_with_joins)  $\textemdash$ just for syntax
```SQL
SELECT Title, Domestic_sales, International_sales
FROM Movies
    JOIN Boxoffice
        ON Movies.id = Boxoffice.movie_id;
```
2. Last exercise on [LEFT JOIN](https://sqlbolt.com/lesson/select_queries_with_outer_joins) $\textemdash$ keep empty buildings
> [! ]- answer
> ```SQL
> SELECT DISTINCT building_name, role 
> FROM buildings 
>   LEFT JOIN employees
   ON building_name = building;
>```
3. Last exercise on [page](https://sqlbolt.com/lesson/select_queries_with_nulls) $\textemdash$ order of execution
	1. `FROM` to `WHERE` $\rightarrow$ `SELECT` $\rightarrow$ `DISTINCT`
> [! ]- example
> ```
> /* Find the names of the buildings that hold no employees */
> SELECT DISTINCT building_name
>   FROM buildings 
>   LEFT JOIN employees
>     ON building_name = building
>   WHERE role IS NULL;
> ```
4. expressions with aggregates
	1. Common aggregate functions![[Pasted image 20261127102609.png]]
	2. **`HAVING group_condition`**: constraints *applied to the grouped rows*.  In other words, if you aren't using the `GROUP BY` clause, a simple `WHERE` clause will suffice.
> [! ]- example
> ```sql
 /* total num of years employed by all engineers */
 SELECT role, SUM(years_employed)
 FROM employees
 GROUP BY role
 HAVING role = "Engineer";
> ```
5. [*Order of execution of a Query*](https://sqlbolt.com/lesson/select_queries_order_of_execution)
	1. `FROM/JOIN` $\rightarrow$ `WHERE` $\rightarrow$ `GROUP BY` $\rightarrow$ `HAVING` $\rightarrow$ `SELECT` $\rightarrow$ `DISTINCT` $\rightarrow$ `ORDER BY` $\rightarrow$ `LIMIT/OFFSET`
	2. Accessing aliases
		1. **Table aliases** are born in `FROM`, so they’re available everywhere after that: `WHERE`, `GROUP BY`, `HAVING`, `SELECT`, `ORDER BY`. 
		2. **Column aliases** are born in `SELECT`, and `SELECT` runs _after_ `WHERE`, `GROUP BY`, and `HAVING` — so in strict logical order they can’t be referenced there. The value doesn’t exist yet at that stage.
			1. `ORDER BY` runs _after_ `SELECT`, so column aliases are legal there. That’s the one place the standard guarantees it.

## HackerRank
### Syntax
1. Multiple order by: `ORDER BY column1 ASC, column2 DESC` 
2. Multiple `WHERE` conditions: `AND`, `OR` and `()`.
3. https://www.hackerrank.com/challenges/earnings-of-employees/
4. [Top Competitors](https://www.hackerrank.com/challenges/full-score/) $\star$
	1. Hint: use alias; group by multiple columns; which hacker_id to use
> [! ]- answer
> ```SQL
> SELECT s.hacker_id, h.name
FROM Submissions s
JOIN Hackers h       ON s.hacker_id = h.hacker_id
JOIN Challenges c   ON s.challenge_id = c.challenge_id
JOIN Difficulty d   ON c.difficulty_level = d.difficulty_level
WHERE s.score = d.score
GROUP BY s.hacker_id, h.name
HAVING COUNT(DISTINCT s.challenge_id) > 1
ORDER BY COUNT(DISTINCT s.challenge_id) DESC, s.hacker_id ASC;
> ```