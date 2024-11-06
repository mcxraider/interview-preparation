
Write a query to find the percentage of orders that do not have a customer ID (i.e., where the customer_id is NULL
``` SQL
SELECT (COUNT(*) FILTER(WHERE customer_id IS NULL) * 100.0 / COUNT(*)) AS null_percentage
FROM orders;
```


Q: Write a SQL query to retrieve the top 5 most sold products in the last 30 days.
``` SQL
SELECT product_id, COUNT(*) AS total_sold
FROM sales
WHERE sale_date >= NOW() - INTERVAL '30 days'
GROUP BY product_id
ORDER BY total_sold DESC
LIMIT 5;
```

Q: Write a query to calculate the average revenue generated per order in the last year.
```SQL
SELECT AVG(total_amount) AS avg_order_revenue
FROM orders
WHERE order_date >= NOW() - INTERVAL '1 year';
```

```SQL

UPDATE A TABLE
UPDATE salary SET sex =
CASE sex
    WHEN 'm' THEN 'f'
    ELSE 'm'
END;
```

REGEX:
```SQL
where column REGEXP '^[A-Za-z][A-Za-z0-9._-]*@leetcode\\.com$'
```
The ^ and $ are to start and end the string respectively, and first character must be a string, the rest can be anything. The * is for when u want the subsequent chars to include the relevant characters.


GROUP CONCAT:
When u want to concatenate the results from each row into one row 
```SQL
STRING_AGG(DISTINCT candidate, ', ' ORDER BY candidate ASC) AS candidates
```
CONCAT:
```SQL
select user_id, 
concat(upper(substring(name,1,1)),lower(substring(name,2))) as name
from Users
```
Concatenating 2 things, substring parameters are to index a string in SQL.
=> substring(string FROM start FOR length)


FINDING MAX/ SECOND HIGHEST result from column
```SQL
SELECT MAX(salary) 
FROM employees 
WHERE salary < (SELECT MAX(salary) FROM employees);
```


OPERATIONS BETWEEN COLUMNS
```SQL
round(avg(cast(rating as decimal) / position), 2) as quality
```


IF ELSE CONDITIONS
```SQL
round(sum(case when rating <3 then 1 else 0 end) * 100 / count(*), 2) as poor_query_percentage
```

MAX() / MIN()
These return null when there are no maxes in the column


WINDOW FUNCTIONS 
```SQL
SELECT product_id, 
region_id, 
quantity_sold, 
    RANK() OVER (PARTITION BY region_id ORDER BY quantity_sold DESC) AS rank 
FROM sales;
```

REMOVING DUPLICATE:
```SQL
DELETE FROM video_views
WHERE (user_firstname, user_lastname, video_id) IN (
    SELECT user_firstname, user_lastname, video_id
    FROM (
        SELECT user_firstname, user_lastname, video_id,
               ROW_NUMBER() OVER (PARTITION BY user_firstname, user_lastname, video_id ORDER BY user_firstname) AS rn
        FROM video_views
    ) AS numbered_rows
    WHERE rn > 1
);
```


row_number() assigns a row number to each partition, then u filter for where the id is more than one, because u want to keep the first instance only. (keep=‘first’).
  DATA CLEANING

```SQL
SELECT LOWER(customer_name) AS customer_name_lowercase
FROM customers;
```

* LOWER() converts all characters to lowercase. You can use UPPER() to convert to uppercase.


IF ELSE STATEMENTS
```SQL
SELECT DISTINCT business_name,
CASE
WHEN business_name ILIKE ANY(ARRAY['%school%']) THEN 'school'
WHEN business_name ILIKE ANY(ARRAY['%restaurant%', '%bar%]) THEN 'food'
       END AS category
FROM table_name;
```

b) Correcting Numerical Values
```SQL
UPDATE orders
SET total_amount = 0
WHERE total_amount < 0;
```

5. Handling Inconsistent Data
```SQL
SELECT CASE 
           WHEN gender IN ('male', 'Male', 'M') THEN 'Male'
           WHEN gender IN ('female', 'Female', 'F') THEN 'Female'
           ELSE 'Unknown'
       END AS standardized_gender
FROM customers;
```

a) Converting Text to Numeric
```SQL
SELECT CAST(order_total AS DECIMAL(10, 2)) AS order_total_numeric
FROM orders;
```
* CAST() converts order_total from a string to a decimal number.


10. Date and Time Transformations
```SQL
SELECT EXTRACT(YEAR FROM order_date) AS order_year,
       EXTRACT(MONTH FROM order_date) AS order_month
FROM orders;
```







