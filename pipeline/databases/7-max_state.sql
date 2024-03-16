-- query max temperature in a state
SELECT state, MAX(value) 
FROM temperatures 
GROUP BY state 
ORDER BY state;
