-- Creates a stored procedure
CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id INT)
BEGIN
  -- Declare variable for storing the computed weighted average
  DECLARE weighted_avg DECIMAL(10,2);
  
  SELECT SUM(score * weight) / SUM(weight) INTO weighted_avg
  FROM scores
  WHERE user_id = user_id;

  UPDATE users
  SET average_weighted_score = weighted_avg
  WHERE id = user_id;
  
END$$

DELIMITER ;