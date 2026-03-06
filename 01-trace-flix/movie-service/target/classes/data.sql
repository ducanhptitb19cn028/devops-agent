DROP TABLE IF EXISTS movie;

CREATE TABLE movie (
  id INT PRIMARY KEY,
  title VARCHAR(255),
  release_year INT,
  actor_ids INT ARRAY
);

INSERT INTO movie (id, title, release_year, actor_ids) VALUES
(1, 'The Shawshank Redemption', 1994, ARRAY[1, 2]),
(2, 'The Godfather', 1972, ARRAY[3, 4]),
(3, 'The Dark Knight', 2008, ARRAY[5, 6]),
(4, 'Pulp Fiction', 1994, ARRAY[7, 8]),
(5, 'Forrest Gump', 1994, ARRAY[9, 10]),
(6, 'Inception', 2010, ARRAY[11, 12]),
(7, 'Fight Club', 1999, ARRAY[13, 14]),
(8, 'The Matrix', 1999, ARRAY[15, 16]),
(9, 'Gladiator', 2000, ARRAY[17, 18]),
(10, 'Saving Private Ryan', 1998, ARRAY[9, 20]);
