DROP TABLE IF EXISTS review;

CREATE TABLE review (
  id INT PRIMARY KEY,
  movie_id INT NOT NULL,
  rating INT,
  comment TEXT NOT NULL,
  reviewer VARCHAR(100) NOT NULL
);

INSERT INTO review (id, movie_id, rating, comment, reviewer) VALUES
    (1, 1, 5, 'One of those rare movies that gets better every time you watch it.', 'Alice'),
    (2, 1, 4, 'A bit slow at first, but the payoff is absolutely worth it.', 'Marcus'),

    (3, 2, 5, 'A flawless classic. The tension and performances are unmatched.', 'Sophie'),
    (4, 2, 4, 'Great film, but a bit long for my taste.', 'Leo'),

    (5, 3, 5, 'Heath Ledger’s Joker is pure cinematic gold.', 'Patrick'),
    (6, 3, 4, 'Gripping from start to finish, though a bit too dark for some.', 'Daniel'),

    (7, 4, 5, 'Sharp, stylish, and full of memorable lines.', 'Nina'),
    (8, 4, 4, 'Enjoyed it, but had to watch it twice to really appreciate it.', 'Tom'),

    (9, 5, 5, 'Heartwarming and inspiring. Tom Hanks delivers once again.', 'Rachel'),
    (10, 5, 4, 'Some parts are a bit too sentimental, but it works.', 'Sam'),

    (11, 6, 5, 'Blew my mind the first time. Still does.', 'Jake'),
    (12, 6, 4, 'Visually stunning, though the plot takes effort to follow.', 'Anya'),

    (13, 7, 5, 'Raw and powerful. A punch to the gut in the best way.', 'Matt'),
    (14, 7, 5, 'One of the most intense psychological dramas I’ve ever seen.', 'Elena'),

    (15, 8, 5, 'The action, the philosophy, the visuals—just iconic.', 'Josh'),
    (16, 8, 4, 'Cool concept, though the pacing dips in the middle.', 'Lara'),

    (17, 9, 4, 'Great score, great battles. Crowe is a beast.', 'Derek'),
    (18, 9, 4, 'Epic, though a little emotionally distant for me.', 'Meera'),

    (19, 10, 5, 'Realistic and deeply emotional. Spielberg nailed it.', 'Victor'),
    (20, 10, 5, 'Watched it in silence. That beach scene is haunting.', 'Isabella');
