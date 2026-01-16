-- Resurgence optimization: find point configurations with large resurgence
-- The ideal I defined here will be evaluated for its resurgence properties
--
-- Resurgence rho(I) = sup{m/r : I^(m) not contained in I^r}
-- For generic points, resurgence is close to 1
-- Special configurations (Fermat points, star configurations) can have larger resurgence

-- EVOLVE-BLOCK-START
-- Define points in affine 2-space (or projective 2-space)
-- The evolution should find point configurations with high resurgence

R = QQ[x,y,z]

-- Define 6 lines in general position using the moment curve (1, t, t^2)
-- General position means no 3 lines intersect at a single point
-- We use t values and include x (t=0) and z (t=inf) explicitly or via the curve
-- Vectors for lines: (1,0,0), (0,0,1), (1,1,1), (1,2,4), (1,3,9), (1,-1,1)
lines = {
    x,
    z,
    x + y + z,
    x + 2*y + 4*z,
    x + 3*y + 9*z,
    x - y + z
}

-- Compute the points of intersection for every pair of lines
-- For 6 lines in general position, this yields binom(6,2) = 15 points
pts = {}
for i from 0 to #lines-2 do (
    for j from i+1 to #lines-1 do (
        -- The ideal of the intersection of two lines is the sum of their ideals
        -- Since they are linear forms in P^2, this defines a point
        pts = append(pts, ideal(lines#i, lines#j))
    )
)

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END