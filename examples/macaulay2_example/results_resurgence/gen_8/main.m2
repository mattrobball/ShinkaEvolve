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

-- Star configuration of 5 generic lines in P^2
-- This configuration (10 points) is known to have resurgence 5/3 approx 1.667
-- This is higher than the Fermat configuration (1.5) and should fail I^(5) <= I^3
L = {x, y, z, x+y+z, x+2*y+3*z}

-- Generate ideal of the intersection points of all pairs of lines
pts = flatten table(0..#L-2, i ->
    table(i+1..#L-1, j -> ideal(L#i, L#j))
)

I = intersect pts
-- EVOLVE-BLOCK-END