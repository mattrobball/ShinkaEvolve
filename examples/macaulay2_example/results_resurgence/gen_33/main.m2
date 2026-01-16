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

-- Define a Star Configuration of 5 lines in P^2.
-- The resurgence for s lines is 2(s-1)/s.
-- For s=5, rho = 8/5 = 1.6.
-- This is an improvement over the s=4 case (rho=1.5).
-- We expect failures for m/r < 1.6, including 3/2 = 1.5.

-- Define 5 lines in generic position (no three lines intersect at a single point).
-- We use small integer coefficients to ensure efficiency and avoid accidental concurrencies.
L = {
    x,
    y,
    z,
    x + y + z,
    x + 2*y + 3*z
}

-- The points of the configuration are the pairwise intersections of these 5 lines.
-- There are binomial(5, 2) = 10 such points.
pts = flatten apply(5, i -> apply(i, j -> ideal(L#i, L#j)))

-- The ideal I is the intersection of the point ideals.
-- This defines the radical ideal of the point configuration.
I = intersect pts
-- EVOLVE-BLOCK-END