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

-- The Fermat configuration of 12 points in P^2
-- These are the intersection points of the lines defined by x^3-y^3, y^3-z^3, z^3-x^3
-- This configuration is known to provide a counterexample to I^(3) <= I^2
I = ideal(
    x*(y^3-z^3),
    y*(z^3-x^3),
    z*(x^3-y^3)
)

-- Derive the points from the ideal for completeness
pts = decompose I
-- EVOLVE-BLOCK-END