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

-- Define the ideal of a point configuration
-- Using a Star Configuration of 5 generic lines in P^2
-- The star configuration of s lines is expected to have higher resurgence for larger s.
-- For s=5, theoretical resurgence is 1.6, which allows for containment failures such as I^(3) not in I^2.
L = {x, y, z, x+y+z, x+2*y+3*z}
pts = flatten apply(5, i -> apply(i, j -> ideal(L#i, L#j)))

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END