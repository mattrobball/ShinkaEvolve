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
-- Using the singular locus of the B3 reflection arrangement (9 lines)
-- Lines: x, y, z, x+/-y, x+/-z, y+/-z
-- This arrangement creates points with high multiplicity (up to 4 lines intersecting)
-- Such configurations (rational points of high multiplicity arrangements) often have high resurgence
F = x*y*z*(x^2-y^2)*(x^2-z^2)*(y^2-z^2)
J = ideal jacobian ideal F
pts = minimalPrimes J

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END