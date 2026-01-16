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

-- Define a Star Configuration of 6 lines
-- Star configurations of s lines in P^2 are known to have resurgence rho = (s-1)/2
-- For s=6, this gives a resurgence of 2.5, significantly higher than the Fermat configuration

-- Define 6 generic linear forms using a Vandermonde-like structure
-- The coefficients (1, t, t^2) ensure that any 3 lines are linearly independent (Vandermonde determinant)
-- Thus no 3 lines are concurrent, forming a proper star configuration
lines = apply(0..5, t -> x + t*y + t^2*z)

-- The configuration consists of the pairwise intersection points of these 6 lines
-- There are binomial(6,2) = 15 points
-- We construct the ideal as the intersection of these point ideals
ptsList = flatten table(6, i -> table(i, j -> ideal(lines#i, lines#j)))

I = intersect ptsList

-- Define pts variable for compatibility
pts = ptsList
-- EVOLVE-BLOCK-END