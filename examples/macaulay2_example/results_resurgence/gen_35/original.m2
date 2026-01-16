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

-- Define a line arrangement consisting of a 3x3 grid plus diagonals
-- The polynomial F defines the union of 8 lines:
-- x=0, x=z, x=-z (from x^3-xz^2)
-- y=0, y=z, y=-z (from y^3-yz^2)
-- y=x, y=-x      (from x^2-y^2)
F = (x^3 - x*z^2) * (y^3 - y*z^2) * (x^2 - y^2)

-- The singular locus of this arrangement consists of points where >= 2 lines intersect
-- High multiplicity intersections (up to 4 lines at the origin) drive up resurgence
-- We calculate the Jacobian ideal
J = ideal(diff(x,F), diff(y,F), diff(z,F))

-- We obtain the ideal of the points by taking the radical of the Jacobian
-- In this case, we decompose to find the prime ideals of the points and intersect them
-- This ensures I is the radical ideal of the configuration
pts = decompose J
I = intersect pts
-- EVOLVE-BLOCK-END