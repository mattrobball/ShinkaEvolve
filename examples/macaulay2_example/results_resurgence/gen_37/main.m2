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

-- Star configuration of 5 lines in general position
-- The ideal of the star configuration is the ideal of the pairwise intersection points of s lines.
-- For generic lines, this is given by the Jacobian ideal of the product of the linear forms.
-- Star configurations are known to maximize resurgence; for 5 lines, we expect rho(I) > 1.5.

-- Define 5 lines in general position (no 3 concurrent)
L1 = x
L2 = y
L3 = z
L4 = x + y + z
L5 = x + 2*y + 3*z

-- The defining polynomial of the arrangement (product of lines)
F = L1 * L2 * L3 * L4 * L5

-- The ideal I of the configuration (singular locus of the union of lines)
-- Since no 3 lines are concurrent, the singular locus consists exactly of the 10 pairwise intersections.
-- The Jacobian ideal defines this locus.
I = trim ideal(diff(x,F), diff(y,F), diff(z,F))

-- Derive the points from the ideal for completeness
-- decompose returns the prime components (points) of the ideal
pts = decompose I
-- EVOLVE-BLOCK-END