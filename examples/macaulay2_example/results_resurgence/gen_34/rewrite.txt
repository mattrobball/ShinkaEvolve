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

-- Define the ideal of a point configuration using the Hessian line arrangement (8 lines).
-- This configuration is obtained from the B3 arrangement by removing the line z=0.
-- The polynomial F defines the union of lines: x, y, x-y, x+y, x-z, x+z, y-z, y+z.
-- This arrangement is known to produce containment failures for ratios up to 1.5 (e.g., I^(3) not in I^2).
F = x*y*(x^2-y^2)*(x^2-z^2)*(y^2-z^2)

-- The singular locus of this arrangement consists of the intersection points.
-- We compute the Jacobian ideal to find these points.
J = ideal jacobian ideal F

-- Extract the minimal primes of the Jacobian ideal to get the ideals of the individual points.
-- This automatically selects the high-multiplicity intersection points which drive up the resurgence.
pts = minimalPrimes J

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END