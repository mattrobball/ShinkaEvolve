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

-- Strategy: Construct the configuration of points defined by the singular locus
-- of the Type B3 reflection arrangement.
-- The arrangement consists of 9 lines: x=0, y=0, z=0, x=±y, x=±z, y=±z.
-- The polynomial defining the union of these lines is Q.
-- The singular locus (points where at least 2 lines meet) creates a configuration
-- with points of multiplicity 4, 3, and 2.
-- These mixed high multiplicities are key for creating containment failures.

-- Define the defining polynomial of the arrangement
-- Q is degree 9
Q = x*y*z*(x^2-y^2)*(y^2-z^2)*(z^2-x^2)

-- The singular locus is defined by the Jacobian ideal J (vanishing of partial derivatives)
J = ideal(diff(x,Q), diff(y,Q), diff(z,Q))

-- The ideal I of the point configuration is the radical of J
-- We trim it to ensure minimal generators
I = trim radical J

-- Derive the points from the ideal for completeness
-- decompose returns the prime components (points) of the ideal
pts = decompose I
-- EVOLVE-BLOCK-END