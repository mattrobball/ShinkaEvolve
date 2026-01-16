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

-- Star configuration of 6 lines in P^2
-- Defined as the set of pairwise intersections of 6 lines in general position.
-- This yields 15 points.
-- Star configurations are known to exhibit high resurgence.
-- For s lines, resurgence is often related to ratios involving s.
-- With s=6, we target containment failures like I^(5) not in I^3 (ratio 1.66).

-- Define 6 lines in general position (using coefficients to ensure no triple points)
L = {
    x,
    y,
    z,
    x + y + z,
    x - y + z,
    x + 2*y + 3*z
}

-- The ideal I of a star configuration is the intersection of the ideals of all pairs of lines.
-- There are binomial(6,2) = 15 such points.
-- We compute I as the intersection of these pair ideals.
pairIdeals = apply(subsets(L, 2), pair -> ideal(pair#0, pair#1))
I = intersect pairIdeals

-- Derive the points from the ideal for completeness
pts = decompose I
-- EVOLVE-BLOCK-END