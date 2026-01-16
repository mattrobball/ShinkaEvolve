-- Resurgence optimization: find point configurations with large resurgence
-- The ideal I defined here will be evaluated for its resurgence properties
--
-- Resurgence rho(I) = sup{m/r : I^(m) not contained in I^r}
-- For generic points, resurgence is close to 1
-- Special configurations (Fermat points, star configurations) can have larger resurgence

-- EVOLVE-BLOCK-START
-- Define points in affine 2-space (or projective 2-space)
-- The evolution should find point configurations with high resurgence

R = QQ[x,y]

-- Define the ideal of a point configuration
-- Starting with a simple configuration - evolution should find better ones
-- Example: 4 points forming a square
pts = {
    ideal(x, y),           -- origin
    ideal(x-1, y),         -- (1,0)
    ideal(x, y-1),         -- (0,1)
    ideal(x-1, y-1)        -- (1,1)
}

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END
