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

-- Define a Star Configuration of 5 lines in P^2
-- The ideal is the intersection of the 10 points formed by pairwise intersections of 5 lines in general position
-- Lines: x=0, y=0, z=0, x+y+z=0, x+2y+5z=0
-- Such configurations are known to have high resurgence (containment failures like I^(3) not in I^2)
pts = {
    ideal(x, y),                -- L1 n L2: (0:0:1)
    ideal(x, z),                -- L1 n L3: (0:1:0)
    ideal(x, y+z),              -- L1 n L4: (0:1:-1)
    ideal(x, 2*y+5*z),          -- L1 n L5: (0:5:-2)
    ideal(y, z),                -- L2 n L3: (1:0:0)
    ideal(y, x+z),              -- L2 n L4: (1:0:-1)
    ideal(y, x+5*z),            -- L2 n L5: (5:0:-1)
    ideal(z, x+y),              -- L3 n L4: (1:-1:0)
    ideal(z, x+2*y),            -- L3 n L5: (2:-1:0)
    ideal(x+y+z, x+2*y+5*z)     -- L4 n L5: (3:-4:1)
}

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END