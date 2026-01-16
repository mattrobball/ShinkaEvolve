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

-- We construct a configuration of 7 points.
-- First, the 6 vertices of a complete quadrilateral (Star Configuration of 4 lines).
-- The lines are x, y, z, x+y+z.
-- Vertices:
-- V1=(0:0:1), V2=(1:-1:0)  (on x+y=0)
-- V3=(0:1:0), V4=(1:0:-1)  (on x+z=0)
-- V5=(1:0:0), V6=(0:1:-1)  (on y+z=0)
--
-- We add the intersection of the diagonals x+y=0 and x+z=0.
-- P7 = (-1:1:1).
-- This point is collinear with V1,V2 and V3,V4.

pts = {
    ideal(x, y),           -- (0:0:1)
    ideal(x+y, z),         -- (1:-1:0)
    ideal(x, z),           -- (0:1:0)
    ideal(x+z, y),         -- (1:0:-1)
    ideal(y, z),           -- (1:0:0)
    ideal(y+z, x),         -- (0:1:-1)
    ideal(x+y, x+z)        -- (-1:1:1) intersection of diagonals
}

-- The ideal I is the intersection of the point ideals
I = intersect pts
-- EVOLVE-BLOCK-END