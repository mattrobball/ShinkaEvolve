-- EVOLVE-BLOCK-START
-- Example Macaulay2 program to compute Hilbert polynomial
-- This is a simple initial program that will be evolved

R = QQ[x,y,z]
I = ideal(x^2, x*y, y^2)
hilbertPolynomial(I, Projective => false)
-- EVOLVE-BLOCK-END
