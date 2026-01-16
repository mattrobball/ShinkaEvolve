-- Resurgence evaluation template
-- This file loads a program that defines an ideal I, then computes
-- symbolic power containments to estimate resurgence
--
-- Usage: M2 --script resurgence_check.m2 --args <path_to_initial.m2>

needsPackage "SymbolicPowers"

-- Get the program path from command line arguments
programPath = null
args = commandLine
for i from 1 to #args - 1 do (
    if args#i == "--args" and i + 1 < #args then (
        programPath = args#(i+1);
        break;
    )
)

if programPath === null then (
    print "RESURGENCE_ERROR: No program path provided";
    exit 1;
)

-- Load the program which should define ideal I
loadError = null
try (
    load programPath;
) else err -> (
    loadError = err;
)

if loadError =!= null then (
    print("RESURGENCE_ERROR: Failed to load " | programPath);
    print("RESURGENCE_M2_ERROR: " | toString(loadError));
    exit 1;
)

-- Check that I is defined
if not isGlobalSymbol "I" then (
    print "RESURGENCE_ERROR: Program must define an ideal I";
    print "RESURGENCE_HINT: Make sure the code ends with 'I = intersect pts' or 'I = ideal(...)'";
    print "RESURGENCE_HINT: Check that intermediate computations don't fail silently";
    exit 1;
)

I = value getGlobalSymbol "I"

if not instance(I, Ideal) then (
    print "RESURGENCE_ERROR: I must be an ideal";
    exit 1;
)

-- Output basic ideal information
print("IDEAL_NUMGENS " | toString(numgens I))
print("IDEAL_DIM " | toString(dim I))
print("IDEAL_CODIM " | toString(codim I))

-- Check containments I^(m) ⊆ I^r for various (m, r) pairs
-- We check pairs where m/r is potentially interesting for resurgence
--
-- Key containment to check: I^(3) ⊆ I^2 (the famous containment problem)
-- If this fails, resurgence > 3/2

checkContainment = (m, r) -> (
    symPower = symbolicPower(I, m);
    ordPower = I^r;
    contained = isSubset(symPower, ordPower);
    ratio = m / r;
    print("CONTAINMENT " | toString(m) | " " | toString(r) | " " | toString(ratio * 1.0) | " " | toString(contained));
    contained
)

-- List of (m, r) pairs to check, ordered by ratio m/r descending
-- We want to find the largest ratio where containment fails
containmentPairs = {
    (4, 2),   -- ratio 2.0
    (6, 3),   -- ratio 2.0
    (5, 3),   -- ratio 1.667
    (7, 4),   -- ratio 1.75
    (3, 2),   -- ratio 1.5
    (5, 4),   -- ratio 1.25
    (4, 3),   -- ratio 1.333
    (2, 1),   -- ratio 2.0 (should always be contained for radical ideals)
    (3, 1),   -- ratio 3.0 (usually contained, but worth checking)
    (2, 2),   -- ratio 1.0 (always contained, baseline)
    (3, 3),   -- ratio 1.0 (always contained, baseline)
    (4, 4)    -- ratio 1.0 (always contained, baseline)
}

print "CONTAINMENT_START"
bestFailedRatio = 0.0
for pair in containmentPairs do (
    (m, r) = pair;
    try (
        contained = checkContainment(m, r);
        ratio = (m * 1.0) / r;
        if not contained and ratio > bestFailedRatio then (
            bestFailedRatio = ratio;
        )
    ) else (
        print("CONTAINMENT " | toString(m) | " " | toString(r) | " ERROR");
    )
)
print "CONTAINMENT_END"

-- Output the best (largest) ratio where containment failed
-- This is a lower bound on the resurgence
print("RESURGENCE_LOWER_BOUND " | toString(bestFailedRatio))

-- Also compute the Waldschmidt constant as another invariant
-- alpha_hat(I) = lim_{m->inf} alpha(I^(m))/m where alpha is initial degree
try (
    waldschmidt = waldschmidtConstant(I);
    print("WALDSCHMIDT " | toString(waldschmidt * 1.0));
) else (
    print "WALDSCHMIDT ERROR"
)

print "RESURGENCE_COMPLETE"
