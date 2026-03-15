# Part C formulas (only)

## Symbols
`s` = sugar fragment, `i` = interactor fragment.  
All surface areas are in Å².

Let:
- `PSA(tag)` = polar surface area on a chosen surface definition `tag`
- `NPSA(tag)` = nonpolar surface area on the same surface definition `tag`

In Multiwfn (module 12 summary), these correspond to:
- `Polar surface area (|ESP| > X kcal/mol)`
- `Nonpolar surface area (|ESP| <= X kcal/mol)`
(where commonly `X = 10`).  

QMPSA-style PSA partitions a surface into polar vs apolar regions based on ESP ranges.

## Uppercase descriptors
`QPSA = PSA_s(tag_Q) + PSA_i(tag_Q)`

`QNPSA = NPSA_s(tag_Q) + NPSA_i(tag_Q)`

Identity:
`QPSA + QNPSA = OverallSurfaceArea(tag_Q)`

## Lowercase descriptor used in the model
From the main paper:
`q2 = qNPSA = qNPSA_s + qNPSA_i`

## Sources
[1] Nature Chemistry (2021) main paper, Fig. 5 text: defines `q2` as “the sum of the quantum mechanical non-polar surface area (qNPSA) of both fragments”.  
[2] Supplementary methods: ESP-based fragment descriptors computed using Multiwfn and Molden; fragment-pair (subsystem) descriptors obtained by mathematical operations.  
[3] Schaftenaar & de Vlieg, “Quantum mechanical polar surface area”, J. Comput.-Aided Mol. Des. 26, 311–318 (2012): defines QMPSA/PSA by partitioning a molecular surface by ESP ranges.  
[4] Multiwfn documentation/discussion: reports polar/nonpolar surface area using an absolute ESP threshold in kcal/mol (e.g. 10 kcal/mol).
