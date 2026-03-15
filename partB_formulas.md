# Part‑B descriptor formulas (only)

## Fragment‑level (from Multiwfn module 12 “Summary of surface analysis”)

For each fragment f ∈ {sugar (s), interactor (i)}:

- Vol_f = Volume (Å³)
- SA_f = Overall surface area (Å²)
- PSA_f = Positive surface area (Å²)  [ESP > 0]
- NSA_f = Negative surface area (Å²)  [ESP < 0]
- MinVal_f = Minimal value (kcal/mol)
- MaxVal_f = Maximal value (kcal/mol)
- Avg_f = Overall average value (kcal/mol)
- Avg+_f = Positive average value (kcal/mol)
- Avg−_f = Negative average value (kcal/mol)
- Var_f = Overall variance σ²_tot ((kcal/mol)²)
- Var+_f = Positive variance ((kcal/mol)²)
- Var−_f = Negative variance ((kcal/mol)²)
- miu_f = Balance of charges (ν)  [reported by Multiwfn as “nu”]
- (Var·miu)_f = Product of σ²_tot and ν ((kcal/mol)²)
- Pi_f = Internal charge separation Π (kcal/mol)
- MinESP_f = Global surface minimum (a.u.)
- MaxESP_f = Global surface maximum (a.u.)
- qPSA_f = Polar surface area (Å²)  [|ESP| > threshold]
- qNPSA_f = Nonpolar surface area (Å²)  [|ESP| ≤ threshold]
- TSA_f = Total surface area (Å²) = qPSA_f + qNPSA_f

## System‑level (pair) descriptors for sugar–interactor pair (s, i)

- PSAs+NSAi = PSA_s + NSA_i   (q1)
- PSAs*NSAi = PSA_s · NSA_i
- NSAs+PSAi = NSA_s + PSA_i
- NSAs*PSAi = NSA_s · PSA_i

- Overall surface area = SA_s + SA_i
- Positive surface area = PSAs+NSAi
- Negative surface area = Overall surface area − Positive surface area
- PSAs*NSAi/overall surface = (PSAs*NSAi) / (Overall surface area)

- Volume = Vol_s + Vol_i

- Minimal value = min(MinVal_s, MinVal_i)
- Maximal value = max(MaxVal_s, MaxVal_i)
- MinESP = min(MinESP_s, MinESP_i)
- MaxESP = max(MaxESP_s, MaxESP_i)

- Overall average value = Avg_s + Avg_i
- Positive average value = Avg+_s + Avg+_i
- Negative average value = Avg−_s + Avg−_i

- Overall variance (sigma^2_tot) = Var_s + Var_i
- Positive variance = Var+_s + Var+_i
- Negative variance = Var−_s + Var−_i

- Balance of charges (miu) = miu_s + miu_i
- Product of sigma^2_tot and miu = (Var·miu)_s + (Var·miu)_i
- Internal charge separation (Pi) = Pi_s + Pi_i

- qPSA = qPSA_s + qPSA_i
- qNPSA = qNPSA_s + qNPSA_i
- Total Surface area = qPSA + qNPSA

## Sources (for the formulas / terms)

- q1 definition: q1 = PSAsug + NSAint.
- q2 definition: q2 = sum of qNPSA of both fragments.
- “Descriptors of fragment pairs were then combined through mathematical operations to obtain subsystem descriptors.”
- “Molecular volumes … using Multiwfn … with ρ = 0.001 isovalue” (for how Volume is obtained at fragment level).
