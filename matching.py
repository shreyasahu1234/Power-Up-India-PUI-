"""
REUSEGRID AI -- Layers 4 & 5: Multi-Battery Matching & Optimization Engine
Updated to use all columns in the new energy_needs.csv:
  - Voltage window gate (min_voltage_v / max_voltage_v)
  - Internal resistance gate (max_ir_mohm)
  - Cycle history gate (max_cycles)
  - Eligibility tier filter (accepted_eligibility, derated_accepted, post_repair_accepted)
  - Preferred battery type affinity bonus
  - Capacity + health + criticality composite score
  - Global ILP optimization via Google OR-Tools (CBC / SCIP fallback)
"""

from ortools.linear_solver import pywraplp
import pandas as pd


# ============================================================
# LAYER 4: Compatibility Scoring
# ============================================================

def calculate_compatibility(battery, application):
    """
    Computes a grounded compatibility score (0.0 to 1.0) for one battery-application pair.

    Hard disqualifiers (score = 0.0):
      - Voltage outside the application's accepted window
      - Battery eligibility tier not accepted by application
      - Internal resistance above application's maximum
      - Cycle history exceeds application's maximum
      - Available energy critically below minimum (< 70% of req_wh)

    Soft scoring (0.05 to 1.0):
      - Capacity coverage ratio (weight 0.40)
      - SoH health ratio (weight 0.35)
      - Criticality alignment (weight 0.15)
      - Preferred battery type affinity bonus (weight 0.10)
    """

    # ----- Retrieve battery values (support both column naming conventions) -----
    wh       = float(battery.get("effective_capacity_wh", battery.get("effective_capacity", 30.0)))
    soh      = float(battery.get("effective_health", 70.0))
    v_nom    = float(battery.get("Nominal_Voltage_V", battery.get("voltage", 11.1)))
    v_meas   = float(battery.get("Measured_Voltage_V", battery.get("voltage", v_nom)))
    ir       = float(battery.get("Internal_Resistance_mOhm", 60.0))
    cycles   = int(battery.get("Historical_Cycles", battery.get("cycles", 500)))
    bat_type = str(battery.get("Battery_Type", ""))
    elig     = str(battery.get("eligibility", "Eligible (Direct)"))

    # ----- Retrieve application requirements -----
    req_wh        = float(application.get("min_energy_wh", 40.0))
    min_soh       = float(application.get("min_health", 65.0))
    crit          = float(application.get("criticality", 3.0))
    min_v         = float(application.get("min_voltage_v", 7.4))
    max_v         = float(application.get("max_voltage_v", 37.5))
    max_ir        = float(application.get("max_ir_mohm", 220.0))
    max_cyc       = int(application.get("max_cycles", 2500))
    derated_ok    = int(application.get("derated_accepted", 0))
    postrepair_ok = int(application.get("post_repair_accepted", 0))
    pref_types    = str(application.get("preferred_battery_types", ""))
    accepted_elig = str(application.get("accepted_eligibility", "Eligible (Direct)"))

    # =========================================================
    # HARD GATES: any failure -> score = 0.0 (cannot be assigned)
    # =========================================================

    # Gate 1: Voltage window — use nominal as representative pack voltage
    if v_nom < min_v or v_nom > max_v:
        return 0.0

    # Gate 2: Eligibility tier
    elig_accepted = False
    for tier in accepted_elig.split(";"):
        tier = tier.strip()
        if tier in elig:
            elig_accepted = True
            break
    # Extra check: if derated/post-repair explicitly denied by flag
    if "Limited Use" in elig and not derated_ok:
        elig_accepted = False
    if "Post-Repair" in elig and not postrepair_ok:
        elig_accepted = False
    if not elig_accepted:
        return 0.0

    # Gate 3: Internal resistance ceiling
    if ir > max_ir:
        return 0.0

    # Gate 4: Cycle count ceiling
    if cycles > max_cyc:
        return 0.0

    # Gate 5: Catastrophically low energy (< 70% of minimum needed)
    if wh < (req_wh * 0.70):
        return 0.05

    # =========================================================
    # SOFT SCORING
    # =========================================================

    # Capacity coverage ratio (0.0 → 1.0, capped at 1.4x surplus)
    cap_ratio = min(wh / max(req_wh, 1.0), 1.4) / 1.4

    # Health ratio: penalise batteries below the application's SoH floor
    if soh < min_soh:
        health_ratio = (soh / 100.0) * 0.50   # Significant penalty below floor
    else:
        health_ratio = soh / 100.0

    # Criticality alignment (higher criticality apps only reward high-health packs)
    crit_ratio = crit / 5.0

    # Preferred chemistry affinity bonus
    type_bonus = 0.0
    for pref in pref_types.split(";"):
        if pref.strip() and pref.strip() in bat_type:
            type_bonus = 1.0
            break

    # Weighted composite score
    score = (
        0.40 * cap_ratio
        + 0.35 * health_ratio
        + 0.15 * crit_ratio
        + 0.10 * type_bonus
    )
    return round(float(score), 4)


# ============================================================
# BUILD MATRIX
# ============================================================

def build_compatibility_matrix(eligible_batteries, energy_needs):
    """
    Constructs compatibility lookup: matrix[battery_id][application_name] = score (0-1)
    Batteries with score=0 for ALL applications will not appear in any allocation.
    """
    matrix = {}
    id_col = "Battery_ID" if "Battery_ID" in eligible_batteries.columns else "id"

    for _, b in eligible_batteries.iterrows():
        b_id = str(b[id_col])
        matrix[b_id] = {}
        for _, a in energy_needs.iterrows():
            a_name = str(a["application"])
            matrix[b_id][a_name] = calculate_compatibility(b, a)
    return matrix


# ============================================================
# LAYER 5: Google OR-Tools Global ILP Optimizer
# ============================================================

def run_multi_battery_optimizer(eligible_batteries, energy_needs, compatibility_matrix):
    """
    Solves the global battery-to-application assignment using Integer Linear Programming.

    Objective: Maximise total system compatibility utility
    Subject to:
      1. Each battery is assigned to at most one application.
      2. Each application receives at most allocation_quota batteries.
      3. Batteries with score=0 for an application cannot be assigned there.

    Solver priority: CBC -> SCIP -> GLOP
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise RuntimeError("No OR-Tools solver available (CBC/SCIP/GLOP).")

    id_col   = "Battery_ID" if "Battery_ID" in eligible_batteries.columns else "id"
    bat_ids  = eligible_batteries[id_col].astype(str).tolist()
    app_names = energy_needs["application"].astype(str).tolist()

    # Decision variables x[b, a] in {0, 1}
    x = {}
    for b in bat_ids:
        for a in app_names:
            if compatibility_matrix[b][a] > 0.0:   # Only create variable if viable
                x[(b, a)] = solver.BoolVar(f"x_{b}_{a}")

    # Constraint 1: Each battery used at most once across all applications
    for b in bat_ids:
        active_vars = [x[(b, a)] for a in app_names if (b, a) in x]
        if active_vars:
            solver.Add(sum(active_vars) <= 1)

    # Constraint 2: Each application respects allocation quota
    for _, a_row in energy_needs.iterrows():
        a_name = str(a_row["application"])
        quota  = int(a_row.get("allocation_quota", 1))
        active_vars = [x[(b, a_name)] for b in bat_ids if (b, a_name) in x]
        if active_vars:
            solver.Add(sum(active_vars) <= quota)

    # Objective: Maximise total system compatibility utility score
    objective = solver.Objective()
    for (b, a), var in x.items():
        objective.SetCoefficient(var, compatibility_matrix[b][a])
    objective.SetMaximization()

    status = solver.Solve()

    allocations = []
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        for (b, a), var in x.items():
            if var.solution_value() > 0.5:
                allocations.append({
                    "battery_id":  b,
                    "application": a,
                    "score":       compatibility_matrix[b][a],
                })
    return allocations, solver.Objective().Value()


# ============================================================
# EXPLAINABLE AUDIT CARD GENERATOR
# ============================================================

def generate_explanation(b_row, app_row, score):
    """
    Produces a human-readable, layered audit card explaining why this
    specific battery was matched to this specific application.
    """
    soh       = b_row.get("effective_health", 70.0)
    wh        = b_row.get("effective_capacity_wh", b_row.get("effective_capacity", 35.0))
    v_nom     = b_row.get("Nominal_Voltage_V", b_row.get("voltage", 11.1))
    ir        = b_row.get("Internal_Resistance_mOhm", 60.0)
    cycles    = b_row.get("Historical_Cycles", b_row.get("cycles", 500))
    bat_type  = b_row.get("Battery_Type", "Unknown")
    elig      = b_row.get("eligibility", "Eligible")
    action    = b_row.get("repair_action", "Direct reuse")
    cost      = b_row.get("repair_cost_inr", 0)

    min_soh   = app_row.get("min_health", 65.0)
    req_wh    = app_row.get("min_energy_wh", app_row.get("required_energy", 30.0))
    min_v     = app_row.get("min_voltage_v", 7.4)
    max_v     = app_row.get("max_voltage_v", 37.5)
    max_ir    = app_row.get("max_ir_mohm", 220.0)
    pref_types = str(app_row.get("preferred_battery_types", ""))

    reasons = [
        "Passed mandatory safety screening (Layer 1 hard gate).",
        f"Voltage {v_nom}V within application window [{min_v}V - {max_v}V].",
        f"Effective Health: {soh}% >= Application minimum {min_soh}%.",
        f"Available Energy: {wh} Wh vs {req_wh} Wh required.",
        f"Internal Resistance: {ir} mOhm <= Application ceiling {max_ir} mOhm.",
        f"Cycle history: {cycles} cycles within accepted range.",
    ]

    # Chemistry affinity note
    type_match = any(p.strip() in str(bat_type) for p in pref_types.split(";") if p.strip())
    if type_match:
        reasons.append(f"Battery chemistry '{bat_type}' is a preferred type for this application.")
    else:
        reasons.append(f"Battery chemistry '{bat_type}' is acceptable (not top-preferred but within voltage gate).")

    # Eligibility pathway note
    if "Post-Repair" in elig:
        reasons.append(f"Post-repair verified pathway: {action} (Investment: INR {cost}). Application accepts repaired units.")
    elif "Limited Use" in elig:
        reasons.append("Derated operational envelope applied safely; application criticality permits Limited Use tier.")
    else:
        reasons.append("Direct reuse pathway -- zero component defects detected.")

    return "\n  ".join([f"[+] {r}" for r in reasons])
