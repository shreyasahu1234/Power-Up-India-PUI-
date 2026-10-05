"""
REUSEGRID AI — Layer 2: Battery Health Intelligence Engine
Estimates State of Health (SoH %) and usable remaining capacity using Random Forest Regression.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class BatteryHealthModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._train_baseline()

    def _train_baseline(self):
        """
        Trains baseline model on degradation trajectories covering 50 to 1800 cycles.
        """
        np.random.seed(42)
        cycles = np.linspace(40, 1800, 400)
        age = cycles / 280.0
        v_cell = 4.2 - (cycles * 0.00045) + np.random.normal(0, 0.02, len(cycles))
        temp = 24.0 + (cycles * 0.007) + np.random.normal(0, 0.8, len(cycles))
        soh = 100.0 - (cycles * 0.024) - (age * 1.8) + np.random.normal(0, 1.5, len(cycles))
        soh = np.clip(soh, 35.0, 98.0)

        X = np.column_stack([v_cell, temp, cycles, age])
        y = soh
        self.model.fit(X, y)

    def predict(self, row):
        # Unsafe batteries receive zero health evaluation
        if row.get("safety_status") == "FAIL":
            return 0.0

        v = float(row.get("Measured_Voltage_V", row.get("voltage", 3.8)))
        pack_config = str(row.get("Pack_Configuration", "1S"))
        cells = 1
        if "S" in pack_config:
            try:
                cells = int(pack_config.replace("S", ""))
            except ValueError:
                cells = 1

        v_cell = v / max(1, cells)
        temp = float(row.get("Operating_Temperature_C", row.get("temperature", 28.0)))
        cycles = float(row.get("Historical_Cycles", row.get("cycles", 400)))
        age = float(row.get("Estimated_Age_Years", row.get("age_years", 2.0)))

        pred = self.model.predict([[v_cell, temp, cycles, age]])[0]

        # Penalize for dynamic internal resistance surge
        ir = float(row.get("Internal_Resistance_mOhm", 40.0))
        ir_penalty = max(0.0, (ir - (cells * 25.0)) * 0.04)
        pred_soh = np.clip(pred - ir_penalty, 35.0, 98.0)

        return round(float(pred_soh), 1)


def apply_health_model(df, health_model=None):
    if health_model is None:
        health_model = BatteryHealthModel()

    df_out = df.copy()
    df_out["estimated_health"] = df_out.apply(health_model.predict, axis=1)

    # Compute baseline remaining Wh
    def get_remaining_wh(r):
        if r.get("safety_status") == "FAIL":
            return 0.0
        nom_wh = float(r.get("Nominal_Energy_Wh", r.get("rated_capacity", 50.0)))
        return round((r["estimated_health"] / 100.0) * nom_wh, 1)

    df_out["remaining_energy_wh"] = df_out.apply(get_remaining_wh, axis=1)
    return df_out
