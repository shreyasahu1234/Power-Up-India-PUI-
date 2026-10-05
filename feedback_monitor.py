"""
REUSEGRID AI Extension -- Degradation & Operational Feedback Monitor
Simulates continuous operating cycles, tracks cell divergence over time,
and generates predictive maintenance feedback for reconstructed batteries.
"""

import numpy as np
import pandas as pd


class FeedbackMonitor:
    @staticmethod
    def simulate_degradation_trajectory(initial_soh, initial_cap_delta, operating_cycles=500):
        """
        Simulates future capacity retention, impedance growth, and cell imbalance
        over 500 operating cycles based on initial cell balancing quality.
        """
        cycles = np.linspace(0, operating_cycles, 11)
        records = []
        
        # Degradation acceleration factor based on initial imbalance
        deg_rate = 0.032 + (initial_cap_delta * 0.008)
        
        for c in cycles:
            retention = round(max(50.0, 100.0 - (c * deg_rate) + np.random.normal(0, 0.3)), 1)
            soh_proj = round((initial_soh * retention) / 100.0, 1)
            ir_growth = round(c * 0.058 + np.random.normal(0, 0.4), 1)
            delta_v = round(6.0 + (c * 0.035) + (initial_cap_delta * 4.0) + np.random.normal(0, 0.8), 1)
            
            if c < 250:
                health_state = "Optimal Operation (Linear Wear)"
            elif c < 400:
                health_state = "Moderate Wear (Balancing Active)"
            else:
                health_state = "Secondary Retirement Impending"
                
            records.append({
                "Cycle": int(c),
                "Projected_SoH_Percent": soh_proj,
                "Capacity_Retention_Percent": retention,
                "Impedance_Growth_Percent": ir_growth,
                "Cell_Delta_V_mV": delta_v,
                "Operational_State": health_state
            })
            
        return pd.DataFrame(records)
