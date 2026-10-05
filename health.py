"""
REUSEGRID AI -- Health Model Wrapper (backend/health.py)
"""

from backend.health_model import BatteryHealthModel, apply_health_model

_health_model = None

def get_health_model():
    global _health_model
    if _health_model is None:
        _health_model = BatteryHealthModel()
    return _health_model

def calculate_health(battery):
    model = get_health_model()
    return model.predict(battery)
