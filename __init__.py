"""
REUSEGRID AI -- Component Recovery & Multi-Battery Reconstruction Extension Package
Additive extension preserving the original backend and original dataset.
"""

from backend.extension.component_extractor import ComponentExtractor
from backend.extension.component_grading import ComponentGrader
from backend.extension.component_compatibility import ComponentCompatibilityEngine
from backend.extension.reconstruction_optimizer import ReconstructionOptimizer
from backend.extension.candidate_validator import CandidateBatteryValidator
from backend.extension.feedback_monitor import FeedbackMonitor
from backend.extension.extension_pipeline import (
    execute_full_reconstruction_pipeline,
    get_all_reconstruction_targets,
    get_all_components,
    get_inventory_summary
)
