"""plvr-primitives — mine reusable reasoning primitives from your own traces.

Bring traces, get a validated set of candidate operations and a runnable prompt for each.
Composing them into a program is left to you; see the README for why.
"""
from .cluster import Cluster, ClusterResult, mine, stability, sweep
from .prompts import Primitive, build_library, build_primitive, load_library, save_library
from .segment import Segment, build_segments, load_traces, mask, segments_of
from .validate import lift_report

__version__ = "0.1.0"
__all__ = [
    "Segment", "segments_of", "build_segments", "load_traces", "mask",
    "Cluster", "ClusterResult", "mine", "sweep", "stability",
    "lift_report",
    "Primitive", "build_primitive", "build_library", "save_library", "load_library",
]
