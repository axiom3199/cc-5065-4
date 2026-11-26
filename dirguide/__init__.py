"""DirGuide: indirect-direction guiding laboratory.

A small CPU path tracer with an online, spatial directional-distribution
guide.  The guide only produces proposal directions; every estimator uses
the exact mixture pdf, so cached observations are never added as energy.
"""

__version__ = "1.0.0"
