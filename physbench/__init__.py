"""Judge-free physics evaluation for generated video.

The chain is four stages, each replaceable:

    first frame  ->  VDM  ->  tracking  ->  calibration-invariant residuals

Nothing here estimates the unknown metre/pixel scale s or second/frame scale tau.
Every residual is built so those two nuisance factors cancel, which is what makes
the measurement possible on an uncalibrated generated clip.
"""

__version__ = "0.1.0"
