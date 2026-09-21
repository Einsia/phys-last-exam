#!/usr/bin/env python3
"""Synthetic regression checks for period and P9 invariant extraction."""
from pathlib import Path
import math
import numpy as np
import yaml

from evaluate import period_measure


def main():
    cfg=yaml.safe_load((Path(__file__).parent/'config.yaml').read_text())
    rng=np.random.default_rng(9); t=np.arange(124)/24
    Ts=1.60; Tl=Ts*math.sqrt(2)
    ys=.22*np.cos(2*np.pi*t/Ts)+rng.normal(0,.002,len(t))
    yl=.22*np.cos(2*np.pi*t/Tl)+rng.normal(0,.002,len(t))
    # Deterministic missing-coordinate regression.
    ys[[17,18,71]]=np.nan; yl[[33,34]]=np.nan
    ms=period_measure(ys,t,cfg); ml=period_measure(yl,t,cfg)
    assert ms and ml
    assert abs(ms['period_s']/Ts-1)<.05, ms
    assert abs(ml['period_s']/Tl-1)<.05, ml
    m1=abs((ms['period_s']/ml['period_s'])**2/.5-1)
    assert m1<.12, m1
    # Constant signal cannot silently become a zero-residual valid measurement.
    flat=period_measure(np.zeros_like(t),t,cfg)
    assert flat is None or flat['fit_r2'] is None or flat['amplitude_deg']<1
    print({'short_period':ms['period_s'],'long_period':ml['period_s'],'m1':m1})


if __name__=='__main__': main()
