"""LLNL replication of Wang/Khor/Desnoyers 2DIO IRM-class trace synthesis baseline.

This package re-implements the Independent Reference Model (Coffman &
Denning §6.6) and the 2-dimensional time-bucketed IRM (2DIO) extension
as described in the Wang/Khor/Desnoyers literature, for direct
empirical comparison against LLGAN/altgan/newgan generators on the
race corpora.

Models:
- irm.py        : pure i.i.d. sampling from real trace's frequency rank PMF
- irm_2dio.py   : rank PMF conditioned on time-of-trace bucket (4 bins)

CLI:
    python -m desnoyers.irm fit --real <real.csv> --output <model.pkl>
    python -m desnoyers.irm generate --model <model.pkl> --n <N> --seed <S> --output <fake.csv>
"""
