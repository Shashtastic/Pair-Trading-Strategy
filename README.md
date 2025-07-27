# Pair Trading Strategy
This repository contains three main Python scripts that form a sequential workflow for building a quantitative trading portfolio based on cointegration signals:

# Components
Cointegration Checker (Cointegration checker - Industry Agnostic.py)

Threshold Checker (Threshold Checker for Pairs.py)

Final Portfolio Builder (Final Portfolio Builder.py)

# How Does it Work
The output of each stage feeds into the next:

The Cointegration Checker generates a list of cointegrated pairs and their spread ratios.

Those spread ratios are manually fed into the Threshold Checker, which evaluates trading signals against predefined thresholds and formats its output into an Excel file.

The Final Portfolio Builder reads the Excel output to construct and backtest the final portfolio.
