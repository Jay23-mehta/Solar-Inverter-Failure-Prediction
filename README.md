# Solar-Inverter-Failure-Prediction
This is an AI-powered monitoring system for a solar power plant. It predicts which inverters are likely to fail or underperform in the next 7–10 days, explains why in plain English, and displays everything on a live dashboard — so plant operators can take action before a failure actually happens.
# Purpose
Solar inverters fail silently. By the time an operator notices reduced power output, the damage is already done — lost energy, expensive repairs, downtime. This system catches the warning signs early, using sensor data the inverters are already producing.

# System Architecture
The system is divided into four independent components that work together:
Component 1 — ML Prediction Model

The core intelligence of the system. It analyzes historical sensor readings from inverters and learns patterns that appear before a failure. It outputs a risk score between 0 and 100 for each inverter, where higher scores indicate greater likelihood of failure.

Component 2 — Flask REST API
The central communication layer. It acts as a bridge between the ML model, the GenAI layer, and the frontend dashboard. All data passes through the API, which processes requests, validates inputs, runs predictions, and returns structured responses.

Component 3 — Operational Dashboard
The visual interface used by plant operators. It displays risk scores, trend charts, a plant map, and AI-generated explanations in a single screen, allowing operators to monitor all inverters at a glance.

Component 4 — GenAI Insight Engine
The explainability layer. It takes the ML model's risk score and top contributing features, and generates a human-readable 3-sentence summary explaining what is wrong, why it is happening, and what action the operator should take.
