🚘 Edge-Enabled Digital Twin Framework for Connected Autonomous Vehicles
📌 Project Overview

This project implements an Edge-Enabled Digital Twin (DT) Framework designed for Connected and Autonomous Vehicles (CAVs).
It integrates Deep Reinforcement Learning (DRL) and Long Short-Term Memory (LSTM) models to optimize decision-making in dynamic vehicular networks.

The system enables a CAV to:

Predict future vehicle positions using LSTM.

Select the best CoV (Connected Vehicle) or Edge Digital Twin using DRL.

Adapt to varying latency, bandwidth, and trust levels of network connections.

Improve reliability of autonomous driving through context-aware digital twins.

🏗️ Architecture

Input Data: CARLA Simulator-generated vehicular states (position, velocity, etc.).

Prediction Module: LSTM model predicts future positions of the Vehicle Under Observation (VU).

Decision Module: DRL agent (PPO-based) selects between candidate CoVs or the Edge Digital Twin.

Reward Function: Balances latency, fidelity, bandwidth, and trust.

Execution Environment: Gymnasium-compatible simulation environment.

⚙️ Features

✅ LSTM-based trajectory prediction
✅ PPO-based DRL decision-making
✅ Gymnasium-compatible environment for training
✅ Edge vs. CoV offloading strategy
✅ CI/CD integration for reproducibility