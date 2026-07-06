# Modular Deep Learning Core & SQL Preprocessing Pipeline

A production-grade, L-layer deep neural network engine built completely from scratch in NumPy with explicit mathematical matrix tracking, paired with a local SQL database and Pandas data preprocessing pipeline.

This repository serves as a foundational implementation demonstrating how raw mathematical deep learning models integrate with an enterprise-style data access layer.

## System Architecture

The project implements a clean three-tier data separation:
1. **Data Storage Tier (SQLite):** Maintains relational metadata tables tracking image file system paths and classification labels.
2. **Orchestration Tier (Pandas/Pillow):** Queries data frames via SQL, runs spatial array resizing, and scales matrix bounds.
3. **Engine Tier (NumPy/OOP):** An object-oriented deep neural network implementation managing network layers, forward loops, cost functions, backpropagation, and gradient updates.

## File Structure

* `model.py` - Contains the object-oriented `DeepNeuralNetwork` core configuration framework class.
* `init_db.py` - Sets up the SQLite tables and mock raw files to simulate seed states.
* `train_pipeline.py` - Pipeline driver script running queries, handling loading conversions, and optimizing the network.

## Dependencies

Ensure your execution environment has the following packages installed:

```bash
pip install numpy pandas Pillow
