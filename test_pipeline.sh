#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
# pip install -r requirements.txt
echo "Skipping intsalling dependencies..."

# Create directories
mkdir -p models reports/comparisons reports/enhanced_tests

# Train initial models
echo "Training initial ML models..."
python train_initial_models.py

# Run enhanced testing framework
echo "Running enhanced tests..."
python enhanced_testing_framework.py --config ml_hybrid_strategy --timeframe 1h --days 180 --all

# Run optimization
echo "Running optimization..."
python auto_optimizer.py --config ml_hybrid_strategy --days 90

# Compare strategies
echo "Comparing strategies..."
python compare_strategies.py --timeframe 1h --days 180 --debug
