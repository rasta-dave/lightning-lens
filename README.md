# Lightning Lens 🔍⚡

An AI-powered tool for optimizing Lightning Network node liquidity through advanced prediction and analysis.

## Overview

LightningLens uses machine learning to help Lightning Network node operators optimize their channel liquidity. It analyzes network patterns, predicts optimal liquidity levels, and provides actionable recommendations for channel management.

## Features

- 📊 Network Analysis: Collect and analyze Lightning Network data
- 🤖 AI Predictions: Use machine learning to predict optimal liquidity levels
- 💡 Smart Recommendations: Get actionable insights for channel management
- 📈 Visualization: View network statistics and predictions through intuitive visualizations

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate sample data (if needed):

```bash
python -m scripts.generate_sample_data
```

3. Train the model:

```bash
python -m src.scripts.lightning_lens train --data data/processed/features.csv
```

4. Analyze channels and get recommendations:

```bash
python -m src.scripts.lightning_lens analyze --data data/processed/features_new.csv
```

## Directory Structure

```
lightninglens/
├── configs/           # Configuration files
│   └── processed/    # Processed features
├── src/
│   ├── models/       # ML model code
│   ├── scripts/      # Main scripts
│   └── utils/        # Utility functions
├── test/             # Test files
└── visualizations/   # Generated plots
```

## Usage Guide

### Training a Model

Train a new model using historical channel data:

```bash
python -m src.scripts.lightning_lens train --data PATH_TO_TRAINING_DATA.csv
```

The model and scaler will be saved in `data/models/` with timestamps.

### Analyzing Channels

Analyze current channel balances and get rebalancing recommendations:

```bash
python -m src.scripts.lightning_lens analyze --data PATH_TO_CHANNEL_DATA.csv
```

This will:

1. Make predictions using the latest trained model
2. Generate visualizations in the `visualizations/` directory
3. Create a detailed report (`rebalance_report.md`) with recommendations

### Output Files

After analysis, you'll find:

- `visualizations/balance_distribution.png`: Current balance distribution
- `visualizations/optimal_vs_current.png`: Comparison plot
- `visualizations/rebalance_recommendations.png`: Top channels needing rebalancing
- `visualizations/feature_importance.png`: Feature importance analysis
- `visualizations/rebalance_report.md`: Detailed recommendations report

## Advanced Usage

Use a specific model for analysis:

```bash
python -m src.scripts.lightning_lens analyze \
  --data data/processed/features.csv \
  --model data/models/model_TIMESTAMP.pkl
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
