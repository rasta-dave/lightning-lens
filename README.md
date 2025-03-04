# Lightning Lens 🔍⚡

An AI-powered tool for optimizing Lightning Network node liquidity through advanced prediction and analysis.

## Overview

LightningLens uses machine learning to help Lightning Network node operators optimize their channel liquidity. It analyzes network patterns, predicts optimal liquidity levels, and provides actionable recommendations for channel management.

## Features

- 📊 Network Analysis: Collect and analyze Lightning Network data
- 🤖 AI Predictions: Use machine learning to predict optimal liquidity levels
- 💡 Smart Recommendations: Get actionable insights for channel management
- 📈 Visualization: View network statistics and predictions through intuitive visualizations
- 🔄 Real-time Monitoring: Connect to Lightning Network simulations via WebSocket

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

## Real-time Integration with Lightning Network

LightningLens provides two methods for real-time integration with Lightning Network nodes or simulations:

### WebSocket Client

The WebSocket client connects to a Lightning Network simulation or monitoring service that broadcasts channel updates and transactions via WebSocket. This enables real-time analysis and recommendations.

```bash
python -m src.scripts.websocket_client
```

### HTTP Server

The HTTP server provides an API endpoint that can receive updates from Lightning Network nodes or simulations. It processes the data, generates recommendations, and makes them available via API.

```bash
python -m src.scripts.http_server
```

## Data Management

LightningLens collects and processes significant amounts of data during operation. To help manage this data, a cleanup utility is provided:

### Data Cleanup

The cleanup script helps manage the CSV files that accumulate in the data directories:

```bash
# Show what would be deleted without actually deleting anything
python -m scripts.cleanup_data --dry-run

# Delete all CSV files across all data directories
python -m scripts.cleanup_data

# Delete only transaction files
python -m scripts.cleanup_data --type transactions

# Keep the last 7 days of data
python -m scripts.cleanup_data --keep-days 7
```

#### Why Use the Cleanup Script

- **Prevent Disk Space Issues**: Over time, the continuous data collection can fill your disk space
- **Improve Performance**: Large amounts of data can slow down model training and analysis
- **Focus on Recent Data**: For most analyses, recent data is more relevant than historical data
- **Maintain Privacy**: Regular cleanup helps maintain privacy by removing old transaction data

#### Available Options

- `--dry-run`: Preview what would be deleted without actually deleting files
- `--keep-days N`: Keep files newer than N days
- `--type TYPE`: Only delete specific file types:
  - `transactions`: Transaction history files
  - `features`: Processed feature files
  - `channels`: Channel state files
  - `metrics`: Raw metrics files
  - `predictions`: Model prediction files

## Continuous Learning

LightningLens is designed to continuously improve its recommendations by learning from new data. The HTTP server automatically:

1. Saves transaction and channel data every 5 minutes
2. Retrains the model every 30 minutes using the latest data
3. Updates recommendations based on the newly trained model

This ensures that recommendations adapt to changing network conditions and become more accurate over time.

## Visualization

After analysis, you'll find:

- `visualizations/balance_distribution.png`: Current balance distribution
- `visualizations/optimal_vs_current.png`: Comparison plot
- `visualizations/rebalance_recommendations.png`: Top channels needing rebalancing
- `visualizations/feature_importance.png`: Feature importance analysis
- `visualizations/rebalance_report.md`: Detailed recommendations report

## Advanced Usage

### Custom Model Path

Use a specific model for the WebSocket client:

```bash
python -m src.scripts.websocket_client --model data/models/model_TIMESTAMP.pkl
```

### Verbose Logging

Enable detailed logging for troubleshooting:

```bash
python -m src.scripts.websocket_client --verbose
```

### Monitoring Logs

Monitor the activity of both services:

```bash
tail -f lightning_lens_http.log
tail -f lightning_lens_ws.log
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
