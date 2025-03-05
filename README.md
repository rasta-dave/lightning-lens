# Lightning Lens 🔍⚡

An AI-powered tool for optimizing Lightning Network node liquidity through advanced prediction and analysis.

## Overview

LightningLens uses machine learning to help Lightning Network node operators optimize their channel liquidity. It analyzes network patterns, predicts optimal liquidity levels, and provides actionable recommendations for channel management.

## LightningLens and Simulation Interaction Schema

Here's a visual schema showing how the LightningLens ML model interacts with the Lightning Network simulation:

```
┌───────────────────────────────────────────────┐      ┌─────────────────────────────────────────┐
│                                               │      │                                         │
│       Lightning Network Simulation            │      │           LightningLens Model           │
│                                               │      │                                         │
├───────────────────────────────────────────────┤      ├─────────────────────────────────────────┤
│                                               │      │                                         │
│  ┌─────────────┐       ┌─────────────┐        │      │  ┌─────────────┐       ┌─────────────┐  │
│  │             │       │             │        │      │  │             │       │             │  │
│  │  Lightning  │       │  WebSocket  │        │      │  │  WebSocket  │       │    Model    │  │
│  │    Nodes    │◄─────►│   Server    │◄───────┼──────┼─►│   Client    │◄─────►│  Processor  │  │
│  │             │       │             │        │      │  │             │       │             │  │
│  └─────────────┘       └─────────────┘        │      │  └─────────────┘       └─────────────┘  │
│         ▲                     ▲               │      │                               │         │
│         │                     │               │      │                               │         │
│         │                     │               │      │                               │         │
│         ▼                     │               │      │                               ▼         │
│  ┌─────────────┐       ┌─────────────┐        │      │  ┌─────────────┐       ┌─────────────┐  │
│  │             │       │             │        │      │  │             │       │             │  │
│  │ Transaction │       │ Rebalancing │◄───────┼──────┼──┤ Suggestions │◄──────┤   Online    │  │
│  │  Generator  │──────►│   Engine    │        │      │  │  Generator  │       │   Learner   │  │
│  │             │       │             │        │      │  │             │       │             │  │
│  └─────────────┘       └─────────────┘        │      │  └─────────────┘       └─────────────┘  │
│         │                                     │      │                               ▲         │
│         ▼                                     │      │                               │         │
│  ┌─────────────┐       ┌─────────────┐        │      │                               │         │
│  │             │       │             │        │      │                               │         │
│  │   Channel   │──────►│   Feature   │────────┼──────┼─►                             │         │
│  │    State    │       │   Adapter   │        │      │                               │         │
│  │             │       │             │        │      │                               │         │
│  └─────────────┘       └─────────────┘        │      │                               │         │
│                                               │      │                               │         │
└───────────────────────────────────────────────┘      └───────────────────────────────┼─────────┘
                                                                                       │
                                                                              ┌────────┴─────────┐
                                                                              │                  │
                                                                              │  Trained Model   │
                                                                              │    & Scaler      │
                                                                              │                  │
                                                                              └──────────────────┘
```

## Data Flow:

1. **Simulation to Model:**

   - Channel state updates (balances, capacities)
   - Transaction data (sender, receiver, amount, success)
   - Network topology information

2. **Feature Adapter:**

   - Transforms raw channel state data into the format expected by the model
   - Adds required fields like `channel_id` and `balance_ratio`
   - Ensures feature names match those used during model training
   - Sends properly formatted data to the HTTP API

3. **Model to Simulation:**
   - Rebalancing suggestions (source node, target node, amount)
   - Confidence scores for each suggestion
   - Optimal balance ratios for each channel

## Communication Mechanism:

- **WebSocket Protocol:** Real-time bidirectional communication
- **HTTP API:** Alternative communication channel for updates
- **JSON Format:** Structured data exchange between systems
- **Feature Adapter:** Ensures data compatibility between systems

## Learning Cycle:

1. Simulation generates realistic payment patterns
2. Feature adapter transforms raw data into model-compatible format
3. Model observes channel states and transaction outcomes
4. Model learns optimal balance distributions
5. Model suggests rebalancing actions
6. Simulation applies high-confidence suggestions
7. Channel performance improves
8. Model continues learning from new data

This creates a continuous feedback loop where the model improves the simulation's performance, and the simulation provides more data for the model to learn from.

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

## Real-time Learning

LightningLens supports real-time learning from Lightning Network data:

### Online Learning

The system can learn continuously from transaction data as it arrives:

```bash
# Start HTTP server with online learning enabled
python -m src.scripts.http_server

# Start WebSocket client with online learning
python -m src.scripts.websocket_client
```

### Hybrid Learning Approach

LightningLens uses a hybrid approach that combines:

1. **Online Learning**: Continuously updates the model as new data arrives
2. **Batch Learning**: Periodically retrains the model using accumulated data

This provides both immediate adaptation to network changes and long-term stability.

### Training an Initial Model

For best results, train an initial model before starting online learning:

```bash
# Train initial model from existing data
python -m scripts.train_initial_model

# Use a specific features file
python -m scripts.train_initial_model --features data/processed/features_20250304_180000.csv
```

## License

This project is licensed under the MIT License

////////////////////////////////////////

## Running the Complete System

1. **Generate Initial Transaction Data**

   ```bash
   # Run the simulation for a short period (about 10 minutes) to generate transaction data
   ```

   This will create a transaction data file like `lightning_simulation_TIMESTAMP.csv` that will end up in the /data/processed directory.

2. **Generate Features from Transaction Data**

   ```bash
   # Generate features from the initial transaction data
   python scripts/generate_features.py \
     --input data/processed/transactions_initial.csv \
     --output data/processed/features_new.csv
   ```

   Replace the `transactions_initial.csv` name with the newly generated transaction data file.

   This will create a features file that can be used for model training.

3. **Train an Initial Model**

   ```bash
   # Train the model using:
   python -m scripts.train_initial_model
   ```

   This will save the model to the /data/models directory in the form of two .pkl files:

   ```bash
   # For example:
   model_initial_20250305_144628.pkl
   and
   scaler_initial_20250305_144628.pkl
   ```

   These are the two necessary files to move on to the next step.

4. **Start the HTTP Server with the Model**

   ```bash
   # Start the HTTP server with your trained model
   python -m src.scripts.http_server --model data/models/YOUR_MODEL_FILENAME.pkl
   ```

5. **Start the WebSocket Client**

   ```bash
   # Start the WebSocket client with the same model
   python -m src.scripts.websocket_client \
     --uri ws://localhost:6789 \
     --model data/models/YOUR_MODEL_FILENAME.pkl \
     --verbose
   ```

6. **Start the Simulation with Feature Adapter**

   ```bash
   # Run the simulation with the feature adapter
   python path/to/simulation.py
   ```

7. **Monitor the System**

   ```bash
   # In a new terminal window
   tail -f lightning_lens_http.log

   # In another terminal window
   tail -f lightning_lens_ws.log
   ```

This complete workflow ensures that:

1. You generate initial transaction data from the simulation
2. You generate proper features from your transaction data
3. You train a model on those features
4. Both the HTTP server and WebSocket client use the same trained model
5. The feature adapter in the simulation ensures data compatibility
6. You can monitor the system's operation in real-time

### First-Time Setup vs. Ongoing Operation

The steps above describe the complete first-time setup. For ongoing operation:

- Steps 1-4 only need to be done once to create the initial model
- Steps 5-8 are run each time you want to use the system
- The model will continue to learn and improve as more data is collected
