"""
Model Runner - Train and utilize the LightningLens model
"""
import os
import argparse
import pandas as pd
from datetime import datetime
from src.models.trainer import ModelTrainer
from src.models.features import FeatureProcessor
from src.utils.config import load_config

def train_model(config_path, data_path, output_dir, target_column="balance_ratio"):
    """ Train a new model using processed feature data 
    
    Args:
        config_path (str): Path to configuration file
        data_path (str): Path to processed features CSV
        output_dir (str): Directory to save model artifacts

    """
    # Load configuration ...
    config = load_config(config_path)
    model_config = config.get("model", {})

    # Load data ...
    print(f"Loading data from {data_path}...")
    features_df = pd.read_csv(data_path)

    # Initialize the ModelTrainer ...
    trainer = ModelTrainer(
        test_size=config.get("training", {}).get("validation_split", 0.2),
        random_state=42
    )

    # Training the model ...
    print("Training model ...")
    model_params = {
        'n_estimators': model_config.get("n_estimators", 100),
        'max_depth': model_config.get("max_depth", 10),
        'min_samples_split': model_config.get("min_sampples_split", 2),
        'random_state': 42
    }

    model, metrics = trainer.train_model(
        features_df,
        target_column=target_column,
        model_params=model_params
    )

    # Printing metrics ...
    print("\nModel Performance:")
    print(f" MAE: {metrics['mae']:.4f}")
    print(f" RMSE: {metrics['rmse']:.4f}")
    print(f" R2: {metrics['r2']:.4f}")

    # Save model and scaler ...
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"model_{timestamp}.pkl")
    scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(model_path, scaler_path)



def predict_optimal_ratios(model_path: str, data_path: str, output_dir: str = "data/predictions"):
    """Make predictions using a trained model
    
    Args:
        model_path (str): Path to saved model
        data_path (str): Path to input data
        output_dir (str): Directory to save predictions (default: data/predictions)
    """
    # Create predictions directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Load model and make predictions
    trainer = ModelTrainer()
    model = trainer.load_model(model_path)
    predictions = trainer.predict(model, df)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    
    # Save predictions
    result_df = df.copy()
    result_df['predicted_optimal_ratio'] = predictions
    result_df['adjustment_needed'] = predictions - result_df['balance_ratio']
    result_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    
    # Print top rebalancing recommendations
    print("\nTop rebalancing recommendations:")
    top_rebalance = result_df.loc[abs(result_df['adjustment_needed']).nlargest(5).index]
    for _, row in top_rebalance.iterrows():
        channel_id = row['channel_id']
        adjustment = row['adjustment_needed']
        direction = "Push funds OUT" if adjustment < 0 else "Pull funds IN"
        print(f" Channel {channel_id}: {direction} by {abs(adjustment):.2f}")


#############################################

def main():
    parser = argparse.ArgumentParser(description='Train LightningLens model')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--data', required=True, help='Path to processed features CSV')
    parser.add_argument('--output', default='data/models', help='Output directory for model')
    
    args = parser.parse_args()
    train_model(args.config, args.data, args.output)

if __name__ == "__main__":
    main()

