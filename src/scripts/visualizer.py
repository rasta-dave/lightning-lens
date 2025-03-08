#!/usr/bin/env python3
"""
LightningLens Prediction Visualizer

This script provides comprehensive visualizations for Lightning Network channel
balance predictions, helping node operators understand optimal liquidity distribution.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LightningLens Prediction Visualizer')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to the predictions CSV file')
    parser.add_argument('--output', type=str, default='data/visualizations',
                        help='Directory to save visualizations (default: data/visualizations)')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate more detailed visualizations')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top channels to highlight (default: 10)')
    return parser.parse_args()

def create_visualizations(predictions_file, output_dir, detailed=False, top_n=10):
    """Create visualizations from prediction data"""
    print(f"LightningLens Prediction Visualizer")
    print(f"==================================")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    try:
        df = pd.read_csv(predictions_file)
        print(f"Loaded predictions for {len(df)} channels from {predictions_file}")
    except Exception as e:
        print(f"Error loading predictions file: {str(e)}")
        return False
    
    # Basic data validation
    required_columns = ['channel_id', 'current_ratio', 'optimal_ratio', 'adjustment_needed']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in predictions file: {missing_columns}")
        return False
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Distribution of Current vs Optimal Balance Ratios
    plt.figure(figsize=(12, 6))
    sns.histplot(df['current_ratio'], kde=True, color='blue', alpha=0.5, label='Current Ratio')
    sns.histplot(df['optimal_ratio'], kde=True, color='green', alpha=0.5, label='Optimal Ratio')
    plt.title('Distribution of Current vs Optimal Balance Ratios')
    plt.xlabel('Balance Ratio (Local / Total)')
    plt.ylabel('Number of Channels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'balance_distribution_{timestamp}.png'))
    print(f"Generated balance distribution visualization")
    
    # 2. Adjustment Recommendations
    plt.figure(figsize=(12, 6))
    sns.histplot(df['adjustment_needed'], kde=True, bins=30)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Recommended Balance Adjustments')
    plt.xlabel('Adjustment Needed (+ means add local funds, - means remove)')
    plt.ylabel('Number of Channels')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'adjustment_distribution_{timestamp}.png'))
    print(f"Generated adjustment distribution visualization")
    
    # 3. Top Channels Needing Adjustment
    df['abs_adjustment'] = abs(df['adjustment_needed'])
    top_channels = df.nlargest(top_n, 'abs_adjustment')
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(top_channels['channel_id'].astype(str), top_channels['adjustment_needed'])
    
    # Color bars based on direction (add or remove funds)
    for i, bar in enumerate(bars):
        if top_channels.iloc[i]['adjustment_needed'] > 0:
            bar.set_color('green')  # Add funds
        else:
            bar.set_color('red')    # Remove funds
    
    plt.title(f'Top {top_n} Channels Needing Adjustment')
    plt.xlabel('Adjustment Needed')
    plt.ylabel('Channel ID')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_adjustments_{timestamp}.png'))
    print(f"Generated top adjustments visualization")
    
    # 4. Current vs Optimal Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(df['current_ratio'], df['optimal_ratio'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.title('Current vs Optimal Balance Ratios')
    plt.xlabel('Current Ratio')
    plt.ylabel('Optimal Ratio')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, f'current_vs_optimal_{timestamp}.png'))
    print(f"Generated current vs optimal visualization")
    
    # Generate detailed visualizations if requested
    if detailed:
        # 5. Balance Ratio Heatmap
        plt.figure(figsize=(12, 10))
        
        # Create bins for current and optimal ratios
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        df['current_bin'] = pd.cut(df['current_ratio'], bins, labels=bins[:-1])
        df['optimal_bin'] = pd.cut(df['optimal_ratio'], bins, labels=bins[:-1])
        
        # Create a cross-tabulation
        heatmap_data = pd.crosstab(df['current_bin'], df['optimal_bin'])
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Current vs Optimal Balance Ratio Distribution')
        plt.xlabel('Optimal Ratio Bin')
        plt.ylabel('Current Ratio Bin')
        plt.savefig(os.path.join(output_dir, f'ratio_heatmap_{timestamp}.png'))
        print(f"Generated ratio heatmap visualization")
        
        # 6. Adjustment vs Current Ratio
        plt.figure(figsize=(12, 6))
        plt.scatter(df['current_ratio'], df['adjustment_needed'], alpha=0.5)
        plt.title('Adjustment Needed vs Current Balance Ratio')
        plt.xlabel('Current Balance Ratio')
        plt.ylabel('Adjustment Needed')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'adjustment_vs_current_{timestamp}.png'))
        print(f"Generated adjustment vs current visualization")
    
    print(f"All visualizations saved to {output_dir}")
    return True

def main():
    """Main function"""
    args = parse_args()
    create_visualizations(args.predictions, args.output, args.detailed, args.top)

if __name__ == "__main__":
    main() 