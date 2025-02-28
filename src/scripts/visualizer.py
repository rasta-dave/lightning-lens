"""
Channel Balance Visualizer

Creates visualizations for Lightning Network channel balance analysis and rebalancing recommendations.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

def create_output_directory(output_dir):
    """ Create directory for output visualizations if it doesn't exist """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data(predictions_csv):
    """ Load predictions data from CSV file """
    try:
        df = pd.read_csv(predictions_csv)
        print(f"Loaded data with {len(df)} channels")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def plot_balance_distribution(df, output_dir):
    """ Create histogram of current balance ratios """
    plt.figure(figsize=(10, 6))

    # Create main histogram ...
    sns.histplot(df['balance_ratio'], bins=20, kde=True)

    # Add styling and labels
    plt.title('Distribution of Current Channel Balance Ratios', fontsize=16)
    plt.xlabel('Balance Ratio (0=Remote Side Full, 1=Local Side Full)', fontsize=12)
    plt.ylabel('Number of Channels', fontsize=12)
    plt.grid(alpha=0.3)

    # Add annotation about balanced channels
    balanced_channels = ((df['balance_ratio'] >= 0.4) & (df['balance_ratio'] <= 0.6)).sum()
    plt.annotate(f'{balanced_channels} channels ({balanced_channels/len(df)*100:.1f}%) have balanced ratios (0.4-0.6)',
                 xy=(0.5, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                 )
    
    # Save figure ...
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'balance_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Created balance distribution plot: {output_path}")
    return output_path

def plot_optimal_vs_current(df, output_dir):
    """ Create scatter plot of optimal vs current balance ratios """
    plt.figure(figsize=(10, 8))

    # Create scatter plot ...
    plt.scatter(df['balance_ratio'], df['predicted_optimal_ratio'],
                alpha=0.6, edgecolor='w', s=100)
    
    # Add diagonal line (no change line)
    min_val = min(df['balance_ratio'].min(), df['predicted_optimal_ratio'].min())
    max_val = max(df['balance_ratio'].max(), df['predicted_optimal_ratio'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
             label='No Change Needed'
             )
    
    # Add styling and labels ...
    plt.title('Current vs Predicted Optimal Channel Balance Ratios', fontsize=16)
    plt.xlabel('Current Balance Ratio', fontsize=12)
    plt.ylabel('Predicted Optimal Balance Ratio', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    # Add annotations ...
    need_more_local = (df['adjustment_needed'] > 0.1).sum()
    need_more_remote = (df['adjustment_needed'] < -0.1).sum()

    plt.annotate(f'{need_more_local} channels need more local balance',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="gray", alpha=0.8))
    
    plt.annotate(f'{need_more_remote} channels need more remote balance',
                 xy=(0.05, 0.87), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f8d5db", ec="gray", alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'optimal_vs_current.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Return the output path
    print(f"✓ Created optimal vs current plot: {output_path}")
    return output_path

def plot_rebalance_recommendations():
    pass