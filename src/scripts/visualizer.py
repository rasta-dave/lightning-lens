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

def plot_rebalance_recommendations(df, output_dir, top_n=10):
    """ Create bar chart of top channels needing rebalancing """
    # Sort by absolute adjustment needed and get top N ...
    top_channels = df.loc[abs(df['adjustment_needed']).nlargest(top_n).index].copy()

    # Sort by adjustment value for better visualization ...
    top_channels = top_channels.sort_values('adjustment_needed')

    # Set up plot ...
    plt.figure(figsize=(12, 8))

    # Create colormap based on adjustment direction ...
    colors = ['#2a9d8f' if x > 0 else '#e63946' for x in top_channels['adjustment_needed']]

    # Create horizontal bar chart
    plt.barh(range(len(top_channels)), top_channels['adjustment_needed'], color=colors)

    # Improve channel ID formatting
    def format_channel_id(ch_id):
        ch_id = str(ch_id)
        if len(ch_id) > 12:
            return f"Ch...{ch_id[-6:]} ({ch_id[:6]}...)"
        return f"Ch {ch_id}"

    labels = [format_channel_id(ch_id) for ch_id in top_channels['channel_id']]

    # Add styling and labels
    plt.yticks(range(len(top_channels)), labels)
    plt.title(f'Top {top_n} Channels Needing Rebalancing', fontsize=16)
    plt.xlabel('Adjustment Needed (+ means pull funds in, - means push funds out)', fontsize=12)
    plt.ylabel('Channel ID', fontsize=12)
    plt.grid(axis='x', alpha=0.3)

    # Add a zero line...
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)

    # Add value labels on the bars
    for i, adjustment in enumerate(top_channels['adjustment_needed']):
        x_pos = adjustment + (0.01 if adjustment > 0 else -0.01)
        ha = 'left' if adjustment > 0 else 'right'
        plt.text(x_pos, i, f'{adjustment:.2f}', 
                va='center', ha=ha,
                fontsize=10, color='black')

    # Save figure ...
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rebalance_recommendations.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Created rebalance recommendations plot: {output_path}")
    return output_path
    
def plot_feature_importance(df, output_dir, target_column='predicted_optimal_ratio'):
    """ Create bar chart of feature importance """
    # Skip if there's no model prediction data ...
    if target_column not in df.columns:
        print("No model predictions found, skipping feature importance plot")
        return None
    
    # Select feature columns (include non-features)
    non_features = ['channel_id', 'timestamp', 'predicted_optimal_ratio',
                    'current_ratio', 'adjustment_needed']
    feature_cols = [col for col in df.columns if col not in non_features]

    if not feature_cols:
        print("No feature columns found, skipping feature importance plot")
        return None
    
    # Create and train a simple model to get feature importance ...
    X = df[feature_cols]
    y = df[target_column]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Get feature importance ...
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create plot with updated seaborn syntax
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=feature_importance,
        x='Importance',
        y='Feature',
        hue='Feature',
        legend=False,
        palette='viridis'
    )

    # Add value labels to the bars
    for i, v in enumerate(feature_importance['Importance']):
        ax.text(v, i, f'{v:.3f}', va='center')

    # Add styling and labels ...
    plt.title('Feature Importance for Balance Prediction', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.tight_layout()

    # Save figure ...
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Created feature importance plot: {output_path}")
    return output_path

def create_summary_report(df, output_dir, visualizations):
    """ Create a text summary report of findings """
    # Calculate key statistics ...
    total_channels = len(df)
    significant_rebalance = (abs(df['adjustment_needed']) > 0.1).sum()
    pull_in_count = (df['adjustment_needed'] > 0.1).sum()
    push_out_count = (df['adjustment_needed'] < -0.1).sum()

    # Calculate percentages ...
    pct_significant = significant_rebalance / total_channels * 100
    pct_pull = pull_in_count / total_channels * 100
    pct_push = push_out_count / total_channels * 100

    # Build report
    report = [
        "# Lightning Lens Rebalancing Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Summary Statistics",
        f"Total Channels Analyzed: {total_channels}",
        f" - Channels Needing More Local Balance: {pull_in_count} ({pct_pull:.1f}%)",
        f" - Channels Needing More Remote Balance: {push_out_count} ({pct_push:.1f}%)",
        f"\n## Top 5 Rebalancing Recommendations"
    ]

    # Add top 5 recommendations with improved formatting
    top_5 = df.loc[abs(df['adjustment_needed']).nlargest(5).index]
    for _, row in top_5.iterrows():
        ch_id = str(row['channel_id'])
        ch_id_short = f"{ch_id[:6]}...{ch_id[-6:]}" if len(ch_id) > 12 else ch_id
        current = row['balance_ratio']
        optimal = row['predicted_optimal_ratio']
        adjustment = row['adjustment_needed']

        direction = "Pull funds IN" if adjustment > 0 else "Push funds OUT"

        report.append(f"- Channel {ch_id_short}:")
        report.append(f"  • Current ratio: {current:.2f}")
        report.append(f"  • Target ratio: {optimal:.2f}")
        report.append(f"  • Action: {direction} by {abs(adjustment):.2f}")
        report.append("")  # Add blank line between entries

    # Add visualization paths to report ...
    report.append("\n## Visualizations Generated")
    for desc, path in visualizations.items():
        if path:
            report.append(f"- {desc}: {os.path.basename(path)}")

    # Write report to file ...
    report_path = os.path.join(output_dir, 'rebalance_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Created summary report: {report_path}")
    return report_path

def create_visualizations(predictions_csv, output_dir):
        """ Create all visualizations from predictions data """
        # Create output directory if needed
        output_dir = create_output_directory(output_dir)

        # Load data ...
        df = load_data(predictions_csv)
        if df is None:
            return

        print("\nGenerating visualizations ...")
        visualizations = {}

        # Create standard plots ...
        visualizations['Balance Distribution'] = plot_balance_distribution(df, output_dir)
        visualizations['Optimal vs Current Balance'] = plot_optimal_vs_current(df, output_dir)
        visualizations['Rebalance Recommendations'] = plot_rebalance_recommendations(df, output_dir)

        # Create feature importance plot if data available ...
        visualizations['Feature Importance'] = plot_feature_importance(df, output_dir)

        # Create summary report ...
        report_path = create_summary_report(df, output_dir, visualizations)

        print(f"\n✓ All visualizations and report created successfully in {output_dir}")
        print(f"Summary report: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for Lightning Network channel balance analysis")

    parser.add_argument("--predictions", required=True,
                        help="Path to CSV file with model predictions")
    parser.add_argument("--output", default="visualizations",
                        help="Directory to save visualizations (default: visualizations/)")
        
    args = parser.parse_args()

    create_visualizations(args.predictions, args.output)

if __name__ == "__main__":
    main()




    

    