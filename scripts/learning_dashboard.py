#!/usr/bin/env python3
"""
Learning Dashboard for LightningLens

This script creates a simple dashboard to monitor the continuous learning
progress of the LightningLens model.
"""

import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_dashboard():
    """Generate a dashboard to monitor learning progress"""
    print("LightningLens Learning Dashboard")
    print("================================")
    
    # Create output directory
    os.makedirs("data/dashboard", exist_ok=True)
    
    # 1. Model Performance Metrics
    try:
        if os.path.exists('data/models/performance_metrics.json'):
            with open('data/models/performance_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            if metrics:
                # Extract metrics for plotting
                timestamps = []
                balanced_channels = []
                avg_adjustments = []
                
                for m in metrics:
                    if 'timestamp' in m and 'after' in m:
                        ts = m['timestamp']
                        time_obj = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                        formatted_time = time_obj.strftime("%m-%d %H:%M")
                        
                        timestamps.append(formatted_time)
                        balanced_channels.append(m['after'].get('balanced_channels', 0))
                        avg_adjustments.append(m['after'].get('avg_adjustment', 0))
                
                # Plot balanced channels over time
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, balanced_channels, marker='o')
                plt.title('Balanced Channels Over Time')
                plt.xlabel('Time')
                plt.ylabel('Proportion of Balanced Channels')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('data/dashboard/balanced_channels.png')
                
                # Plot average adjustment needed over time
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, avg_adjustments, marker='o', color='orange')
                plt.title('Average Adjustment Needed Over Time')
                plt.xlabel('Time')
                plt.ylabel('Average Adjustment')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('data/dashboard/avg_adjustment.png')
                
                print("Generated performance metric charts")
    except Exception as e:
        print(f"Error generating performance metrics: {str(e)}")
    
    # 2. Model Evolution
    try:
        if os.path.exists('data/visualizations/model_evolution.csv'):
            evolution_df = pd.read_csv('data/visualizations/model_evolution.csv')
            
            # Get unique channels
            channels = evolution_df['channel'].unique()
            
            # Plot optimal ratio convergence
            plt.figure(figsize=(12, 6))
            
            for channel in channels[:5]:  # Limit to 5 channels for readability
                channel_data = evolution_df[evolution_df['channel'] == channel]
                if not channel_data.empty:
                    plt.plot(
                        channel_data['timestamp'], 
                        abs(channel_data['adjustment_needed']),
                        marker='o',
                        label=channel
                    )
            
            plt.title('Convergence of Channel Recommendations')
            plt.xlabel('Time')
            plt.ylabel('Absolute Adjustment Needed')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/dashboard/convergence.png')
            
            print("Generated model evolution charts")
    except Exception as e:
        print(f"Error generating model evolution charts: {str(e)}")
    
    # 3. Model Retraining Frequency
    try:
        model_files = glob.glob("data/models/model_retrained_*.pkl")
        if model_files:
            # Extract timestamps
            timestamps = []
            for file in model_files:
                ts = file.split("model_retrained_")[1].split(".pkl")[0]
                time_obj = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                timestamps.append(time_obj)
            
            # Sort timestamps
            timestamps.sort()
            
            # Calculate time differences in minutes
            time_diffs = []
            formatted_times = []
            
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                time_diffs.append(diff)
                formatted_times.append(timestamps[i-1].strftime("%m-%d %H:%M"))
            
            # Plot retraining frequency
            plt.figure(figsize=(12, 6))
            plt.bar(formatted_times, time_diffs, color='green')
            plt.title('Time Between Model Retrainings')
            plt.xlabel('Retraining Time')
            plt.ylabel('Minutes Since Last Retraining')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/dashboard/retraining_frequency.png')
            
            print("Generated retraining frequency chart")
    except Exception as e:
        print(f"Error generating retraining frequency chart: {str(e)}")
    
    # 4. Generate HTML dashboard
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LightningLens Learning Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .chart-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>LightningLens Learning Dashboard</h1>
        <p>Last updated: {}</p>
        
        <div class="chart-container">
            <h2>Model Performance</h2>
            <img src="balanced_channels.png" alt="Balanced Channels">
            <img src="avg_adjustment.png" alt="Average Adjustment">
        </div>
        
        <div class="chart-container">
            <h2>Model Evolution</h2>
            <img src="convergence.png" alt="Convergence">
        </div>
        
        <div class="chart-container">
            <h2>Retraining Frequency</h2>
            <img src="retraining_frequency.png" alt="Retraining Frequency">
        </div>
    </body>
    </html>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open("data/dashboard/index.html", "w") as f:
        f.write(html_content)
    
    print("Generated HTML dashboard at data/dashboard/index.html")
    
    return True

if __name__ == "__main__":
    generate_dashboard() 