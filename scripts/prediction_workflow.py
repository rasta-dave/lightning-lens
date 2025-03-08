#!/usr/bin/env python3
"""
LightningLens Prediction Workflow

This script provides a complete workflow for generating predictions,
creating visualizations, and managing prediction files.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PredictionWorkflow")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LightningLens Prediction Workflow')
    parser.add_argument('--count', type=int, default=1,
                        help='Number of predictions to generate (default: 1)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Interval between predictions in minutes (default: 5)')
    parser.add_argument('--keep', type=int, default=20,
                        help='Number of prediction files to keep (default: 20)')
    parser.add_argument('--archive', action='store_true',
                        help='Archive old files instead of deleting them')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed visualizations')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip prediction generation (only visualize and archive)')
    parser.add_argument('--skip-visualize', action='store_true',
                        help='Skip visualization (only generate and archive)')
    parser.add_argument('--skip-archive', action='store_true',
                        help='Skip archiving (only generate and visualize)')
    return parser.parse_args()

def run_workflow():
    """Run the complete prediction workflow"""
    args = parse_args()
    
    logger.info("Starting LightningLens Prediction Workflow")
    
    # Step 1: Generate predictions (if not skipped)
    if not args.skip_generate:
        logger.info("Generating predictions")
        try:
            from scripts.auto_generate_predictions import auto_generate_predictions
            
            # Save original sys.argv
            original_argv = sys.argv.copy()
            
            # Set up arguments for auto_generate_predictions
            generate_args = ['auto_generate_predictions.py']
            generate_args.extend(['--count', str(args.count)])
            generate_args.extend(['--interval', str(args.interval)])
            sys.argv = generate_args
            
            # Run the auto-generate function
            auto_generate_predictions()
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            logger.info("Successfully generated predictions")
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return False
    
    # Step 2: Create visualizations (if not skipped)
    if not args.skip_visualize:
        logger.info("Creating visualizations")
        try:
            # First, run the model evolution visualizer
            from scripts.visualize_model_evolution import visualize_model_evolution
            visualize_model_evolution()
            logger.info("Generated model evolution visualizations")
            
            # Then, run the detailed visualizer
            from src.scripts.visualizer import create_visualizations
            create_visualizations(
                "data/predictions/latest_predictions.csv",
                "data/visualizations",
                detailed=args.detailed
            )
            logger.info("Generated detailed visualizations")
            
            # Finally, generate the dashboard
            from scripts.learning_dashboard import generate_dashboard
            generate_dashboard()
            logger.info("Generated learning dashboard")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    # Step 3: Archive old prediction files (if not skipped)
    if not args.skip_archive:
        logger.info("Archiving old prediction files")
        try:
            from scripts.archive_predictions import archive_predictions
            
            # Save original sys.argv
            original_argv = sys.argv.copy()
            
            # Set up arguments for archive_predictions
            archive_args = ['archive_predictions.py', '--keep', str(args.keep)]
            if args.archive:
                archive_args.append('--archive')
            sys.argv = archive_args
            
            # Run the archive function
            archive_predictions()
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            logger.info("Successfully archived old prediction files")
        except Exception as e:
            logger.error(f"Error archiving prediction files: {str(e)}")
    
    logger.info("Prediction workflow complete")
    return True

if __name__ == "__main__":
    run_workflow() 