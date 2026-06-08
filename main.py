"""
Main entry point for CVX Trading Bot
"""
import logging
import argparse
import sys
from datetime import datetime
from train import main as train_main
from predict import main as predict_main
from backtest import main as backtest_main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train():
    """
    Train machine learning models.
    """
    logger.info("Starting model training...")
    lstm_model, xgb_model, lstm_metrics, xgb_metrics = train_main()
    logger.info("Model training completed successfully")
    return lstm_model, xgb_model


def predict():
    """
    Generate trading predictions and signals.
    """
    logger.info("Starting prediction engine...")
    results = predict_main()
    logger.info("Prediction completed successfully")
    return results


def backtest():
    """
    Run backtest on historical data.
    """
    logger.info("Starting backtest...")
    engine = backtest_main()
    logger.info("Backtest completed successfully")
    return engine


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description='CVX Political Trading Bot')
    parser.add_argument(
        'command',
        choices=['train', 'predict', 'backtest', 'full'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info(f"CVX TRADING BOT - {args.command.upper()} MODE")
    logger.info(f"Started: {datetime.now()}")
    logger.info("="*80)
    
    try:
        if args.command == 'train':
            train()
        elif args.command == 'predict':
            predict()
        elif args.command == 'backtest':
            backtest()
        elif args.command == 'full':
            logger.info("Running full pipeline: train -> backtest -> predict")
            train()
            backtest()
            predict()
        
        logger.info("\n" + "="*80)
        logger.info("OPERATION COMPLETED SUCCESSFULLY")
        logger.info(f"Finished: {datetime.now()}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
