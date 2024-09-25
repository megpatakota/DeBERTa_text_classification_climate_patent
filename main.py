# src/main.py
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import coloredlogs
from rich.logging import RichHandler
import logging
import torch
import yaml
import os
import argparse
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from src.data_utils import load_data, preprocess_data, sample_data, split_data
from src.train import get_training_arguments, train_model
from src.model_utils import create_datasets, predict_proba
from src.visual_results import generate_evaluation_reports_and_plots


def setup_logging(config):
    logging_level = getattr(logging, config["logging"]["level"])

    logging.basicConfig(
        level=logging_level,
        format=config["logging"]["format"],
        datefmt="[%X]",
        handlers=[RichHandler()],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a DeBERTa model on climate patent data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )


def main():
    args = parse_args()
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set up logging
    setup_logging(config)

    # Ensure output directories exist
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    os.makedirs(config["training"]["logging_dir"], exist_ok=True)
    os.makedirs(config["training"]["trained_model_dir"], exist_ok=True)

    # Load tokenizer and model
    logging.info("Loading tokenizer and model...")
    tokenizer = DebertaTokenizer.from_pretrained(config["model"]["name"])
    model = DebertaForSequenceClassification.from_pretrained(
        config["model"]["name"], num_labels=config["model"]["num_labels"]
    )

    # Load and preprocess data
    df = load_data(config)
    df = preprocess_data(df, config)
    df = sample_data(df, config)
    # Split data into train, validation, and test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = (
        split_data(df)
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        tokenizer,
        config,
    )
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training arguments
    training_args = get_training_arguments(config)

    # Train model
    trainer = train_model(
        model, train_dataset, val_dataset, test_dataset, training_args
    )

    # Save the trained model and tokenizer
    trained_model_dir = config["training"]["trained_model_dir"]
    logging.info(f"Saving the model to {trained_model_dir}...")
    model.save_pretrained(trained_model_dir)
    tokenizer.save_pretrained(trained_model_dir)

    logging.info("Model and tokenizer saved successfully.")

    # Evaluate model
    logging.info("Evaluating the model on the test set...")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    logging.info(f"Test set metrics: {metrics}")

    # # Example prediction
    # example_patent = df["full_text"].iloc[0]
    # prediction = predict_proba(example_patent, model, tokenizer, device, config)
    # logging.info(f"Prediction for the example patent: {prediction}")

    # Predict on all data
    logging.info("Predicting on all data...")
    df["predicted_yo2"] = predict_proba(
        df["full_text"].tolist(), model, tokenizer, device, config
    )

    # Generate evaluation reports and plots
    generate_evaluation_reports_and_plots(
        df=df, model=model, tokenizer=tokenizer, device=device, config=config
    )

    # Display sample predictions
    columns_to_view = ["title", "abstract", "claims", "yo2", "predicted_yo2"]
    logging.info("Displaying sample predictions:")
    logging.info(df[columns_to_view].head())


if __name__ == "__main__":
    main()
