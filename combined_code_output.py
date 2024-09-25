# --- model_utils.py ---
# src/model_utils.py

import logging
import torch
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from datasets import Dataset
import numpy as np


def tokenize_function(texts, tokenizer, config):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    pr_auc = average_precision_score(labels, logits[:, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    # Compute ROC-AUC, but only if both classes are present
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(
            labels, logits[:, 1]
        )  # Only if both classes are present
    else:
        roc_auc = None  # Cannot compute ROC-AUC with only one class

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def create_datasets(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    test_texts,
    test_labels,
    tokenizer,
    config,
):
    logging.info("Tokenizing training data...")
    train_inputs = tokenize_function(train_texts, tokenizer, config)

    logging.info("Tokenizing validation data...")
    val_inputs = tokenize_function(
        val_texts, tokenizer, config
    )  # Tokenize validation data

    logging.info("Tokenizing testing data...")
    test_inputs = tokenize_function(test_texts, tokenizer, config)

    logging.info("Creating training, validation, and testing datasets...")
    train_dataset = Dataset.from_dict(
        {
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"],
            "labels": torch.tensor(train_labels),
        }
    )

    val_dataset = Dataset.from_dict(
        {
            "input_ids": val_inputs["input_ids"],
            "attention_mask": val_inputs["attention_mask"],
            "labels": torch.tensor(val_labels),
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "input_ids": test_inputs["input_ids"],
            "attention_mask": test_inputs["attention_mask"],
            "labels": torch.tensor(test_labels),
        }
    )

    return train_dataset, val_dataset, test_dataset  # Return all three datasets


def predict_proba(texts, model, tokenizer, device, config):
    """
    Returns the predicted probabilities for the positive class.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize input texts
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt",
    )

    # Move inputs to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ensure the model is on the correct device
    model.to(device)
    model.eval()

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    # Return the probability for the positive class
    return probs[:, 1].cpu().numpy()  # Probability of the positive class


# --- visual_results.py ---
import numpy as np
import os
import logging
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from src.model_utils import predict_proba


def generate_evaluation_reports_and_plots(df, model, tokenizer, device, config):
    """
    Generates evaluation reports and plots, saving them to the specified output directory.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data and predictions.
        model: The trained model.
        tokenizer: The tokenizer used for the model.
        device: The device ('cpu' or 'cuda') on which to run the model.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    try:
        logging.info("Generating evaluation reports and plots...")

        # Get true labels
        true_labels = df["yo2"]

        # Get predicted probabilities
        logging.info("Calculating predicted probabilities...")
        df["predicted_proba"] = predict_proba(
            texts=df["full_text"].tolist(),
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )

        # Convert predicted probabilities to binary labels (threshold of 0.5)
        df["predicted_yo2"] = (df["predicted_proba"] >= 0.5).astype(int)

        # Get binary predicted labels
        predicted_labels = df["predicted_yo2"]

        # Ensure the evaluation output directory exists
        output_dir = config["evaluation"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save confusion matrix
        plot_confusion_matrix(true_labels, predicted_labels, output_dir)

        # Generate and save classification report
        generate_classification_report(true_labels, predicted_labels, output_dir)

        # Generate and save ROC curve
        plot_roc_curve(true_labels, df["predicted_proba"], output_dir)

        # Generate and save Precision-Recall curve
        plot_precision_recall_curve(true_labels, df["predicted_proba"], output_dir)

        logging.info("Evaluation reports and plots generated successfully.")

    except Exception as e:
        logging.error(f"Failed to generate evaluation reports and plots: {e}")


def plot_confusion_matrix(true_labels, predicted_labels, output_dir):
    try:
        unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))
        if len(unique_labels) == 1:
            cm = confusion_matrix(
                true_labels, predicted_labels, labels=[unique_labels[0]]
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=[str(unique_labels[0])]
            )
        else:
            cm = confusion_matrix(true_labels, predicted_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {plot_path}.")
    except Exception as e:
        logging.error(f"Failed to generate confusion matrix: {e}")


def generate_classification_report(true_labels, predicted_labels, output_dir):
    try:
        unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))
        if len(unique_labels) == 1:
            report = classification_report(
                true_labels,
                predicted_labels,
                labels=[unique_labels[0]],
                target_names=[str(unique_labels[0])],
            )
        else:
            report = classification_report(true_labels, predicted_labels)

        report_file = os.path.join(output_dir, "classification_report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        logging.info(f"Classification report saved to {report_file}.")
    except Exception as e:
        logging.error(f"Failed to generate classification report: {e}")


def plot_roc_curve(true_labels, predicted_probabilities, output_dir):
    try:
        fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC Curve (AUC = {roc_auc:0.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"ROC curve saved to {plot_path}.")
    except Exception as e:
        logging.error(f"Failed to generate ROC curve: {e}")


def plot_precision_recall_curve(true_labels, predicted_probabilities, output_dir):
    try:
        precision, recall, _ = precision_recall_curve(
            true_labels, predicted_probabilities
        )
        avg_precision = average_precision_score(true_labels, predicted_probabilities)
        plt.figure()
        plt.plot(
            recall,
            precision,
            color="b",
            lw=2,
            label=f"Avg Precision = {avg_precision:0.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plot_path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Precision-Recall curve saved to {plot_path}.")
    except Exception as e:
        logging.error(f"Failed to generate Precision-Recall curve: {e}")


# --- data_utils.py ---
# src/data_utils.py

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def load_data(config):
    data_file = config["data"]["data_file"]
    if not os.path.exists(data_file):
        logging.error(f"Data file {data_file} not found.")
        raise FileNotFoundError(f"Data file {data_file} not found.")
    else:
        logging.info(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
    return df


def preprocess_data(df, config):

    # Check for necessary columns
    required_columns = config["data"]["required_columns"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in the dataset: {missing_columns}")
        raise KeyError(f"Missing columns in the dataset: {missing_columns}")

    return df


def sample_data(df, config):
    sample_size = config["data"]["sample_size"]
    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        logging.info(f"Sampled {sample_size} entries from the dataset.")
    else:
        logging.info("Using all data in the dataset.")
    return df


def split_data(df):
    logging.info("Splitting data into training, validation, and testing sets...")
    # First, split into train + validation and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df["full_text"].tolist(), df["yo2"].tolist(), test_size=0.2, random_state=42
    )

    # Now, split the train + validation into actual train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=0.25,  # 25% of train + validation set for validation
        random_state=42,
    )

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


# --- train.py ---
# src/train.py

import logging
import torch
from transformers import Trainer, TrainingArguments
from src.model_utils import compute_metrics
import torch.nn.functional as F
from tqdm import tqdm


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(focal_loss)


def get_training_arguments(config):
    return TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        use_mps_device=True,
        logging_steps=config["training"]["logging_steps"],
        logging_first_step=config["training"]["logging_first_step"],
    )


def train_model(model, train_dataset, val_dataset, test_dataset, training_args):
    logging.info("Initializing the Trainer with validation...")

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Tune alpha and gamma here
            focal_loss = FocalLoss(alpha=0.5, gamma=2)  # Adjust alpha as needed
            loss = focal_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use validation set for evaluation
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training with validation...")
    trainer.train()

    logging.info("Training completed. Evaluating on the validation set...")
    validation_metrics = trainer.evaluate(eval_dataset=val_dataset)
    logging.info(f"Validation set metrics: {validation_metrics}")

    return trainer


# --- main.py ---
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
    # save the dataframe
    df.to_csv("data/predictions.csv", index=False)


if __name__ == "__main__":
    main()


