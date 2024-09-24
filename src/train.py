# src/train.py

import logging
import torch
from transformers import Trainer, TrainingArguments
from src.model_utils import compute_metrics


def get_training_arguments(config):
    return TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        no_cuda=not torch.cuda.is_available(),
        logging_steps=config["training"]["logging_steps"],
        logging_first_step=config["training"]["logging_first_step"],
    )


def train_model(model, train_dataset, test_dataset, training_args):
    logging.info("Initializing the Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    try:
        logging.info("Starting training...")
        trainer.train()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e
    return trainer
