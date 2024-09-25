# Climate Change Mitigation Technology Identification Using Patent Data with DeBERTa

[![Personal Project](https://img.shields.io/badge/Project-Personal-green)](https://meg-patakota.github.io)
[![by Meg Patakota](https://img.shields.io/badge/by-Meg%20Patakota-blue)](https://meg-patakota.github.io)
[![Project Status](https://img.shields.io/badge/Status-In%20Development-orange)](https://github.com/yourusername/patent-classification-deberta)

> ⚠️ **Disclaimer:** This project is currently under active development. Features, code, and documentation are subject to change as it evolves.

This project utilizes the **DeBERTa** model for classifying patent data into **climate change mitigation technologies**, as defined by the **IPCC** (Intergovernmental Panel on Climate Change) classification system. This tool is intended to assist in identifying technologies that contribute to climate change mitigation efforts, leveraging advanced machine learning techniques for large-scale patent analysis.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Methodology](#model-methodology)
- [Contributing](#contributing)
- [License](#license)

## Features

- **DeBERTa-based text classification**: The model is fine-tuned on patent texts to classify them according to the IPCC climate change mitigation categories.
- **Efficient patent data processing**: Handles large-scale datasets and performs model training with minimal manual intervention.
- **Advanced evaluation metrics**: Includes metrics like accuracy, ROC-AUC, precision, recall, and f1-score for assessing model performance on imbalanced data.
- **Custom tokenization**: Implements tokenization with dynamic padding and truncation to handle long patent texts.
- **Domain-specific model fine-tuning**: The model is optimized for identifying technologies in the climate change mitigation space, using a variety of metrics tailored to imbalanced datasets.

## Installation

Clone the repository and install dependencies using **Poetry**:

```bash
git clone https://github.com/yourusername/patent-classification-deberta.git
cd patent-classification-deberta
poetry install
```

## Usage

Once the dependencies are installed, you can run the model as follows:

### Training the Model

```bash
poetry run python src/train.py
```

### Evaluating the Model

```bash
poetry run python src/evaluate.py
```

### Tokenization

This project uses a custom tokenizer function defined in `model_utils.py`, which tokenizes the patent texts using the DeBERTa tokenizer with dynamic padding and truncation. To tokenize a dataset:

```python
from src.model_utils import tokenize_function

tokenized_data = tokenize_function(texts, tokenizer, config)
```

### Model Fine-Tuning

To fine-tune the DeBERTa model on your own dataset, simply update the configuration in the `config.json` file and run:

```bash
poetry run python src/train.py --config config.json
```

## Model Methodology

### Tokenization

The tokenization process leverages the `transformers` library's DeBERTa tokenizer. It ensures efficient tokenization of large patent documents by applying dynamic padding and truncation. The model processes the text by converting it into token embeddings that the DeBERTa model can understand.

### Metrics & Model Evaluation

The model's performance is assessed using several metrics, which are implemented in the `compute_metrics()` function within the `model_utils.py` file. These include:
- **Accuracy**: Measures the overall correctness of predictions.
- **Precision, Recall, and F1-Score**: Important for imbalanced datasets, capturing the trade-off between false positives and false negatives.
- **ROC-AUC and PR-AUC**: Provide insight into the model's ability to distinguish between classes, especially useful for imbalanced data.

The model outputs ROC-AUC only if both classes are present in the dataset, ensuring meaningful evaluation on imbalanced patent data. The average precision score is also calculated to assess the model's performance in ranking relevant patents.

### Handling Imbalanced Data

The dataset used in this project is highly imbalanced, with one class significantly outweighing the other. To address this, techniques such as:
- Weighted loss functions
- Over/Under-sampling strategies
- Evaluation metrics suited for imbalanced data (e.g., Precision-Recall AUC)

are employed to ensure that the model can generalize well across both classes.

### Trainer

The `transformers` Trainer API is used to streamline the training and evaluation process, with built-in capabilities for logging, checkpointing, and metric tracking. This makes the workflow more efficient and customizable.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---