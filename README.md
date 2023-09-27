# Translation using Large Language Models

## Introduction

This repository is dedicated to the exploration and application of Large Language Models (LLMs) for translation tasks. It showcases the process of fine-tuning a model, making it available on the Hugging Face Model Hub, and providing a demonstration for translating English to French.

## Demonstration

- **Demo Link**: [Insert Your Demo Link Here]

- **Fine-Tuned Model Link**: [Hugging Face Model Hub Link](https://huggingface.co/Safeer143/marian-finetuned-kde4-en-to-fr-4-epochs-10000-samples)

## Repository Contents

This repository consists of the following essential files and modules:

### 1. Datasets

The Datasets module focuses on data handling and preparation. It includes the following functions:

- `load_dataset_from_hf(dataset_name, lang1='en', lang2='fr')`: This function loads a dataset from the Hugging Face Model Hub.
- `tokenized_ds(dataset)`: It handles dataset tokenization.

### 2. Model Training

The Model Training module is responsible for fine-tuning the LLM model and saving it to the Hugging Face Model Hub. It provides the following functions:

- `fine_tune_model(model_name, dataset_tokenized, batch_size, model_name_to_save)`: This function fine-tunes the model using specified parameters.
- `compute_metrics(eval_preds)`: It calculates evaluation metrics for the model's predictions.
- `training_args(model_checkpoint, batch_size)`: This function provides training arguments for the fine-tuning process.

### 3. Fine-Tuning the Model

This module integrates the dataset and fine-tuning module to prepare the dataset for training. The fine-tuned model is available on the Hugging Face Model Hub and is utilized for demonstrations.

### 4. Translating with Model (EN to FR)

The 'Translating with Model' module loads the fine-tuned model and deploys it for translating English text to French. It leverages Gradio for easy and interactive demonstrations.

## Repository Structure

- `FineTuning_the_model.ipynb`: This Jupyter notebook integrates various aspects of the model and dataset. It serves as the workspace for experimentation and testing.

## Getting Started

To explore and utilize this repository, follow these steps:

1. Fine-tune your model using the provided dataset.
2. Save the fine-tuned model to the Hugging Face Model Hub.
3. Explore the `FineTuning_the_model` module to understand the integration process.
4. Leverage the fine-tuned model for translation tasks.
5. Use Gradio to create interactive demos.

## Conclusion

This repository provides a comprehensive guide to working with LLMs for translation tasks. Feel free to explore, fine-tune, and utilize the provided model for your language translation needs. Enjoy the journey of harnessing the power of language models.

For any questions or support, please don't hesitate to reach out.

Happy translating!
