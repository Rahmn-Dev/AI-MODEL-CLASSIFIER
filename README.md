# Stroke Prediction AI Model using KNeighborsClassifier
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

## Overview

This project implements a stroke prediction model using the KNeighborsClassifier algorithm. The model is trained on a dataset containing features related to strokes, and it aims to predict whether an individual is at risk of experiencing a stroke based on given parameters.

## Dataset

The dataset used for training and testing the model is available in the 'dataset' directory. It includes various features such as age, hypertension, heart disease, etc., which are used to predict the likelihood of a stroke occurrence.

### Dataset Source

The dataset is obtained from [Kaggle - Health Dataset](https://www.kaggle.com/datasets/prosperchuks/health-dataset).

## Project Structure

- `data-test`: Directory for test data.
- `data-train`: Directory for training data.
- `dataset`: Original dataset directory.
- `model_results`: Directory for storing model-related results.
- `report_data`: Directory for any report-related data.
- `model_classifier.py`: Python script containing the KNeighborsClassifier implementation for the stroke prediction model.
- `Pipfile`: Pipenv configuration file.
- `Pipfile.lock`: Pipenv lock file.
- `requirements.txt`: Plain Python requirements file.

## Model Evaluation Results

### Training Data Results:

- **Precision (Class 0):** 100%
- **Recall (Class 0):** 100%
- **F1-Score (Class 0):** 100%
- **Support (Class 0):** 16,428

- **Precision (Class 1):** 100%
- **Recall (Class 1):** 100%
- **F1-Score (Class 1):** 100%
- **Support (Class 1):** 16,300

- **Accuracy:** 100%
- **Macro Avg Precision, Recall, F1-Score:** 100%
- **Weighted Avg Precision, Recall, F1-Score:** 100%

### Testing Data Results:

- **Precision (Class 0):** 97.76%
- **Recall (Class 0):** 79.19%
- **F1-Score (Class 0):** 87.50%
- **Support (Class 0):** 4,022

- **Precision (Class 1):** 83.00%
- **Recall (Class 1):** 98.25%
- **F1-Score (Class 1):** 89.98%
- **Support (Class 1):** 4,160

- **Accuracy:** 88.88%
- **Macro Avg Precision:** 90.38%
- **Macro Avg Recall:** 88.72%
- **Macro Avg F1-Score:** 88.74%

- **Weighted Avg Precision:** 90.26%
- **Weighted Avg Recall:** 88.88%
- **Weighted Avg F1-Score:** 88.76%

## Interpretation:

- The model achieved perfect performance on the training data.

- On the testing data, the model shows high accuracy, especially for individuals at risk of stroke (Class 1).

- Consider the specific context and requirements of your problem when interpreting these results.

Feel free to reach out if you have any specific questions or need further assistance!

## Additional Notes:

- Adjust hyperparameters in the model training script (`model_classifier.py`) for further optimization.
- Experiment with feature engineering and selection to improve model performance.

## Issues and Contributions:

Feel free to raise issues or contribute to the project by submitting pull requests. Your feedback and contributions are highly appreciated!

## License:

This project is licensed under the [MIT License](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.
