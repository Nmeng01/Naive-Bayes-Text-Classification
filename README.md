# Naive Bayes Text Classification

This project implements a Naive Bayes classifier to categorize biographies based on their content. The classifier is trained using a dataset of biographies and evaluates its performance on a test set of biographies.

## Features

- **Training and Testing**:
  - Uses the first n entries from the corpus as the training set.
  - Evaluates the classifier on the remaining entries in the test set.
- **Text Preprocessing**:
  - Normalizes all text to lowercase.
  - Removes stop words that do not contribute much to classification.
- **Probability Computation**:
  - Computes probabilities with Laplacian correction.
  - Avoids underflow by using negative log probabilities.
- **Prediction**:
  - Calculates the likelihood for each category and predicts the one with the highest probability.
  - Provides a breakdown of probabilities for all categories.

## Input Format

The input file should contain short biographies formatted as:
1. First line: Name of the person.
2. Second line: Category.
3. Remaining lines: Biography text.

Biographies are separated by one or more blank lines.

## Output

The program outputs:
- Predicted category for each person in the test set.
- Probabilities for all categories.
- A statement indicating whether the prediction was correct.
- Overall accuracy of the classifier.
