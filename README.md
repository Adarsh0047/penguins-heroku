# Penguin Species Classifier

A Streamlit app that predicts a penguin species from physical measurements.

## Features

- Enter island, sex, bill length, bill depth, flipper length, and body mass in the sidebar.
- Upload a CSV for batch input.
- View the predicted species and class probabilities.

The app uses the saved classifier in `penguins_classifier.pkl` and the sample data in `penguins_cleaned.csv`.

## Run locally

Install `requirements.txt`, then run:

```bash
streamlit run penguins_app.py
```
