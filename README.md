Fake News Detection using NLP
## Project Overview
This project aims to develop and compare multiple Natural Language Processing (NLP) models for classifying news articles as real or fake. We explore different text preprocessing techniques and feature extraction methods, evaluating the models based on accuracy, confusion matrix and F1-score to identify the most effective approach.
 ## Dataset
The dataset used in this project is assumed to be named `training_data_lowercase.csv`. Training dataset consist of news headlines, which is evenly distribute between Fake and real news. It is expected to be a tab-separated file with one column, the column contain headline text and labels(0 or 1).
## Project Workflow
The project follows these key steps:
1. **Text Preprocessing**:
    *  We split the original dataset into two columns:
    *  The first column containing the `label` (either 'fake'=0 or 'real'=1).
    *  The second column containing the `text` of the news article.
    *  Cleaning and preparing the text data by tokenizing words, removing stopwords, eliminating punctuation and numbers, and lemmatizing tokens.
2. **Feature Extraction**: Converting the cleaned text into numerical representations suitable for machine learning models using:
    *   CountVectorizer (Bag-of-Words)
    *   TF-IDF Vectorizer
3. **Model Development and Evaluation**: Implementing and evaluating following different model combinations:
    *   CountVectorizer + Multinomial Naive Bayes (with hyperparameter tuning)
    *   TF-IDF + Random Forest (using default parameters)
    *   TF-IDF + Multinomial Naive Bayes (using the optimized Naive Bayes model)
    *   CountVectorizer + Logistic Regression
    *  	TFIDF	CountVectorizer	+ Logistic Regression
    *  	CountVectorizer + XGBoost
    *  	TFIDF	CountVectorizer	+ XGBoost
    *   CountVectorizer + Random Forest (using default parameters)
    We compare the models using accuracy, confusion matrices, and classification reports (including precision, recall, and F1-score).
4. **Model Analysis**: Analyzing the performance of each model, including investigating reasons for potential underperformance (e.g., overfitting in the Random Forest       model).
5. **Model Saving**: Saving the best performing model (TF-IDF + Naive Bayes based on the notebook's analysis) and its corresponding vectorizer for future use.
## Results
    *  The notebook includes detailed evaluations and comparisons of the implemented models. Based on the analysis, the TF-IDF + Naive Bayes model demonstrated strong          performance on the validation set after hyperparameter tuning. Our final model is saved in "Prediction_test_NB_Vectrorizer" notebook. Our prediction of test data is saved in "test_predictions.csv" file.
## Saved Models
  The trained TF-IDF Vectorizer and the best performing Naive Bayes model are saved using `pickle` and can be found in the repository (or downloaded from the notebook     execution):
  *   `tfidf_vectorizer.pkl`
  *   `best_nb.pkl`
  These files can be loaded to make predictions on new data without retraining the model.