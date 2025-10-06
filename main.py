import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import joblib
import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Colab-specific import
try:
    from google.colab import files
    COLAB = True
except ImportError:
    COLAB = False

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_and_prepare_data(fake_path='Fake (2).csv', true_path='True (1).csv'):
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"Files {fake_path} or {true_path} not found.")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    if 'text' not in df.columns:
        raise ValueError("The CSV files must contain a 'text' column.")
    df = df[['text', 'label']]
    df['text'] = df['text'].fillna('')
    
    # Data quality check
    if len(df) < 100:
        raise ValueError("Dataset is too small. Ensure sufficient samples in CSV files.")
    df = df[df['text'].str.strip() != '']
    
    # Balance dataset
    fake = df[df['label'] == 0]
    real = df[df['label'] == 1]
    if len(fake) > len(real):
        fake = resample(fake, replace=False, n_samples=len(real), random_state=42)
    elif len(real) > len(fake):
        real = resample(real, replace=False, n_samples=len(fake), random_state=42)
    df = pd.concat([fake, real], ignore_index=True)
    return df

def train_fake_news_detector(fake_path='Fake (2).csv', true_path='True (1).csv'):
    df = load_and_prepare_data(fake_path, true_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Add sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Scale sentiment feature
    scaler = StandardScaler()
    X_train_sentiment = scaler.fit_transform(df.loc[X_train.index, 'sentiment'].values.reshape(-1, 1))
    X_test_sentiment = scaler.transform(df.loc[X_test.index, 'sentiment'].values.reshape(-1, 1))
    X_train_tfidf = hstack([X_train_tfidf, X_train_sentiment])
    X_test_tfidf = hstack([X_test_tfidf, X_test_sentiment])
    
    model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    
    # Cross-validation
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Fake News Detection')
    plt.show()
    
    # Feature importance
    feature_names = list(tfidf.get_feature_names_out()) + ['sentiment']
    coef = model.coef_[0]
    top_fake = sorted(zip(coef, feature_names), reverse=True)[:10]
    top_real = sorted(zip(coef, feature_names))[:10]
    feature_output = "\nTop 10 Features for Fake News:\n" + "\n".join([f"{f}: {c:.4f}" for c, f in top_fake])
    feature_output += "\n\nTop 10 Features for Real News:\n" + "\n".join([f"{f}: {c:.4f}" for c, f in top_real])
    
    # Save model, vectorizer, and scaler
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    joblib.dump(scaler, 'sentiment_scaler.pkl')
    
    return model, tfidf, scaler, feature_output

def predict_news(model, tfidf, scaler, title, text, subject, date):
    combined_text = f"{title or ''} {subject or ''} {text or ''}".strip()
    if not combined_text:
        return "Error: At least one of Title, Text, or Subject must be provided.", 0.0
    
    processed_text = preprocess_text(combined_text)
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(combined_text)['compound']
    text_tfidf = tfidf.transform([processed_text])
    sentiment_scaled = scaler.transform([[sentiment]])
    text_tfidf = hstack([text_tfidf, sentiment_scaled])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][1]
    label = "Real" if prediction == 1 else "Fake"
    return label, probability

def predict_news_batch(model, tfidf, scaler, batch_inputs):
    results = []
    confidences = []
    analyzer = SentimentIntensityAnalyzer()
    
    for entry in batch_inputs:
        title, text, subject, date = entry
        combined_text = f"{title or ''} {subject or ''} {text or ''}".strip()
        if not combined_text:
            continue
        
        processed_text = preprocess_text(combined_text)
        sentiment = analyzer.polarity_scores(combined_text)['compound']
        text_tfidf = tfidf.transform([processed_text])
        sentiment_scaled = scaler.transform([[sentiment]])
        text_tfidf = hstack([text_tfidf, sentiment_scaled])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0][1]
        label = "Real" if prediction == 1 else "Fake"
        results.append({
            "Title": title[:50] + "..." if title and len(title) > 50 else title or "",
            "Text": text[:50] + "..." if text and len(text) > 50 else text or "",
            "Subject": subject or "",
            "Date": date or "",
            "Prediction": label,
            "Confidence": f"{probability:.2%}"
        })
        confidences.append(probability * 100)
    
    total_confidence = sum(confidences) if confidences else 0
    return pd.DataFrame(results), total_confidence

def gradio_interface(title, text, subject, date, batch_input):
    model = joblib.load('fake_news_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('sentiment_scaler.pkl')
    
    # Single prediction
    single_result = ""
    if any([title, text, subject]):
        label, probability = predict_news(model, tfidf, scaler, title, text, subject, date)
        single_result = (
            f'<div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #ccc; font-family: Arial, sans-serif;">'
            f'<p><strong>Prediction:</strong> {label} (Confidence: {probability:.2%})</p>'
            f'<p><strong>Title:</strong> {title or "N/A"}</p>'
            f'<p><strong>Text:</strong> {text[:100] + "..." if text and len(text) > 100 else text or "N/A"}</p>'
            f'<p><strong>Subject:</strong> {subject or "N/A"}</p>'
            f'<p><strong>Date:</strong> {date or "N/A"}</p>'
            f'</div>'
        )
    
    # Batch prediction
    batch_result = None
    total_confidence = 0
    if batch_input:
        batch_inputs = []
        lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
        for line in lines:
            parts = [part.strip() for part in line.split(',', 3)]
            if len(parts) < 4:
                parts.extend([''] * (4 - len(parts)))
            batch_inputs.append(parts)
        
        if batch_inputs:
            batch_df, total_confidence = predict_news_batch(model, tfidf, scaler, batch_inputs)
            batch_result = (
                f'<div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #ccc; overflow-x: auto;">'
                f'{batch_df.to_html(index=False, classes="table table-striped", border=0)}'
                f'<p style="margin-top: 10px;"><strong>Total Confidence (Sum of Real Probabilities):</strong> {total_confidence:.2f}%</p>'
                f'</div>'
            )
    
    return single_result or "Enter at least one of Title, Text, or Subject.", batch_result

def main():
    try:
        if COLAB:
            print("Using uploaded CSV files: Fake (2).csv and True (1).csv")
            if not os.path.exists('/content/Fake (2).csv') or not os.path.exists('/content/True (1).csv'):
                print("Please upload Fake (2).csv and True (1).csv files.")
                uploaded = files.upload()
                if 'Fake (2).csv' not in uploaded or 'True (1).csv' not in uploaded:
                    raise FileNotFoundError("Both Fake (2).csv and True (1).csv must be uploaded.")
            fake_path = '/content/Fake (2).csv'
            true_path = '/content/True (1).csv'
        else:
            fake_path = 'Fake (2).csv'
            true_path = 'True (1).csv'
        
        model, tfidf, scaler, feature_output = train_fake_news_detector(fake_path=fake_path, true_path=true_path)
        
        # Print feature importance
        print("\nModel Training Metrics and Feature Importance:")
        print(feature_output)
        
        # Gradio interface
        css = """
        .gradio-container {
            font-family: Arial, sans-serif;
            background-color: #e9ecef;
            padding: 20px;
        }
        .input-text {
            border-radius: 5px;
            border: 1px solid #ced4da;
            padding: 10px;
            background-color: #ffffff;
            color: #212529;
        }
        .output-text {
            font-size: 16px;
            color: #212529;
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #ced4da;
            margin-top: 10px;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            background-color: #ffffff;
            font-size: 14px;
        }
        .table th, .table td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
            color: #212529;
        }
        .table th {
            background-color: #198754;
            color: #ffffff;
            font-weight: bold;
        }
        .table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .table tr:hover {
            background-color: #e9ecef;
        }
        button {
            background-color: #198754;
            color: #ffffff;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            background-color: #146c43;
        }
        div {
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        """
        
        iface = gr.Interface(
            fn=gradio_interface,
            inputs=[
                gr.Textbox(lines=2, placeholder="Enter article title...", label="Title"),
                gr.Textbox(lines=5, placeholder="Enter article text...", label="Text"),
                gr.Textbox(lines=2, placeholder="Enter article subject (e.g., Politics, Science)...", label="Subject"),
                gr.Textbox(lines=1, placeholder="Enter article date (e.g., 2025-05-25)...", label="Date"),
                gr.Textbox(lines=5, placeholder="Enter multiple articles (one per line, format: Title,Text,Subject,Date)...", label="Batch News Articles")
            ],
            outputs=[
                gr.HTML(label="Prediction for Single Article"),
                gr.HTML(label="Batch Prediction Results")
            ],
            css=css,
            title="Fake News Detector",
            description="Enter article details (Title, Text, Subject, Date) to predict if they are Real or Fake. For batch input, use CSV format: Title,Text,Subject,Date per line.",
            submit_btn="Predict News"
        )
        iface.launch()
        
        return model, tfidf, scaler
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files 'Fake (2).csv' and 'True (1).csv' are uploaded (in Colab) or in the correct directory.")
        return None, None, None
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check that the CSV files contain a 'text' column.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

if _name_ == "_main_":
    model, tfidf, scaler = main()
