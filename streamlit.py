import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import emoji
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import os

df = pd.read_csv('fixedNotClean_all_temp.csv')
df_fb = pd.read_csv('facebook_posts.csv')
df_tw = pd.read_csv('tweets.csv')
#merge the datasets
df = df.rename(columns={'tweet': 'text'})
df_tw= df_tw.rename(columns={'tweet': 'text'})


def remove_emoticons(text):
    emoticon_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoticon_pattern.sub(r'', text)

def clean_tweet_id(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r':([a-zA-Z0-9_]+):', r'\1', text)
    text = text.replace('_', ' ')
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_emoticons(text)
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    return text


stemmer = StemmerFactory().create_stemmer()
wpt = nltk.WordPunctTokenizer()
stopword_factory = StopWordRemoverFactory()

stopword = stopword_factory.get_stop_words()

def uncapitalize(doc):
    return doc.lower()


def normalize_document(doc):
    doc = uncapitalize(doc)
    doc = doc.strip()
    doc = wpt.tokenize(doc)
    doc = [stemmer.stem(word) for word in doc]
    doc = [word for word in doc if word not in stopword]
    doc = " ".join(doc)
    return doc


def preprocess_data():
    status = st.empty()
    bar = st.progress(0.0)
    
    status.text("Step 1: Lowercasing and cleaning text...")
    df_encoded['text'] = df_encoded['text'].apply(clean_tweet_id)
    bar.progress(0.3)

    status.text("Step 2: Normalizing and stemming...")
    df_encoded['text'] = df_encoded['text'].apply(normalize_document)
    bar.progress(0.7)

    status.text("Step 3: Saving cleaned data...")
    df_encoded.to_csv('cleaned_and_normalized_data.csv')
    st.session_state.cleaned_state = True
    bar.progress(1.0)

    status.success("Preprocessing complete!")
    if st.button("Oversample Data"):
        oversample_data()

def oversample_data():
    from sklearn.utils import resample
    df_encoded = pd.read_csv('cleaned_and_normalized_data.csv')
    status = st.empty()
    bar = st.progress(0.0)
    minority_labels = [
    "Future_Neutral",
    "Recyclability (PET)_Negative",
    "Recyclability_Negative"
]
    oversampled_rows = []
    status.text("Oversampling the data...")
    target_count = 600 

    for label in minority_labels:
        label_positive = df_encoded[df_encoded[label] == 1]
        if len(label_positive) < target_count:
            label_upsampled = resample(
                label_positive,
                replace=True,
                n_samples=target_count - len(label_positive),   
                random_state=42
            )
            oversampled_rows.append(label_upsampled)

    # Combine original data with oversampled rows
    df_balanced = pd.concat([df_encoded] + oversampled_rows)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced.to_csv('oversampled_data.csv', index=False)

    status.success("Oversampling complete!")
    bar.progress(1.0)
    st.text("Training will now use the oversampled data.")
    label_sums = df_balanced[[
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]].sum()

    label_df = label_sums.reset_index()
    label_df.columns = ['label', 'count']
    label_df['aspect'] = label_df['label'].apply(lambda x: x.split('_')[0])
    label_df['sentiment'] = label_df['label'].apply(lambda x: x.split('_')[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=label_df, x='aspect', y='count', hue='sentiment', palette='Set2', ax=ax)
    ax.set_title("Label Distribution (Oversampled Data)")
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Aspect")
    ax.legend(title='Sentiment')
    st.pyplot(fig)



columns_to_keep = ['sentimenteffective', 'sentimentpet', 'sentimentprocess', 'sentimentrecycle', 'text']
df_fb = df_fb[columns_to_keep]
df_tw = df_tw[columns_to_keep]
df = df[columns_to_keep]

# Combine the two DataFrames
combined_df = pd.concat([df_tw, df_fb,df], ignore_index=True)
#Drop empty Values 
combined_df = combined_df.dropna(subset=['text'])
df = combined_df.dropna(subset=['sentimentprocess', 'sentimentpet', 'sentimenteffective', 'sentimentrecycle'])

st.write("Current Data, Not Cleaned")
df.rename(columns={'sentimenteffective': 'Recyclability'}, inplace=True)
df.rename(columns={'sentimentpet': 'Recyclability (PET)'}, inplace=True)
df.rename(columns={'sentimentprocess': 'Recycling'}, inplace=True)
df.rename(columns={'sentimentrecycle': 'Future'}, inplace=True)
df

st.write("Current Label Distribution")
df_encoded = pd.get_dummies(df, columns=['Recyclability', 'Recyclability (PET)', 'Recycling', 'Future'])

if "df_encoded" not in st.session_state:
    st.session_state.df_encoded = df_encoded

label_sums = df_encoded[[
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]].sum()

label_df = label_sums.reset_index()
label_df.columns = ['label', 'count']
label_df['aspect'] = label_df['label'].apply(lambda x: x.split('_')[0])
label_df['sentiment'] = label_df['label'].apply(lambda x: x.split('_')[1])

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=label_df, x='aspect', y='count', hue='sentiment', palette='Set2', ax=ax)
ax.set_title("Label Distribution Grouped by Aspect")
ax.set_ylabel("Number of Samples")
ax.set_xlabel("Aspect")
ax.legend(title='Sentiment')
st.pyplot(fig)

if('cleaned_state' not in st.session_state):
    if(os.path.exists('cleaned_and_normalized_data.csv')):
        st.session_state.cleaned_state = True
    else:
        st.session_state.cleaned_state = False

st.title("Preprocessing and Training Pipeline")

with st.container():
    if st.session_state.get('cleaned_state', True):
        st.text("Cleaned and normalized data is available.")
        if os.path.exists('oversampled_data.csv'):
            st.text("Oversampled data is available.")
            df_balanced = pd.read_csv('oversampled_data.csv')
            label_sums = df_balanced[[
                "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
                "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
                "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
                "Future_Negative", "Future_Neutral", "Future_Positive"
            ]].sum()

            label_df = label_sums.reset_index()
            label_df.columns = ['label', 'count']
            label_df['aspect'] = label_df['label'].apply(lambda x: x.split('_')[0])
            label_df['sentiment'] = label_df['label'].apply(lambda x: x.split('_')[1])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=label_df, x='aspect', y='count', hue='sentiment', palette='Set2', ax=ax)
            ax.set_title("Label Distribution (Oversampled Data)")
            ax.set_ylabel("Number of Samples")
            ax.set_xlabel("Aspect")
            ax.legend(title='Sentiment')
            st.pyplot(fig)

            if(st.button("Delete oversampled data")):
                os.remove('oversampled_data.csv')
                st.text("Oversampled data deleted.")
        else:
            if( st.button("Oversample Data")):
                oversample_data()
    else:
        st.text("Cleaned and normalized data is not available. Please run the preprocessing step first.")
        st.text("You can also oversample the data to balance the classes.")

if st.button("Run Preprocessing"):  
    preprocess_data()

    


