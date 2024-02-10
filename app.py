# Imports
import requests
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import streamlit as st

# Functions

def extract_text(url):
    # Send a GET request to the URL and retrieve the response
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Use BeautifulSoup to parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all of the text in the HTML document
        text = soup.get_text()

        # Return the text
        return text
    else:
        # If the request was not successful, return an empty string
        return ''

def preprocess(texts):
    preprocessed_texts = []
    for text in texts:
        try:
            # clean and normalize the text as desired
            article_text = text.lower().strip()
            article_text = re.sub(r'[^a-zA-Z0-9\s]', '', article_text)

            preprocessed_texts.append(article_text)
        except:
            print(f"Failed to preprocess text: {text}")

    return preprocessed_texts

def cluster_news_articles(links):
    # Extract the text from each of the links
    texts = [extract_text(link) for link in links]

    # Preprocess the text (e.g. remove stop words, stem words, etc.)
    preprocessed_texts = preprocess(texts)

    print(preprocessed_texts)

    # Convert the preprocessed text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)

    # Use KMeans to cluster the articles into groups based on their content
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)

    # Return the cluster labels for each article
    return kmeans.labels_

# Load vectorizer and kmeans models
vectorizer_path = 'tfidf_vectorizer.joblib'
kmeans_path = 'kmeans_model.joblib'
vectorizer = joblib.load(vectorizer_path)
kmeans = joblib.load(kmeans_path)

# Streamlit app
def main():
    # Title and description
    st.title("News Article Clustering")
    st.write("This web app clusters news articles based on their content similarity.")

    # Provide a list of news article links
    links = st.text_area("Enter news article links (one per line)", """
    https://www.gorillatoursafrica.com/things-to-do-in-kigali/
    https://www.thetravelersbuddy.com/2022/08/30/best-things-rwanda/
    # Add more links as needed
    """)

    # Convert input to a list of links
    links = links.split("\n")

    # Cluster the articles
    cluster_labels = cluster_news_articles(links)

    # Display the cluster labels for each article
    for cluster_label in set(cluster_labels):
        st.subheader(f"Cluster {cluster_label}:")
        for i, link in enumerate(links):
            if cluster_labels[i] == cluster_label:
                st.write(f"\tArticle {i + 1} ({link})")
        st.write("\n")

# Run the Streamlit app
if __name__ == "__main__":
    main()
