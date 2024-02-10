# Imports
import requests
import re

try:
    from bs4 import BeautifulSoup
except :
from BeautifulSoup import BeautifulSoup 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import streamlit as st

# Functions (as in your original script)

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
