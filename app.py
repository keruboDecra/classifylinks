# Imports
import requests
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import streamlit as st

# Functions

from urllib.parse import urlparse, urlunparse
# ... (rest of the code)


# Load vectorizer and kmeans models
vectorizer_path = 'tfidf_vectorizer.joblib'
kmeans_path = 'kmeans_model.joblib'
vectorizer = joblib.load(vectorizer_path)
kmeans = joblib.load(kmeans_path)
def extract_text(url):
    try:
        # Remove leading and trailing whitespaces from the URL
        url = url.strip()

        # Parse the URL to validate and properly format it
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            # If the scheme is missing, add 'http'
            parsed_url = parsed_url._replace(scheme='http')

        # Reconstruct the URL
        url = urlunparse(parsed_url)

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
    except requests.exceptions.ConnectionError as e:
        st.warning(f"Connection error for URL {url}: {e}")
        return ''
    except Exception as e:
        st.warning(f"Error for URL {url}: {e}")
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

# ... (previous code)

def cluster_news_articles(links):
    # Filter out empty or invalid URLs
    valid_links = [link for link in links if link.strip()]

    # Check if there are any valid links
    if not valid_links:
        st.warning("No valid URLs provided.")
        return []

    # Extract the text from each of the valid links
    texts = [extract_text(link) for link in valid_links]

    # Preprocess the text (e.g., remove stop words, stem words, etc.)
    preprocessed_texts = preprocess(texts)

    # Check if there are any valid preprocessed texts
    if not preprocessed_texts:
        st.warning("No valid texts extracted from the provided URLs.")
        return []

    # Convert the preprocessed text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)

    # Check the shape of X
    print(f"Shape of X: {X.shape}")

    # Check if there are any valid features
    if X.shape[0] == 0 or X.shape[1] == 0:
        st.warning("No valid features extracted from the preprocessed texts.")
        return []

    # Use KMeans to cluster the articles into groups based on their content
    kmeans = KMeans(n_clusters=10, random_state=0)

    # Try to fit the KMeans model
    try:
        kmeans.fit(X)
    except Exception as e:
        print(f"Error while fitting KMeans: {e}")
        return []

    # Return the cluster labels for each article
    return kmeans.labels_



def main():
    # Title and description
    st.title("News Article Clustering")
    st.write("This web app clusters news articles based on their content similarity.")

    # Provide a list of news article links
    links = st.text_area("Enter news article links (one per line)", """
    https://www.gorillatoursafrica.com/things-to-do-in-kigali/
    https://www.thetravelersbuddy.com/2022/08/30/best-things-rwanda/
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

    # Save the vectorizer and kmeans models
    vectorizer_path = 'tfidf_vectorizer.joblib'
    kmeans_path = 'kmeans_model.joblib'
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(kmeans, kmeans_path)

# Run the Streamlit app
if __name__ == "__main__":
    main()
