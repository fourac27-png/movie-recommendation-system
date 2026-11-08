# CineMatch — Movie Recommendation System

A beginner-friendly AI/ML mini project built with Python, Streamlit, Pandas and Scikit-learn.

## What it does
Choose a movie and CineMatch recommends similar movies using:
- **Content-Based Filtering**
- **TF-IDF Vectorization** to convert movie descriptions into numerical features
- **Cosine Similarity** to compare movies

This is a learning/demo dataset containing 40 movies. It does not use Netflix data or user watch histories.

## Run locally
1. Install Python 3.9 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy free using GitHub + Streamlit Community Cloud
1. Create a free account at https://github.com/
2. Create a new **public** repository, for example `movie-recommendation-system`.
3. Upload these three files to the repository root:
   - `app.py`
   - `movies.csv`
   - `requirements.txt`
4. Create/sign in at https://share.streamlit.io/ using GitHub.
5. Choose **Create app**, select your repository, branch `main`, and main file path `app.py`.
6. Click **Deploy**. Streamlit Community Cloud will build the app and provide a public website URL.

Free hosting availability and limits are controlled by Streamlit and may change.

## Project report notes
- **Title:** CineMatch — Content-Based Movie Recommendation System
- **Problem:** Finding movies similar to a movie a user already likes can be time-consuming.
- **Objective:** Recommend similar movies from their genres, plot descriptions and keywords.
- **AI/ML concepts:** Text feature extraction, TF-IDF, vector similarity, content-based recommendation.
- **Tools:** Python, Pandas, Scikit-learn, Streamlit, GitHub.
- **Limitations:** The demo dataset is small and recommendations depend on the text metadata. It is not a trained deep-learning model and does not learn from user ratings.

## Important
Use this as a starter project. Run it, test different movies, understand the code, and personalize it before submitting. Be ready to explain TF-IDF and cosine similarity during your demonstration.
