# Movie Recommendation System

A web-based movie recommendation system that uses machine learning to suggest similar movies based on user input.

## Setup Instructions

### 1. Install Python Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Prepare Your Data Files
Make sure you have these files in the same directory as `app.py`:
- `movies.csv` - Your movie dataset with columns: title, year, genre, rating, overview, index
- `similarity_matrix.npy` - Your pre-computed similarity matrix

### 3. Run the Flask Server
\`\`\`bash
python app.py
\`\`\`
The server will start on `http://localhost:5000`

### 4. Access the Website
Simply open your browser and navigate to:
\`\`\`
http://localhost:5000
\`\`\`
The website is now served directly by Flask!

## Features

- **Smart Search**: Uses fuzzy matching to find movies even with slight spelling errors
- **ML-Powered Recommendations**: Returns top 5 similar movies based on your similarity matrix
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Results**: Instant recommendations as you search
- **Similarity Scores**: Shows how closely matched each recommendation is

## API Endpoints

- `GET /recommend?movie=<movie_name>` - Get recommendations for a movie
- `GET /health` - Check if the API is running

## CSV Format Expected

Your `movies.csv` should have these columns:
- `title` - Movie title
- `year` - Release year
- `genre` - Movie genres
- `rating` - Movie rating (0-10)
- `overview` - Movie description
- `index` - Unique index for similarity matrix
