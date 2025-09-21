# Movie Rating Prediction Project
# Predict movie ratings based on genre, director, and actors

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#  Load dataset
df = pd.read_csv("movies.csv")
print("First 5 rows of dataset:")
print(df.head())

#  Handle missing values
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['director'] = df['director'].fillna('Unknown')
df['genre'] = df['genre'].fillna('Unknown')
df['actors'] = df['actors'].fillna('Unknown')

#  Feature Engineering
# Extract multiple actors and primary genre for modeling
df['main_actor'] = df['actors'].apply(lambda x: x.split('|')[0])
df['main_genre'] = df['genre'].apply(lambda x: x.split('|')[0])

# You can also create a feature like number of genres
df['num_genres'] = df['genre'].apply(lambda x: len(x.split('|')))
# And number of actors
df['num_actors'] = df['actors'].apply(lambda x: len(x.split('|')))

#  Select features and target
features = ['director', 'main_genre', 'main_actor', 'num_genres', 'num_actors']
X = df[features]
y = df['rating']

#  Encode categorical features
categorical_features = ['director', 'main_genre', 'main_actor']
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep numeric features like num_genres, num_actors
)

X_encoded = encoder.fit_transform(X)

#  Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

#  Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R2 Score: {r2:.2f}")

#  Predict rating for a new movie
new_movie = pd.DataFrame([{
    'director': 'Steven Spielberg',
    'main_genre': 'Action',
    'main_actor': 'Tom Hanks',
    'num_genres': 1,
    'num_actors': 1
}])

# Encode new movie using same encoder
new_movie_encoded = encoder.transform(new_movie)
predicted_rating = model.predict(new_movie_encoded)
print(f"Predicted Rating for the new movie: {predicted_rating[0]:.2f}")
