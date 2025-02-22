import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.cluster import KMeans
import os
import json

question = "https://techathon-06ht.onrender.com/api/questions/get-questions"

def load_onet_data():
    print("Loading O*NET dataset...")
    occupations = pd.read_excel("ONET/Occupation Data.xlsx")
    skills = pd.read_excel("ONET/Skills.xlsx") 
    interests = pd.read_excel("ONET/Interests.xlsx")
    values = pd.read_excel("ONET/Work Values.xlsx")
    
    print("O*NET data loaded successfully!")
    return occupations, skills, interests, values

def preprocess_data(occupations, skills, interests, values):
    print("Preprocessing O*NET data...")
    # Create pivot tables for each category
    skills_pivot = pd.pivot_table(skills, values='Data Value', 
                                index='O*NET-SOC Code', 
                                columns='Element Name',
                                fill_value=0)
    
    interests_pivot = pd.pivot_table(interests, values='Data Value',
                                   index='O*NET-SOC Code',
                                   columns='Element Name',
                                   fill_value=0)
    
    values_pivot = pd.pivot_table(values, values='Data Value',
                                 index='O*NET-SOC Code',
                                 columns='Element Name',
                                 fill_value=0)
    
    # Get basic occupation information
    merged_data = occupations[['O*NET-SOC Code', 'Title', 'Description']].set_index('O*NET-SOC Code')
    
    # Merge all data
    for pivot in [skills_pivot, interests_pivot, values_pivot]:
        if not pivot.empty:
            merged_data = merged_data.join(pivot, how='left')
    
    merged_data = merged_data.fillna(0)
    
    # Store feature names in order
    feature_columns = [col for col in merged_data.columns if col not in ['Title', 'Description']]
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    print("Data preprocessing complete!")
    return merged_data

def train_ml_model(merged_data):
    print("Training machine learning model...")
    # Load feature columns to ensure consistency
    feature_columns = joblib.load('feature_columns.pkl')
    features = merged_data[feature_columns]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=20, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    merged_data['Cluster'] = clusters
    
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("ML model training complete!")
    return merged_data, scaler, kmeans

def load_quiz_questions():
    try:
        question
    except FileNotFoundError:
        print("Error: question not found!")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in questions!")
        return None

def run_career_quiz():
    questions = load_quiz_questions()
    if not questions:
        return None
    
    user_profile = {}
    
    print("\nCAREER RECOMMENDATION QUIZ")
    print("Answer these questions to receive personalized career recommendations.\n")
    
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}\n")
        for j, option in enumerate(q['options'], 1):
            print(f"{j}. {option}")

        num_options = len(q['options'])
        
        while True:
            try:
                choice = int(input("\nYour choice (in numbers): "))
                if 1 <= choice <= num_options:
                    break
                print(f"Please enter a number between 1 and {num_options}.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_option = q['options'][choice-1]
        attributes = q['mapping'][selected_option]
        
        for attr, value in attributes.items():
            if attr in user_profile:
                user_profile[attr] = (user_profile[attr] + value) / 2
            else:
                user_profile[attr] = value
        print()
    
    return user_profile

def generate_recommendations(user_profile, merged_data, scaler, kmeans, top_n=5):
    print("Generating career recommendations...")
    feature_columns = joblib.load('feature_columns.pkl')
    user_features = np.zeros(len(feature_columns))
    feature_to_index = {feat: i for i, feat in enumerate(feature_columns)}
    for attr, value in user_profile.items():
        if attr in feature_to_index:
            user_features[feature_to_index[attr]] = value
        else:
            print(f"Warning: Feature '{attr}' from user profile not found in training data")
    
    print(f"Number of features in user profile: {len(user_profile)}")
    print(f"Number of features in training data: {len(feature_columns)}")
    print("Missing features in user profile:", [feat for feat in feature_columns if feat not in user_profile])
    
    user_features = user_features.reshape(1, -1)
    scaled_features = scaler.transform(user_features)
    user_cluster = kmeans.predict(scaled_features)[0]
    cluster_occupations = merged_data[merged_data['Cluster'] == user_cluster]
    similarities = []
    
    for idx, row in cluster_occupations.iterrows():
        occ_features = row[feature_columns].values.reshape(1, -1)
        similarity = cosine_similarity(scaled_features, scaler.transform(occ_features))[0][0]
        similarities.append((idx, row['Title'], similarity, row['Description']))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_n]

def main():
    occupations, skills, interests, values = load_onet_data()
    merged_data = preprocess_data(occupations, skills, interests, values)
    merged_data, scaler, kmeans = train_ml_model(merged_data)
    joblib.dump(merged_data, 'merged_data.pkl')

    user_profile = run_career_quiz()
    if user_profile is None:
        return

    recommendations = generate_recommendations(user_profile, merged_data, scaler, kmeans)
    
    print("\nYOUR CAREER RECOMMENDATIONS")
    print("-" * 30 + "\n")
    for i, (_, title, score, description) in enumerate(recommendations, 1):
        print(f"{i}. {title}")
        print(f"   Match Score: {score*100:.1f}%")
        print(f"   Description: {description[:]}...")
        print()

if __name__ == "__main__":
    main()