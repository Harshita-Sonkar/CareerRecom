import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.cluster import KMeans
import os
import json

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
    skills_pivot = skills.pivot_table(
        index='O*NET-SOC Code', 
        columns='Element Name', 
        values='Data Value',
        fill_value=0
    )
    interests_pivot = interests.pivot_table(
        index='O*NET-SOC Code',
        columns='Element Name',
        values='Data Value',
        fill_value=0
    )
    values_pivot = values.pivot_table(
        index='O*NET-SOC Code',
        columns='Element Name',
        values='Data Value',
        fill_value=0
    )

    occupation_basics = occupations[['O*NET-SOC Code', 'Title', 'Description']]
    merged_data = occupation_basics.set_index('O*NET-SOC Code')
    
    if not skills_pivot.empty:
        merged_data = merged_data.join(skills_pivot, how='left')
    
    if not interests_pivot.empty:
        merged_data = merged_data.join(interests_pivot, how='left')
    
    if not values_pivot.empty:
        merged_data = merged_data.join(values_pivot, how='left')
    
    merged_data = merged_data.fillna(0)
    
    print("Data preprocessing complete!")
    return merged_data

def train_ml_model(merged_data):
    print("Training machine learning model...")
    
    feature_columns = merged_data.columns.difference(['Title', 'Description'])
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
        with open('questions.json', 'r') as file:
            data = json.load(file)
            return data['questions']
    except FileNotFoundError:
        print("Error: career_questions.json not found!")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in career_questions.json!")
        return None

def run_career_quiz():
    questions = load_quiz_questions()
    if not questions:
        print("Could not load quiz questions. Exiting...")
        return None
    
    user_profile = {}
    
    print("CAREER RECOMMENDATION QUIZ")
    print("Answer these scenario-based questions to receive personalized career recommendations.")
    print("Your choices will help match your interests, skills, and values to potential career paths.\n")
    
    for i, q in enumerate(questions):
        print(f"Question {i+1}: {q['question']}")
        for j, option in enumerate(q['options']):
            print(f"  {j+1}. {option}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1-4): "))
                if 1 <= choice <= 4:
                    break
                else:
                    print("Please enter a number between 1 and 4.")
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
    print("Generating career recommendations based on your preferences...")
    
    feature_columns = merged_data.columns.difference(['Title', 'Description', 'Cluster'])
    user_vector = pd.Series(0, index=feature_columns)
    
    for attr, value in user_profile.items():
        if attr in user_vector.index:
            user_vector[attr] = value

    user_vector = user_vector[feature_columns]    
    scaled_user_vector = scaler.transform(user_vector.values.reshape(1, -1))
    user_cluster = kmeans.predict(scaled_user_vector)[0]
    cluster_occupations = merged_data[merged_data['Cluster'] == user_cluster]
    similarities = []
    for idx, row in cluster_occupations.iterrows():
        occupation_vector = row[feature_columns].values.reshape(1, -1)
        similarity = cosine_similarity(scaled_user_vector, scaler.transform(occupation_vector))[0][0]
        similarities.append((idx, row['Title'], similarity, row['Description']))
    similarities.sort(key=lambda x:x[2], reverse=True)
    top_recommendations = similarities[:top_n]
    return top_recommendations

def main():
    if os.path.exists('kmeans_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('merged_data.pkl'):
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        merged_data = joblib.load('merged_data.pkl')
    else:
        occupations, skills, interests, values = load_onet_data()
        merged_data = preprocess_data(occupations, skills, interests, values)
        merged_data, scaler, kmeans = train_ml_model(merged_data)
        
        joblib.dump(merged_data, 'merged_data.pkl')
    
    user_profile = run_career_quiz()
    if user_profile is None:
        return
    
    recommendations = generate_recommendations(user_profile, merged_data, scaler, kmeans)
    
    print("YOUR CAREER RECOMMENDATIONS")
    for i, (_, title, score, description) in enumerate(recommendations):
        print(f"{i+1}. {title}")
        print(f"   Match Score: {score*100:.1f}%")
        print(f"   Description: {description[:200]}...")
        print()

if __name__ == "__main__":
    main()