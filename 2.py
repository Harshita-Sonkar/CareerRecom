from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

question = "https://techathon-06ht.onrender.com/api/questions/get-questions"

app = FastAPI()

try:
    feature_columns = joblib.load("feature_columns.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    merged_data = joblib.load("merged_data.pkl")
    scaler = joblib.load("scaler.pkl")
    response = requests.get(question)
    response.raise_for_status()  
    quiz_data = response.json()['data']['questions'] 
except Exception as e:
    print(f"Error loading required files: {e}")
    raise RuntimeError("Required files not found")

class Question(BaseModel):
    id: int
    question: str
    options: List[str]

class QuizResponse(BaseModel):
    answers: Dict[int, int]  

class CareerRecommendation(BaseModel):
    title: str
    match_score: float
    description: str

class QuizResult(BaseModel):
    recommendations: List[CareerRecommendation]

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Career Quiz and Recommendation API",
        "endpoints": {
            "/questions": "GET - Retrieve quiz questions",
            "/submit-quiz": "POST - Submit quiz answers and get recommendations",
        }
    }

@app.get("/questions", response_model=List[Question])
async def get_questions():
    """Get all quiz questions with their options"""
    try:
        questions = []
        for i, q in enumerate(quiz_data['questions'], 1):
            questions.append(Question(
                id=i,
                question=q['question'],
                options=q['options']
            ))
        return questions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving questions: {str(e)}"
        )

@app.post("/submit-quiz", response_model=QuizResult)
async def submit_quiz(quiz_response: QuizResponse):
    try:
        if not quiz_response.answers:
            raise HTTPException(
                status_code=400,
                detail="No answers provided"
            )

        user_profile = {}
        for question_id, selected_option in quiz_response.answers.items():
            question_idx = question_id - 1
            if question_idx >= len(quiz_data['questions']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid question ID: {question_id}"
                )

            question = quiz_data['questions'][question_idx]
            if selected_option >= len(question['options']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid option for question {question_id}"
                )

            selected_option_text = question['options'][selected_option]
            attributes = question['mapping'][selected_option_text]

            for attr, value in attributes.items():
                if attr in user_profile:
                    user_profile[attr] = (user_profile[attr] + value) / 2
                else:
                    user_profile[attr] = value

        user_features = np.zeros(len(feature_columns))
        feature_to_index = {feat: i for i, feat in enumerate(feature_columns)}
        
        for attr, value in user_profile.items():
            if attr in feature_to_index:
                user_features[feature_to_index[attr]] = value

        user_features = user_features.reshape(1, -1)
        scaled_features = scaler.transform(user_features)
        user_cluster = kmeans.predict(scaled_features)[0]
        
        cluster_occupations = merged_data[merged_data['Cluster'] == user_cluster]
        recommendations = []

        for idx, row in cluster_occupations.iterrows():
            occ_features = row[feature_columns].values.reshape(1, -1)
            similarity = cosine_similarity(scaled_features, scaler.transform(occ_features))[0][0]
            recommendations.append(
                CareerRecommendation(
                    title=row['Title'],
                    match_score=float(similarity),
                    description=row['Description']
                )
            )

        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        recommendations = recommendations[:5] 

        return QuizResult(
            recommendations=recommendations
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing quiz: {str(e)}"
        )