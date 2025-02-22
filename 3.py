import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.cluster import KMeans
import os
import json
import datetime
import uuid
from typing import List, Dict, Any

# Data Loading Functions
def load_onet_data():
    print("Loading O*NET dataset...")
    occupations = pd.read_excel("ONET/Occupation Data.xlsx")
    skills = pd.read_excel("ONET/Skills.xlsx") 
    interests = pd.read_excel("ONET/Interests.xlsx")
    values = pd.read_excel("ONET/Work Values.xlsx")
    education = pd.read_excel("ONET/Education, Training, and Experience.xlsx")
    tech_skills = pd.read_excel("ONET/Technology Skills.xlsx")
    print("O*NET data loaded successfully!")
    return occupations, skills, interests, values, education, tech_skills

def load_coursera_data():
    print("\nLoading Coursera course data...")
    coursera_df = pd.read_csv("Coursera.csv")
    
    coursera_df['skills'] = coursera_df['skills'].apply(
        lambda x: [skill.strip().lower() for skill in x.split(',')] if pd.notnull(x) else []
    )
    
    skill_course_map = {}
    for _, row in coursera_df.iterrows():
        course_info = {
            'title': row['course'],
            'partner': row['partner'],
            'skills': row['skills'],
            'rating': row['rating'],
            'level': row['level'],
            'duration': row['duration'],
            'certificate_type': row['certificatetype'],
            'review_count': row['reviewcount']
        }
        for skill in row['skills']:
            if skill not in skill_course_map:
                skill_course_map[skill] = []
            skill_course_map[skill].append(course_info)
    
    print(f"Loaded {len(coursera_df)} courses with {len(skill_course_map)} unique skills")
    return coursera_df, skill_course_map

def preprocess_data(occupations, skills, interests, values, education, tech_skills):
    print("\nPreprocessing O*NET data...")
    
    pivots = []
    for df, category in zip([skills, interests, values, education],
                          ['skills', 'Interests', 'Values', 'Education']):
        pivot = df.pivot_table(
            index='O*NET-SOC Code', 
            columns='Element Name', 
            values='Data Value',
            fill_value=0
        )
        pivots.append(pivot)
    
    tech_skills_grouped = tech_skills.groupby('O*NET-SOC Code')['Example'].apply(list).reset_index()
    tech_skills_dict = dict(zip(tech_skills_grouped['O*NET-SOC Code'], tech_skills_grouped['Example']))
    
    occupation_basics = occupations[['O*NET-SOC Code', 'Title', 'Description']]
    merged_data = occupation_basics.set_index('O*NET-SOC Code')
    
    for pivot in pivots:
        if not pivot.empty:
            merged_data = merged_data.join(pivot, how='left')
    
    merged_data = merged_data.fillna(0)
    merged_data['tech_skills'] = merged_data.index.map(lambda x: json.dumps(tech_skills_dict.get(x, [])))
    
    print("Data preprocessing complete!")
    return merged_data

def train_ml_model(merged_data):
    print("\nTraining machine learning model...")
    
    feature_columns = merged_data.columns.difference(['Title', 'Description', 'tech_skills'])
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

def run_career_quiz():
    questions = [
{
        "question": "You're a project manager leading a team developing a new mobile app.  The app's core feature relies on real-time location data, but testing reveals significant inaccuracies in certain areas.  The launch date is fast approaching. What's your next move?",
        "options": [
            "Immediately inform stakeholders about the issue and propose a revised launch date to allow for further testing and development.",
            "Assemble the development team to brainstorm potential solutions and implement quick fixes, even if they're not perfect.",
            "Analyze the location data to pinpoint the specific problem areas and prioritize fixing those with the highest user impact.",
            "Contact a third-party location data provider to explore alternative solutions or partnerships to improve accuracy."
        ],
        "mapping": {
            "Immediately inform stakeholders about the issue and propose a revised launch date to allow for further testing and development.": {"Project Management": 5, "Communication": 4, "Critical Thinking": 4, "Conventional": 4},
            "Assemble the development team to brainstorm potential solutions and implement quick fixes, even if they're not perfect.": {"Problem-Solving": 5, "Teamwork": 4, "Adaptability": 4, "Realistic": 4},
            "Analyze the location data to pinpoint the specific problem areas and prioritize fixing those with the highest user impact.": {"Data Analysis": 5, "Critical Thinking": 5, "Investigative": 4, "Realistic": 5},
            "Contact a third-party location data provider to explore alternative solutions or partnerships to improve accuracy.": {"Negotiation": 4, "Problem-Solving": 4, "Technology Design": 5, "Enterprising": 4}
        }
    },
    {
        "question": "You are a customer service representative. A customer calls, furious because their recent order arrived damaged. They're demanding a full refund and threatening to leave negative reviews online. How do you respond?",
        "options": [
            "Listen empathetically to the customer's complaint, apologize for the inconvenience, and assure them you'll do everything to resolve the issue.",
            "Explain the company's return policy and offer a replacement product, even if it's slightly more expensive than what they originally ordered.",
            "Offer a full refund immediately, without requiring the customer to return the damaged product, to de-escalate the situation quickly.",
            "Document the customer's complaint thoroughly and escalate it to a supervisor for further handling."
        ],
        "mapping": {
            "Listen empathetically to the customer's complaint, apologize for the inconvenience, and assure them you'll do everything to resolve the issue.": {"Active Listening": 5, "Service Orientation": 5, "Social": 4, "Realistic": 4},
            "Explain the company's return policy and offer a replacement product, even if it's slightly more expensive than what they originally ordered.": {"Problem-Solving": 4, "Customer Service": 4, "Negotiation": 4, "Conventional": 4},
            "Offer a full refund immediately, without requiring the customer to return the damaged product, to de-escalate the situation quickly.": {"Decision Making": 4, "Service Orientation": 5, "Adaptability": 4, "Enterprising": 4},
            "Document the customer's complaint thoroughly and escalate it to a supervisor for further handling.": {"Record Keeping": 4, "Communication": 4, "Conventional": 5}
        }
    },
    {
        "question": "You're a marketing manager launching a new product.  Initial sales are lower than projected.  What's your first step?",
        "options": [
            "Conduct market research to understand why the product isn't resonating with customers and identify any unmet needs.",
            "Increase advertising spend and explore new marketing channels to reach a wider audience.",
            "Analyze sales data to identify any patterns or trends that might explain the lower-than-expected performance.",
            "Refine the product's messaging and value proposition to better communicate its benefits to potential customers."
        ],
        "mapping": {
            "Conduct market research to understand why the product isn't resonating with customers and identify any unmet needs.": {"Market Research": 5, "Critical Thinking": 4, "Investigative": 5, "Realistic": 4},
            "Increase advertising spend and explore new marketing channels to reach a wider audience.": {"Marketing": 4, "Sales Strategy": 4, "Enterprising": 4, "Conventional": 4},
            "Analyze sales data to identify any patterns or trends that might explain the lower-than-expected performance.": {"Data Analysis": 5, "Critical Thinking": 5, "Investigative": 4, "Realistic": 5},
            "Refine the product's messaging and value proposition to better communicate its benefits to potential customers.": {"Marketing": 4, "Communication": 4, "Creativity": 4, "Artistic": 4}
        }
    },
    {
        "question": "You are a software developer. You've been tasked with implementing a new feature in a complex software system.  You realize that the current system architecture isn't well-suited for this new functionality. What do you do?",
        "options": [
            "Discuss the architectural limitations with the project lead and suggest potential solutions, including refactoring or redesigning parts of the system.",
            "Try to implement the feature within the existing architecture, even if it's not ideal, to meet the immediate deadline.",
            "Research best practices for implementing similar features in other systems and propose a plan for adapting those practices to the current project.",
            "Document the architectural challenges and their potential impact on the project's long-term maintainability and scalability."
        ],
        "mapping": {
            "Discuss the architectural limitations with the project lead and suggest potential solutions, including refactoring or redesigning parts of the system.": {"Problem-Solving": 5, "Communication": 4, "Systems Analysis": 5, "Conventional": 4},
            "Try to implement the feature within the existing architecture, even if it's not ideal, to meet the immediate deadline.": {"Adaptability": 4, "Time Management": 4, "Realistic": 4},
            "Research best practices for implementing similar features in other systems and propose a plan for adapting those practices to the current project.": {"Research Skills": 5, "Technology Design": 4, "Investigative": 5},
            "Document the architectural challenges and their potential impact on the project's long-term maintainability and scalability.": {"Documentation": 4, "Systems Analysis": 4, "Critical Thinking": 4, "Conventional": 5}
        }
    },
    {
        "question": "You're an HR manager.  Several employees have reported feeling stressed and overwhelmed due to heavy workloads and tight deadlines. What actions do you take?",
        "options": [
            "Organize team-building activities and social events to improve morale and foster a more supportive work environment.",
            "Review workload distribution and project timelines to identify potential imbalances and adjust them accordingly.",
            "Offer stress management workshops or resources to help employees cope with pressure and improve their well-being.",
            "Conduct anonymous surveys to gather feedback on workload and identify any underlying issues contributing to stress."
        ],
        "mapping": {
            "Organize team-building activities and social events to improve morale and foster a more supportive work environment.": {"Teamwork": 4, "Social": 4, "Leadership": 4, "Enterprising": 4},
            "Review workload distribution and project timelines to identify potential imbalances and adjust them accordingly.": {"Project Management": 4, "Management of Personnel Resources": 5, "Conventional": 4, "Realistic": 4},
            "Offer stress management workshops or resources to help employees cope with pressure and improve their well-being.": {"Counseling and Guidance": 4, "Human Resources": 5, "Social": 4},
            "Conduct anonymous surveys to gather feedback on workload and identify any underlying issues contributing to stress.": {"Social Perceptiveness": 5, "Data Analysis": 4, "Investigative": 4, "Conventional": 4}
        }
    }
    ]
    
    user_profile = {}
    print("\nCAREER RECOMMENDATION QUIZ")
    print("Answer scenario-based questions for personalized recommendations.\n")
    
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
            user_profile[attr] = user_profile.get(attr, 0) + value
    
    for attr in user_profile:
        user_profile[attr] = user_profile[attr]/len(questions)
    
    return user_profile

def generate_recommendations(user_profile, merged_data, scaler, kmeans, top_n=5):
    print("\nGenerating career recommendations...")
    
    feature_columns = scaler.feature_names_in_
    
    user_vector = pd.Series(0.0, index=feature_columns, dtype=float)
    for attr, value in user_profile.items():
        if attr in feature_columns:
            user_vector[attr] = float(value)
    scaled_user_vector = scaler.transform(user_vector.values.reshape(1, -1))
    user_cluster = kmeans.predict(scaled_user_vector)[0]
    cluster_occupations = merged_data[merged_data['Cluster'] == user_cluster]
    
    similarities = []
    for idx, row in cluster_occupations.iterrows():
        occupation_vector = row[feature_columns].values.astype(float).reshape(1, -1)
        similarity = cosine_similarity(scaled_user_vector, scaler.transform(occupation_vector))[0][0]
        similarities.append((idx, row['Title'], similarity, row['Description']))
    
    return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]

def get_career_details(career_title, merged_data):
    career_info = merged_data[merged_data['Title'] == career_title]
    if career_info.empty:
        return {
            "title": career_title,
            "description": "Career information not found in O*NET database.",
            "skills": [],
            "tech_skills": [],
            "education_level": "Not specified"
        }
    
    career_row = career_info.iloc[0]
    education_cols = [col for col in merged_data.columns if 'Education' in col]
    education_level = "Not specified"
    
    if education_cols:
        education_series = career_row[education_cols]
        if not education_series.empty:
            max_col = education_series.idxmax()
            education_level = max_col.split(' - ')[-1]
    
    skill_cols = [col for col in merged_data.columns if 'Skill' in col or 'Ability' in col]
    skills = []
    if skill_cols:
        skills_series = career_row[skill_cols].sort_values(ascending=False)
        skills = [s.split(' - ')[-1] for s in skills_series.head(10).index.tolist()]
    
    try:
        tech_skills = json.loads(career_row['tech_skills'])
    except:
        tech_skills = []
    
    return {
        "title": career_title,
        "description": career_row['Description'],
        "skills": skills,
        "tech_skills": tech_skills,
        "education_level": education_level
    }
def enhance_career_with_courses(career_details, skill_course_map, top_n=3):
    enhanced = career_details.copy()
    required_skills = [s.lower() for s in enhanced['skills'] + enhanced['tech_skills']]
    
    course_scores = {}
    for skill in required_skills:
        for course in skill_course_map.get(skill, []):
            course_title = course['title']
            if course_title not in course_scores:
                course_scores[course_title] = {
                    'course': course,
                    'score': 0,
                    'matched_skills': set()
                }
            course_scores[course_title]['score'] += 1
            course_scores[course_title]['matched_skills'].add(skill)
    
    sorted_courses = sorted(course_scores.values(), 
                          key=lambda x: (-x['score'], -x['course']['rating'], -x['course']['review_count']))
    
    enhanced['recommended_courses'] = []
    for course in sorted_courses[:top_n]:
        enhanced['recommended_courses'].append({
            'title': course['course']['title'],
            'partner': course['course']['partner'],
            'matched_skills': list(course['matched_skills'])[:3],
            'rating': course['course']['rating'],
            'level': course['course']['level'],
            'duration': course['course']['duration'],
            'certificate_type': course['course']['certificate_type'],
            'url': f"https://www.coursera.org/learn/{course['course']['title'].lower().replace(' ', '-')}"
        })
    
    return enhanced

def recommend_learning_pathway(career_title, merged_data, skill_course_map):
    career_details = get_career_details(career_title, merged_data)
    career_details = enhance_career_with_courses(career_details, skill_course_map)
    
    milestones = [
        {
            "id": "foundation",
            "title": "Core Skill Development",
            "tasks": [],
            "completion_percentage": 0
        },
        {
            "id": "technical",
            "title": "Technical Specialization",
            "tasks": [],
            "completion_percentage": 0
        },
        {
            "id": "professional",
            "title": "Career Preparation",
            "tasks": [],
            "completion_percentage": 0
        }
    ]
    
    for skill in career_details['skills'][:3]:
        matching_courses = [c for c in career_details['recommended_courses']
                          if skill.lower() in c['matched_skills']]
        
        milestones[0]['tasks'].append({
            "id": str(uuid.uuid4()),
            "title": f"Master {skill}",
            "description": f"Develop core competency in {skill}",
            "estimated_hours": 40,
            "resources": matching_courses[:2],
            "completed": False
        })
    
    for tech in career_details['tech_skills'][:3]:
        matching_courses = [c for c in career_details['recommended_courses']
                          if any(tech.lower() in s for s in c['matched_skills'])]
        
        milestones[1]['tasks'].append({
            "id": str(uuid.uuid4()),
            "title": f"{tech} Proficiency",
            "description": f"Develop practical skills with {tech}",
            "estimated_hours": 30,
            "resources": matching_courses[:2],
            "completed": False
        })
    
    milestones[2]['tasks'].extend([
        {
            "id": str(uuid.uuid4()),
            "title": "Build Professional Portfolio",
            "description": "Create portfolio showcasing key projects",
            "estimated_hours": 20,
            "resources": [],
            "completed": False
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Interview Preparation",
            "description": "Practice common interview questions",
            "estimated_hours": 15,
            "resources": [],
            "completed": False
        }
    ])
    
    return {
        "career": career_details,
        "learning_pathway": {
            "milestones": milestones,
            "overall_progress": 0
        }
    }


def main():
    onet_data = load_onet_data()
    merged_data = preprocess_data(*onet_data)
    merged_data, scaler, kmeans = train_ml_model(merged_data)
    coursera_df, skill_course_map = load_coursera_data()
    
    user_profile = run_career_quiz()
    
    recommendations = generate_recommendations(user_profile, merged_data, scaler, kmeans)
    
    top_career = recommendations[0][1]
    learning_pathway = recommend_learning_pathway(top_career, merged_data, skill_course_map)
    
    
    print("\n" + "="*50)
    print(f"Top Career Recommendation: {top_career}")
    print(f"Description: {learning_pathway['career']['description'][:500]}...")
    
    print("\nLearning Pathway:")
    for milestone in learning_pathway['learning_pathway']['milestones']:
        print(f"\n{milestone['title']}:")
        for task in milestone['tasks']:
            print(f" - {task['title']}")
            if task['resources']:
                print("   Recommended Courses:")
                for course in task['resources']:
                    print(f"    * {course['title']} by {course['partner']}")
                    print(f"      Rating: {course['rating']} | Level: {course['level']}")
    

if __name__ == "__main__":
    main()