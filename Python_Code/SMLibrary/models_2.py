import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
import joblib
import os
import plotly.io as pio
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.svm import * 

def train_and_evaluate_pipeline(all_data, n_components, output_dir):
    """Train and evaluate pipelines with different models."""
    
    # Prepare features and target variable
    X = all_data.drop(['Class','File','RT','Sum'], axis=1)
    y = all_data['Class']
    
    # Encode target variable if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Identify numeric columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Handle missing values
                ('scaler', StandardScaler())  # Apply scaling
            ]), numeric_features)
        ],
        remainder='passthrough'  # Pass through any other features unchanged
    )
    
    # Initialize TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    
    models = {
        'SGD': SGDClassifier(),
        # 'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=30),
        # 'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, algorithm="SAMME"),
        'Ridge Classifier': RidgeClassifierCV()
        # 'SVC': SVC(C=1.0, kernel='rbf', degree=3, gamma='scale'),
        # 'Gaussian Naive Bayes': GaussianNB(),
        # 'Decision Tree Classifier': DecisionTreeClassifier(),
        # 'Hist Gradient Boosting Classifier': HistGradientBoostingClassifier(max_iter=100),
        # 'BernoulliNB': BernoulliNB()
    }
    
    pipelines = {}
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('svd', svd),
            ('model', model)
        ])
        
        # Fit pipeline on the entire dataset
        pipeline.fit(X, y)
        
        # Cross-validation prediction
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=10)
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Calculate classification report
        report = classification_report(y, y_pred)
        
        # Cross-validation scores
        scores = cross_val_score(pipeline, X, y, cv=cv)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Create interactive confusion matrix using plotly
        classes = np.unique(y)  # Get unique class labels
        fig = ff.create_annotated_heatmap(
            z=conf_matrix.tolist(),  # Convert to list to ensure correct format
            x=classes.tolist(),  # Convert to list to ensure correct format
            y=classes.tolist(),  # Convert to list to ensure correct format
            colorscale='Viridis', 
            showscale=True
        )
        fig.update_layout(title=f'Matrice de confusion pour le modèle {name}')
        
        # Save the plotly figure as a JSON
        plot_path = os.path.join(output_dir, f'{name}_confusion_matrix.json')
        pio.write_json(fig, plot_path)
        
        # Store the results
        results[name] = {
            'classification_report': report,
            'mean_score': mean_score,
            'std_score': std_score,
            'confusion_matrix_json': plot_path
        }
        
        # Store the pipeline
        pipelines[name] = pipeline
        
    return pipelines, numeric_features, results


import json

def display_confusion_matrix(json_file_path):
    """Display the confusion matrix from a saved JSON file."""
    
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        fig_json = json.load(file)  # Use json.load to parse JSON
    
    # Convert JSON dictionary to JSON string
    fig_json_str = json.dumps(fig_json)
    
    # Create the figure from the JSON string
    fig = pio.from_json(fig_json_str)
    
    # Display the figure
    fig.show()

def display_classification_report(model_name, results):
    """Display the classification report for a specific model."""
    if model_name not in results:
        raise ValueError(f"Le modèle '{model_name}' n'existe pas dans les résultats.")
    
    # Extract and print the classification report
    report = results[model_name]['classification_report']
    print(f"Classification report pour le modèle {model_name} :\n{report}")

def save_selected_models(pipelines, name='', output_dir=''):
    if name in pipelines:
        save_path = os.path.join(output_dir, f'{name}_pipeline.pkl')
        joblib.dump(pipelines[name], save_path)
        print(f"Pipeline {name} enregistré sous le nom '{name}_pipeline.pkl'.")
    else:
        print(f"Modèle {name} non trouvé dans les pipelines disponibles.")

def save_mzstrain(numeric_features, mass_range_name='mzs_glioblastome', output_dir=''):
    # Crée le chemin complet pour la sauvegarde
    save_path = os.path.join(output_dir, f'{mass_range_name}.pkl')
    # Sauvegarde des données dans le fichier spécifié
    joblib.dump(numeric_features, save_path)