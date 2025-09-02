"""
Heart Failure Prediction Machine Learning Pipeline
A comprehensive and modular ML pipeline for predicting heart failure events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, classification_report, 
                           confusion_matrix, precision_recall_curve, 
                           average_precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartFailurePipeline:
    """A comprehensive ML pipeline for heart failure prediction."""
    
    def __init__(self, data_path='heart_failure_clinical_records.csv', random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_and_validate_data(self):
        """Load and validate the heart failure dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            
            # Basic validation
            if 'DEATH_EVENT' not in self.df.columns:
                raise ValueError("Target variable 'DEATH_EVENT' not found in dataset")
                
            # Check for missing values
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print("Warning: Missing values found:")
                print(missing_values[missing_values > 0])
                
            # Check class distribution
            class_dist = self.df['DEATH_EVENT'].value_counts()
            print(f"\nClass distribution:\n{class_dist}")
            print(f"Class imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self, test_size=0.2):
        """Preprocess the data and split into train/test sets."""
        try:
            # Separate features and target
            self.X = self.df.drop('DEATH_EVENT', axis=1)
            self.y = self.df['DEATH_EVENT']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                stratify=self.y,
                random_state=self.random_state
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f"Data split: Train={self.X_train.shape}, Test={self.X_test.shape}")
            return True
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return False
    
    def handle_class_imbalance(self, method='smote'):
        """Handle class imbalance using various techniques."""
        try:
            if method == 'smote':
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(self.X_train_scaled, self.y_train)
                print("Applied SMOTE for class imbalance handling")
                
            elif method == 'undersample':
                undersampler = RandomUnderSampler(random_state=self.random_state)
                X_res, y_res = undersampler.fit_resample(self.X_train_scaled, self.y_train)
                print("Applied Random UnderSampling")
                
            else:
                X_res, y_res = self.X_train_scaled, self.y_train
                print("No imbalance handling applied")
                
            return X_res, y_res
            
        except Exception as e:
            print(f"Error in handling class imbalance: {e}")
            return self.X_train_scaled, self.y_train
    
    def train_models(self, X_res, y_res):
        """Train multiple models and select the best one."""
        try:
            # Define base models
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state,
                    class_weight='balanced'
                )),
                ('xgb', XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss',
                    random_state=self.random_state,
                    scale_pos_weight=len(y_res[y_res==0])/len(y_res[y_res==1])
                ))
            ]
            
            # Stacking ensemble
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(),
                cv=5
            )
            
            # Hyperparameter tuning for stacking
            param_grid = {
                'final_estimator__C': [0.1, 1, 10],
                'final_estimator__solver': ['liblinear', 'lbfgs']
            }
            
            grid_search = GridSearchCV(
                stacking_model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            print("Training models with hyperparameter tuning...")
            grid_search.fit(X_res, y_res)
            
            self.best_model = grid_search.best_estimator_
            self.models['stacking'] = self.best_model
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Also train individual models for comparison
            for name, model in base_models:
                model.fit(X_res, y_res)
                self.models[name] = model
            
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def evaluate_models(self):
        """Evaluate all trained models on test data."""
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(self.X_test_scaled)
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Metrics
                auc_roc = roc_auc_score(self.y_test, y_proba)
                report = classification_report(self.y_test, y_pred)
                
                results[name] = {
                    'auc_roc': auc_roc,
                    'classification_report': report,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                print(f"\n=== {name.upper()} Model Results ===")
                print(f"AUC-ROC: {auc_roc:.4f}")
                print("Classification Report:")
                print(report)
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return results
    
    def explain_model(self, model_name='xgb'):
        """Explain model predictions using SHAP."""
        try:
            if model_name not in self.models:
                print(f"Model {model_name} not found")
                return
            
            model = self.models[model_name]
            
            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test_scaled)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")
            plt.title(f'SHAP Feature Importance - {model_name.upper()}')
            plt.tight_layout()
            plt.savefig(f'shap_feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Force plot for first prediction
            plt.figure(figsize=(12, 6))
            shap.force_plot(explainer.expected_value, shap_values[0], self.X_test.iloc[0], matplotlib=True)
            plt.title(f'SHAP Force Plot - First Prediction ({model_name.upper()})')
            plt.tight_layout()
            plt.savefig(f'shap_force_plot_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"SHAP explanations saved as images for {model_name}")
            
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
    
    def plot_evaluation_metrics(self, results):
        """Plot evaluation metrics and comparison."""
        try:
            # Create comparison plot
            model_names = list(results.keys())
            auc_scores = [results[name]['auc_roc'] for name in model_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, auc_scores, color=['blue', 'green', 'orange'])
            plt.title('Model Comparison - AUC-ROC Scores')
            plt.ylabel('AUC-ROC Score')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, auc_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # ROC Curve for best model
            best_model_name = max(results.items(), key=lambda x: x[1]['auc_roc'])[0]
            y_proba_best = results[best_model_name]['probabilities']
            
            fpr, tpr, _ = roc_curve(self.y_test, y_proba_best)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {results[best_model_name]["auc_roc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Best Model')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('roc_curve_best_model.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
    
    def save_model(self, filename='heart_failure_model.pkl'):
        """Save the best model to disk."""
        try:
            if self.best_model:
                joblib.dump({
                    'model': self.best_model,
                    'scaler': self.scaler,
                    'feature_names': self.X.columns.tolist()
                }, filename)
                print(f"Model saved successfully as {filename}")
            else:
                print("No model to save")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def run_pipeline(self):
        """Run the complete ML pipeline."""
        print("=" * 50)
        print("HEART FAILURE PREDICTION ML PIPELINE")
        print("=" * 50)
        
        # Step 1: Load and validate data
        if not self.load_and_validate_data():
            return
        
        # Step 2: Preprocess data
        if not self.preprocess_data():
            return
        
        # Step 3: Handle class imbalance
        X_res, y_res = self.handle_class_imbalance('smote')
        
        # Step 4: Train models
        if not self.train_models(X_res, y_res):
            return
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Model explanation
        self.explain_model('xgb')
        
        # Step 7: Plot evaluation metrics
        self.plot_evaluation_metrics(results)
        
        # Step 8: Save best model
        self.save_model()
        
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)

# Main execution
if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = HeartFailurePipeline()
    pipeline.run_pipeline()
