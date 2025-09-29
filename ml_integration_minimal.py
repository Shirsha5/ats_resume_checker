# ml_integration_minimal.py
"""
Minimal ML integration that works alongside your existing ATS
Non-disruptive enhancement - existing system keeps working unchanged
Updated to work with LightGBM model that only uses CGPA feature
"""

import os
import sys
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import ML components, fail gracefully if not available
try:
    import numpy as np
    import pandas as pd
    import lightgbm as lgb  # Updated for LightGBM
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not installed. Run: pip install scikit-learn pandas numpy lightgbm")

class SimpleMLEnhancer:
    """
    Minimal ML enhancement for existing ATS system
    Falls back gracefully if ML components are not available
    Updated for LightGBM with CGPA-only model
    """
    
    def __init__(self, ml_models_dir='ml_models'):
        self.ml_models_dir = ml_models_dir
        self.ml_enabled = False
        self.classifier = None
        self.feature_names = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if ML_AVAILABLE:
            self._try_load_models()
    
    def _try_load_models(self):
        """Try to load ML models, fail gracefully if not available"""
        try:
            if os.path.exists(self.ml_models_dir):
                # Look for any classifier model
                model_files = [f for f in os.listdir(self.ml_models_dir) 
                              if 'classifier' in f.lower() and f.endswith('.pkl')]
                
                if model_files:
                    model_path = os.path.join(self.ml_models_dir, model_files[0])
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.classifier = model_data['model']
                        self.feature_names = model_data.get('selected_features', ['cgpa'])
                    else:
                        self.classifier = model_data
                        self.feature_names = ['cgpa']  # Default fallback
                    
                    self.ml_enabled = True
                    self.logger.info(f"✅ ML model loaded: {model_files[0]}")
                    self.logger.info(f"📋 Using features: {self.feature_names}")
                else:
                    self.logger.info("ℹ️ No ML models found - using rule-based evaluation only")
            else:
                self.logger.info("ℹ️ ML models directory not found - using rule-based evaluation only")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load ML models: {e}")
            self.ml_enabled = False
    
    def extract_features_for_model(self, parsed_resume: Dict, raw_text: str) -> Dict[str, float]:
        """
        Extract only the features that the model actually uses
        Updated to match your trained model's feature set
        """
        features = {}
        
        # Extract CGPA (the main feature your model uses)
        cgpa_raw = parsed_resume.get('cgpa', 0) or 0
        features['cgpa'] = float(cgpa_raw)
        
        # Extract other features in case future models use them
        features['academic_year'] = int(parsed_resume.get('academic_year', 0) or 0)
        features['company_law'] = 1 if parsed_resume.get('company_law', False) else 0
        features['contract_law'] = 1 if parsed_resume.get('contract_law', False) else 0
        
        # Experience features
        experience = parsed_resume.get('experience', {})
        features['internship_count'] = len(experience.get('internships', []))
        features['legal_research'] = 1 if experience.get('legal_research', False) else 0
        features['moot_court'] = 1 if experience.get('moot_court', False) else 0
        features['tier_firm'] = 1 if experience.get('tier_firm_internship', False) else 0
        features['publications'] = len(experience.get('publications', []))
        
        return features
    
    def enhance_evaluation(self, rule_based_result: Dict, parsed_resume: Dict, raw_text: str) -> Dict:
        """
        Enhance existing rule-based evaluation with ML insights
        
        Args:
            rule_based_result: Your existing evaluation result
            parsed_resume: Parsed resume data
            raw_text: Raw resume text
            
        Returns:
            Enhanced result with ML insights added
        """
        
        # Start with existing result
        enhanced_result = rule_based_result.copy()
        
        # Add ML enhancement section
        enhanced_result['ml_enhancement'] = {
            'ml_available': self.ml_enabled,
            'ml_prediction': None,
            'ml_confidence': None,
            'ml_hire_probability': None,
            'ml_insights': [],
            'features_used': self.feature_names,
            'enhancement_timestamp': datetime.now().isoformat()
        }
        
        # Add ML evaluation if available
        if self.ml_enabled and self.classifier:
            try:
                # Extract features
                all_features = self.extract_features_for_model(parsed_resume, raw_text)
                
                # Create feature vector using only the features the model was trained on
                if self.feature_names:
                    feature_vector = [all_features.get(name, 0) for name in self.feature_names]
                else:
                    # Fallback to CGPA only
                    feature_vector = [all_features.get('cgpa', 0)]
                
                # Convert to DataFrame to preserve feature names for LightGBM
                feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
                
                # Make prediction
                prediction = self.classifier.predict(feature_df)[0]
                
                # Get probabilities if available
                if hasattr(self.classifier, 'predict_proba'):
                    probabilities = self.classifier.predict_proba(feature_df)[0]
                    confidence = max(probabilities)
                    hire_probability = probabilities[1] if len(probabilities) > 1 else prediction
                else:
                    confidence = 0.8  # High confidence for simple model
                    hire_probability = prediction
                
                # Update ML enhancement
                enhanced_result['ml_enhancement'].update({
                    'ml_prediction': int(prediction),
                    'ml_hire_probability': float(hire_probability),
                    'ml_confidence': float(confidence),
                    'feature_values': dict(zip(self.feature_names, feature_vector)),
                    'ml_insights': self._generate_insights(all_features, prediction, hire_probability)
                })
                
                # Add ML recommendation
                enhanced_result['ml_recommendation'] = self._get_ml_recommendation(
                    hire_probability, confidence
                )
                
                self.logger.info(f"✅ ML enhancement applied - Prediction: {prediction}")
                
            except Exception as e:
                self.logger.error(f"❌ ML enhancement failed: {e}")
                enhanced_result['ml_enhancement']['error'] = str(e)
        
        return enhanced_result
    
    def _generate_insights(self, features: Dict, prediction: int, hire_prob: float) -> list:
        """Generate ML insights based on prediction and features"""
        
        insights = []
        
        # Prediction insight
        if hire_prob >= 0.8:
            insights.append("🌟 ML strongly recommends this candidate")
        elif hire_prob >= 0.6:
            insights.append("✅ ML shows positive indicators for this candidate")
        elif hire_prob >= 0.4:
            insights.append("⚠️ ML shows mixed signals - requires human review")
        else:
            insights.append("❌ ML identifies concerns with this candidate")
        
        # Feature-based insights (focusing on CGPA since that's what the model uses)
        cgpa = features.get('cgpa', 0)
        if cgpa >= 8.5:
            insights.append("🎓 Outstanding academic performance (CGPA ≥ 8.5)")
        elif cgpa >= 7.5:
            insights.append("📚 Strong academic performance (CGPA ≥ 7.5)")
        elif cgpa >= 6.5:
            insights.append("📖 Good academic performance (CGPA ≥ 6.5)")
        elif cgpa > 0:
            insights.append("⚠️ Academic performance below optimal threshold")
        
        # Additional context insights
        if features.get('tier_firm', 0) == 1:
            insights.append("💼 Valuable tier firm experience")
        
        if features.get('moot_court', 0) == 1:
            insights.append("⚖️ Moot court experience demonstrates advocacy skills")
        
        if features.get('legal_research', 0) == 1:
            insights.append("🔍 Legal research experience adds value")
        
        return insights[:4]  # Limit to 4 insights
    
    def _get_ml_recommendation(self, hire_prob: float, confidence: float) -> Dict:
        """Get ML recommendation based on hire probability and confidence"""
        
        if hire_prob >= 0.8 and confidence >= 0.7:
            return {
                'recommendation': 'STRONGLY_CONSIDER',
                'reason': 'High ML confidence and hire probability',
                'confidence_level': 'High'
            }
        elif hire_prob >= 0.6:
            return {
                'recommendation': 'CONSIDER',  
                'reason': 'Moderate to high ML hire probability',
                'confidence_level': 'Medium'
            }
        elif hire_prob >= 0.4:
            return {
                'recommendation': 'REVIEW_CAREFULLY',
                'reason': 'Mixed ML signals - human judgment recommended',
                'confidence_level': 'Medium'
            }
        else:
            return {
                'recommendation': 'LIKELY_REJECT',
                'reason': 'Low ML hire probability',
                'confidence_level': 'High'
            }
    
    def get_status(self) -> Dict:
        """Get current ML enhancement status"""
        return {
            'ml_libraries_available': ML_AVAILABLE,
            'ml_models_loaded': self.ml_enabled,
            'models_directory': self.ml_models_dir,
            'features_used': self.feature_names,
            'model_type': 'LightGBM Classifier' if self.ml_enabled else 'None',
            'status_check_time': datetime.now().isoformat()
        }

# Convenience function for easy integration
def create_ml_enhancer(models_dir='ml_models'):
    """Create ML enhancer for easy integration with existing ATS"""
    return SimpleMLEnhancer(models_dir)