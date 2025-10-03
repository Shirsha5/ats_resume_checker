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

# Try to import BERT components for text analysis
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
    print("✅ BERT libraries available")
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT libraries not available. Run: pip install transformers torch")

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
        # BERT components for text analysis
        self.bert_enabled = False
        self.bert_tokenizer = None
        self.bert_model = None
        self.experience_embeddings = None
        self.bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if ML_AVAILABLE:
            self._try_load_models()
        
        # Try to load BERT if available
        if BERT_AVAILABLE:
            self._load_bert_models()

    
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
    
    def _load_bert_models(self):
        """Load BERT model for text analysis"""
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            
            # Create experience embeddings
            self._create_experience_embeddings()
            self.bert_enabled = True
            self.logger.info("✅ BERT text analysis enabled")
        except Exception as e:
            self.logger.warning(f"⚠️ BERT loading failed: {e}")
            self.bert_enabled = False

    def _create_experience_embeddings(self):
        """Create embeddings for ideal M&A experiences"""
        ideal_experiences = [
            "mergers acquisitions corporate law",
            "tier firm internship corporate transactions",
            "company law contract law legal research",
            "moot court corporate law competition",
            "legal research corporate governance",
            "due diligence private equity",
            "corporate compliance securities law",
        ]
        
        embeddings = []
        for exp in ideal_experiences:
            embedding = self._get_text_embedding(exp)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            self.experience_embeddings = np.array(embeddings)

    def _get_text_embedding(self, text):
        """Get BERT embedding for text"""
        if not self.bert_model or not text:
            return None
        try:
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, 
                                       padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        except:
            return None

    def _analyze_resume_text(self, resume_text):
        """Analyze resume text with BERT"""
        if not self.bert_enabled or not resume_text:
            return {'text_quality_score': 0.5, 'bert_insights': []}
        
        try:
            resume_embedding = self._get_text_embedding(resume_text)
            if resume_embedding is None:
                return {'text_quality_score': 0.5, 'bert_insights': []}
            
            similarities = cosine_similarity([resume_embedding], self.experience_embeddings)[0]
            max_similarity = np.max(similarities)
            avg_similarity = np.mean(similarities)
            
            insights = []
            if max_similarity > 0.7:
                insights.append("🌟 Highly relevant M&A experience detected")
            elif max_similarity > 0.5:
                insights.append("✅ Good corporate law background")
            else:
                insights.append("📋 Limited M&A-specific experience")
            
            return {
                'text_quality_score': float(avg_similarity),
                'bert_insights': insights
            }
        except:
            return {'text_quality_score': 0.5, 'bert_insights': []}

    
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
                    lgb_hire_probability = probabilities[1] if len(probabilities) > 1 else prediction
                else:
                    confidence = 0.8  # High confidence for simple model
                    lgb_hire_probability = prediction
                
                # Get BERT text analysis
                bert_analysis = self._analyze_resume_text(raw_text)
                bert_probability = bert_analysis['text_quality_score']
                
                # Combine LightGBM + BERT (60% LGB + 40% BERT)
                if self.bert_enabled:
                    hire_probability = (lgb_hire_probability * 0.6) + (bert_probability * 0.4)
                    method_used = "LightGBM + BERT Hybrid"
                else:
                    hire_probability = lgb_hire_probability
                    method_used = "LightGBM Only"
                
                # Generate LightGBM insights
                lgb_insights = self._generate_insights(all_features, prediction, hire_probability)
                
                # Get BERT insights if available
                bert_insights = bert_analysis.get('bert_insights', []) if self.bert_enabled else []
                
                # Combine all insights
                all_insights = lgb_insights + bert_insights

                # Update ML enhancement
                enhanced_result['ml_enhancement'].update({
                    'ml_prediction': int(prediction),
                    'ml_hire_probability': float(hire_probability),
                    'ml_confidence': float(confidence),
                    'feature_values': dict(zip(self.feature_names, feature_vector)),
                    'ml_insights': all_insights[:8]  # Show top 8 insights total
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

        return insights[:8]  # Limit to 8 insights

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