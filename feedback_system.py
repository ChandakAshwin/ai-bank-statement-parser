"""
Feedback System for Bank Statement Parser AI Agent
Learns from user corrections to improve parsing accuracy
"""

import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import MEMORY_CONFIG, AI_AGENT_CONFIG

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """Feedback system for learning from user corrections"""
    
    def __init__(self):
        self.feedback_data_path = Path("memory") / "feedback_data.pkl"
        self.models_path = Path("models") / "feedback_models.pkl"
        self.learning_rate = MEMORY_CONFIG['learning_rate']
        
        # Load existing feedback data and models
        self.feedback_data = self._load_feedback_data()
        self.models = self._load_models()
        
        # Initialize learning components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training data
        self.training_data = {
            'texts': [],
            'labels': [],
            'features': []
        }
    
    def record_user_correction(self, original_result: Dict, user_correction: Dict, 
                              context: Dict) -> str:
        """
        Record a user correction for learning
        """
        try:
            correction_id = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            correction_record = {
                'id': correction_id,
                'timestamp': datetime.now().isoformat(),
                'original_result': original_result,
                'user_correction': user_correction,
                'context': context,
                'correction_type': self._classify_correction_type(original_result, user_correction),
                'severity': self._assess_correction_severity(original_result, user_correction)
            }
            
            # Store correction
            self.feedback_data['corrections'].append(correction_record)
            
            # Update training data
            self._update_training_data(correction_record)
            
            # Retrain models if enough new data
            if len(self.feedback_data['corrections']) % 10 == 0:  # Retrain every 10 corrections
                self._retrain_models()
            
            # Save updated data
            self._save_feedback_data()
            
            logger.info(f"Recorded user correction: {correction_id}")
            return correction_id
            
        except Exception as e:
            logger.error(f"Error recording user correction: {e}")
            return ""
    
    def get_correction_suggestions(self, parsing_result: Dict, context: Dict) -> List[Dict]:
        """
        Get suggestions for potential corrections based on learned patterns
        """
        try:
            suggestions = []
            
            # Analyze parsing result for potential issues
            potential_issues = self._identify_potential_issues(parsing_result)
            
            for issue in potential_issues:
                # Get similar corrections from history
                similar_corrections = self._find_similar_corrections(issue, context)
                
                if similar_corrections:
                    suggestion = self._create_correction_suggestion(issue, similar_corrections)
                    suggestions.append(suggestion)
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error getting correction suggestions: {e}")
            return []
    
    def apply_learned_corrections(self, parsing_result: Dict, confidence_threshold: float = 0.8) -> Dict:
        """
        Apply learned corrections to parsing result
        """
        try:
            corrected_result = parsing_result.copy()
            applied_corrections = []
            
            # Check each transaction for potential corrections
            for i, transaction in enumerate(corrected_result.get('transactions', [])):
                transaction_corrections = self._get_transaction_corrections(transaction, context={})
                
                for correction in transaction_corrections:
                    if correction.get('confidence', 0) >= confidence_threshold:
                        # Apply correction
                        corrected_result['transactions'][i] = self._apply_correction(
                            transaction, correction
                        )
                        applied_corrections.append({
                            'transaction_index': i,
                            'correction': correction,
                            'confidence': correction.get('confidence', 0)
                        })
            
            corrected_result['applied_corrections'] = applied_corrections
            corrected_result['correction_confidence'] = self._calculate_overall_confidence(applied_corrections)
            
            return corrected_result
            
        except Exception as e:
            logger.error(f"Error applying learned corrections: {e}")
            return parsing_result
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get learning progress and statistics
        """
        try:
            total_corrections = len(self.feedback_data['corrections'])
            
            if total_corrections == 0:
                return {
                    'total_corrections': 0,
                    'accuracy_improvement': 0,
                    'common_correction_types': [],
                    'learning_status': 'no_data'
                }
            
            # Calculate accuracy improvement
            recent_corrections = self.feedback_data['corrections'][-min(50, total_corrections):]
            accuracy_improvement = self._calculate_accuracy_improvement(recent_corrections)
            
            # Analyze common correction types
            correction_types = [c.get('correction_type') for c in self.feedback_data['corrections']]
            common_types = self._get_common_correction_types(correction_types)
            
            return {
                'total_corrections': total_corrections,
                'accuracy_improvement': accuracy_improvement,
                'common_correction_types': common_types,
                'learning_status': 'active' if total_corrections > 10 else 'learning',
                'model_performance': self._get_model_performance()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning progress: {e}")
            return {}
    
    def _classify_correction_type(self, original: Dict, correction: Dict) -> str:
        """Classify the type of correction made"""
        try:
            # Compare original and corrected results
            if 'transactions' in original and 'transactions' in correction:
                orig_txns = original['transactions']
                corr_txns = correction['transactions']
                
                if len(orig_txns) != len(corr_txns):
                    return 'transaction_count'
                
                # Check for field corrections
                field_corrections = []
                for i, (orig, corr) in enumerate(zip(orig_txns, corr_txns)):
                    for field in ['date', 'description', 'amount', 'type', 'category']:
                        if orig.get(field) != corr.get(field):
                            field_corrections.append(field)
                
                if field_corrections:
                    return f"field_correction_{max(set(field_corrections), key=field_corrections.count)}"
            
            # Check for balance corrections
            if original.get('closing_balance') != correction.get('closing_balance'):
                return 'balance_correction'
            
            return 'general_correction'
            
        except Exception as e:
            logger.error(f"Error classifying correction type: {e}")
            return 'unknown'
    
    def _assess_correction_severity(self, original: Dict, correction: Dict) -> str:
        """Assess the severity of a correction"""
        try:
            # Simple severity assessment based on correction type
            correction_type = self._classify_correction_type(original, correction)
            
            high_severity = ['transaction_count', 'balance_correction']
            medium_severity = ['field_correction_amount', 'field_correction_date']
            low_severity = ['field_correction_category', 'field_correction_description']
            
            if correction_type in high_severity:
                return 'high'
            elif correction_type in medium_severity:
                return 'medium'
            elif correction_type in low_severity:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.error(f"Error assessing correction severity: {e}")
            return 'medium'
    
    def _update_training_data(self, correction_record: Dict):
        """Update training data with new correction"""
        try:
            # Extract features from the correction
            features = self._extract_correction_features(correction_record)
            
            # Add to training data
            self.training_data['texts'].append(features['text'])
            self.training_data['labels'].append(correction_record['correction_type'])
            self.training_data['features'].append(features)
            
        except Exception as e:
            logger.error(f"Error updating training data: {e}")
    
    def _extract_correction_features(self, correction_record: Dict) -> Dict[str, Any]:
        """Extract features from correction record for training"""
        try:
            original = correction_record['original_result']
            correction = correction_record['user_correction']
            
            # Create text representation
            text_parts = []
            
            # Add transaction information
            if 'transactions' in original:
                for txn in original['transactions'][:3]:  # First 3 transactions
                    text_parts.append(f"Transaction: {txn.get('description', '')} {txn.get('amount', '')}")
            
            # Add correction context
            text_parts.append(f"Correction type: {correction_record['correction_type']}")
            text_parts.append(f"Severity: {correction_record['severity']}")
            
            return {
                'text': ' '.join(text_parts),
                'correction_type': correction_record['correction_type'],
                'severity': correction_record['severity'],
                'num_transactions': len(original.get('transactions', [])),
                'has_balance': 'closing_balance' in original
            }
            
        except Exception as e:
            logger.error(f"Error extracting correction features: {e}")
            return {'text': '', 'correction_type': 'unknown', 'severity': 'medium'}
    
    def _retrain_models(self):
        """Retrain learning models with updated data"""
        try:
            if len(self.training_data['texts']) < 5:
                logger.info("Not enough training data for retraining")
                return
            
            # Prepare training data
            X_text = self.vectorizer.fit_transform(self.training_data['texts'])
            y = self.training_data['labels']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=0.2, random_state=42
            )
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save models
            self.models = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'accuracy': accuracy,
                'last_trained': datetime.now().isoformat()
            }
            
            self._save_models()
            
            logger.info(f"Models retrained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _identify_potential_issues(self, parsing_result: Dict) -> List[Dict]:
        """Identify potential issues in parsing result"""
        issues = []
        
        try:
            transactions = parsing_result.get('transactions', [])
            
            # Check for missing required fields
            for i, txn in enumerate(transactions):
                missing_fields = []
                for field in ['date', 'description', 'amount']:
                    if not txn.get(field):
                        missing_fields.append(field)
                
                if missing_fields:
                    issues.append({
                        'type': 'missing_fields',
                        'transaction_index': i,
                        'fields': missing_fields,
                        'confidence': 0.9
                    })
            
            # Check for unusual amounts
            amounts = [txn.get('amount', 0) for txn in transactions if txn.get('amount')]
            if amounts:
                avg_amount = sum(amounts) / len(amounts)
                for i, txn in enumerate(transactions):
                    amount = txn.get('amount', 0)
                    if amount and abs(amount - avg_amount) > 3 * avg_amount:
                        issues.append({
                            'type': 'unusual_amount',
                            'transaction_index': i,
                            'amount': amount,
                            'average': avg_amount,
                            'confidence': 0.7
                        })
            
            # Check for duplicate transactions
            seen_descriptions = set()
            for i, txn in enumerate(transactions):
                desc = txn.get('description', '')
                if desc in seen_descriptions:
                    issues.append({
                        'type': 'duplicate_transaction',
                        'transaction_index': i,
                        'description': desc,
                        'confidence': 0.8
                    })
                seen_descriptions.add(desc)
            
        except Exception as e:
            logger.error(f"Error identifying potential issues: {e}")
        
        return issues
    
    def _find_similar_corrections(self, issue: Dict, context: Dict) -> List[Dict]:
        """Find similar corrections from history"""
        try:
            similar_corrections = []
            
            for correction in self.feedback_data['corrections']:
                if correction['correction_type'] == issue['type']:
                    # Calculate similarity score
                    similarity = self._calculate_similarity(issue, correction)
                    if similarity > 0.7:  # Similarity threshold
                        similar_corrections.append({
                            'correction': correction,
                            'similarity': similarity
                        })
            
            # Sort by similarity
            similar_corrections.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_corrections[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Error finding similar corrections: {e}")
            return []
    
    def _calculate_similarity(self, issue: Dict, correction: Dict) -> float:
        """Calculate similarity between current issue and historical correction"""
        try:
            # Simple similarity calculation
            # In a more sophisticated implementation, you'd use embeddings
            
            issue_type = issue.get('type', '')
            correction_type = correction.get('correction_type', '')
            
            if issue_type == correction_type:
                return 0.8
            
            return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _create_correction_suggestion(self, issue: Dict, similar_corrections: List[Dict]) -> Dict:
        """Create a correction suggestion based on similar corrections"""
        try:
            if not similar_corrections:
                return {}
            
            # Use the most similar correction as base
            best_correction = similar_corrections[0]
            
            suggestion = {
                'issue_type': issue['type'],
                'suggested_correction': best_correction['correction']['user_correction'],
                'confidence': best_correction['similarity'] * issue.get('confidence', 0.5),
                'reason': f"Based on {len(similar_corrections)} similar corrections",
                'similarity_score': best_correction['similarity']
            }
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error creating correction suggestion: {e}")
            return {}
    
    def _get_transaction_corrections(self, transaction: Dict, context: Dict) -> List[Dict]:
        """Get potential corrections for a specific transaction"""
        try:
            corrections = []
            
            # Check for common field corrections
            for field in ['date', 'description', 'amount', 'type', 'category']:
                if field in transaction:
                    field_corrections = self._get_field_corrections(field, transaction[field])
                    corrections.extend(field_corrections)
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error getting transaction corrections: {e}")
            return []
    
    def _get_field_corrections(self, field: str, value: Any) -> List[Dict]:
        """Get potential corrections for a specific field"""
        try:
            corrections = []
            
            # Look for similar corrections in history
            for correction in self.feedback_data['corrections']:
                if correction['correction_type'] == f'field_correction_{field}':
                    # Check if the original value is similar
                    if self._is_similar_value(value, correction['original_result']):
                        corrections.append({
                            'field': field,
                            'original_value': value,
                            'suggested_value': correction['user_correction'],
                            'confidence': 0.7
                        })
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error getting field corrections: {e}")
            return []
    
    def _is_similar_value(self, value1: Any, value2: Any) -> bool:
        """Check if two values are similar"""
        try:
            # Simple similarity check
            return str(value1).lower() == str(value2).lower()
        except Exception:
            return False
    
    def _apply_correction(self, transaction: Dict, correction: Dict) -> Dict:
        """Apply a correction to a transaction"""
        try:
            corrected_transaction = transaction.copy()
            
            if 'field' in correction and 'suggested_value' in correction:
                field = correction['field']
                suggested_value = correction['suggested_value']
                corrected_transaction[field] = suggested_value
            
            return corrected_transaction
            
        except Exception as e:
            logger.error(f"Error applying correction: {e}")
            return transaction
    
    def _calculate_overall_confidence(self, applied_corrections: List[Dict]) -> float:
        """Calculate overall confidence of applied corrections"""
        try:
            if not applied_corrections:
                return 1.0
            
            confidences = [correction.get('confidence', 0) for correction in applied_corrections]
            return sum(confidences) / len(confidences)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _calculate_accuracy_improvement(self, recent_corrections: List[Dict]) -> float:
        """Calculate accuracy improvement over time"""
        try:
            if len(recent_corrections) < 2:
                return 0.0
            
            # Simple accuracy improvement calculation
            # In a real implementation, you'd track actual accuracy metrics
            
            early_corrections = recent_corrections[:len(recent_corrections)//2]
            late_corrections = recent_corrections[len(recent_corrections)//2:]
            
            early_severity = sum(1 for c in early_corrections if c.get('severity') == 'high')
            late_severity = sum(1 for c in late_corrections if c.get('severity') == 'high')
            
            if len(early_corrections) > 0:
                early_rate = early_severity / len(early_corrections)
                late_rate = late_severity / len(late_corrections)
                return early_rate - late_rate
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating accuracy improvement: {e}")
            return 0.0
    
    def _get_common_correction_types(self, correction_types: List[str]) -> List[Tuple[str, int]]:
        """Get most common correction types"""
        try:
            from collections import Counter
            counter = Counter(correction_types)
            return counter.most_common(5)
        except Exception as e:
            logger.error(f"Error getting common correction types: {e}")
            return []
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            if 'accuracy' in self.models:
                return {
                    'accuracy': self.models['accuracy'],
                    'last_trained': self.models.get('last_trained', 'unknown'),
                    'training_samples': len(self.training_data['texts'])
                }
            return {'status': 'not_trained'}
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'status': 'error'}
    
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load feedback data from disk"""
        try:
            if self.feedback_data_path.exists():
                with open(self.feedback_data_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
        
        return {'corrections': []}
    
    def _save_feedback_data(self):
        """Save feedback data to disk"""
        try:
            with open(self.feedback_data_path, 'wb') as f:
                pickle.dump(self.feedback_data, f)
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
    
    def _load_models(self) -> Dict[str, Any]:
        """Load trained models from disk"""
        try:
            if self.models_path.exists():
                with open(self.models_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        
        return {}
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(self.models_path, 'wb') as f:
                pickle.dump(self.models, f)
        except Exception as e:
            logger.error(f"Error saving models: {e}") 