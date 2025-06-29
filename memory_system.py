"""
Memory System for Bank Statement Parser AI Agent
Tracks past documents and learns from them to improve future extractions
"""

import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from config import MEMORY_CONFIG

logger = logging.getLogger(__name__)

class MemorySystem:
    """Memory system for storing and retrieving past parsing experiences"""
    
    def __init__(self):
        self.vector_db_path = MEMORY_CONFIG['vector_db_path']
        self.embedding_model = SentenceTransformer(MEMORY_CONFIG['embedding_model'])
        self.max_memory_size = MEMORY_CONFIG['max_memory_size']
        self.similarity_threshold = MEMORY_CONFIG['similarity_threshold']
        self.learning_rate = MEMORY_CONFIG['learning_rate']
        self.model_save_path = MEMORY_CONFIG['model_save_path']
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections
        self.document_collection = self.client.get_or_create_collection("documents")
        self.parsing_collection = self.client.get_or_create_collection("parsing_patterns")
        self.error_collection = self.client.get_or_create_collection("parsing_errors")
        
        # Load learned models
        self.learned_models = self._load_learned_models()
    
    def store_document_experience(self, file_path: str, analysis_result: Dict, 
                                parsing_result: Dict, validation_result: Dict) -> str:
        """
        Store a document parsing experience in memory
        """
        try:
            # Create document embedding
            doc_content = self._create_document_content(analysis_result, parsing_result)
            embedding = self.embedding_model.encode(doc_content)
            
            # Store in vector database
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(file_path).stem}"
            
            self.document_collection.add(
                embeddings=[embedding.tolist()],
                documents=[doc_content],
                metadatas=[{
                    'file_path': file_path,
                    'bank_identification': analysis_result.get('bank_identification'),
                    'document_format': analysis_result.get('document_format'),
                    'confidence': analysis_result.get('confidence', 0),
                    'validation_score': validation_result.get('validation_score', 0),
                    'timestamp': datetime.now().isoformat(),
                    'num_transactions': len(parsing_result.get('transactions', [])),
                    'closing_balance': parsing_result.get('closing_balance')
                }],
                ids=[doc_id]
            )
            
            # Store parsing patterns
            self._store_parsing_patterns(doc_id, analysis_result, parsing_result)
            
            # Store errors if any
            if validation_result.get('issues_found'):
                self._store_parsing_errors(doc_id, validation_result)
            
            # Update learned models
            self._update_learned_models(analysis_result, parsing_result, validation_result)
            
            logger.info(f"Stored document experience: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document experience: {e}")
            return ""
    
    def find_similar_documents(self, analysis_result: Dict, limit: int = 5) -> List[Dict]:
        """
        Find similar documents from memory based on current analysis
        """
        try:
            # Create query embedding
            query_content = self._create_document_content(analysis_result, {})
            query_embedding = self.embedding_model.encode(query_content)
            
            # Search similar documents
            results = self.document_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where={"confidence": {"$gte": self.similarity_threshold}}
            )
            
            similar_docs = []
            for i in range(len(results['ids'][0])):
                similar_docs.append({
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': results['distances'][0][i] if results['distances'] else 0,
                    'content': results['documents'][0][i]
                })
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def get_parsing_recommendations(self, analysis_result: Dict) -> Dict[str, Any]:
        """
        Get parsing recommendations based on similar past experiences
        """
        try:
            similar_docs = self.find_similar_documents(analysis_result)
            
            if not similar_docs:
                return self._get_default_recommendations()
            
            # Analyze patterns from similar documents
            recommendations = {
                'suggested_strategy': self._analyze_strategy_patterns(similar_docs),
                'expected_challenges': self._analyze_challenge_patterns(similar_docs),
                'success_rate': self._calculate_success_rate(similar_docs),
                'confidence_boost': self._calculate_confidence_boost(similar_docs),
                'special_handlers': self._get_special_handlers(similar_docs)
            }
            
            logger.info(f"Generated recommendations with {len(similar_docs)} similar documents")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting parsing recommendations: {e}")
            return self._get_default_recommendations()
    
    def learn_from_feedback(self, doc_id: str, user_feedback: Dict) -> bool:
        """
        Learn from user feedback to improve future parsing
        """
        try:
            # Store feedback
            feedback_content = json.dumps(user_feedback)
            feedback_embedding = self.embedding_model.encode(feedback_content)
            
            self.error_collection.add(
                embeddings=[feedback_embedding.tolist()],
                documents=[feedback_content],
                metadatas=[{
                    'doc_id': doc_id,
                    'feedback_type': user_feedback.get('type', 'general'),
                    'timestamp': datetime.now().isoformat(),
                    'severity': user_feedback.get('severity', 'medium')
                }],
                ids=[f"feedback_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
            )
            
            # Update learned models with feedback
            self._update_models_with_feedback(user_feedback)
            
            logger.info(f"Learned from feedback for document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
            return False
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from learned patterns and experiences
        """
        try:
            insights = {
                'total_documents_processed': self.document_collection.count(),
                'success_rate': self._calculate_overall_success_rate(),
                'common_challenges': self._get_common_challenges(),
                'bank_specific_patterns': self._get_bank_patterns(),
                'improvement_areas': self._get_improvement_areas(),
                'recent_performance': self._get_recent_performance()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}
    
    def _create_document_content(self, analysis_result: Dict, parsing_result: Dict) -> str:
        """Create a text representation of the document for embedding"""
        content_parts = []
        
        # Add analysis information
        content_parts.append(f"Bank: {analysis_result.get('bank_identification', 'Unknown')}")
        content_parts.append(f"Format: {analysis_result.get('document_format', 'Unknown')}")
        content_parts.append(f"Confidence: {analysis_result.get('confidence', 0)}")
        
        # Add table structure
        table_structure = analysis_result.get('table_structure', {})
        content_parts.append(f"Tables: {table_structure.get('num_tables', 0)}")
        content_parts.append(f"Column patterns: {', '.join(table_structure.get('column_patterns', []))}")
        
        # Add parsing results if available
        if parsing_result:
            content_parts.append(f"Transactions: {len(parsing_result.get('transactions', []))}")
            content_parts.append(f"Closing balance: {parsing_result.get('closing_balance', 'Unknown')}")
        
        return " | ".join(content_parts)
    
    def _store_parsing_patterns(self, doc_id: str, analysis_result: Dict, parsing_result: Dict):
        """Store parsing patterns for future reference"""
        try:
            patterns = {
                'bank_identification': analysis_result.get('bank_identification'),
                'document_format': analysis_result.get('document_format'),
                'table_structure': analysis_result.get('table_structure'),
                'parsing_strategy': analysis_result.get('parsing_strategy'),
                'transaction_count': len(parsing_result.get('transactions', [])),
                'successful_parsing': True
            }
            
            pattern_content = json.dumps(patterns)
            pattern_embedding = self.embedding_model.encode(pattern_content)
            
            self.parsing_collection.add(
                embeddings=[pattern_embedding.tolist()],
                documents=[pattern_content],
                metadatas=[{
                    'doc_id': doc_id,
                    'pattern_type': 'successful_parsing',
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[f"pattern_{doc_id}"]
            )
            
        except Exception as e:
            logger.error(f"Error storing parsing patterns: {e}")
    
    def _store_parsing_errors(self, doc_id: str, validation_result: Dict):
        """Store parsing errors for learning"""
        try:
            errors = {
                'issues_found': validation_result.get('issues_found', []),
                'missing_data': validation_result.get('missing_data', []),
                'recommendations': validation_result.get('recommendations', [])
            }
            
            error_content = json.dumps(errors)
            error_embedding = self.embedding_model.encode(error_content)
            
            self.error_collection.add(
                embeddings=[error_embedding.tolist()],
                documents=[error_content],
                metadatas=[{
                    'doc_id': doc_id,
                    'error_type': 'parsing_validation',
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[f"error_{doc_id}"]
            )
            
        except Exception as e:
            logger.error(f"Error storing parsing errors: {e}")
    
    def _update_learned_models(self, analysis_result: Dict, parsing_result: Dict, validation_result: Dict):
        """Update learned models with new experience"""
        try:
            # Update bank-specific patterns
            bank_id = analysis_result.get('bank_identification')
            if bank_id:
                if bank_id not in self.learned_models['bank_patterns']:
                    self.learned_models['bank_patterns'][bank_id] = {
                        'count': 0,
                        'success_rate': 0,
                        'common_issues': [],
                        'best_strategies': []
                    }
                
                bank_pattern = self.learned_models['bank_patterns'][bank_id]
                bank_pattern['count'] += 1
                
                # Update success rate
                validation_score = validation_result.get('validation_score', 0)
                bank_pattern['success_rate'] = (
                    (bank_pattern['success_rate'] * (bank_pattern['count'] - 1) + validation_score) 
                    / bank_pattern['count']
                )
                
                # Update best strategies
                strategy = analysis_result.get('parsing_strategy', {})
                if strategy.get('primary_method') not in bank_pattern['best_strategies']:
                    bank_pattern['best_strategies'].append(strategy.get('primary_method'))
            
            # Save updated models
            self._save_learned_models()
            
        except Exception as e:
            logger.error(f"Error updating learned models: {e}")
    
    def _analyze_strategy_patterns(self, similar_docs: List[Dict]) -> str:
        """Analyze successful strategy patterns from similar documents"""
        strategies = []
        for doc in similar_docs:
            # Extract strategy from metadata or content
            if 'parsing_strategy' in doc.get('metadata', {}):
                strategies.append(doc['metadata']['parsing_strategy'])
        
        if strategies:
            # Return most common strategy
            return max(set(strategies), key=strategies.count)
        return 'standard'
    
    def _analyze_challenge_patterns(self, similar_docs: List[Dict]) -> List[str]:
        """Analyze common challenges from similar documents"""
        challenges = []
        for doc in similar_docs:
            # Extract challenges from content
            if 'challenges' in doc.get('metadata', {}):
                challenges.extend(doc['metadata']['challenges'])
        
        return list(set(challenges))  # Remove duplicates
    
    def _calculate_success_rate(self, similar_docs: List[Dict]) -> float:
        """Calculate success rate from similar documents"""
        if not similar_docs:
            return 0.5
        
        success_scores = []
        for doc in similar_docs:
            validation_score = doc.get('metadata', {}).get('validation_score', 0.5)
            success_scores.append(validation_score)
        
        return sum(success_scores) / len(success_scores)
    
    def _calculate_confidence_boost(self, similar_docs: List[Dict]) -> float:
        """Calculate confidence boost based on similar experiences"""
        if not similar_docs:
            return 0.0
        
        # Average similarity score
        similarities = [doc.get('similarity', 0) for doc in similar_docs]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost confidence based on similarity and success rate
        success_rate = self._calculate_success_rate(similar_docs)
        return min(0.3, avg_similarity * success_rate)
    
    def _get_special_handlers(self, similar_docs: List[Dict]) -> List[str]:
        """Get special handlers based on similar documents"""
        handlers = []
        for doc in similar_docs:
            bank_id = doc.get('metadata', {}).get('bank_identification')
            if bank_id:
                handlers.append(f'bank_{bank_id.lower()}_handler')
        
        return list(set(handlers))  # Remove duplicates
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Get default recommendations when no similar documents found"""
        return {
            'suggested_strategy': 'standard',
            'expected_challenges': ['unknown_format'],
            'success_rate': 0.5,
            'confidence_boost': 0.0,
            'special_handlers': []
        }
    
    def _load_learned_models(self) -> Dict[str, Any]:
        """Load learned models from disk"""
        try:
            if Path(self.model_save_path).exists():
                with open(self.model_save_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading learned models: {e}")
        
        return {
            'bank_patterns': {},
            'error_patterns': {},
            'success_patterns': {}
        }
    
    def _save_learned_models(self):
        """Save learned models to disk"""
        try:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(self.learned_models, f)
        except Exception as e:
            logger.error(f"Error saving learned models: {e}")
    
    def _update_models_with_feedback(self, user_feedback: Dict):
        """Update models based on user feedback"""
        # Implementation for feedback-based learning
        pass
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate from all stored documents"""
        try:
            all_docs = self.document_collection.get()
            if not all_docs['metadatas']:
                return 0.5
            
            scores = [doc.get('validation_score', 0.5) for doc in all_docs['metadatas']]
            return sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"Error calculating overall success rate: {e}")
            return 0.5
    
    def _get_common_challenges(self) -> List[str]:
        """Get common challenges from stored documents"""
        # Implementation for extracting common challenges
        return []
    
    def _get_bank_patterns(self) -> Dict[str, Any]:
        """Get bank-specific patterns"""
        return self.learned_models.get('bank_patterns', {})
    
    def _get_improvement_areas(self) -> List[str]:
        """Get areas for improvement based on stored data"""
        # Implementation for identifying improvement areas
        return []
    
    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics"""
        # Implementation for recent performance analysis
        return {} 