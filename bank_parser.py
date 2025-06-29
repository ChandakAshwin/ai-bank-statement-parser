"""
BankStatementParser: Main orchestrator for parsing bank statements
Enhanced with AI Agent capabilities
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from utils import (
    validate_file_path,
    is_supported_file,
    get_file_extension,
    save_output
)
from pdf_processor import PDFProcessor
from excel_processor import ExcelProcessor
from data_extractor import DataExtractor
from ai_reasoning import AIReasoningEngine
from memory_system import MemorySystem
from autonomous_system import AutonomousSystem
from feedback_system import FeedbackSystem
from config import OUTPUT_CONFIG, AI_AGENT_CONFIG
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class BankStatementParser:
    """Enhanced AI agent for parsing bank statements with learning capabilities"""
    
    def __init__(self, enable_ai_agent: bool = True):
        # Core parsing components
        self.pdf_processor = PDFProcessor()
        self.excel_processor = ExcelProcessor()
        self.data_extractor = DataExtractor()
        
        # AI Agent components
        self.enable_ai_agent = enable_ai_agent
        if enable_ai_agent:
            self.reasoning_engine = AIReasoningEngine()
            self.memory_system = MemorySystem()
            self.autonomous_system = AutonomousSystem()
            self.feedback_system = FeedbackSystem()
            
            # Set parser in autonomous system to avoid circular import
            self.autonomous_system.set_parser(self)
            
            # Start autonomous mode if configured
            if AI_AGENT_CONFIG['autonomous_mode']:
                self.autonomous_system.start_autonomous_mode()
        
        # Performance tracking
        self.parsing_history = []
        self.learning_metrics = {}

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a bank statement file and return structured transactions"""
        result = self.parse_file_with_balance(file_path)
        return result['transactions']

    def parse_file_with_balance(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a bank statement file with AI agent enhancements
        Returns: {'transactions': List[Dict], 'closing_balance': Optional[float], 'ai_insights': Dict}
        """
        if not validate_file_path(file_path):
            logger.error(f"File not found: {file_path}")
            return {'transactions': [], 'closing_balance': None, 'ai_insights': {}}
        
        if not is_supported_file(file_path):
            logger.error(f"Unsupported file type: {file_path}")
            return {'transactions': [], 'closing_balance': None, 'ai_insights': {}}
        
        # Initialize result structure
        result = {
            'transactions': [],
            'closing_balance': None,
            'ai_insights': {},
            'parsing_strategy': {},
            'confidence_score': 0.0,
            'learning_applied': False
        }
        
        try:
            # Step 1: Extract raw data
            ext = get_file_extension(file_path)
            text_content = []
            all_tables = []
            
            if ext == '.pdf':
                all_tables = self.pdf_processor.extract_tables_from_pdf(file_path)
                text_content = self.pdf_processor.extract_text(file_path)
            elif ext in ['.xls', '.xlsx', '.xlsm']:
                excel_tables = self.excel_processor.extract_tables_from_excel(file_path)
                for table in excel_tables:
                    if table is not None and not table.empty:
                        text_content.append(' '.join([str(cell) for cell in table.values.flatten() if pd.notna(cell)]))
                        table_list = [table.columns.tolist()] + table.values.tolist()
                        all_tables.append(table_list)
            
            # Step 2: AI Reasoning (if enabled)
            if self.enable_ai_agent:
                result['ai_insights'] = self._apply_ai_reasoning(text_content, all_tables, file_path)
                result['parsing_strategy'] = result['ai_insights'].get('parsing_strategy', {})
                result['confidence_score'] = result['ai_insights'].get('confidence', 0.5)
            
            # Step 3: Apply learned corrections
            if self.enable_ai_agent and AI_AGENT_CONFIG['learning_enabled']:
                result = self._apply_learned_corrections(result)
            
            # Step 4: Extract transactions
            transactions = []
            for table in all_tables:
                # Use table extraction method for both PDF and Excel
                txns = self.data_extractor.extract_transactions_from_table(table)
                transactions.extend(txns)
            
            result['transactions'] = transactions
            
            # Step 5: Extract closing balance
            result['closing_balance'] = self.data_extractor.extract_closing_balance(text_content, all_tables)
            
            # Step 6: AI Validation and Learning
            if self.enable_ai_agent:
                result = self._apply_ai_validation_and_learning(result, file_path)
            
            # Step 7: Update performance tracking
            self._update_parsing_history(result, file_path)
            
            logger.info(f"Successfully parsed {len(transactions)} transactions with confidence {result['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            result['transactions'] = []
            result['closing_balance'] = None
        
        return result

    def _apply_ai_reasoning(self, text_content: List[str], tables: List[List[List[str]]], file_path: str) -> Dict[str, Any]:
        """Apply AI reasoning to analyze document and determine parsing strategy"""
        try:
            # Analyze document structure
            analysis_result = self.reasoning_engine.analyze_document_structure(text_content, tables)
            
            # Get parsing recommendations from memory
            memory_recommendations = self.memory_system.get_parsing_recommendations(analysis_result)
            
            # Determine parsing strategy
            parsing_strategy = self.reasoning_engine.determine_parsing_strategy(analysis_result)
            
            # Combine insights
            ai_insights = {
                'document_analysis': analysis_result,
                'memory_recommendations': memory_recommendations,
                'parsing_strategy': parsing_strategy,
                'confidence': analysis_result.get('confidence', 0.5),
                'bank_identification': analysis_result.get('bank_identification'),
                'expected_challenges': analysis_result.get('challenges', [])
            }
            
            # Apply confidence boost from memory
            if memory_recommendations.get('confidence_boost', 0) > 0:
                ai_insights['confidence'] = min(1.0, ai_insights['confidence'] + memory_recommendations['confidence_boost'])
            
            return ai_insights
            
        except Exception as e:
            logger.error(f"Error applying AI reasoning: {e}")
            return {
                'document_analysis': {},
                'memory_recommendations': {},
                'parsing_strategy': {'method': 'standard'},
                'confidence': 0.3,
                'bank_identification': None,
                'expected_challenges': ['ai_reasoning_failed']
            }

    def _apply_learned_corrections(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned corrections from feedback system"""
        try:
            # Get correction suggestions
            suggestions = self.feedback_system.get_correction_suggestions(result, {})
            
            if suggestions:
                # Apply corrections with high confidence
                corrected_result = self.feedback_system.apply_learned_corrections(
                    result, confidence_threshold=0.8
                )
                
                if corrected_result.get('applied_corrections'):
                    result.update(corrected_result)
                    result['learning_applied'] = True
                    logger.info(f"Applied {len(corrected_result['applied_corrections'])} learned corrections")
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying learned corrections: {e}")
            return result

    def _apply_ai_validation_and_learning(self, result: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Apply AI validation and store experience for learning"""
        try:
            # Validate parsing results
            validation_result = self.reasoning_engine.validate_parsing_result(
                result['transactions'], result['ai_insights'].get('document_analysis', {})
            )
            
            result['validation_result'] = validation_result
            
            # Store experience in memory for future learning
            if AI_AGENT_CONFIG['memory_enabled']:
                self.memory_system.store_document_experience(
                    file_path, 
                    result['ai_insights'].get('document_analysis', {}),
                    result,
                    validation_result
                )
            
            # Update confidence based on validation
            validation_score = validation_result.get('validation_score', 0.5)
            result['confidence_score'] = (result['confidence_score'] + validation_score) / 2
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying AI validation and learning: {e}")
            return result

    def record_user_feedback(self, file_path: str, original_result: Dict, 
                           user_corrections: Dict, feedback_type: str = 'correction') -> bool:
        """
        Record user feedback for learning
        """
        try:
            if not self.enable_ai_agent or not AI_AGENT_CONFIG['feedback_loop_enabled']:
                return False
            
            # Record correction in feedback system
            correction_id = self.feedback_system.record_user_correction(
                original_result, user_corrections, {'file_path': file_path, 'type': feedback_type}
            )
            
            # Learn from feedback in memory system
            self.memory_system.learn_from_feedback(correction_id, {
                'type': feedback_type,
                'severity': 'medium',
                'corrections': user_corrections
            })
            
            logger.info(f"Recorded user feedback: {correction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            return False

    def get_ai_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive AI insights and learning progress
        """
        try:
            if not self.enable_ai_agent:
                return {'ai_agent_enabled': False}
            
            insights = {
                'ai_agent_enabled': True,
                'autonomous_status': self.autonomous_system.get_status(),
                'learning_progress': self.feedback_system.get_learning_progress(),
                'memory_insights': self.memory_system.get_learning_insights(),
                'parsing_performance': self._get_parsing_performance(),
                'recommendations': self._get_ai_recommendations()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {'ai_agent_enabled': False, 'error': str(e)}

    def start_autonomous_mode(self):
        """Start autonomous monitoring and processing"""
        if self.enable_ai_agent:
            self.autonomous_system.start_autonomous_mode()
            logger.info("Autonomous mode started")

    def stop_autonomous_mode(self):
        """Stop autonomous monitoring"""
        if self.enable_ai_agent:
            self.autonomous_system.stop_autonomous_mode()
            logger.info("Autonomous mode stopped")

    def _update_parsing_history(self, result: Dict[str, Any], file_path: str):
        """Update parsing history for performance tracking"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'transaction_count': len(result.get('transactions', [])),
                'confidence_score': result.get('confidence_score', 0),
                'learning_applied': result.get('learning_applied', False),
                'bank_identification': result.get('ai_insights', {}).get('bank_identification')
            }
            
            self.parsing_history.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.parsing_history) > 100:
                self.parsing_history = self.parsing_history[-100:]
                
        except Exception as e:
            logger.error(f"Error updating parsing history: {e}")

    def _get_parsing_performance(self) -> Dict[str, Any]:
        """Get parsing performance metrics"""
        try:
            if not self.parsing_history:
                return {'total_files': 0, 'average_confidence': 0}
            
            total_files = len(self.parsing_history)
            avg_confidence = sum(entry['confidence_score'] for entry in self.parsing_history) / total_files
            learning_applied_count = sum(1 for entry in self.parsing_history if entry['learning_applied'])
            
            return {
                'total_files': total_files,
                'average_confidence': avg_confidence,
                'learning_applied_rate': learning_applied_count / total_files if total_files > 0 else 0,
                'recent_performance': self.parsing_history[-10:] if len(self.parsing_history) >= 10 else self.parsing_history
            }
            
        except Exception as e:
            logger.error(f"Error getting parsing performance: {e}")
            return {'total_files': 0, 'average_confidence': 0}

    def _get_ai_recommendations(self) -> List[str]:
        """Get AI recommendations for improvement"""
        try:
            recommendations = []
            
            # Get recommendations from feedback system
            learning_progress = self.feedback_system.get_learning_progress()
            if learning_progress.get('total_corrections', 0) < 5:
                recommendations.append("More user feedback needed for better learning")
            
            # Get recommendations from memory system
            memory_insights = self.memory_system.get_learning_insights()
            if memory_insights.get('success_rate', 0) < 0.7:
                recommendations.append("Consider reviewing parsing strategies for better accuracy")
            
            # Performance-based recommendations
            performance = self._get_parsing_performance()
            if performance.get('average_confidence', 0) < 0.6:
                recommendations.append("Low confidence scores detected - consider manual review")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return ["Enable detailed logging for better analysis"]

    def save_parsed_output(self, transactions: List[Dict[str, Any]], output_path: Optional[str] = None, closing_balance: Optional[float] = None) -> str:
        """Save parsed transactions to file (json/csv/excel) with optional closing balance"""
        if not transactions and closing_balance is None:
            logger.warning("No transactions or closing balance to save.")
            return ""
        
        if output_path is None:
            output_path = str(Path.cwd() / f"parsed_output.{OUTPUT_CONFIG['output_format']}")
        
        # If we have closing balance, include it in the output
        if closing_balance is not None:
            # Add closing balance as a summary entry
            summary_data = {
                'transactions': transactions,
                'summary': {
                    'total_transactions': len(transactions),
                    'closing_balance': closing_balance
                }
            }
            # For structured output with summary, save as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
        else:
            save_output(transactions, output_path, format=OUTPUT_CONFIG['output_format'])
        
        return output_path 