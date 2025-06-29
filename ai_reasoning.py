"""
AI Reasoning Layer for Bank Statement Parser
Uses LLM to analyze documents and decide parsing strategies
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from config import AI_AGENT_CONFIG

logger = logging.getLogger(__name__)

class AIReasoningEngine:
    """AI reasoning engine for document analysis and parsing strategy selection"""
    
    def __init__(self):
        self.client = OpenAI(api_key=AI_AGENT_CONFIG['openai_api_key'])
        self.model = AI_AGENT_CONFIG['openai_model']
        self.temperature = AI_AGENT_CONFIG['temperature']
        self.max_tokens = AI_AGENT_CONFIG['max_tokens']
        
    def analyze_document_structure(self, text_content: List[str], tables: List[List[List[str]]]) -> Dict[str, Any]:
        """
        Analyze document structure using AI to determine parsing strategy
        """
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(text_content, tables)
            
            prompt = f"""
            You are an expert bank statement analyzer. Analyze the following document structure and determine the best parsing strategy.
            
            Document Context:
            {context}
            
            Please analyze this document and provide:
            1. Bank identification (if possible)
            2. Document format type (PDF native, PDF scanned, Excel, etc.)
            3. Table structure analysis
            4. Recommended parsing strategy
            5. Confidence level (0-1)
            6. Potential challenges
            
            Respond in JSON format:
            {{
                "bank_identification": "string or null",
                "document_format": "string",
                "table_structure": {{
                    "num_tables": int,
                    "table_types": ["list of table types"],
                    "column_patterns": ["list of identified column patterns"]
                }},
                "parsing_strategy": {{
                    "primary_method": "string",
                    "fallback_method": "string",
                    "special_handling": ["list of special cases"]
                }},
                "confidence": float,
                "challenges": ["list of potential issues"],
                "recommendations": ["list of recommendations"]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"AI analysis completed with confidence: {result.get('confidence', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._get_fallback_analysis(text_content, tables)
    
    def determine_parsing_strategy(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best parsing strategy based on AI analysis
        """
        strategy = {
            'method': 'standard',
            'confidence': analysis_result.get('confidence', 0.5),
            'special_handlers': [],
            'preprocessing_steps': [],
            'postprocessing_steps': []
        }
        
        # Determine primary method
        if analysis_result.get('document_format') == 'PDF scanned':
            strategy['method'] = 'ocr_enhanced'
            strategy['preprocessing_steps'].append('image_enhancement')
        elif analysis_result.get('document_format') == 'PDF native':
            strategy['method'] = 'table_extraction'
        elif analysis_result.get('document_format') == 'Excel':
            strategy['method'] = 'excel_processing'
        
        # Add special handlers based on bank identification
        bank_id = analysis_result.get('bank_identification')
        if bank_id:
            strategy['special_handlers'].append(f'bank_{bank_id.lower()}_handler')
        
        # Add special handling for identified challenges
        challenges = analysis_result.get('challenges', [])
        if 'multi_currency' in challenges:
            strategy['special_handlers'].append('currency_handler')
        if 'complex_layout' in challenges:
            strategy['preprocessing_steps'].append('layout_analysis')
        
        return strategy
    
    def validate_parsing_result(self, transactions: List[Dict], original_analysis: Dict) -> Dict[str, Any]:
        """
        Validate parsing results using AI to check for consistency and completeness
        """
        try:
            # Prepare validation context
            validation_context = self._prepare_validation_context(transactions, original_analysis)
            
            prompt = f"""
            You are validating the results of a bank statement parsing operation.
            
            Original Analysis:
            {json.dumps(original_analysis, indent=2)}
            
            Parsed Transactions (first 5 for context):
            {json.dumps(transactions[:5], indent=2)}
            
            Total transactions parsed: {len(transactions)}
            
            Please validate the parsing results and provide:
            1. Data quality assessment
            2. Missing data detection
            3. Inconsistency detection
            4. Confidence in the results
            5. Recommendations for improvement
            
            Respond in JSON format:
            {{
                "validation_score": float,
                "data_quality": {{
                    "completeness": float,
                    "accuracy": float,
                    "consistency": float
                }},
                "issues_found": ["list of issues"],
                "missing_data": ["list of missing fields"],
                "recommendations": ["list of recommendations"],
                "overall_confidence": float
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Validation completed with score: {result.get('validation_score', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return self._get_fallback_validation(transactions)
    
    def suggest_improvements(self, validation_result: Dict, user_feedback: Optional[str] = None) -> List[str]:
        """
        Suggest improvements based on validation results and user feedback
        """
        try:
            context = f"""
            Validation Results:
            {json.dumps(validation_result, indent=2)}
            
            User Feedback: {user_feedback or 'None provided'}
            """
            
            prompt = f"""
            Based on the validation results and user feedback, suggest specific improvements for the parsing system.
            
            {context}
            
            Provide actionable recommendations in order of priority:
            1. Immediate fixes needed
            2. Medium-term improvements
            3. Long-term enhancements
            
            Respond as a JSON array of recommendation strings.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            return ["Enable detailed logging for better debugging"]
    
    def _prepare_analysis_context(self, text_content: List[str], tables: List[List[List[str]]]) -> str:
        """Prepare context for AI analysis"""
        context_parts = []
        
        # Add text content summary
        if text_content:
            text_summary = " ".join(text_content[:3])  # First 3 text blocks
            context_parts.append(f"Text Content Preview: {text_summary[:500]}...")
        
        # Add table structure information
        if tables:
            context_parts.append(f"Number of tables found: {len(tables)}")
            for i, table in enumerate(tables[:3]):  # First 3 tables
                if table and len(table) > 0:
                    headers = table[0] if len(table) > 0 else []
                    context_parts.append(f"Table {i+1} headers: {headers}")
        
        return "\n".join(context_parts)
    
    def _prepare_validation_context(self, transactions: List[Dict], original_analysis: Dict) -> str:
        """Prepare context for validation"""
        context_parts = []
        
        # Add transaction summary
        if transactions:
            context_parts.append(f"Total transactions: {len(transactions)}")
            
            # Sample transaction structure
            sample_txn = transactions[0] if transactions else {}
            context_parts.append(f"Sample transaction structure: {list(sample_txn.keys())}")
            
            # Check for common fields
            fields_present = set()
            for txn in transactions[:10]:  # Check first 10 transactions
                fields_present.update(txn.keys())
            context_parts.append(f"Fields present: {list(fields_present)}")
        
        return "\n".join(context_parts)
    
    def _get_fallback_analysis(self, text_content: List[str], tables: List[List[List[str]]]) -> Dict[str, Any]:
        """Fallback analysis when AI is not available"""
        return {
            "bank_identification": None,
            "document_format": "unknown",
            "table_structure": {
                "num_tables": len(tables),
                "table_types": ["unknown"],
                "column_patterns": []
            },
            "parsing_strategy": {
                "primary_method": "standard",
                "fallback_method": "ocr",
                "special_handling": []
            },
            "confidence": 0.3,
            "challenges": ["AI analysis unavailable"],
            "recommendations": ["Use standard parsing with manual review"]
        }
    
    def _get_fallback_validation(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Fallback validation when AI is not available"""
        return {
            "validation_score": 0.5,
            "data_quality": {
                "completeness": 0.5,
                "accuracy": 0.5,
                "consistency": 0.5
            },
            "issues_found": ["AI validation unavailable"],
            "missing_data": [],
            "recommendations": ["Manual review recommended"],
            "overall_confidence": 0.5
        } 