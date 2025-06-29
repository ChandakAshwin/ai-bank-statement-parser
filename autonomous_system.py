"""
Autonomous System for Bank Statement Parser AI Agent
Enables autonomous behavior for fetching, parsing, and analyzing bank statements
"""

import logging
import time
import threading
import imaplib
import email
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from config import AUTONOMOUS_CONFIG, AI_AGENT_CONFIG
from ai_reasoning import AIReasoningEngine
from memory_system import MemorySystem

logger = logging.getLogger(__name__)

class AutonomousSystem:
    """Autonomous system for monitoring and processing bank statements"""
    
    def __init__(self, parser=None):
        self.parser = parser  # Will be set later to avoid circular import
        self.reasoning_engine = AIReasoningEngine()
        self.memory_system = MemorySystem()
        self.is_running = False
        self.monitoring_thread = None
        
        # Email monitoring
        self.email_config = AUTONOMOUS_CONFIG['email_monitoring']
        self.imap_server = None
        self.imap_port = None
        
        # Cloud integration
        self.cloud_config = AUTONOMOUS_CONFIG['cloud_integration']
        self.drive_service = None
        
        # Auto analysis
        self.analysis_config = AUTONOMOUS_CONFIG['auto_analysis']
        
        # Initialize services
        self._initialize_services()
    
    def set_parser(self, parser):
        """Set the parser instance to avoid circular import"""
        self.parser = parser
    
    def start_autonomous_mode(self):
        """Start autonomous monitoring and processing"""
        if self.is_running:
            logger.warning("Autonomous mode is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Autonomous mode started")
    
    def stop_autonomous_mode(self):
        """Stop autonomous monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Autonomous mode stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for autonomous operation"""
        while self.is_running:
            try:
                # Check email for new statements
                if self.email_config['enabled']:
                    self._check_email_for_statements()
                
                # Check cloud storage for new files
                if self.cloud_config['enabled']:
                    self._check_cloud_storage()
                
                # Wait before next check
                time.sleep(self.email_config['check_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_email_for_statements(self):
        """Check email for new bank statements"""
        try:
            if not self._connect_email():
                return
            
            # Search for emails with bank statement keywords
            keywords = self.email_config['keywords']
            search_criteria = ' OR '.join([f'SUBJECT "{keyword}"' for keyword in keywords])
            
            # Search for recent emails (last 24 hours)
            date_since = (datetime.now() - timedelta(days=1)).strftime('%d-%b-%Y')
            search_query = f'(SINCE {date_since}) AND ({search_criteria})'
            
            _, message_numbers = self.imap_server.search(None, search_query)
            
            if message_numbers[0]:
                message_list = message_numbers[0].split()
                
                for num in message_list:
                    self._process_email_message(num)
            
            self._disconnect_email()
            
        except Exception as e:
            logger.error(f"Error checking email: {e}")
    
    def _connect_email(self) -> bool:
        """Connect to email server"""
        try:
            self.imap_server = imaplib.IMAP4_SSL(
                self.email_config['imap_server'],
                self.email_config['imap_port']
            )
            self.imap_server.login(
                self.email_config['email_address'],
                self.email_config['email_password']
            )
            self.imap_server.select('INBOX')
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to email: {e}")
            return False
    
    def _disconnect_email(self):
        """Disconnect from email server"""
        try:
            if self.imap_server:
                self.imap_server.close()
                self.imap_server.logout()
        except Exception as e:
            logger.error(f"Error disconnecting from email: {e}")
    
    def _process_email_message(self, message_num: bytes):
        """Process a single email message"""
        try:
            _, msg_data = self.imap_server.fetch(message_num, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            # Check for attachments
            for part in email_message.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                
                filename = part.get_filename()
                if filename:
                    # Check if it's a bank statement file
                    if self._is_bank_statement_file(filename):
                        self._download_and_process_attachment(part, filename)
            
        except Exception as e:
            logger.error(f"Error processing email message: {e}")
    
    def _is_bank_statement_file(self, filename: str) -> bool:
        """Check if file is likely a bank statement"""
        filename_lower = filename.lower()
        
        # Check file extension
        supported_extensions = ['.pdf', '.xls', '.xlsx', '.xlsm']
        if not any(filename_lower.endswith(ext) for ext in supported_extensions):
            return False
        
        # Check filename for bank statement indicators
        bank_indicators = [
            'statement', 'account', 'transaction', 'bank', 'credit', 'debit',
            'summary', 'report', 'activity'
        ]
        
        return any(indicator in filename_lower for indicator in bank_indicators)
    
    def _download_and_process_attachment(self, part, filename: str):
        """Download and process email attachment"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(part.get_payload(decode=True))
                temp_file_path = temp_file.name
            
            # Process the file
            self._process_bank_statement(temp_file_path, f"email_{filename}")
            
            # Clean up
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error downloading and processing attachment: {e}")
    
    def _check_cloud_storage(self):
        """Check cloud storage for new bank statement files"""
        try:
            if self.drive_service:
                # Check Google Drive
                self._check_google_drive()
            
            # Add other cloud storage providers here
            # self._check_dropbox()
            
        except Exception as e:
            logger.error(f"Error checking cloud storage: {e}")
    
    def _check_google_drive(self):
        """Check Google Drive for new bank statement files"""
        try:
            folder_id = self.cloud_config['google_drive_folder_id']
            if not folder_id:
                return
            
            # Query for files in the specified folder
            query = f"'{folder_id}' in parents and (mimeType contains 'application/pdf' or mimeType contains 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')"
            
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, modifiedTime)',
                orderBy='modifiedTime desc'
            ).execute()
            
            files = results.get('files', [])
            
            for file in files:
                # Check if file was modified recently (last hour)
                modified_time = datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00'))
                if datetime.now(modified_time.tzinfo) - modified_time < timedelta(hours=1):
                    self._download_and_process_drive_file(file)
            
        except Exception as e:
            logger.error(f"Error checking Google Drive: {e}")
    
    def _download_and_process_drive_file(self, file_info: Dict):
        """Download and process Google Drive file"""
        try:
            # Download file
            request = self.drive_service.files().get_media(fileId=file_info['id'])
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_info['name']).suffix) as temp_file:
                temp_file.write(file_content.getvalue())
                temp_file_path = temp_file.name
            
            # Process the file
            self._process_bank_statement(temp_file_path, f"drive_{file_info['name']}")
            
            # Clean up
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error downloading and processing Drive file: {e}")
    
    def _process_bank_statement(self, file_path: str, source: str):
        """Process a bank statement file autonomously"""
        try:
            logger.info(f"Processing bank statement from {source}: {file_path}")
            
            # Parse the file
            result = self.parser.parse_file_with_balance(file_path)
            transactions = result['transactions']
            closing_balance = result['closing_balance']
            
            if not transactions:
                logger.warning(f"No transactions found in {source}")
                return
            
            # Analyze document structure
            analysis_result = self.reasoning_engine.analyze_document_structure(
                [], []  # Will be populated by parser
            )
            
            # Validate results
            validation_result = self.reasoning_engine.validate_parsing_result(
                transactions, analysis_result
            )
            
            # Store experience in memory
            self.memory_system.store_document_experience(
                file_path, analysis_result, result, validation_result
            )
            
            # Generate insights if enabled
            if self.analysis_config['enabled']:
                insights = self._generate_insights(transactions, closing_balance)
                self._save_insights(insights, source)
            
            logger.info(f"Successfully processed {source}: {len(transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error processing bank statement {source}: {e}")
    
    def _generate_insights(self, transactions: List[Dict], closing_balance: Optional[float]) -> Dict[str, Any]:
        """Generate insights from parsed transactions"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'transaction_count': len(transactions),
            'closing_balance': closing_balance,
            'summary': {},
            'trends': {},
            'anomalies': [],
            'recommendations': []
        }
        
        if not transactions:
            return insights
        
        # Basic summary
        total_debits = sum(t.get('amount', 0) for t in transactions if t.get('type') == 'debit')
        total_credits = sum(t.get('amount', 0) for t in transactions if t.get('type') == 'credit')
        
        insights['summary'] = {
            'total_debits': total_debits,
            'total_credits': total_credits,
            'net_change': total_credits - total_debits,
            'average_transaction': sum(t.get('amount', 0) for t in transactions) / len(transactions)
        }
        
        # Category analysis
        if self.analysis_config['spending_patterns']:
            insights['trends']['categories'] = self._analyze_spending_categories(transactions)
        
        # Anomaly detection
        if self.analysis_config['anomaly_detection']:
            insights['anomalies'] = self._detect_anomalies(transactions)
        
        # Budget tracking
        if self.analysis_config['budget_tracking']:
            insights['trends']['budget_status'] = self._track_budget(transactions)
        
        return insights
    
    def _analyze_spending_categories(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending patterns by category"""
        categories = {}
        
        for transaction in transactions:
            category = transaction.get('category', 'uncategorized')
            amount = transaction.get('amount', 0)
            
            if category not in categories:
                categories[category] = {'count': 0, 'total': 0}
            
            categories[category]['count'] += 1
            categories[category]['total'] += amount
        
        return categories
    
    def _detect_anomalies(self, transactions: List[Dict]) -> List[Dict]:
        """Detect anomalous transactions"""
        anomalies = []
        
        if len(transactions) < 2:
            return anomalies
        
        # Calculate average transaction amount
        amounts = [t.get('amount', 0) for t in transactions]
        avg_amount = sum(amounts) / len(amounts)
        std_amount = (sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)) ** 0.5
        
        # Detect outliers (transactions > 2 standard deviations from mean)
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            if abs(amount - avg_amount) > 2 * std_amount:
                anomalies.append({
                    'transaction': transaction,
                    'reason': 'unusual_amount',
                    'severity': 'high' if abs(amount - avg_amount) > 3 * std_amount else 'medium'
                })
        
        return anomalies
    
    def _track_budget(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Track budget against transactions"""
        # This is a simplified budget tracking
        # In a real implementation, you'd load budget categories from user preferences
        
        budget_status = {
            'total_spent': sum(t.get('amount', 0) for t in transactions if t.get('type') == 'debit'),
            'budget_remaining': 'unknown',  # Would be calculated from user budget
            'over_budget_categories': []
        }
        
        return budget_status
    
    def _save_insights(self, insights: Dict[str, Any], source: str):
        """Save generated insights to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"insights_{source}_{timestamp}.json"
            filepath = Path("output") / filename
            
            with open(filepath, 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.info(f"Saved insights to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
    
    def _initialize_services(self):
        """Initialize email and cloud services"""
        try:
            # Initialize Google Drive service if configured
            if (self.cloud_config['enabled'] and 
                self.cloud_config['google_drive_folder_id']):
                self._initialize_google_drive()
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
    
    def _initialize_google_drive(self):
        """Initialize Google Drive API service"""
        try:
            # This is a simplified implementation
            # In a real implementation, you'd need proper OAuth2 setup
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            
            creds = None
            # The file token.json stores the user's access and refresh tokens
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            self.drive_service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive service initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Google Drive: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of autonomous system"""
        return {
            'is_running': self.is_running,
            'email_monitoring': self.email_config['enabled'],
            'cloud_integration': self.cloud_config['enabled'],
            'auto_analysis': self.analysis_config['enabled'],
            'last_check': datetime.now().isoformat()
        } 