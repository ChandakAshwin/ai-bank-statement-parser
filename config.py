"""
Configuration settings for the Bank Statement Parser AI Agent
"""

import os
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
MEMORY_DIR = BASE_DIR / "memory"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, MEMORY_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# AI Agent Configuration
AI_AGENT_CONFIG = {
    'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
    'openai_model': 'gpt-4-turbo-preview',  # or 'gpt-3.5-turbo' for cost efficiency
    'temperature': 0.1,
    'max_tokens': 2000,
    'memory_enabled': True,
    'autonomous_mode': False,  # Set to True for autonomous operation
    'learning_enabled': True,
    'feedback_loop_enabled': True,
    'confidence_threshold': 0.8,
    'max_retries': 3
}

# Memory and Learning Configuration
MEMORY_CONFIG = {
    'vector_db_path': str(MEMORY_DIR / "chroma_db"),
    'embedding_model': 'all-MiniLM-L6-v2',
    'max_memory_size': 10000,  # Maximum number of documents to remember
    'similarity_threshold': 0.85,
    'learning_rate': 0.1,
    'model_save_path': str(MODELS_DIR / "learned_models.pkl")
}

# Autonomous Behavior Configuration
AUTONOMOUS_CONFIG = {
    'email_monitoring': {
        'enabled': False,
        'email_address': os.getenv('EMAIL_ADDRESS', ''),
        'email_password': os.getenv('EMAIL_PASSWORD', ''),
        'imap_server': 'imap.gmail.com',
        'imap_port': 993,
        'check_interval': 300,  # 5 minutes
        'keywords': ['bank statement', 'account statement', 'transaction summary']
    },
    'cloud_integration': {
        'enabled': False,
        'google_drive_folder_id': os.getenv('GOOGLE_DRIVE_FOLDER_ID', ''),
        'dropbox_access_token': os.getenv('DROPBOX_ACCESS_TOKEN', ''),
        'sync_interval': 600  # 10 minutes
    },
    'auto_analysis': {
        'enabled': True,
        'trend_analysis': True,
        'anomaly_detection': True,
        'spending_patterns': True,
        'budget_tracking': True
    }
}

# Supported file types
SUPPORTED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'excel': ['.xls', '.xlsx', '.xlsm'],
    'all': ['.pdf', '.xls', '.xlsx', '.xlsm']
}

# Date formats commonly found in bank statements
DATE_FORMATS = [
    '%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%d/%m/%y',
    '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y',
    '%b %d, %Y', '%B %d, %Y',
    '%d %b %Y', '%d %B %Y',
    '%d-%b-%Y', '%d-%B-%Y'  # Added for IDFC format like "01-Mar-2025"
]

# Common column headers and their variations
COLUMN_MAPPINGS = {
    'date': [
        'date', 'transaction date', 'posting date', 'value date',
        'date posted', 'transaction date', 'date of transaction',
        'value dt', 'posting dt', 'transaction dt',  # Added for IDFC
        'date', 'transaction date', 'value date'  # Generic variations
    ],
    'description': [
        'description', 'transaction description', 'details', 'narration',
        'particulars', 'memo', 'reference', 'transaction details',
        'merchant', 'payee', 'description of transaction',
        'transaction particulars', 'narration',  # Added for IDFC
        'remarks', 'transaction id', 'narration', 'particulars',  # More variations
        'description', 'details', 'narration',  # Generic variations
        'remarks'  # Specific for third statement
    ],
    'amount': [
        'amount', 'transaction amount', 'debit', 'credit',
        'withdrawal', 'deposit', 'transaction', 'sum', 'total',
        'debit amount', 'credit amount',  # Added for IDFC
        'amount(rs.)', 'amount (rs.)', 'amount(rs)', 'amount (rs)',  # Amount with currency
        'debit', 'credit', 'dr', 'cr',  # Debit/Credit variations
        'amount', 'transaction amount'  # Generic variations
    ],
    'debit': [
        'debit', 'withdrawal', 'payment', 'out', 'dr', 'debit amount',
        'amount debited', 'withdrawn', 'debit', 'dr'
    ],
    'credit': [
        'credit', 'deposit', 'receipt', 'in', 'cr', 'credit amount',
        'amount credited', 'deposited', 'credit', 'cr'
    ],
    'balance': [
        'balance', 'running balance', 'closing balance', 'available balance',
        'account balance', 'current balance', 'balance after transaction',
        'balance', 'running balance'  # Generic variations
    ]
}

# Transaction categories for automatic categorization
TRANSACTION_CATEGORIES = {
    'income': [
        'salary', 'deposit', 'transfer in', 'refund', 'interest',
        'dividend', 'payment received', 'credit'
    ],
    'shopping': [
        'walmart', 'target', 'amazon', 'ebay', 'online purchase',
        'retail', 'store', 'shop', 'mall'
    ],
    'food': [
        'restaurant', 'mcdonalds', 'starbucks', 'pizza', 'food',
        'dining', 'cafe', 'coffee', 'grocery'
    ],
    'transportation': [
        'uber', 'lyft', 'taxi', 'gas', 'fuel', 'parking',
        'toll', 'public transport', 'metro', 'bus'
    ],
    'utilities': [
        'electric', 'water', 'gas', 'internet', 'phone',
        'utility', 'bill', 'service'
    ],
    'entertainment': [
        'netflix', 'spotify', 'movie', 'theater', 'concert',
        'game', 'entertainment', 'streaming'
    ],
    'healthcare': [
        'pharmacy', 'doctor', 'hospital', 'medical', 'health',
        'dental', 'vision', 'insurance'
    ]
}

# OCR settings
OCR_CONFIG = {
    'language': 'eng',
    'config': '--oem 3 --psm 6',
    'timeout': 30
}

# PDF processing settings
PDF_CONFIG = {
    'table_settings': {
        'vertical_strategy': 'text',
        'horizontal_strategy': 'text',
        'intersection_y_tolerance': 10,
        'intersection_x_tolerance': 10
    },
    'ocr_settings': {
        'dpi': 300,
        'format': 'PNG'
    }
}

# Excel processing settings
EXCEL_CONFIG = {
    'sheet_name': 0,  # First sheet by default
    'header': 0,      # First row as header
    'skiprows': None,
    'na_values': ['', 'nan', 'NaN', 'N/A', 'n/a']
}

# Output settings
OUTPUT_CONFIG = {
    'date_format': '%Y-%m-%d',
    'amount_precision': 2,
    'include_category': True,
    'output_format': 'json'  # 'json', 'csv', 'excel'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'parser.log'
}

# Create logs directory
LOGGING_CONFIG['file'].parent.mkdir(exist_ok=True) 