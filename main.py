"""
CLI entry point for the Bank Statement Parser AI Agent
Enhanced with AI agent features
"""

import argparse
import logging
import json
from pathlib import Path
from bank_parser import BankStatementParser

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Bank Statement Parser AI Agent")
    parser.add_argument('--file', type=str, required=True, help='Path to the bank statement file (PDF or Excel)')
    parser.add_argument('--output', type=str, default=None, help='Path to save the parsed output (optional)')
    parser.add_argument('--ai-agent', action='store_true', default=True, help='Enable AI agent features (default: True)')
    parser.add_argument('--insights', action='store_true', help='Show AI insights and learning progress')
    parser.add_argument('--autonomous', action='store_true', help='Start autonomous monitoring mode')
    parser.add_argument('--feedback', type=str, help='Provide feedback file path for learning')
    args = parser.parse_args()

    # Initialize AI agent
    agent = BankStatementParser(enable_ai_agent=args.ai_agent)
    
    # Start autonomous mode if requested
    if args.autonomous and args.ai_agent:
        print("Starting autonomous monitoring mode...")
        agent.start_autonomous_mode()
    
    # Parse the file
    print(f"Parsing file: {args.file}")
    result = agent.parse_file_with_balance(args.file)
    
    transactions = result['transactions']
    closing_balance = result['closing_balance']
    confidence_score = result.get('confidence_score', 0)
    ai_insights = result.get('ai_insights', {})
    
    if not transactions:
        print("No transactions found or failed to parse the file.")
        return
    
    print(f"\n=== PARSING RESULTS ===")
    print(f"Parsed {len(transactions)} transactions.")
    print(f"Confidence Score: {confidence_score:.2f}")
    
    if closing_balance is not None:
        print(f"Closing Balance: ₹{closing_balance:,.2f}")
    else:
        print("Closing balance not found in the statement.")
    
    # Show AI insights if requested
    if args.insights and args.ai_agent:
        print(f"\n=== AI INSIGHTS ===")
        insights = agent.get_ai_insights()
        
        if insights.get('ai_agent_enabled'):
            print(f"AI Agent Status: Enabled")
            
            # Bank identification
            bank_id = ai_insights.get('bank_identification')
            if bank_id:
                print(f"Bank Identified: {bank_id}")
            
            # Parsing strategy
            strategy = ai_insights.get('parsing_strategy', {})
            if strategy:
                print(f"Parsing Strategy: {strategy.get('method', 'standard')}")
            
            # Memory recommendations
            memory_recs = ai_insights.get('memory_recommendations', {})
            if memory_recs.get('suggested_strategy'):
                print(f"Memory Suggestion: {memory_recs['suggested_strategy']}")
            
            # Learning progress
            learning = insights.get('learning_progress', {})
            if learning.get('total_corrections', 0) > 0:
                print(f"Learning Progress: {learning['total_corrections']} corrections recorded")
                print(f"Accuracy Improvement: {learning.get('accuracy_improvement', 0):.2f}")
            
            # Performance metrics
            performance = insights.get('parsing_performance', {})
            if performance.get('total_files', 0) > 0:
                print(f"Total Files Processed: {performance['total_files']}")
                print(f"Average Confidence: {performance.get('average_confidence', 0):.2f}")
            
            # Recommendations
            recommendations = insights.get('recommendations', [])
            if recommendations:
                print(f"\nRecommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
        else:
            print("AI Agent Status: Disabled")
    
    # Handle user feedback
    if args.feedback and args.ai_agent:
        print(f"\n=== PROCESSING FEEDBACK ===")
        try:
            with open(args.feedback, 'r') as f:
                feedback_data = json.load(f)
            
            success = agent.record_user_feedback(
                args.file, 
                result, 
                feedback_data, 
                feedback_type='correction'
            )
            
            if success:
                print("Feedback recorded successfully for learning!")
            else:
                print("Failed to record feedback.")
                
        except Exception as e:
            print(f"Error processing feedback: {e}")
    
    # Save output
    output_path = agent.save_parsed_output(transactions, args.output, closing_balance)
    print(f"\nOutput saved to: {output_path}")
    
    # Show confidence and recommendations
    if confidence_score < 0.6:
        print(f"\n⚠️  Low confidence score ({confidence_score:.2f}). Consider manual review.")
        if args.ai_agent:
            print("Enable AI insights (--insights) for detailed analysis and recommendations.")
    
    if args.autonomous and args.ai_agent:
        print(f"\n🔄 Autonomous mode is running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            agent.stop_autonomous_mode()
            print("\nAutonomous mode stopped.")

if __name__ == '__main__':
    main() 