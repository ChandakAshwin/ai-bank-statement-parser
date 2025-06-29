"""
Example usage of the Bank Statement Parser AI Agent
Demonstrates all AI agent features and capabilities
"""

import json
from pathlib import Path
from bank_parser import BankStatementParser

def main():
    """Demonstrate AI agent capabilities"""
    
    print("🚀 Bank Statement Parser AI Agent Demo")
    print("=" * 50)
    
    # Initialize AI agent
    agent = BankStatementParser(enable_ai_agent=True)
    
    # Example file path (replace with actual file)
    file_path = "example_statement.pdf"
    
    if not Path(file_path).exists():
        print(f"⚠️  Example file '{file_path}' not found.")
        print("Please provide a valid bank statement file path.")
        return
    
    print(f"📄 Processing file: {file_path}")
    print("-" * 30)
    
    # Parse with AI agent
    result = agent.parse_file_with_balance(file_path)
    
    # Display results
    transactions = result['transactions']
    closing_balance = result['closing_balance']
    confidence_score = result.get('confidence_score', 0)
    ai_insights = result.get('ai_insights', {})
    
    print(f"✅ Parsed {len(transactions)} transactions")
    print(f"🎯 Confidence Score: {confidence_score:.2f}")
    
    if closing_balance:
        print(f"💰 Closing Balance: ₹{closing_balance:,.2f}")
    
    # Show AI insights
    print("\n🤖 AI Agent Insights:")
    print("-" * 20)
    
    if ai_insights:
        bank_id = ai_insights.get('bank_identification')
        if bank_id:
            print(f"🏦 Bank Identified: {bank_id}")
        
        strategy = ai_insights.get('parsing_strategy', {})
        if strategy:
            print(f"📋 Parsing Strategy: {strategy.get('method', 'standard')}")
        
        challenges = ai_insights.get('expected_challenges', [])
        if challenges:
            print(f"⚠️  Expected Challenges: {', '.join(challenges)}")
    
    # Get comprehensive insights
    print("\n📊 Learning & Performance Insights:")
    print("-" * 35)
    
    insights = agent.get_ai_insights()
    
    if insights.get('ai_agent_enabled'):
        # Learning progress
        learning = insights.get('learning_progress', {})
        total_corrections = learning.get('total_corrections', 0)
        accuracy_improvement = learning.get('accuracy_improvement', 0)
        
        print(f"📚 Learning Progress: {total_corrections} corrections recorded")
        print(f"📈 Accuracy Improvement: {accuracy_improvement:.2f}")
        
        # Performance metrics
        performance = insights.get('parsing_performance', {})
        total_files = performance.get('total_files', 0)
        avg_confidence = performance.get('average_confidence', 0)
        
        print(f"📁 Total Files Processed: {total_files}")
        print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    # Demonstrate feedback learning
    print("\n🔄 Feedback Learning Demo:")
    print("-" * 25)
    
    # Create example feedback
    example_feedback = {
        "transactions": [
            {
                "index": 0,
                "corrections": {
                    "description": "Corrected merchant name",
                    "amount": 150.00,
                    "category": "shopping"
                }
            }
        ],
        "closing_balance": 125450.00,
        "feedback_type": "correction",
        "notes": "User corrected transaction details"
    }
    
    # Record feedback
    success = agent.record_user_feedback(
        file_path, 
        result, 
        example_feedback, 
        feedback_type='correction'
    )
    
    if success:
        print("✅ Feedback recorded successfully!")
        print("   The AI agent will learn from this correction")
        print("   Future parsing of similar documents will be improved")
    else:
        print("❌ Failed to record feedback")
    
    # Demonstrate autonomous features
    print("\n🤖 Autonomous Features Demo:")
    print("-" * 30)
    
    autonomous_status = insights.get('autonomous_status', {})
    if autonomous_status.get('is_running'):
        print("🔄 Autonomous mode is currently running")
        print("   - Monitoring email for new statements")
        print("   - Checking cloud storage for updates")
        print("   - Generating automatic insights")
    else:
        print("⏸️  Autonomous mode is not running")
        print("   Use --autonomous flag to enable")
    
    # Save results
    output_path = agent.save_parsed_output(transactions, "demo_output.json", closing_balance)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Show sample transactions
    if transactions:
        print(f"\n📋 Sample Transactions (first 3):")
        print("-" * 40)
        
        for i, txn in enumerate(transactions[:3]):
            print(f"{i+1}. {txn.get('date', 'N/A')} - {txn.get('description', 'N/A')}")
            print(f"   Amount: ₹{txn.get('amount', 0):,.2f}")
            print(f"   Category: {txn.get('category', 'N/A')}")
            print()
    
    print("🎉 Demo completed!")
    print("\n💡 Tips for better results:")
    print("   • Use --insights flag for detailed AI analysis")
    print("   • Provide feedback to improve learning")
    print("   • Enable autonomous mode for continuous monitoring")
    print("   • Check confidence scores for quality assessment")

def demonstrate_autonomous_mode():
    """Demonstrate autonomous monitoring"""
    print("\n🤖 Autonomous Mode Demo")
    print("=" * 30)
    
    agent = BankStatementParser(enable_ai_agent=True)
    
    print("Starting autonomous monitoring...")
    print("This will:")
    print("  • Monitor email for bank statements")
    print("  • Check cloud storage for new files")
    print("  • Automatically process and analyze")
    print("  • Generate insights and trends")
    
    # Start autonomous mode
    agent.start_autonomous_mode()
    
    print("\n🔄 Autonomous mode is now running!")
    print("Press Ctrl+C to stop...")
    
    try:
        import time
        while True:
            time.sleep(10)
            # In a real scenario, you'd see periodic updates here
    except KeyboardInterrupt:
        agent.stop_autonomous_mode()
        print("\n⏹️  Autonomous mode stopped.")

if __name__ == "__main__":
    main()
    
    # Uncomment to demonstrate autonomous mode
    # demonstrate_autonomous_mode() 