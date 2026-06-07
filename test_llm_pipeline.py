import json
from core.llm_generator import ComplianceGenerator
from core.llm_critic import ComplianceCritic

def main():
    print("1. Preparing mock context chunks...")
    mock_chunks = [
        {
            "clause_id": "4.2",
            "section": "TERMINATION",
            "text": "The Vendor may terminate this Agreement at any time without notice. In the event of termination by the Vendor, the Client shall pay a penalty fee of $50,000."
        },
        {
            "clause_id": "7.1",
            "section": "LIABILITY",
            "text": "Under no circumstances shall the Vendor's aggregate liability exceed the total amount paid by the Client in the preceding one (1) month."
        }
    ]
    query = "Are there any unfair termination penalties or severe liability caps?"
    
    print("\n2. Initializing Compliance Generator...")
    # Using default model (llama3-70b-8192) via Groq API
    generator = ComplianceGenerator()
    
    print("\n3. Generating Risk Analysis (calling Ollama)...")
    try:
        analysis = generator.generate(mock_chunks, query)
        print("\n=== GENERATED ANALYSIS ===")
        print(f"Summary: {analysis.summary}")
        for r in analysis.risks:
            print(f"- [{r.severity}] {r.risk_type} (Clause {r.relevant_clause_id}): {r.description}")
            
    except Exception as e:
        print(f"\n[ERROR] Generation failed. Is Ollama running with the llama3 model? Error: {e}")
        return

    print("\n4. Initializing Compliance Critic...")
    critic = ComplianceCritic()
    
    print("\n5. Running Critic Validation...")
    try:
        validation = critic.validate(mock_chunks, analysis)
        print("\n=== CRITIC VALIDATION ===")
        print(f"Hallucinated: {validation.is_hallucinated}")
        print(f"Confidence Score: {validation.confidence_score}/10")
        print(f"Feedback: {validation.feedback}")
        
        if validation.is_hallucinated and validation.corrected_analysis:
            print("\n=== CORRECTED ANALYSIS ===")
            print(f"Summary: {validation.corrected_analysis.summary}")
            
    except Exception as e:
        print(f"\n[ERROR] Critic validation failed. Error: {e}")

if __name__ == "__main__":
    main()
