import os
import json
import requests
from typing import Dict, Any, List

from schemas.generation import RiskAnalysisOutput, CriticValidationOutput

# Attempt to load dotenv if available, otherwise rely on os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

class ComplianceCritic:
    """
    Validates the generated compliance analysis against the original context using Groq's fast inference API.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_url: str = "https://api.groq.com/openai/v1/chat/completions"):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in your .env file.")

    def _build_prompt(self, context_chunks: List[Dict[str, Any]], generated_analysis: RiskAnalysisOutput) -> str:
        """Constructs the system prompt for the critic."""
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            clause = chunk.get("clause_id", "N/A")
            text = chunk.get("text", "")
            context_str += f"Clause ID: {clause}\nText: {text}\n---\n"
            
        analysis_json_str = generated_analysis.model_dump_json(indent=2)
            
        prompt = f"""You are a senior legal auditor evaluating an AI-generated compliance risk report.
Your job is to strictly verify if the proposed risks are actually supported by the source text. 

Source Contract Text:
{context_str}

AI Generated Risk Analysis:
{analysis_json_str}

Evaluate the analysis for hallucinations, unsupported claims, or exaggerations. 
You MUST return a raw JSON object adhering EXACTLY to the following schema structure:
{{
  "is_hallucinated": true or false,
  "confidence_score": integer between 1 and 10,
  "feedback": "string explaining your reasoning",
  "corrected_analysis": {{
    "summary": "string",
    "risks": [
      {{
        "risk_type": "string",
        "description": "string",
        "severity": "High" | "Medium" | "Low",
        "relevant_clause_id": "string or null"
      }}
    ]
  }} // ONLY INCLUDE THIS FIELD IF is_hallucinated IS TRUE, OTHERWISE NULL
}}

Output ONLY the JSON object. Do not include markdown formatting, code blocks, or conversational text.
"""
        return prompt

    def validate(self, context_chunks: List[Dict[str, Any]], generated_analysis: RiskAnalysisOutput) -> CriticValidationOutput:
        """
        Calls the Groq API to validate the analysis.
        """
        prompt = self._build_prompt(context_chunks, generated_analysis)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0 # Zero temperature for absolute determinism in grading
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result_json = response.json()
            response_text = result_json["choices"][0]["message"]["content"]
            
            # Parse raw text into Python dict
            parsed_dict = json.loads(response_text)
            
            # If corrected_analysis is provided but empty/null in JSON, ensure it is None in Pydantic
            if parsed_dict.get("is_hallucinated") is False:
                parsed_dict["corrected_analysis"] = None
            
            # Validate and convert via Pydantic
            return CriticValidationOutput(**parsed_dict)
            
        except requests.exceptions.RequestException as e:
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f"\nResponse: {e.response.text}"
            raise RuntimeError(f"Groq API request failed: {err_msg}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode Critic output into JSON: {e}\nRaw output: {response_text}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse or validate Critic output: {e}")
