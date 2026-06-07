import os
import json
import requests
from typing import Dict, Any, List

from schemas.generation import RiskAnalysisOutput

# Attempt to load dotenv if available, otherwise rely on os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

class ComplianceGenerator:
    """
    Generates structured compliance risk analysis using Groq's fast inference API.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_url: str = "https://api.groq.com/openai/v1/chat/completions"):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in your .env file.")

    def _build_prompt(self, context_chunks: List[Dict[str, Any]], query: str) -> str:
        """Constructs the strict system prompt for JSON output."""
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            clause = chunk.get("clause_id", "N/A")
            section = chunk.get("section", "N/A")
            text = chunk.get("text", "")
            context_str += f"\n--- Chunk {i+1} ---\nSection: {section}\nClause ID: {clause}\nText: {text}\n"
            
        prompt = f"""You are an expert legal auditor and compliance officer.
Your task is to analyze the following extracted legal contract clauses and identify compliance risks based on the user's query.

User Query: "{query}"

Retrieved Context:
{context_str}

You MUST return a raw JSON object adhering EXACTLY to the following schema structure:
{{
  "summary": "Overall summary of compliance posture",
  "risks": [
    {{
      "risk_type": "string",
      "description": "string",
      "severity": "High" | "Medium" | "Low",
      "relevant_clause_id": "string or null"
    }}
  ]
}}

Output ONLY the JSON object. Do not include markdown formatting, code blocks, or conversational text.
"""
        return prompt

    def generate(self, context_chunks: List[Dict[str, Any]], query: str) -> RiskAnalysisOutput:
        """
        Calls the Groq API in JSON mode and parses the result into Pydantic.
        """
        prompt = self._build_prompt(context_chunks, query)
        
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
            "temperature": 0.1 # Low temperature for analytical consistency
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result_json = response.json()
            response_text = result_json["choices"][0]["message"]["content"]
            
            # Parse raw text into Python dict
            parsed_dict = json.loads(response_text)
            
            # Validate and convert via Pydantic
            return RiskAnalysisOutput(**parsed_dict)
            
        except requests.exceptions.RequestException as e:
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f"\nResponse: {e.response.text}"
            raise RuntimeError(f"Groq API request failed: {err_msg}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode LLM output into JSON: {e}\nRaw output: {response_text}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse or validate LLM output: {e}")
