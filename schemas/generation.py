from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class RiskItem(BaseModel):
    """
    Represents a single compliance risk identified within the text.
    """
    risk_type: str = Field(description="The category of the risk (e.g., 'Uncapped Liability', 'Auto-renewal')")
    description: str = Field(description="A detailed explanation of why this constitutes a risk")
    severity: Literal["High", "Medium", "Low"] = Field(description="The severity level of the risk")
    relevant_clause_id: Optional[str] = Field(default=None, description="The specific clause ID associated with the risk, if available")

class RiskAnalysisOutput(BaseModel):
    """
    The structured output expected from the LLM Generator.
    """
    risks: List[RiskItem] = Field(default_factory=list, description="A list of identified risks")
    summary: str = Field(description="A brief overall summary of the compliance posture for the provided context")

class CriticValidationOutput(BaseModel):
    """
    The structured output expected from the LLM Critic.
    """
    is_hallucinated: bool = Field(description="True if the critic believes the initial analysis contains hallucinations or unsupported claims")
    corrected_analysis: Optional[RiskAnalysisOutput] = Field(default=None, description="The corrected risk analysis if hallucinations were found")
    confidence_score: int = Field(ge=1, le=10, description="Confidence score from 1 to 10 on the final analysis")
    feedback: str = Field(description="Explanation for the validation decision")
