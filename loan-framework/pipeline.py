from agents import (
    stage_1_document_verification,
    stage_2_credit_scoring,
    stage_3_risk_assessment,
    stage_4_final_approval,
)
from langgraph.graph import END, StateGraph


def create_workflow():
    """Create LangGraph workflow"""
    workflow = StateGraph(dict)  # Use dict for TypedDict compatibility
    
    workflow.add_node("stage_1", stage_1_document_verification)
    workflow.add_node("stage_2", stage_2_credit_scoring)
    workflow.add_node("stage_3", stage_3_risk_assessment)
    workflow.add_node("stage_4", stage_4_final_approval)
    
    workflow.add_edge("stage_1", "stage_2")
    workflow.add_edge("stage_2", "stage_3")
    workflow.add_edge("stage_3", "stage_4")
    workflow.add_edge("stage_4", END)
    
    workflow.set_entry_point("stage_1")
    
    return workflow.compile()

pipeline = create_workflow()
print("✓ Pipeline created successfully")