from langgraph.graph import END, StateGraph

from loan_framework.cso.agents import (
    stage_1_document_verification,
    stage_2_credit_assessment_with_discovered_factors,
    stage_3_risk_assessment_with_discovered_factors,
    stage_4_final_decision_with_discovered_factors,
)
from loan_framework.cso.cso_framework import ContextStateObject


def create_workflow():
    """Create LangGraph workflow with CSO-aware agents"""
    workflow = StateGraph(ContextStateObject)

    workflow.add_node("stage_1", stage_1_document_verification)
    workflow.add_node("stage_2", stage_2_credit_assessment_with_discovered_factors)
    workflow.add_node("stage_3", stage_3_risk_assessment_with_discovered_factors)
    workflow.add_node("stage_4", stage_4_final_decision_with_discovered_factors)

    workflow.add_edge("stage_1", "stage_2")
    workflow.add_edge("stage_2", "stage_3")
    workflow.add_edge("stage_3", "stage_4")
    workflow.add_edge("stage_4", END)

    workflow.set_entry_point("stage_1")

    return workflow.compile()


pipeline = create_workflow()
print("✓ Phase 2 Pipeline created with CSO-aware agents")
