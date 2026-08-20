def baseline_stage_1(applicant_data):
    """No CSO - just rule-based"""
    verification_pass = (
        not applicant_data['bankruptcy_history'] and
        applicant_data['experience_years'] >= 2
    )
    return {"status": "pass" if verification_pass else "fail"}

def baseline_stage_2(applicant_data):
    """No prior context"""
    credit_score = applicant_data['credit_score']
    if credit_score >= 740:
        band = "Good"
    elif credit_score >= 670:
        band = "Fair"
    else:
        band = "Poor"
    return {"band": band}

def baseline_stage_3(applicant_data):
    """No prior context"""
    risk_score = applicant_data['risk_score']
    if risk_score < 40:
        classification = "Low"
    elif risk_score < 60:
        classification = "Medium"
    else:
        classification = "High"
    return {"classification": classification}

def baseline_stage_4(applicant_data):
    """No prior context"""
    criteria_met = (
        not applicant_data['bankruptcy_history'] and
        applicant_data['risk_score'] < 60 and
        applicant_data['debt_to_income'] <= 0.55
    )
    return {"decision": "APPROVED" if criteria_met else "REJECTED"}

def run_baseline_pipeline(applicant_data):
    """Run all 4 stages without context propagation"""
    s1 = baseline_stage_1(applicant_data)
    s2 = baseline_stage_2(applicant_data)
    s3 = baseline_stage_3(applicant_data)
    s4 = baseline_stage_4(applicant_data)
    return {
        'stage_1': s1,
        'stage_2': s2,
        'stage_3': s3,
        'stage_4': s4,
        'final_decision': s4['decision']
    }