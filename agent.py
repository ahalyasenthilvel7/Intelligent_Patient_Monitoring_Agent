
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END


class PatientState(TypedDict, total=False):

    readings: dict
    history: list
    model: Any
    encoder: Any

    prediction: str
    confidence: float

    previous_flags: int

    priority: str
    message: str
    activity: list


# TOOL 1: ML prediction tool

def ml_prediction_tool(state):

    readings = state["readings"]

    features = [[
        readings["heart_rate"],
        readings["spo2"],
        readings["temperature"],
        readings["systolic_bp"],
        readings["diastolic_bp"],
        readings["respiratory_rate"]
    ]]

    model = state["model"]
    encoder = state["encoder"]

    probabilities = model.predict_proba(features)[0]

    predicted_number = int(probabilities.argmax())

    prediction = encoder.inverse_transform(
        [predicted_number]
    )[0]

    confidence = float(probabilities[predicted_number])

    return {
        "prediction": prediction,
        "confidence": confidence,
        "activity": ["ML prediction completed"]
    }


# TOOL 2: Patient history tool

def patient_history_tool(state):

    history = state.get("history", [])

    recent_flags = sum(
        item["prediction"] != "Normal"
        for item in history[-5:]
    )

    return {
        "previous_flags": recent_flags,
        "activity": ["Patient history reviewed"]
    }


# TOOL 3: Decision-making tool

def decision_tool(state):

    prediction = state["prediction"]
    previous_flags = state["previous_flags"]

    if prediction == "Urgent Review":

        priority = "Urgent Human Review"

        message = (
            "The ML model flagged an unusual patient "
            "reading. Prompt review by qualified "
            "medical staff is required."
        )

    elif prediction == "Review" and previous_flags >= 2:

        priority = "Repeated Review Alert"

        message = (
            "Repeated unusual readings detected. "
            "Please arrange human review."
        )

    elif prediction == "Review":

        priority = "Review Required"

        message = (
            "The ML model flagged an unusual reading. "
            "Please review the patient measurements."
        )

    else:

        priority = "Routine Monitoring"

        message = (
            "No unusual pattern was flagged by the "
            "demonstration model. Continue monitoring."
        )

    return {
        "priority": priority,
        "message": message,
        "activity": ["Decision-making completed"]
    }


# TOOL 4: Notification tool

def notification_tool(state):

    notification = (
        f"{state['priority']}: {state['message']}"
    )

    return {
        "message": notification,
        "activity": ["Notification generated"]
    }


# LANGGRAPH WORKFLOW

def build_agent():

    graph = StateGraph(PatientState)

    graph.add_node(
        "ml_analysis",
        ml_prediction_tool
    )

    graph.add_node(
        "history_analysis",
        patient_history_tool
    )

    graph.add_node(
        "decision",
        decision_tool
    )

    graph.add_node(
        "notification",
        notification_tool
    )

    graph.set_entry_point("ml_analysis")

    graph.add_edge(
        "ml_analysis",
        "history_analysis"
    )

    graph.add_edge(
        "history_analysis",
        "decision"
    )

    graph.add_edge(
        "decision",
        "notification"
    )

    graph.add_edge(
        "notification",
        END
    )

    return graph.compile()
