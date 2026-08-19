from state import State


def second_node(state: State):

    print("Executing second node")

    print("Message received:", state["message"])

    return {
        "message": state["message"] + " → Node 2"
    }