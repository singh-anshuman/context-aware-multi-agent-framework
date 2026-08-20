from langgraph.graph import END, START, StateGraph

from firstNode import first_node
from secondNode import second_node
from state import State

builder = StateGraph(State)

builder.add_node("first", first_node)
builder.add_node("second", second_node)

builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

graph = builder.compile()

result = graph.invoke({
    "message": "Starting application"
})

print(result)