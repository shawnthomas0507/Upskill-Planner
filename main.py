from graph import app
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


user_input = input("Enter your name (type 'quit' to exit): ")
thread={"configurable":{"thread_id":"1"}}
for event in app.stream({"name": user_input, "messages": [HumanMessage(content=user_input)]},thread,stream_mode="values"):
    event["messages"][-1].pretty_print()

    