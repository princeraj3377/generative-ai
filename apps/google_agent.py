from dotenv import load_dotenv
load_dotenv()


from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver



search= GoogleSerperAPIWrapper()


model = ChatOllama(model = "gemma4:31b-cloud"  )

agent = create_agent(model=model, tools=[search.run],system_prompt="you are a helpful assistant that can answer questions and perform searches using the Google Search API. Use the search tool when necessary to find information.",
                     checkpointer=MemorySaver())


while True:
    question = input("Please enter your question: ")
    if(question.lower()=="exit"):
        break
    response = agent.invoke({"messages":[{"role":"user","content":question}]},
                            {"configurable":{"thread_id":"1"}})
    print(response['messages'][-1].content)
    print("------------------------------------------------------------")
