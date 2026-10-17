from dotenv import load_dotenv

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from importlib.metadata import version

core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version:", {core_version})
print(f"langgraph version:", {lg_version})


def main():
    print("Hello from langc-pro!")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("say 'setup complete! in one word'")
    print(f"Open AI Response:", {response.content})

    # llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    # response = llm.invoke("say 'setup complete! in one word'")
    # print(f"Anthropic Response:", {response.content})


if __name__ == "__main__":
    main()
