# LangChain Syntax Cheat Sheet (Simple Guide)

## Overview

LangChain helps you build LLM applications by combining models, prompts,
messages, chains, and runnables.

``` text
LangChain
│
├── Models
├── Prompts
├── Messages
├── Chains (built using LCEL)
└── Runnables (foundation)
```

------------------------------------------------------------------------

# 1. Models

## Theory

A **Model** is the AI brain. It receives input and generates a response.

### Create a model

``` python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)
```

### Invoke

``` python
response = llm.invoke("Explain AI")
print(response.content)
```

**Simple explanation:** `invoke()` sends one request and returns one
response.

### Batch

``` python
responses = llm.batch(["What is AI?","What is ML?"])
```

Runs many requests together.

### Stream

``` python
for chunk in llm.stream("Explain LangChain"):
    print(chunk.content, end="")
```

Returns tokens gradually.

------------------------------------------------------------------------

# 2. Prompts

## Theory

A prompt defines **how you talk to the model**.

### PromptTemplate

``` python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
)

prompt.invoke({"topic":"LangChain"})
```

`{topic}` is replaced at runtime.

### ChatPromptTemplate

``` python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful trainer."),
    ("human","Explain {topic}")
])
```

Use for chat models.

------------------------------------------------------------------------

# 3. Messages

## Theory

Messages represent a conversation.

``` python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are helpful"),
    HumanMessage("Explain AI")
]

llm.invoke(messages)
```

-   SystemMessage = instructions
-   HumanMessage = user input
-   AIMessage = model response
-   ToolMessage = tool output

------------------------------------------------------------------------

# 4. Chains

## Theory

A chain connects multiple components.

``` python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
```

The `|` operator passes output to the next step.

``` python
result = chain.invoke({"topic":"LangChain"})
```

------------------------------------------------------------------------

# 5. Runnables

## Theory

Everything in modern LangChain is a Runnable.

### RunnableLambda

``` python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())

uppercase.invoke("hello")
```

Wraps any Python function.

### RunnableParallel

``` python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "summary": summary_chain,
    "quiz": quiz_chain
})
```

Runs multiple chains simultaneously.

### Useful Runnable methods

``` python
chain.invoke(data)
chain.batch(data)
chain.stream(data)
chain.with_retry()
chain.with_config(run_name="Demo")
```

------------------------------------------------------------------------

# 6. Output Parsers

Convert model output.

``` python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
```

------------------------------------------------------------------------

# 7. Tools

``` python
from langchain_core.tools import tool

@tool
def multiply(a:int,b:int):
    return a*b
```

Tools let an LLM call Python functions.

------------------------------------------------------------------------

# 8. Agents

``` python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[multiply]
)
```

Agents decide which tool to use automatically.

------------------------------------------------------------------------

# 9. Memory

Older:

``` python
ConversationBufferMemory()
```

Recommended with LangGraph:

``` python
MemorySaver()
```

Memory preserves conversation context.

------------------------------------------------------------------------

# 10. RAG

Pipeline:

    Loader
     ↓
    Splitter
     ↓
    Embeddings
     ↓
    Vector Store
     ↓
    Retriever
     ↓
    LLM

Common syntax:

``` python
PyPDFLoader()
RecursiveCharacterTextSplitter()
GoogleGenerativeAIEmbeddings()
FAISS.from_documents()
retriever = vectorstore.as_retriever()
```

------------------------------------------------------------------------

# 11. Structured Output

``` python
llm.with_structured_output(MySchema)
```

Returns structured objects instead of plain text.

------------------------------------------------------------------------

# 12. LCEL

``` python
prompt | llm | parser
```

LCEL (LangChain Expression Language) connects components using the pipe
operator.

------------------------------------------------------------------------

# 13. Common Methods

  Method          Purpose
  --------------- --------------------
  invoke()        One request
  batch()         Many requests
  stream()        Streaming response
  ainvoke()       Async invoke
  with_retry()    Retry failures
  with_config()   Add metadata
  bind()          Bind parameters

------------------------------------------------------------------------

# Complete Example

``` python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert trainer."),
    ("human","Explain {topic}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic":"LangChain"}))
```

## Summary

-   Model = AI engine
-   Prompt = Instructions
-   Messages = Conversation
-   Runnable = Building block
-   Chain = Connected runnables
-   Tool = Python function callable by LLM
-   Agent = Chooses tools
-   Memory = Remembers context
-   RAG = Uses external knowledge
