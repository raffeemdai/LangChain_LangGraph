# LangChain Syntax Cheat Sheet (Simple Guide)

## Overview

LangChain helps you build LLM applications by combining models, prompts,
messages, chains, and runnables.

```
LangChain
│
├── Models
├── Prompts
├── Messages
├── Chains (built using LCEL)
└── Runnables (foundation)
```

> **🧠 Analogy:** Think of LangChain like a **kitchen for cooking with AI**.
> - **Models** = the stove (the thing that actually does the "cooking"/thinking)
> - **Prompts** = the recipe card (tells the stove what to make)
> - **Messages** = the conversation between you and the chef
> - **Chains** = an assembly line where dishes pass from one station to the next
> - **Runnables** = the standard-shaped containers every station can use, so anything can plug into anything

---

# 1. Models

## Theory

A **Model** is the AI brain. It receives input and generates a response.

> **🧠 Simple explanation:** A model is just a very smart "text-in, text-out" machine. You give it words, it gives you words back.
>
> **Analogy:** It's like a super knowledgeable friend on the phone — you ask a question, they answer. They don't remember the call once it ends unless you write it down (that's what Memory, in section 9, is for).

### Create a model

```
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)
```

> **Simple explanation:** This line "hires" a specific AI model to work for you. `temperature=0` means "give me consistent, no-surprises answers" — like asking someone to always take the safest, most predictable route rather than experimenting.

> **📌 Update if you're using Euri (as in `prompt_engineering_notes_euri.md`):**
> Euri is a **free, OpenAI-compatible** API — meaning instead of a Google-specific class, you use LangChain's generic `ChatOpenAI` class and simply point it at Euri's URL and key:
> ```
> import os
> from dotenv import load_dotenv
> from langchain_openai import ChatOpenAI
>
> load_dotenv()
>
> llm = ChatOpenAI(
>     model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
>     api_key=os.getenv("EURI_API_KEY"),
>     base_url=os.getenv("EURI_BASE_URL"),   # https://api.euron.one/api/v1/euri
> )
> ```
> **Analogy:** `ChatGoogleGenerativeAI` is like calling **Google's private phone line directly**. `ChatOpenAI` + Euri is like using a **universal phone adapter** — the same standard "OpenAI-style" plug works no matter which provider (Gemini, Llama, GPT-family) is actually answering on the other end, because Euri translates the call for you.
> Everything downstream (`invoke`, `batch`, `stream`, chains, `|` pipes) works **exactly the same** — only the setup line changes. That's the whole point of Runnables (see Section 5): swap the model, keep the rest of the pipeline untouched.

### Invoke

```
response = llm.invoke("Explain AI")
print(response.content)
```

**Simple explanation:** `invoke()` sends one request and returns one
response.

> **Analogy:** `invoke()` is like sending a single text message and waiting for one reply.

### Batch

```
responses = llm.batch(["What is AI?","What is ML?"])
```

Runs many requests together.

> **Analogy:** `batch()` is like sending several separate text messages to the same friend at once, and getting a stack of replies back — one answer per question, all handled together instead of one at a time.

### Stream

```
for chunk in llm.stream("Explain LangChain"):
    print(chunk.content, end="")
```

Returns tokens gradually.

> **Analogy:** `stream()` is like watching someone type a reply in real time (the "..." typing bubble), instead of waiting for the whole message to be written before you see anything.

---

# 2. Prompts

## Theory

A prompt defines **how you talk to the model**.

> **Analogy:** A prompt is the **recipe card** you hand to the chef (the model). A vague recipe ("make food") gives an unpredictable dish. A clear recipe ("make a 2-egg omelet with cheese") gives a predictable result. Good prompts = clear recipes.

### PromptTemplate

```
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
)

prompt.invoke({"topic":"LangChain"})
```

`{topic}` is replaced at runtime.

> **Analogy:** `{topic}` is a **blank in a fill-in-the-blank sentence**. The template is reusable — you can plug in "LangChain" today and "gravity" tomorrow without rewriting the whole sentence.

### ChatPromptTemplate

```
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful trainer."),
    ("human","Explain {topic}")
])
```

Use for chat models.

> **Analogy:** This is like handing an actor a **script with roles labeled** — one line says "here's your character/personality" (system), and another says "here's what the other person says to you" (human). It sets the scene before the conversation starts.

---

# 3. Messages

## Theory

Messages represent a conversation.

```
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are helpful"),
    HumanMessage("Explain AI")
]

llm.invoke(messages)
```

- SystemMessage = instructions
- HumanMessage = user input
- AIMessage = model response
- ToolMessage = tool output

> **Analogy:** Think of a **group chat with labeled speakers**:
> - **SystemMessage** = the sticky note pinned at the top of the chat ("rules of this conversation")
> - **HumanMessage** = what *you* typed
> - **AIMessage** = what the *AI* replied
> - **ToolMessage** = a message from a "bot" in the chat reporting back results (like a calculator bot posting an answer)

---

# 4. Chains

## Theory

A chain connects multiple components.

```
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
```

The `|` operator passes output to the next step.

```
result = chain.invoke({"topic":"LangChain"})
```

> **Analogy:** A chain is a **factory assembly line**. Each station does one job and passes the product to the next station:
> `prompt` (writes the recipe) → `llm` (cooks the dish) → `StrOutputParser` (plates it nicely for serving).
>
> The `|` symbol is literally the conveyor belt connecting each station.

---

# 5. Runnables

## Theory

Everything in modern LangChain is a Runnable.

> **Analogy:** A Runnable is like a **standard-sized shipping container**. Because every component (models, prompts, parsers, functions) is packaged the same way, they can all be stacked, connected, and shipped through the same "ports" (`invoke`, `batch`, `stream`) no matter what's inside.

### RunnableLambda

```
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())

uppercase.invoke("hello")
```

Wraps any Python function.

> **Analogy:** `RunnableLambda` is an **adapter plug**. It takes your regular Python function (which doesn't naturally fit into a chain) and wraps it so it can plug into the same "socket" as everything else in the assembly line.

### RunnableParallel

```
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "summary": summary_chain,
    "quiz": quiz_chain
})
```

Runs multiple chains simultaneously.

> **Analogy:** Instead of one assembly line, this is **two assembly lines running side by side at the same time**, both starting from the same raw input — like two chefs cooking different dishes from the same grocery delivery, simultaneously instead of one after another.

### Useful Runnable methods

```
chain.invoke(data)
chain.batch(data)
chain.stream(data)
chain.with_retry()
chain.with_config(run_name="Demo")
```

> **Quick cheat sheet in plain English:**
> - `invoke()` → do it once
> - `batch()` → do it for a whole list at once
> - `stream()` → show me the answer as it's being written
> - `with_retry()` → "if it fails, try again automatically" (like a printer auto-retrying a jammed print job)
> - `with_config()` → attach a label/name so you can track this run later (like writing a name on a shipping box)

---

# 6. Output Parsers

Convert model output.

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
```

> **Analogy:** The model's raw output is like **handwriting** — messy and unstructured. A parser is like a **translator/typist** who converts that handwriting into a clean, usable format: plain text (`StrOutputParser`), a spreadsheet-style JSON object (`JsonOutputParser`), or a strict, validated form (`PydanticOutputParser`).

---

# 7. Tools

```
from langchain_core.tools import tool

@tool
def multiply(a:int,b:int):
    return a*b
```

Tools let an LLM call Python functions.

> **Analogy:** A tool is like giving the AI a **calculator or a phone to call a specialist**. The AI is smart at language but bad at precise math — so instead of guessing, it "calls" the `multiply` tool the same way you'd hand a tricky math problem to a calculator instead of doing it in your head.

---

# 8. Agents

```
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[multiply]
)
```

Agents decide which tool to use automatically.

> **Analogy:** If a chain is a fixed assembly line, an agent is more like a **smart manager/receptionist**. You give it a task, and it *decides on its own* which tool(s) to use and in what order — like a receptionist deciding whether to transfer your call to billing, tech support, or handle it themselves.

---

# 9. Memory

Older:

```
ConversationBufferMemory()
```

Recommended with LangGraph:

```
MemorySaver()
```

Memory preserves conversation context.

> **Analogy:** Without memory, every message to the model is like talking to someone with **short-term amnesia** — they forget you the second the conversation ends. Memory is a **notebook** that gets handed back to the model each time, so it can "remember" what was said earlier.

---

# 10. RAG

Pipeline:

```
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
```

Common syntax:

```
PyPDFLoader()
RecursiveCharacterTextSplitter()
GoogleGenerativeAIEmbeddings()
FAISS.from_documents()
retriever = vectorstore.as_retriever()
```

> **🧠 Simple explanation:** RAG (Retrieval-Augmented Generation) lets the model answer using *your own documents* instead of only what it memorized during training.
>
> **Analogy:** Think of it like an **open-book exam** instead of a closed-book one:
> - **Loader** = bringing your textbook into the exam room
> - **Splitter** = tearing the textbook into small index cards (so it's easier to search)
> - **Embeddings** = writing a "topic code" on each index card so similar cards can be found quickly
> - **Vector Store** = the filing cabinet where all the index cards live
> - **Retriever** = the librarian who pulls out the most relevant cards for your question
> - **LLM** = the student who reads those cards and writes the final answer

---

# 11. Structured Output

```
llm.with_structured_output(MySchema)
```

Returns structured objects instead of plain text.

> **Analogy:** Normally the model hands you a **free-written paragraph**. `with_structured_output` is like handing it a **fill-in form** instead ("Name: ___, Age: ___") so you always get back a clean, predictable object your code can use directly — no need to manually parse messy text.

---

# 12. LCEL

```
prompt | llm | parser
```

LCEL (LangChain Expression Language) connects components using the pipe
operator.

> **Analogy:** LCEL is like **Lego bricks that all snap together the same way**. The `|` (pipe) is the snap connector — no matter which pieces you're using (prompt, model, parser, custom function), they click together in a predictable, readable line.

---

# 13. Common Methods

| Method | Purpose | Plain-English Analogy |
|---|---|---|
| `invoke()` | One request | Send one text, get one reply |
| `batch()` | Many requests | Send a stack of texts, get a stack of replies |
| `stream()` | Streaming response | Watch the reply being typed live |
| `ainvoke()` | Async invoke | Send a text and keep doing other things while you wait for the reply |
| `with_retry()` | Retry failures | Auto-redial if the call drops |
| `with_config()` | Add metadata | Put a name tag on this specific run |
| `bind()` | Bind parameters | Pre-set some settings so you don't have to repeat them every time |

---

# Complete Example

```
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

> **Walkthrough in plain English:** "Hire" a model → write a script defining its role and what to ask it → connect script → model → text-cleaner into one assembly line → run the whole line once with `"LangChain"` filled into the blank → print the final, cleaned-up answer.

### Same example, using Euri instead of Google Gemini directly

```
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert trainer."),
    ("human","Explain {topic}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic":"LangChain"}))
```

> **Only two things changed:** the import (`langchain_openai` instead of `langchain_google_genai`) and how the model object is created (`ChatOpenAI(...)` reading Euri's key/URL from `.env` instead of `ChatGoogleGenerativeAI(...)`). The prompt, chain, `|` pipes, and `.invoke()` call are untouched — a nice real-world proof that Runnables really are interchangeable "shipping containers."

## Summary

| Concept | What it is | Analogy |
|---|---|---|
| Model | AI engine | The stove / the smart friend on the phone |
| Prompt | Instructions | The recipe card |
| Messages | Conversation | Labeled speakers in a group chat |
| Runnable | Building block | A standard shipping container |
| Chain | Connected runnables | A factory assembly line |
| Tool | Python function callable by LLM | A calculator the AI can pick up and use |
| Agent | Chooses tools | A receptionist deciding where to route your call |
| Memory | Remembers context | A notebook handed back each turn |
| RAG | Uses external knowledge | An open-book exam |
