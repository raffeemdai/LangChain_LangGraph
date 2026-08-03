# LangChain — Complete Study Guide
*(Merged & reorganized from "LangChain By Me.pdf" + "LangChain_Syntax_Theory_Guide.md" — grouped topic-wise for easy studying)*

---

## Table of Contents

1. [Overview — What & Why LangChain](#1-overview--what--why-langchain)
2. [Quick Reference — Core Concepts Map](#2-quick-reference--core-concepts-map)
3. [Models](#3-models)
4. [Prompts](#4-prompts)
5. [Messages](#5-messages)
6. [Chains & LCEL](#6-chains--lcel)
7. [Runnables (Deep Dive)](#7-runnables-deep-dive)
8. [Output Parsers](#8-output-parsers)
9. [Tools](#9-tools)
10. [Memory](#10-memory)
11. [Indexes & RAG](#11-indexes--rag)
12. [Agents](#12-agents)
13. [Structured Output](#13-structured-output)
14. [What You Can Build With LangChain](#14-what-you-can-build-with-langchain)
15. [Common Runnable Methods — Cheat Sheet](#15-common-runnable-methods--cheat-sheet)
16. [Complete Working Example](#16-complete-working-example)
17. [Prompt Engineering Deep Dive](#17-prompt-engineering-deep-dive)
18. [Advanced Reasoning Prompting Techniques](#18-advanced-reasoning-prompting-techniques)
19. [Prompt Design vs Prompt Engineering](#19-prompt-design-vs-prompt-engineering)
20. [Image Prompting](#20-image-prompting)
21. [Interview Q&A — Runnables](#21-interview-qa--runnables)
22. [Final Summary — One-Line Memory Hooks](#22-final-summary--one-line-memory-hooks)

---

## 1. Overview — What & Why LangChain

LangChain is an open-source framework for building applications powered by large language models (LLMs). It gives you modular, reusable building blocks instead of making you write everything from scratch for each provider.

**🍳 Analogy:** A raw foundation model (GPT, Gemini, Claude) is like a powerful industrial oven — a user just bakes bread in it. A *builder* designs the entire bakery: the recipes, the assembly line, the packaging. LangChain is that bakery blueprint.

### Key Benefits
1. Supports all the major LLMs (OpenAI, Anthropic, Google, HuggingFace, etc.)
2. Simplifies building LLM-based applications
3. Ready-made integrations for most external tools
4. Open source, free, actively developed
5. Covers all major GenAI use cases — chatbots, RAG, agents, automation

> 💡 One framework. Every model. Every use case.

### The Problem Without LangChain
Every provider (OpenAI, Anthropic, HuggingFace) has its own SDK, its own syntax, its own quirks. Switching models means rewriting your whole codebase.

### The LangChain Solution
A single, unified `model.invoke()` interface, regardless of provider:

```python
# OpenAI via LangChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4', temperature=0)
result = model.invoke("How divide the result by 1.5?")
print(result.content)
```

```python
# Anthropic Claude via LangChain — same interface, different model!
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
model = ChatAnthropic(model='claude-3-opus-20240229')
result = model.invoke("Hi who are you")
print(result.content)
```

> 💡 **Key idea:** Swap `ChatOpenAI` for `ChatAnthropic` and everything else stays the same. That's *model-agnostic* development.

---

## 2. Quick Reference — Core Concepts Map

LangChain is built around **6 core components** — think of them as organs of a body, each with a job, all working together:

```
LangChain
│
├── 1. Models
├── 2. Prompts
├── 3. Chains
├── 4. Memory
├── 5. Indexes (RAG)
└── 6. Agents
```

*(Under the hood, everything modern is built on Runnables + LCEL, and Messages/Output Parsers/Tools support the six components above.)*

### The 7 Core Concepts at a Glance

| Concept | What it does | When you need it |
|---|---|---|
| **Chain (LCEL)** | Composes LLM calls + data transforms using `\|` | Every LLM application |
| **Prompt Template** | Structures input sent to the LLM | Whenever you need consistent, reusable prompts |
| **Retriever** | Fetches relevant documents from a data source | RAG applications |
| **Agent** | LLM decides which tools to call, and in what order | Dynamic, multi-step workflows |
| **Tool** | A function the agent is allowed to call | When the agent needs to *do* something external |
| **Memory** | Stores conversation history | Chatbots, multi-turn interactions |
| **Output Parser** | Converts free LLM text into typed/structured data | Whenever downstream code needs a specific shape |

### Rule of Thumb — Layered Complexity

| Layer | Add This | What It Does | When You Need It | Why |
|---|---|---|---|---|
| **1. Baseline** | Chain (LCEL) + Prompt Template | Structures input, wires input → LLM → output into one flow | Every LLM app, no exceptions | Minimum viable pipeline |
| **2. Knowledge-Grounded** | + Retriever | Fetches relevant data from your own docs/DB so answers use real context | RAG apps — "chat with your PDF" | Prevents hallucination on your specific data |
| **3. Action-Taking** | + Tool + Agent | Tool = a callable function; Agent = decides which tool & when | Apps that must *do* something (send emails, query live systems) | A static Chain always runs the same sequence; an Agent adapts |

> **Rule of thumb (summary):** Every app needs **Chain + Prompt Template** → add **Retriever** for RAG → add **Tool + Agent** for actions, not just answers.

---

## 3. Models

**➡️ What it is:** The interface to talk to any AI model.

In LangChain, *models* are the interfaces through which you interact with AI models.

The evolution of language models: `NLP → NLU → LLMs → Internet scale` (billions of parameters, >100GB).

### Create a model
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0
)
```

### Invoke
```python
response = llm.invoke("Explain AI")
print(response.content)
```
**Simple explanation:** `invoke()` sends one request and returns one response.

### Batch
```python
responses = llm.batch(["What is AI?", "What is ML?"])
```
Runs many requests together.

### Stream
```python
for chunk in llm.stream("Explain LangChain"):
    print(chunk.content, end="")
```
Returns tokens gradually.

📌 **What Happens Next?** Once you have a model, you need to talk to it intelligently. That's where **Prompts** come in.

---

## 4. Prompts

**➡️ What it is:** Reusable, dynamic templates for talking to LLMs.

LLMs take **input → prompt → output**. A raw string works, but it's fragile. LangChain makes prompt management powerful, reusable, and structured.

**🍳 Analogy:** A raw string prompt is like shouting an order at a chef. A `PromptTemplate` is like handing them a proper recipe card — structured, consistent, and repeatable every time.

### 4.1 Dynamic & Reusable Prompts — `PromptTemplate`
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
)

prompt.invoke({"topic": "LangChain"})
```
`{topic}` is replaced at runtime.

### 4.2 Role-Based Prompts — `ChatPromptTemplate`
Give your LLM a persona — Doctor, Lawyer, Code Reviewer, etc.:
```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Hi you are an experienced {profession}"),
    ("user", "Tell me about {topic}"),
])

formatted_messages = chat_prompt.format_messages(
    profession="Doctor",
    topic="Viral Fever"
)
```

| Method | Use Case | Input |
|---|---|---|
| `from_template()` | Single-turn, plain-text prompt | One string with `{variables}` |
| `from_messages()` | Multi-turn chat prompt (system/user/assistant roles) | List of `(role, text)` tuples |

### 4.3 Few-Shot Prompting
Teach the model by example before asking your real question:
```python
examples = [
    {"input": "I was charged twice for my subscription this month.", "output": "Billing Issue"},
    {"input": "The app crashes every time I try to log in.", "output": "Technical Problem"},
    {"input": "Can you explain how to upgrade my plan?", "output": "General Inquiry"},
    {"input": "I need a refund for a payment I didn't authorize.", "output": "Billing Issue"},
]

example_template = """
Ticket: {input}
Category: {output}
"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    ),
    prefix="Classify the following customer support tickets into one of the categories: "
           "'Billing Issue', 'Technical Problem', or 'General Inquiry'.\n\n",
    suffix="Ticket: {user_input}\nCategory:",
    input_variables=["user_input"],
)

final_prompt = few_shot_prompt.format(user_input="My payment page keeps freezing.")
print(final_prompt)
```

### 4.4 Prompt Templates — Practical Setup (Euri / OpenAI-compatible API)

A Prompt Template is a reusable prompt with `{placeholders}` — the fixed structure is defined once, and dynamic values are filled in at runtime. This gives reusability, validation, and composability into chains, instead of scattering raw f-strings across a codebase.

**The two core template types:**

| Type | What It Produces | When To Use |
|---|---|---|
| `PromptTemplate` | One formatted string | Simple, single-turn prompts |
| `ChatPromptTemplate` | List of role-tagged messages | Chat models needing persona / system behavior |

**Setup — reusable client** (works with any OpenAI-compatible endpoint, e.g. the free Euri API):
```python
# euri_client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)
```

**`PromptTemplate` example:**
```python
from langchain_core.prompts import PromptTemplate
from euri_client import llm

template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in 2 sentences."
)
chain = template | llm
result = chain.invoke({"topic": "vector embeddings", "audience": "5-year-old"})
print(result.content)
```

**`ChatPromptTemplate` example:**
```python
from langchain_core.prompts import ChatPromptTemplate
from euri_client import llm

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a concise programming tutor who always gives a code example."),
    ("human", "Explain {concept} in Python."),
])
chain = chat_template | llm
result = chain.invoke({"concept": "decorators"})
print(result.content)
```

**Reusable prompt library pattern** (recommended for production):
```python
# prompts.py
from langchain_core.prompts import ChatPromptTemplate

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize input text in exactly 3 bullet points."),
    ("human", "{text}"),
])

# main.py
from prompts import SUMMARIZER_PROMPT
from euri_client import llm

chain = SUMMARIZER_PROMPT | llm
print(chain.invoke({"text": "LangChain is a framework for building LLM apps."}).content)
```

> **Interview one-liner:** A LangChain prompt template is a parameterized prompt — `PromptTemplate` for plain strings, `ChatPromptTemplate` for role-based chat messages. Fill placeholders at runtime and pipe the result into any LLM via LCEL (`template | llm`) — the LLM itself is swappable (OpenAI, Anthropic, Euri) because LangChain abstracts the provider behind a common interface.

📌 **What Happens Next?** A prompt is what you send. But real conversations go back and forth — that's where **Messages** come in.

---

## 5. Messages

**➡️ What it is:** LangChain's typed way of organizing a conversation.

A real conversation isn't just one prompt and one response — it can go back and forth, involve tool calls, etc. LangChain generalizes every piece of a conversation into a **Message** object.

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are helpful"),
    HumanMessage("Explain AI")
]

llm.invoke(messages)
```

| LangChain Class | Represents | Simple explanation |
|---|---|---|
| `SystemMessage` | System Prompt | Instructions given once, before the conversation starts |
| `HumanMessage` | User Prompt | What the human typed |
| `AIMessage` | LLM's response | What the model replied with |
| `ToolMessage` | Result of a tool call | Carries a tool's answer back to the LLM |

**Why bother with these classes instead of plain strings?** Different providers (OpenAI, Gemini, Anthropic) each expect conversation history in slightly different formats internally. LangChain's Message classes are a common, provider-agnostic way to represent a conversation — write it once, and LangChain translates it for whichever model you're using.

📌 **What Happens Next?** Now that a model, prompt, and messages exist, they need to be *connected* into a workflow — that's a **Chain**.

---

## 6. Chains & LCEL

**➡️ What it is:** Pipelines that connect LLMs with other components.

Chains = **Pipelines**. They are the heart of LangChain (hence the name). Instead of calling a model once, you chain multiple calls and operations together into a workflow.

```
Start → Step 1 → LLM → ... → Step N → End
```

**🍳 Analogy:** A single LLM call is like one chef making one dish. A Chain is the entire restaurant kitchen — prep cook → head chef → plating station — each step feeding the next.

### 6.1 The Core Idea
The most basic way to talk to an LLM:
```
prompt → llm → response
```
LangChain chains these pieces together with the pipe operator `|` — the same mental model as Unix pipes, where the output of one step feeds directly into the next:
```python
chain = prompt | llm
```
**Simple picture:** an assembly line. The prompt template fills in your question → passes it to the LLM → the LLM's answer comes out the other end.

**Adding structured output:** if you don't want plain text but a structured object, add another link:
```python
chain = prompt | llm | json_parser
```
The raw LLM response is just text. A parser at the end of the chain converts that text into something usable in code — like a Python dictionary — instead of you manually cleaning up the string.

```python
result = chain.invoke({"topic": "LangChain"})
```

### 6.2 What is LCEL?
The `|` notation is called **LCEL — LangChain Expression Language**. It's LangChain's syntax for composing components (prompts, models, parsers, retrievers, tools) into a pipeline, where every piece speaks the same interface (`invoke`, `stream`, `batch`) so they can be linked together with `|`.

**Simple explanation:** LCEL is a clean, readable way of saying "do this, then this, then this" — instead of writing manual glue code, you snap components together like Lego blocks.

### 6.3 Types of Chains

**1. Sequential Chains** — steps run one after another.
*Example: Translate a 1000-word English text → produce a 100-word Hindi summary.*
```
Create LLM Chain → Define prompt template → Build key-value list → Send to LLM Chain → Execute → Receive results
```

**2. Parallel Chains** — multiple LLM calls run simultaneously, results combined by an aggregator.
*Example: Generate a report from two expert LLMs simultaneously, then merge.*
```
In → [LLM Call 1, LLM Call 2, LLM Call 3] → Aggregator → Out
```

**3. Conditional Chains** — route based on output.
*Example: An AI feedback agent — good feedback → "Thank you!"; bad feedback → send an email alert.*

### Quick Recap (interview-ready)
- **LCEL (`|`)** chains components together — prompt → llm → parser — because they all implement the same `Runnable` interface (see next section).
- **Prompt** = input to the LLM: a system part (role/behavior) + a user part (the actual ask).
- **Messages** are LangChain's typed, provider-agnostic way of representing a conversation: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- Once basic invocation (`llm.invoke([HumanMessage(...)])`) works, LCEL (`prompt | llm | parser`) is the natural next step — reusable prompts, cleanly structured output.

📌 **What Happens Next?** Chains are stateless — they don't remember previous conversations. To build a real chatbot, you need **Memory**. But first, let's understand *why* `|` works at all — the **Runnable** interface.

---

## 7. Runnables (Deep Dive)

### 7.1 What is a Runnable?

A **Runnable** is just a rule LangChain follows:

> "If something is a Runnable, it can always be run using the same 3–4 commands — `invoke()`, `batch()`, `stream()` (and `ainvoke()`) — no matter what it actually does inside."

`Runnable` is the **base class** almost everything in LangChain inherits from. Anything that is a `Runnable` can be executed the exact same way, regardless of what it does internally. Everything in modern LangChain is a Runnable.

**Examples of Runnables:**
- A prompt template
- A chat model (OpenAI, Gemini, Euri, etc.)
- An output parser
- A retriever
- A custom Python function (once wrapped)
- An entire chain (`prompt | llm | parser` is itself one big Runnable)

**🍳 Analogy:** It's like a USB port. A mouse, a keyboard, and a USB drive are completely different devices — but they all plug into the same USB port and get recognized the same way. A Runnable is LangChain's "USB port": one common way to plug anything in.

### 7.2 The 4 Core Methods Every Runnable Shares

| Method | Purpose | Everyday example |
|---|---|---|
| `invoke()` | Run once, synchronously (single input → single output) | Ask one question, get one answer |
| `ainvoke()` | Async version of `invoke()` | Same, but non-blocking |
| `batch()` | Run multiple inputs in parallel | Ask a list of questions, get a list of answers |
| `stream()` | Yield the output incrementally as it's generated | Watch a reply being typed live |

**Why this matters:** because a prompt, a model, and a parser all support these same methods, you can chain them with `|`:
```python
chain = prompt | llm | parser
chain.invoke({"topic": "AI"})
```

**🍳 Analogy:** Think of Lego bricks — every brick has the same connector shape, so any brick snaps onto any other. Runnables all share the same "connector" (invoke/batch/stream), so any Runnable can snap onto any other Runnable using `|`.

### 7.3 Is It a Runnable? Quick Check

| Yes, it's a Runnable ✅ | No, it's not ❌ |
|---|---|
| Prompt Template | A `Document` object (just data) |
| Chat Model | A `TextSplitter` (one-time setup tool) |
| Output Parser | A `DocumentLoader` (one-time setup tool) |
| Retriever (`vectorstore.as_retriever()`) | `VectorStore` itself (before calling `.as_retriever()`) |
| A whole chain (`prompt \| llm \| parser`) | The final answer/response you get back |

**Simple rule:** If it does work *at the moment you ask a question* → it's a Runnable. If it's raw data, or a one-time setup step (loading files, splitting text, building a database) → it's not.

### 7.4 RunnableLambda
Wraps any plain Python function so it can join a chain.
```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())
uppercase.invoke("hello")
```
A normal Python function doesn't implement the Runnable interface on its own, so it can't be piped with `|` — `RunnableLambda` wraps it so it can be.

### 7.5 RunnableParallel
Runs multiple chains simultaneously, returning a dictionary of results.
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "summary": summary_chain,
    "quiz": quiz_chain
})
```

### 7.6 RunnablePassthrough
Forwards the input unchanged. Common in RAG chains, so the original user question reaches the final prompt alongside retrieved documents:
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```

---

## 8. Output Parsers

Convert raw model output (text) into a usable, structured shape.
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
```
A parser sits at the end of a chain and converts free text into something usable in code (e.g. a Python dict), instead of you manually cleaning up strings.

---

## 9. Tools

Tools let an LLM call Python functions to do real work (calculations, API calls, lookups).
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int):
    return a * b
```

---

## 10. Memory

**➡️ What it is:** Giving your LangChain app the ability to remember.

Without memory, every API call is stateless — like talking to someone with amnesia who forgets you the moment you stop speaking.

LangChain's memory components let you **persist and retrieve conversation history**, making chatbots feel natural and context-aware.

**🍳 Analogy:** Memory is like a notepad your assistant keeps on the desk. Every time you talk, they jot down what was said. Next time you walk in, they already know your name, your preferences, and what you discussed last week.

**Older approach:**
```python
ConversationBufferMemory()
```
**Recommended (with LangGraph):**
```python
MemorySaver()
```

📌 **What Happens Next?** Memory handles conversation history — but what if your app needs to search through *thousands* of your own documents? That's where **Indexes (RAG)** come in.

---

## 11. Indexes & RAG

**➡️ What it is:** Connecting your LLM to external knowledge — such as PDFs, websites, or databases.

This is the foundation of **RAG (Retrieval Augmented Generation)** — the most powerful pattern in modern AI apps.

**The Problem:** LLMs are trained on general internet data. They know nothing about *your* company's internal documents, *your* codebase, or *your* PDF notes.

**The RAG Solution:** Don't fine-tune the model — just give it your documents at query time.

### 11.1 The Full RAG Pipeline
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
LLM → Response
```

Flow: **User Documents** → Vector Embeddings → **Vector Database** → (a **Query** is also embedded) → Retrieved Context → **LLM** → **Response**.

**Common syntax:**
```python
PyPDFLoader()
RecursiveCharacterTextSplitter()
GoogleGenerativeAIEmbeddings()
FAISS.from_documents()
retriever = vectorstore.as_retriever()
```

### 11.2 Understanding Embeddings & Semantic Search

- **Traditional (keyword) search:** matches exact words/tokens, e.g. "Virat" → `[372, 961]` (just index positions). Elasticsearch/BM25-style.
- **Semantic search:** converts text into **vectors** — high-dimensional numbers that capture *meaning*, not just spelling.

| Search type | How it matches | "How many runs?" vs "total score of" |
|---|---|---|
| Keyword search | Matches exact words/tokens | Finds neither — no shared words |
| Semantic search | Matches meaning, via vector embeddings | Finds both — both live near the same region in embedding space |

> 💡 **Key insight:** "How many runs?" and "total score of" mean the same thing — semantic search finds both. Keyword search finds neither unless the exact word matches.

📌 **What Happens Next?** RAG gives your app access to documents. But what if you want your app to *act* — search the web, call an API, book a flight? That's what **Agents** do.

---

## 12. Agents

**➡️ What it is:** LLMs that can think, plan, and use tools.

Agents are AI systems that combine:
- 🧠 **Reasoning capabilities** (the LLM brain — chain of thought)
- 🔧 **Tools** (external actions it can call)

**🍳 Analogy:** A chatbot is like a very knowledgeable librarian — it can answer questions from memory. An AI Agent is like a personal assistant with a phone — it can answer questions *and* actually call the airline, book the hotel, and send you a confirmation.

### How Agents Work
```
User Question → LLM (Planner & Reasoner) → Parser → Tool → Observation → Output
                        ↑_____________________loop until final response____________|
```

### Step-by-Step Example
*"Can you multiply today's temperature in Delhi with 3?"*

1. Agent reasons — "I need Delhi's current temperature. I have a Weather API tool."
2. Agent calls **Weather API** → gets Delhi temp: **32°C**
3. Agent reasons — "Now I need to multiply 32 × 3. I have a Calculator tool."
4. Agent calls **Calculator** → 96
5. Agent returns: "Today's temperature in Delhi is 32°C. Multiplied by 3 = **96**." ✅

No hardcoding. No manual steps. Pure autonomous reasoning.

### Creating an Agent
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[multiply]
)
```
Agents decide which tool to use automatically.

📌 **What Happens Next?** Now that you know all 6 components, let's see what you can actually *build* with them!

---

## 13. Structured Output

```python
llm.with_structured_output(MySchema)
```
Returns structured objects instead of plain text — useful when downstream code needs a specific, typed shape rather than free-form text.

---

## 14. What You Can Build With LangChain

| Application Type | Real Example |
|---|---|
| Conversational Chatbots | Scalable customer support bot that handles 10,000 queries/day |
| AI Knowledge Assistants | Q&A over your company's 500-page internal docs |
| AI Agents | "Make my trip" — searches flights, books hotels, sends itinerary |
| Workflow Automation | Multi-step pipelines: scrape → summarize → email → log |
| Summarization/Research Helpers | ChatPDF, research paper summarizer, legal doc analyzer |

---

## 15. Common Runnable Methods — Cheat Sheet

| Method | Purpose |
|---|---|
| `invoke()` | One request |
| `batch()` | Many requests |
| `stream()` | Streaming response |
| `ainvoke()` | Async invoke |
| `with_retry()` | Retry failures |
| `with_config()` | Add metadata |
| `bind()` | Bind parameters |

Example usage:
```python
chain.invoke(data)
chain.batch(data)
chain.stream(data)
chain.with_retry()
chain.with_config(run_name="Demo")
```

---

## 16. Complete Working Example

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert trainer."),
    ("human", "Explain {topic}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "LangChain"}))
```

### Core Component Summary
- **Model** = AI engine
- **Prompt** = Instructions
- **Messages** = Conversation
- **Runnable** = Building block
- **Chain** = Connected runnables
- **Tool** = Python function callable by the LLM
- **Agent** = Chooses tools
- **Memory** = Remembers context
- **RAG** = Uses external knowledge

---

## 17. Prompt Engineering Deep Dive

### What is Prompt Engineering?

LLMs are powerful, but they don't automatically know what you want. Prompt engineering is the simplest way to control them — think of it as **the steering wheel for the LLM**. Small adjustments completely shift the direction of the output.

You're not changing weights (the learned parameters inside the model) — you're changing instructions, and that changes everything.

A good prompt helps the model:
- Think step-by-step
- Follow constraints
- Stay focused
- Avoid shallow answers

It's the fastest, lowest-effort way to get better results from any model.

### Role of Prompt Engineering
- A **prompt** is text containing instructions or requests sent to an LLM.
- **Prompt Engineering** is the process of designing and refining prompts that guide an AI model's behavior to produce the desired outputs.

### Basic Prompting Rules (5 rules for effective prompts)
1. Assign a Persona
2. Core task and subtasks
3. Provide Context
4. Explain expected output
5. Describe any potential limitations

**Example:**
> "You are an expert personal finance advisor. Explain different types of investment options available in India. Assume the audience has basic banking schemes knowledge. Provide the explanation and give me the tabular output of all options. Focus on the data from the last 5 years."

### 17.1 Zero-Shot Prompting

**Definition:** Ask the model to complete a task or answer a question without providing any examples or prior context. The model relies solely on its training to infer what's being asked.

**Example:**
> Prompt: "What is the capital of Japan?"
> Expected Output: "Tokyo"

### 17.2 Few-Shot Prompting

**Definition:** Give the model a few input-output examples (typically 2–5) directly in the prompt before asking it to do something similar. This is "in-context learning" — the model isn't retrained, but uses the examples to infer pattern, format, style, or reasoning.

**Example:**
> Prompt: "Here are some translations from English to French: English: 'Good morning.' → French: 'Bonjour.' | English: 'How are you?' → French: 'Comment ça va?' Translate this sentence to French: 'What is your name?'"
> Expected Output: "Quel est ton nom?"

### 17.3 Chain-of-Thought (CoT) Prompting

**Definition:** Ask the model to reason through a problem step by step, rather than jumping straight to a conclusion. Especially effective for arithmetic, logic, or multi-step reasoning — it mirrors how humans work through a problem and reduces errors from skipping steps. Can be triggered explicitly ("Let's think step by step") or via worked-out reasoning in few-shot examples.

**Example:**
> Prompt: "Solve this math problem step by step: What is 12 multiplied by 9?"
> Expected Output:
> - Step 1: "First, recall that 12 × 9 means adding 12, nine times."
> - Step 2: "So, 12+12+12+12+12+12+12+12+12 = 108."
> - Final answer: "12 multiplied by 9 is 108."

**Standard vs. CoT Prompting — Why It Matters**

| | Standard Prompting | Chain-of-Thought Prompting |
|---|---|---|
| Multi-step word problem | Model jumps to answer, often wrong (e.g. gives 27 instead of 9) | Model shows reasoning at each step and reaches the correct answer |

### Summary of the Three Core Techniques

| Technique | Description | Example |
|---|---|---|
| **Zero-shot** | Model does the task without examples; relies purely on pretrained knowledge | "What is the capital of France?" |
| **Few-shot** | A few examples guide the model before the new output is requested | "Translate 'Good morning' to French" (with examples) |
| **Chain-of-thought** | Model works through the task step by step — useful for logical/math problems | "Solve this math problem step by step." |

---

## 18. Advanced Reasoning Prompting Techniques

Three prompting techniques that significantly improve an LLM's reasoning ability:

### 18.1 Chain of Thought (CoT) — Recap
The simplest and most widely used technique. Instead of asking the LLM to jump straight to the answer, you nudge it to reason step by step. This often improves accuracy because the model can walk through its logic before committing to a final output.

*Example:* "If John has 3 apples and gives away 1, how many are left? Let's think step by step:"

For starters, *zero-shot prompting* means giving the model a task with no intermediate steps or examples — just ask and expect an answer.

**Definition:** Ask the model to reason step by step instead of jumping straight to the final answer. This breaks a complex problem into smaller, logical steps and often improves accuracy on math/logic tasks.

### 18.2 Self-Consistency (Majority Voting over CoT)

CoT is useful but not always consistent — prompting the same question multiple times can yield different answers depending on the temperature setting. **Self-consistency** embraces this variation: you ask the LLM to generate *multiple* reasoning paths, then select the most common final answer.

It's a simple idea: when in doubt, ask the model several times and trust the majority. This technique often leads to more robust results on ambiguous or complex tasks. However, it doesn't evaluate *how* the reasoning was done — only whether the final answer is consistent across paths.

### 18.3 Tree of Thoughts (ToT)

While Self-Consistency varies the *final answer*, Tree of Thoughts varies the *steps of reasoning* at each point and then picks the best overall path.

At every reasoning step, the model explores multiple possible directions. These branches form a tree, and a separate evaluation process judges which path seems most promising at each point. It evaluates which branches look promising, drops the weak ones, and can backtrack — helping it solve problems that need planning or trial-and-error better than a single straight chain of reasoning.

Think of it like a search algorithm over reasoning paths, trying to find the most logical and coherent trail to the solution. It's more compute-intensive, but in most cases significantly outperforms basic CoT.

### Quick Comparison

| Technique | Compute cost | Best for |
|---|---|---|
| Chain-of-Thought | Low | General multi-step reasoning |
| Self-Consistency | Medium–High (multiple samples) | Math/logic where errors are easy to catch by voting |
| Tree-of-Thought | High (search + evaluation) | Planning, puzzles, tasks needing exploration/backtracking |

### Easy Way to Remember (Analogy: reaching a destination)

- **🚶 Chain of Thought:** You follow *one road*, carefully, step by step.
  `Home → Road → Destination`

- **👥 Self-Consistency:** You ask *three friends* for directions. If they all suggest the same route, you trust it.
  `Friend A ⌐, Friend B ─┤ Same Route → Go, Friend C ⌐`

- **🌳 Tree of Thoughts:** You open Google Maps. It shows *multiple routes*, compares traffic, distance, and time, and helps you choose the best one.
  `Route A, Route B, Route C → Compare → Best Route`

### Summary
- **Chain of Thought (CoT):** Ask the model to reason step by step. Simple, fast, and effective for math, coding, and logic.
- **Self-Consistency:** Generate multiple independent reasoning paths and choose the answer that appears most consistently. Improves reliability on difficult reasoning tasks.
- **Tree of Thoughts (ToT):** Explore multiple solution branches, evaluate each one, and select the best path. Especially useful for planning, design, and strategy where there isn't one obvious solution.

---

## 19. Prompt Design vs Prompt Engineering

**Prompt Design** refers to creating and structuring the input given to a language model to achieve a desired output — it focuses on how you *phrase and format* your prompts to get the best results.

### Prompt Design Examples
**Goal:** crafting clear, simple, direct prompts to get the desired output.

- *Simple question:* "What is the capital of France?" → Clear, direct, no ambiguity. Model responds "Paris."
- *Task-oriented instruction:* "Write a short paragraph about the benefits of exercise." → Straightforward instruction, no complex reasoning required.

### Prompt Engineering Examples
**Goal:** using more sophisticated techniques to refine the prompt and control the model's behavior — few-shot, chain-of-thought, system instructions, temperature control.

- **Few-Shot:** "Translate the following English sentences to French. English: 'I am going to the store.' French: 'Je vais au magasin.' English: 'She is reading a book.' French: 'Elle lit un livre.' English: 'How are you today?'" — the model uses the pattern shown to translate the final sentence correctly.

- **Chain-of-Thought:** "Solve this math problem step by step: What is 256 divided by 8?" → Step 1: 256 ÷ 8 = 32. Final answer: 32. Ensures the model doesn't skip important steps.

- **System Prompting (behavior control):** "You are an expert nutritionist. Your task is to give healthy eating advice based on someone's age and lifestyle. Please respond in a friendly, supportive tone." → Sets both role and tone.

- **Temperature Control (creativity):** "Write a creative story about a time-traveling cat." with temperature = 0.8 (higher creativity) vs. 0.2 (more predictable/formal). Higher temperature → more varied, imaginative output; good for creative writing.

### Design vs Engineering — Same Task Compared

**Task:** Generate a business idea for a startup.

- **Prompt Design (Simple):** "Give me a business idea for a tech startup." → "A software platform that helps businesses manage their projects more efficiently." (Clear and direct, but may not be highly innovative or specific.)

- **Prompt Engineering (Few-Shot + Chain-of-Thought):** "Here are some examples of successful tech startup ideas: 1. A platform that connects freelancers with businesses. 2. An app that helps users learn new skills via micro-courses. Based on these, give me a new business idea in the tech industry that focuses on improving remote team collaboration." → "A cloud-based tool that uses AI to automate team task management and enhance communication for remote teams." (More tailored and focused, by combining examples, narrowed industry context, and guided reasoning.)

### Key Takeaways
- **Prompt Design** focuses on clarity and simplicity, ensuring the model understands exactly what you're asking for.
- **Prompt Engineering** is more advanced — involving optimization techniques like few-shot learning, chain-of-thought reasoning, and adjusting parameters like temperature to control output creativity.

In simple terms: Prompt Design is about creating clear and effective prompts; Prompt Engineering is about optimizing those prompts to guide the model's behavior and improve results.

---

## 20. Image Prompting

Image prompting is a form of prompt engineering used to guide an image-generation model to produce a specific image output.

An image prompt generally has three parts, following this pattern:

```
[main subject of the image, description of action, state, mood],
[art form, art style, artist references, if any],
[additional settings, such as lighting, colors, framing]
```

**Areas to think about, depending on the model:**
- **Art medium** — Drawing, painting, ink, origami, mosaic, pottery, realistic and glazed
- **Camera** — Lens and perspective, camera settings
- **Display and resolution** — 8K, 4K, HD, 256×256, 512×512, 768×768
- **Lighting** — Types, display
- **Material** — Metal, cloth, glass, wood, liquids

**Example prompt:**
> "Lion roaring in a dense forest, angry mood, other animals in the distant background, realistic, colorful, 8K"

---

## 21. Interview Q&A — Runnables

**Q1. What is a Runnable in LangChain?**
A Runnable is the base interface almost everything in LangChain implements. It guarantees any component — a prompt, model, parser, retriever, or full chain — can be run the same way, using `invoke()`, `batch()`, `stream()`. This shared interface lets different components connect predictably.

**Q2. What is LCEL, and how does the `|` operator actually work?**
LCEL (LangChain Expression Language) is the syntax for composing Runnables using `|`. It works because every component on either side of `|` implements the Runnable interface, so LangChain knows how to feed the output of one step as the input to the next. Since the resulting chain is itself a Runnable, you can call `.invoke()`, `.batch()`, or `.stream()` on the whole pipeline.

**Q3. Is a VectorStore a Runnable? Why or why not?**
No. A raw `VectorStore` (like Chroma or FAISS) is just a database of vectors, not a Runnable. You must call `.as_retriever()` on it first, which wraps it into a `Retriever` — that *is* a Runnable, because it exposes `.invoke(query)` and returns relevant documents.

**Q4. If you swap `ChatOpenAI` for `ChatAnthropic`, why doesn't the rest of your code need to change?**
Because all chat model classes inherit from the same base Runnable interface (through `BaseChatModel`). `.invoke()`, `.batch()`, `.stream()` behave identically regardless of provider. Only the model-creation line (API key, base URL, model name) changes — prompt, chain, and parsing logic stay the same.

**Q5. How do you use a plain Python function inside a LangChain chain?**
Wrap it with `RunnableLambda`. A normal function doesn't implement the Runnable interface on its own, so it can't be piped with `|`.
```python
from langchain_core.runnables import RunnableLambda
uppercase = RunnableLambda(lambda x: x.upper())
```

**Q6. How do you run two chains at the same time and combine their results?**
Use `RunnableParallel`. It runs multiple Runnables concurrently against the same input and returns a dictionary with each result labeled by key.
```python
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel({"summary": summary_chain, "quiz": quiz_chain})
```

**Q7. In a RAG chain, how does the original user question reach the final prompt alongside retrieved documents?**
Using `RunnablePassthrough()` — it forwards the input unchanged. The retriever runs on the question to get context, while `RunnablePassthrough()` carries the original question through untouched, so both are available together for the final prompt.
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```

**Q8. What's the difference between a Runnable and a non-Runnable component, in one sentence?**
A Runnable does work *at query time* (takes an input, produces an output, right when you ask something); a non-Runnable is either raw data (like a `Document`) or a one-time setup tool used *before* the chain runs (like a `TextSplitter` or `DocumentLoader`).

---

## 22. Final Summary — One-Line Memory Hooks

| Term | One-liner |
|---|---|
| **Runnable** | Same interface (invoke/batch/stream) for everything |
| `invoke()` | Run once |
| `batch()` | Run a list at once |
| `stream()` | Get the answer piece by piece |
| **LCEL (`\|`)** | Snaps Runnables together, like Lego |
| `RunnableLambda` | Wraps a plain function so it can join a chain |
| `RunnableParallel` | Runs multiple chains at the same time |
| `RunnablePassthrough` | Forwards input untouched |
| Not a Runnable | Raw data / one-time setup (`Document`, `Loader`, `Splitter`, `VectorStore`) |

> **If you remember only one thing:** Runnable = the shared interface that lets LangChain snap different pieces together with `|`, and run any of them the same way.

### Core Component Recap
- **Model** = AI engine
- **Prompt** = Instructions
- **Messages** = Conversation
- **Runnable** = Building block
- **Chain** = Connected runnables (via LCEL)
- **Tool** = Python function callable by the LLM
- **Agent** = Chooses tools, reasons, acts
- **Memory** = Remembers context
- **Indexes / RAG** = Uses external knowledge
- **Output Parser** = Shapes free text into structured data

---

*Study order suggestion: Sections 1→13 build the mental model of LangChain top to bottom (Overview → Models → Prompts → Messages → Chains/LCEL → Runnables → Parsers → Tools → Memory → RAG → Agents → Structured Output). Sections 17–20 go deep on prompt engineering as a standalone skill. Sections 21–22 are for quick interview revision.*
