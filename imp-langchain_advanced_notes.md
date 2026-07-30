# 🦜 LangChain — Advanced Notes (Runnable, RAG Pipeline & Production Theory)

> Part 2 of your LangChain notes. Combined from three GitHub study-note files
> (`RAG_BY_ME` repo by raffeemdai — chaining/prompts/messages, Runnable + GCP/RAG setup,
> and document loaders) plus the theory sections of TokenMix's
> ["LangChain Tutorial 2026"](https://tokenmix.ai/blog/langchain-tutorial-2026).
> This file goes **deeper** on how LCEL actually works under the hood, and walks
> the **full RAG pipeline** end-to-end (Document → Loader → Splitter → Embedding → Vector Store).
>
> Read alongside your first notes file (`langchain_complete_notes.md`) — that one
> covers Models, Agents, Structured Output, and Guardrails; this one focuses on
> **Runnable/LCEL internals, prompt/parser mechanics, and RAG document handling.**

---

## 0. Quick Reference — The 7 Core Concepts

Before the deep dive, the simplest possible map of what LangChain is built from:

| Concept | What it does | When you need it |
|---|---|---|
| **Chain (LCEL)** | Composes LLM calls with data transformations using `\|` | Every LLM application |
| **Prompt Template** | Structures input sent to the LLM | Whenever you need consistent, reusable prompts |
| **Retriever** | Fetches relevant documents from a data source | RAG applications |
| **Agent** | LLM decides which tools to call, and in what order | Dynamic, multi-step workflows |
| **Tool** | A function the agent is allowed to call | When the agent needs to *do* something external |
| **Memory** | Stores conversation history | Chatbots, multi-turn interactions |
| **Output Parser** | Converts free LLM text into typed/structured data | Whenever downstream code needs a specific shape |

**Rule of thumb:** every LLM app needs **Chain + Prompt Template**. Add a **Retriever** for RAG. Add **Tool + Agent** once the app needs to take actions, not just answer.

---

## 1. LangChain Basics: Chaining, Prompts, Messages

The three simplest ideas in LangChain — worth locking in before anything else.

### 1.1 Chaining — the core idea

The most basic way to talk to an LLM is:
```
prompt → llm → response
```
LangChain lets you **chain** these pieces together using the pipe operator `|` — the same mental model as Unix pipes, where the output of one step feeds straight into the next:
```python
chain = prompt | llm
```
**Simple picture:** think of an assembly line. The prompt template fills in your question → passes it to the LLM → the LLM's answer comes out the other end.

If you don't want a plain-text answer but a structured one, you add another link:
```python
chain = prompt | llm | json_parser
```
A raw LLM response is just text. A parser sits at the end of the chain and turns that text into something usable in code — like a Python dictionary — instead of you cleaning up the string by hand.

**What is LCEL?** This `|` notation is called **LCEL — LangChain Expression Language**. It's LangChain's syntax for composing components (prompts, models, parsers, retrievers, tools) into a pipeline, where every piece speaks the same interface (`invoke`, `stream`, `batch`) so they can be linked with `|`. It's a clean, readable way of saying "do this, then this, then this" — instead of writing manual glue code, you snap components together like Lego blocks.

### 1.2 Prompt — what you send to the model

A **prompt** is the input given to the LLM. Underneath, LLMs recognize different *types* of prompts, based on role:

| Type | Purpose | Simple picture |
|---|---|---|
| **System Prompt** | Defines the model's role/behavior | Like giving an employee their job description before they start work — "You are a customer support agent, only answer politely and from company docs." |
| **User Prompt** | The actual question being asked | This is you, typing your actual question. |

Sending this prompt to the model gets you a **response** back.

### 1.3 Messages — organizing a conversation

A real conversation isn't just one prompt and one response — it can go back and forth, involve tools, etc. LangChain generalizes every piece of a conversation into a **Message** object, with **4 core types**:

| LangChain Class | Represents | Simple picture |
|---|---|---|
| `SystemMessage` | System Prompt | The "instructions" given once, before the conversation starts |
| `HumanMessage` | User Prompt | What the human typed |
| `AIMessage` | LLM's response | What the model replied with |
| `ToolMessage` | Result of a tool call | If the LLM asked to use a tool (calculator, search...), this carries the tool's answer back to the LLM |

**Why bother with these classes instead of plain strings?** Different LLM providers (OpenAI, Gemini, Anthropic) each expect conversation history in slightly different formats behind the scenes. LangChain's Message classes are a common, provider-agnostic way to represent a conversation — write it once, and LangChain translates it for whichever model you're using.

### 1.4 Quick recap (interview-ready)

- **LCEL** (`|`) chains components together — prompt → llm → parser — because they all implement the same `Runnable` interface (next section).
- **Prompt** = input to the LLM: a **system** part (role/behavior) + a **user** part (the actual ask).
- **Messages** are LangChain's typed, provider-agnostic way of representing a conversation: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- Once basic invocation (`llm.invoke([HumanMessage(...)])`) works, LCEL (`prompt | llm | parser`) is the natural next step — reusable prompts, cleanly structured output.

---

## 2. `Runnable` — The Interface That Makes LCEL Possible

`Runnable` is the **base class** almost everything in LangChain inherits from. Anything that is a `Runnable` can be executed the exact same way, no matter what it actually does internally.

### The 4 core methods every Runnable shares
| Method | Purpose |
|---|---|
| `invoke()` | Run once, synchronously (single input → single output) |
| `ainvoke()` | Async version of `invoke()` |
| `batch()` | Run multiple inputs in parallel |
| `stream()` | Yield the output incrementally as it's generated |

**Why this matters:** once you know these 4 methods, you can run *anything* in LangChain — a model, a prompt, a parser, a retriever, or an entire multi-step chain — the same way.

### The inheritance chain (for chat models)
```
Runnable
   ↑
BaseChatModel        (adds chat-specific logic: messages, roles, generation config)
   ↑
ChatOpenAI / ChatGoogleGenerativeAI / ChatAnthropic / ...
```
Because of this chain, `llm.invoke([HumanMessage(content="Hi")])` works **identically** no matter which provider class you use — swap `ChatOpenAI` for `ChatAnthropic` and nothing else changes.

### Is it a Runnable? (cheat sheet)
| Component | Runnable? | Notes |
|---|---|---|
| `PromptTemplate` / `ChatPromptTemplate` | ✅ Yes | `.invoke(dict)` → formatted prompt |
| Chat model (`ChatOpenAI`, etc.) | ✅ Yes | `.invoke(prompt)` → `AIMessage` |
| Output Parser (`StrOutputParser`, etc.) | ✅ Yes | `.invoke(AIMessage)` → parsed value |
| Retriever (`vectorstore.as_retriever()`) | ✅ Yes | `.invoke(query)` → `list[Document]` |
| A plain Python function | ✅ if wrapped | Wrap with `RunnableLambda(fn)` to make it pipeable |
| `RunnablePassthrough()` | ✅ Yes | Passes input through unchanged (used to forward the original question alongside retrieved context) |
| A dict of Runnables `{"context": retriever, "question": RunnablePassthrough()}` | ✅ Yes | LCEL auto-converts this into a `RunnableParallel` |
| `VectorStore` itself (e.g. `Chroma`) | ❌ No | Call `.as_retriever()` first to get a Runnable wrapper |
| `Document` object | ❌ No | Just a data container, not a processing step |
| `TextSplitter` / `DocumentLoader` | ❌ No | One-time data-prep utilities, used *before* the chain, not inside it |
| The **response** you get back (e.g. an `AIMessage`) | ❌ No | It's *output data*, not something you invoke further |

**Mental model:**
> `Runnable` = things that **do work** (input → output)
> Non-Runnable = the **data** flowing through the pipeline (messages, responses, one-time setup steps)

**Rule of thumb:** anything in the **request/response flow at query time** (prompt → retriever → LLM → parser) is Runnable. Anything in **one-time data preparation** (loading files, splitting text, building the vector store) is not — you run those once, upfront.

---

## 3. LCEL Recap — Composing Runnables with `|`

```python
chain = prompt | llm | parser
```
**Rule:** every item in an LCEL expression must itself be a `Runnable`. Because they all share the same interface, the resulting `chain` is *also* a `Runnable` — so `.invoke()`, `.batch()`, and `.stream()` work on the whole pipeline, not just each piece.

Think of it as an assembly line: the prompt fills in your question → passes it to the LLM → the LLM's answer comes out the other end. Add a parser as one more link if you want structured output instead of a raw message:
```python
chain = prompt | llm | json_parser
```

---

## 4. Prompt vs Prompt Template — Why the Object Matters

| | Plain string / f-string | `PromptTemplate` / `ChatPromptTemplate` |
|---|---|---|
| What it is | Text you build yourself, e.g. `f"Answer: {q}"` | A LangChain object holding a template with placeholders |
| Is it a Runnable? | ❌ No — plain `str`, no `.invoke()` | ✅ Yes — has `.invoke()`, `.batch()`, `.stream()` |
| Usable in an LCEL chain (`\|`)? | No — you'd have to `.format()` manually first | Yes — slots directly into a chain |

**Why it matters for RAG:** because `PromptTemplate` is a Runnable, you can chain it directly with a retriever, an LLM, and a parser using `|` — no manual glue code. A plain string breaks that chain.

### `PromptTemplate` — simple, single-block prompts
```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "What is the capital of {country}? Also show their achievements in {sport}."
)
chain = prompt_template | llm
response = chain.invoke({"country": "India", "sport": "Hockey"})
```

### `ChatPromptTemplate` — role-based prompts (system/user/assistant)
Chat models expect a **list of role-tagged messages**, not one text block:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}."),
    ("user", "What is the significance of {topic} for {audience}?")
])
chain = prompt | llm
response = chain.invoke({"topic": "Sports", "audience": "students"})
```
- `system` sets persona/behavior for the whole conversation.
- `user` is the human's message.
- `assistant` can also be included — useful for few-shot examples or continuing a prior conversation.

### Two template syntaxes
| Format | Syntax | Best for |
|---|---|---|
| f-string | `{variable}` | Simple, flat substitution (default for plain LangChain code) |
| mustache | `{{variable}}` | Nested data, loops (`{{#items}}...{{/items}}`), conditionals — mostly relevant in LangSmith's playground/evaluators |

**Rule of thumb:** stick to f-string style unless you specifically need loops or nested-object access.

### Few-shot prompting
Give the model a handful of example input→output pairs before the real question, so it learns the *pattern* you want (e.g. classifying support tickets) rather than guessing from a bare instruction.

---

## 5. Output Parsers & Structured Output

**The problem:** an LLM's raw response is a `BaseMessage` (e.g. `AIMessage`) — free-form text with no guaranteed shape. Output parsers are `Runnable`s that sit at the *end* of a chain and convert that raw output into a predictable structure.

| Class | Purpose |
|---|---|
| `StrOutputParser` | Extracts just the plain string content |
| `JsonOutputParser` | Parses output into a JSON/Python `dict` |
| `XMLOutputParser` | Parses output as XML |
| `CommaSeparatedListOutputParser` | Splits a comma-separated string into a Python list |
| `PydanticToolsParser` / `JsonOutputKeyToolsParser` | Parses tool-call-style structured output |
| `BaseOutputParser` | Base class for writing your own custom parser |

### Example: `StrOutputParser`
```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"topic": "Sports", "audience": "students"})
# response is now a plain string, not an AIMessage object
```

### Example: `JsonOutputParser` + Pydantic schema
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    score: int = Field(description="1-10 rating")
    summary: str = Field(description="One sentence summary")

parser = JsonOutputParser(pydantic_object=ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze this product review.\n{format_instructions}"),
    ("human", "{review}")
])

chain = prompt | llm | parser
result = chain.invoke({
    "review": "Great product, fast shipping, exactly as described.",
    "format_instructions": parser.get_format_instructions()
})
# result -> {"sentiment": "positive", "score": 9, "summary": "..."}
```
`parser.get_format_instructions()` is injected into the prompt so the model knows exactly what JSON shape to produce.

### Example: `.with_structured_output()` (guaranteed schema, no manual parsing)
```python
from pydantic import BaseModel, Field

class SupportTicket(BaseModel):
    issue_category: str = Field(description="One of: billing, technical, shipping, other")
    resolved: bool = Field(description="Whether the issue appears resolved")
    next_action: str = Field(description="What should happen next, one short sentence")

chain = prompt | llm.with_structured_output(SupportTicket, method="json_schema")
response = chain.invoke({})
# response is a validated SupportTicket object: response.issue_category, response.resolved, ...
```

### Which one should you use?
| Approach | What you get | When to use |
|---|---|---|
| No parser | Raw `AIMessage` | Quick experiments |
| `StrOutputParser` | Plain string | You just want the text |
| `JsonOutputParser` | Python `dict` | Need JSON, don't need strict validation |
| `.with_structured_output(PydanticModel)` | Validated Pydantic object | Need guaranteed, typed fields for downstream code/APIs/DB |

---

## 6. RAG — The Document Pipeline (Load → Split → Embed → Store)

### 6.1 The `Document` object
Every piece of source data in a RAG pipeline becomes a `Document` — just a labeled box of text:
- `page_content` — the actual text.
- `metadata` — extra info about that text (source, page number, author, date...), heavily used to **filter** results before similarity search.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is a sample doc",
    metadata={"source": "https://example.com/1.html", "author": "khaja"}
)
```
In real projects you almost never build these by hand — you **load** them.

### 6.2 Document Loaders
A **Document Loader** reads from a source (file, folder, website, database) and converts it into one or more `Document` objects. `.load()` always returns a **list**, even for a single file — this consistent return type lets every loader plug into the rest of the pipeline the same way.

| Loader | Produces | Notes |
|---|---|---|
| `TextLoader` | 1 Document per file | Simplest — reads a `.txt` file whole |
| `PyPDFLoader` | 1 Document **per page** | `len(docs)` = number of pages |
| `CSVLoader` | 1 Document **per row** | Each row's columns become `key: value` text |
| `DirectoryLoader` | Combines many files | Applies a chosen `loader_cls` to every file matching a `glob` pattern |

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, DirectoryLoader

# Single files
TextLoader("./data/intro.txt", autodetect_encoding=True).load()
PyPDFLoader("./data/sample.pdf").load()
CSVLoader("./data/sample.csv").load()

# Whole folder (glob picks which files to include)
DirectoryLoader(path="./data/pdf", glob="**/*.pdf", loader_cls=PyPDFLoader).load()
```
`glob` patterns: `*.pdf` (all PDFs), `**/*.pdf` (PDFs including subfolders), `*.*` (everything). **Simple way to remember `DirectoryLoader`:** it isn't a file-reader itself — it's a *multiplier* that runs any single-file loader across a whole folder.

### 6.3 Text Splitters — breaking documents into chunks
A whole PDF page or text file is usually **too big** to embed or send in a prompt directly — so you split it into smaller **chunks**.

| Splitter | How it splits | Best for |
|---|---|---|
| `CharacterTextSplitter` | Fixed character count, one separator | Simple/uniform text |
| `RecursiveCharacterTextSplitter` | Tries separators in priority order (`\n\n` → `\n` → `" "` → `""`) | **Recommended default** — keeps chunks semantically coherent |
| `MarkdownHeaderTextSplitter` | Splits by `#`/`##`/`###` headings | Markdown docs — keeps each section together |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```
- `chunk_overlap` repeats a bit of the previous chunk's tail at the start of the next chunk, so a sentence split across a chunk boundary doesn't lose context.
- `.split_text(raw_string)` works the same way directly on plain strings (not `Document` objects).
- Shortcut: `loader.load_and_split(splitter)` loads and splits in one call.

### 6.4 Embeddings — text → meaning-vectors
An **embedding** converts a chunk of text into a numeric vector capturing its *meaning* — texts with similar meaning get numerically similar vectors. This is what makes semantic search work (e.g. searching "money" can retrieve a chunk about "refund").

```python
from langchain_openai import OpenAIEmbeddings
# or: from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("What is the refund policy?")   # one string
vectors = embeddings.embed_documents([doc.page_content for doc in chunks])  # many strings
```
- `embed_query()` — embeds a single string (the user's question).
- `embed_documents()` — embeds a batch (your chunks, before storing them).
- Different embedding models produce vectors of different dimension counts — don't mix models between indexing and querying.

**Embedding vs Indexing — don't confuse them:**
| | Embedding | Indexing |
|---|---|---|
| What it does | Converts text → vector | Stores/organizes vectors for fast search |
| Input | Raw text chunk | Vectors + metadata |
| Tool | Embedding model | Vector database |
| Analogy | Translating a book into a "meaning code" | Building a library catalog so you can find it fast |

### 6.5 Vector Store — storing & searching embeddings
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk so it survives restarts
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```
`Chroma.from_documents(...)` actually does **both** steps at once: it embeds every chunk *and* stores it in the index. Popular vector stores: **FAISS** (local, no server), **Chroma** (lightweight, great for prototyping), **Weaviate** (hybrid search), **Pinecone** (managed cloud), **pgvector** (Postgres extension).

### 6.6 The full RAG pipeline, end to end
```
Raw files (.txt/.pdf/.csv/folder)
       │
       ▼
Document Loader   ──►  Document objects (page_content + metadata)
       │
       ▼
Text Splitter     ──►  smaller chunks
       │
       ▼
Embedding model   ──►  numeric vectors
       │
       ▼
Vector Store      ──►  stores chunks + vectors, supports similarity search
       │
       ▼
Retriever.invoke(query) ──► most relevant chunks for the question
       │
       ▼
Prompt + LLM      ──►  final answer, grounded in retrieved context
```

### 6.7 Building the RAG chain (LCEL, all Runnable)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only this context. If it doesn't contain the "
               "answer, say you don't have that information.\n\nContext: {context}"),
    ("human", "{question}")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the refund policy?")
```
Because **every step** here is a `Runnable`, the whole `rag_chain` becomes one Runnable — you can `.invoke()`, `.stream()`, or `.batch()` the entire pipeline as a single unit. That composability through a shared interface is the whole point of LCEL.

---

## 7. Agents with Tool Use (Classic `AgentExecutor` Pattern)

An alternate, still-common way to build tool-using agents (an older, explicit pattern alongside the newer `create_agent`):

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72F, sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))   # use a safer evaluator in production

tools = [get_weather, calculate]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({
    "input": "What's 15% of 8500, and what's the weather in Tokyo?",
    "chat_history": []
})
print(result["output"])
```
The agent automatically decides to call both `calculate(...)` and `get_weather(...)` to answer both halves of the question — no manual orchestration needed.

---

## 8. Production Best Practices

Five habits worth adopting once you move past a prototype:

1. **Trace every chain with LangSmith.** Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to capture inputs, outputs, latency, tokens, and cost for every step.
2. **Add fallbacks for resilience.**
   ```python
   primary = ChatOpenAI(model="gpt-4o")
   fallback = ChatAnthropic(model="claude-sonnet-4-6-20250514")
   llm_with_fallback = primary.with_fallbacks([fallback])
   ```
3. **Cache repeated calls** (in-memory, Redis, or SQLite) to avoid paying for identical prompts twice:
   ```python
   from langchain_core.globals import set_llm_cache
   from langchain_community.cache import SQLiteCache
   set_llm_cache(SQLiteCache(database_path=".langchain.db"))
   ```
4. **Stream responses** for user-facing apps instead of waiting for the full completion:
   ```python
   for chunk in chain.stream({"question": "Explain RAG"}):
       print(chunk, end="", flush=True)
   ```
5. **Set retries and timeouts** to cap runaway cost/latency:
   ```python
   llm = ChatOpenAI(model="gpt-4o-mini", max_retries=2, request_timeout=30)
   ```

---

## 9. Choosing an Architecture for Your App

| Your application | Architecture | Key components |
|---|---|---|
| Simple chatbot | Chain + Memory | `ChatPromptTemplate` + LLM + conversation memory |
| Q&A over documents | RAG Chain | Loader + Splitter + Embeddings + VectorStore + Retriever |
| Customer support bot | Agent + RAG + Tools | Agent with a retriever tool + API tools + ticket creation |
| Data extraction | Chain + Output Parser | Prompt + LLM + `JsonOutputParser`/`.with_structured_output()` |
| Multi-step workflow | LangGraph | `StateGraph` with conditional edges and tool nodes |
| Content generation | Chain + Templates | Multiple prompt templates chained together |

**Starting-point advice:** a plain LCEL chain (`prompt | llm | parser`) handles most simple use cases. Add a **Retriever** when you need external knowledge (RAG). Add **Tool + Agent** when the app needs to take actions. Move to **LangGraph** only once you need branching, cycles, or multi-agent orchestration.

---

## 10. LangChain vs Alternatives (When to Reach for What)

| Feature | LangChain | LlamaIndex | Haystack | Direct Provider SDK |
|---|---|---|---|---|
| Primary focus | General LLM apps | RAG / data indexing | Production NLP | Raw API access |
| RAG support | Comprehensive | Best-in-class | Good | DIY |
| Agent framework | Strong (+ LangGraph) | Basic | Moderate | DIY |
| Provider support | Widest | Fewer | Fewest | One per SDK |
| Best for | Full-stack LLM apps, agents, multi-model | Data-intensive/deep RAG | Production NLP pipelines | Simple, single-model integrations |

**Skip LangChain when:** you only need simple, single-model API calls, want maximum control over every HTTP request, or the abstraction adds more complexity than it removes.

---

## 11. Hands-On Walkthrough — Your First LangChain Project (`hello_llms`)

A full, step-by-step setup using `uv` (a fast Python package/project manager — an alternative to pip + venv):

**Step 1 — scaffold the project**
```bash
mkdir hello_llms
cd hello_llms
uv init .
```
This creates `pyproject.toml`, `.python-version`, etc.

**Step 2 — add LangChain**
```bash
uv add langchain
```
(`uv add` behaves like `pip install`, but also records the dependency in `pyproject.toml`.)

**Step 3 — get a Gemini API key**
Go to Google AI Studio and generate a key — this is how your code authenticates against Google's models.

**Step 4 — add `python-dotenv`**
```bash
uv add python-dotenv
```
Lets your code load secrets from a `.env` file instead of hardcoding them.

**Step 5 — create a `.env` file**
```
GEMINI_API_KEY='paste your api key here'
```
Keeps the key out of source code and out of anything you might commit to GitHub.

**Step 6 — add the Gemini integration package**
```bash
uv add langchain-google-genai
```

**Step 7 — write the first working code**

Resulting `pyproject.toml` dependencies looked like:
```toml
[project]
name = "hello-llms"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "langchain>=1.3.11",
    "langchain-google-genai>=4.2.6",
    "python-dotenv>=1.2.2",
]
```

`main.py`:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from .env into the environment

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
prompt = HumanMessage(content='What is capital of France?')
response = llm.invoke([prompt])
response.pretty_print()
```
Run it with:
```bash
uv run main.py
```

Line by line: `load_dotenv()` loads the key → `ChatGoogleGenerativeAI(...)` creates the model object pointed at a specific Gemini variant → `HumanMessage(...)` wraps your question in LangChain's standard message format → `.invoke([prompt])` sends it and returns an `AIMessage` → `.pretty_print()` nicely formats the answer to the console.

**Step 8 — upgrade to LCEL chaining**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful geography assistant."),
    ("user", "What is the capital of {country}?")
])

chain = prompt_template | llm | StrOutputParser()
result = chain.invoke({"country": "France"})
print(result)
```
Instead of manually building a `HumanMessage` and calling `.invoke()` yourself, `ChatPromptTemplate` fills in the blank, pipes it to the model, and `StrOutputParser()` extracts clean text — one readable `prompt | llm | parser` line.

**Working with Jupyter notebooks (`.ipynb`) instead of `.py` files:**
1. Create a file with the `.ipynb` extension.
2. Install the Jupyter kernel dependency: `uv add ipykernel`
3. Select your project's `.venv` as the kernel inside VS Code before running cells.

---

## 12. Setting Up GCP / Vertex AI Instead of a Direct API Key

For enterprise use (billing, IAM, org-level control), you can route Gemini calls through **Google Cloud Platform / Vertex AI** instead of a plain Google AI Studio key.

**Steps:**
1. **Install the Google Cloud SDK (`gcloud`)** — needed to authenticate your machine with GCP.
2. **Open the GCP Console**, note your **Project ID**, and check the Agent Platform overview.
3. **Enable required APIs** if prompted (Vertex AI / Generative AI APIs).
4. **Authenticate from the terminal:**
   ```bash
   gcloud init
   gcloud auth application-default login
   ```
   - `gcloud init` — selects/sets up your GCP project and default config.
   - `gcloud auth application-default login` — creates local "Application Default Credentials" so SDKs like LangChain can authenticate as you, without hardcoding keys.
5. **Set up the project with `uv`:**
   ```bash
   mkdir rag_learning_gcp
   cd rag_learning_gcp
   uv init .
   uv add langchain python-dotenv langchain-google-genai ipykernel
   ```
6. Select the new interpreter in VS Code (`Ctrl+Shift+P` → *Python: Select Interpreter*).
7. **Create a `.env` file:**
   ```
   GOOGLE_CLOUD_PROJECT=''
   GOOGLE_CLOUD_LOCATION='us-central1'
   GOOGLE_GENAI_USE_VERTEXAI=true
   ```
   - `GOOGLE_CLOUD_PROJECT` — your GCP project ID.
   - `GOOGLE_CLOUD_LOCATION` — the Vertex AI region (e.g. `us-central1`).
   - `GOOGLE_GENAI_USE_VERTEXAI=true` — tells `langchain-google-genai` to route through Vertex AI, authenticating via your `gcloud` login instead of an API key.

**Suggested project layout:**
```
rag_learning_gcp/
├── .python-version
├── pyproject.toml
├── main.py
├── utils.py                                  # reusable model-builder helper
└── prompts_models_structured_output.ipynb
```

`utils.py` — keep model-creation logic in one place instead of repeating it everywhere:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_model_from_gcp(model_name: str = "gemini-2.5-flash-lite") -> ChatGoogleGenerativeAI:
    """Returns a chat model configured to use GCP/Vertex AI."""
    return ChatGoogleGenerativeAI(model=model_name)
```

`main.py`:
```python
from utils import get_model_from_gcp

def main():
    llm = get_model_from_gcp()
    result = llm.invoke("What is 2+2?")
    result.pretty_print()

if __name__ == "__main__":
    main()
```

---

## 13. Extended Document-Loader Comparison Table

| Aspect | Document Loader (general) | TextLoader | PyPDFLoader | CSVLoader | DirectoryLoader |
|---|---|---|---|---|---|
| What it is | Umbrella category — any class that reads a source into `Document`s | Loader for plain `.txt` files | Loader for `.pdf` files | Loader for `.csv` files | Wrapper applying another loader to many files |
| Package | `langchain_community.document_loaders` | same | same | same | same |
| Input source | Files, URLs, DBs, APIs (varies) | A single `.txt` path | A single `.pdf` path | A single `.csv` path | A folder path + glob pattern |
| Granularity | Depends on the loader | ~1 Document for the whole file | 1 Document per page | 1 Document per row | Combines results from every matched file |
| Typical metadata | Varies | `source` (file path) | `source`, `page` | `source`, `row` | Whatever the inner `loader_cls` adds, per file |
| Extra dependency? | Sometimes | No | Yes (PDF parsing lib) | No | Depends on `loader_cls` |
| Key parameter(s) | — | `file_path`, `autodetect_encoding` | `file_path` | `file_path` | `path`, `glob`, `loader_cls` |
| Return type of `.load()` | `list[Document]` | length 1 | length = # pages | length = # rows | combined across all matched files |

**Simple way to remember it:** `DirectoryLoader` isn't a file-reader itself — it's a *multiplier* that takes any single-file loader and runs it across every matching file in a folder, merging all results into one list.

---

## 14. Recap — What's New in This File vs Part 1

| Topic | Covered here | Covered in Part 1 (`langchain_complete_notes.md`) |
|---|---|---|
| Runnable interface & LCEL internals | ✅ Deep dive | Brief overview |
| Prompt Template vs plain string | ✅ Detailed | Brief |
| Output parsers (`Str`/`Json`/structured) | ✅ Detailed with code | Pydantic/TypedDict for agents |
| Document Loaders, Splitters, Embeddings, Vector Store | ✅ Full pipeline | High-level RAG concept only |
| Agents (`create_agent`, tool loop, middleware, guardrails) | Brief mention | ✅ Full coverage |
| Production practices (tracing, fallback, caching, streaming) | ✅ | Not covered |
| Architecture decision table / LangChain vs alternatives | ✅ | Not covered |

**Bottom line:** start simple with a plain LCEL chain — it handles most use cases. Add a Retriever when you need your own documents (RAG). Add Tools + an Agent when the app needs to *act*. Reach for LangGraph only once you need stateful, branching, multi-agent workflows.
