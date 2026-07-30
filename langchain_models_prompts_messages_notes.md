# LangChain v1.x — Module 2, 3, 4
## Models · Prompts · Messages
### (with free Euri API examples)

> Based on the `LangChain_LangGraph` roadmap by raffeemdai — Modules 2–4 of the
> LangChain learning path: **Models → Prompts → Messages**, the three
> foundational building blocks that everything else in LangChain (Chains,
> Runnables, Agents, RAG) sits on top of.

---

## 0. Why these three modules come first

```
                         LangChain
                              |
        -----------------------------------------
        |                  |                    |
      Models             Prompts             Messages
        |
        |
      Chains
        |
        |
    Runnables
```

Every LangChain application is, at its core, a pipeline that:
1. Builds a **Prompt** (the instructions/question you want answered)
2. Wraps it into **Messages** (the structured, role-based format LLMs expect)
3. Sends it to a **Model** (the LLM that generates a response)

Get these three right, and Chains/Agents/RAG are just composition on top of
them.

---

## 0.1 Setting up Euri (free API) once, for all examples

[Euri](https://euron.one/euri) gives you a **free tier (≈200,000 tokens/day,
no credit card)** across many open models (Gemini, Llama, GPT-family mirrors,
etc.) through a single **OpenAI-compatible** endpoint. Because it's
OpenAI-compatible, it plugs straight into LangChain's `ChatOpenAI` class — no
special LangChain integration needed.

**1. Get a free key:** sign up at https://euron.one/euri and copy your API key.

**2. Install packages:**

```bash
pip install langchain langchain-openai langchain-core python-dotenv
```

**3. Create a `.env` file** (never hardcode keys in code):

```bash
EURI_API_KEY=your_api_key_here
EURI_BASE_URL=https://api.euron.one/api/v1/euri
EURI_MODEL=gemini-2.5-flash
```

**4. Load it in every script:**

```python
import os
from dotenv import load_dotenv

load_dotenv()

EURI_API_KEY  = os.getenv("EURI_API_KEY")
EURI_BASE_URL = os.getenv("EURI_BASE_URL")
EURI_MODEL    = os.getenv("EURI_MODEL", "gemini-2.5-flash")
```

Free models commonly available on Euri (check the dashboard for the current
list, it changes): `gemini-2.5-flash`, `gpt-4.1-nano`, `llama-3.3-70b`,
`deepseek-r1-distill-llama-70b`. We'll use `gemini-2.5-flash` throughout since
it's fast and free.

---

# MODULE 2 — Models

## 2.1 Theory: what "Models" means in LangChain

In LangChain, a **Model** is the object that wraps an LLM provider's API and
gives you a **uniform interface** — `.invoke()`, `.stream()`, `.batch()` —
regardless of which company built the underlying model. This is the whole
point of the abstraction: swap Euri for OpenAI, Anthropic, or a local Ollama
model, and 95% of your code doesn't change.

LangChain divides models into two families:

| Type | Class | Input | Output | Use case |
|---|---|---|---|---|
| **LLM** (legacy, text-completion style) | `LLM` / `BaseLLM` | raw string | raw string | older, mostly deprecated for chat use cases |
| **Chat Model** (modern, message-based) | `BaseChatModel` (e.g. `ChatOpenAI`) | list of **Messages** | an `AIMessage` object | virtually all modern LLMs (GPT, Gemini, Claude, Llama-chat) |

Since LangChain v1.x standardizes around **chat models**, this module focuses
on `ChatOpenAI`-style models. Almost every model provider you'll use today
(Euri included) is a **chat model**.

### 2.1.1 Key model parameters

| Parameter | What it controls |
|---|---|
| `model` | which underlying model to call (e.g. `gemini-2.5-flash`) |
| `temperature` | 0 = deterministic/focused, 1+ = creative/random |
| `max_tokens` | cap on response length |
| `top_p` | nucleus sampling — alternative to temperature |
| `streaming` | whether to stream tokens as they're generated |
| `api_key` / `base_url` | credentials + endpoint (this is how we redirect to Euri) |

### 2.1.2 The Runnable interface (why models feel consistent everywhere)

Every LangChain chat model implements the **Runnable** protocol, so all
models support the same core methods:

- `model.invoke(input)` → single call, returns one response
- `model.stream(input)` → yields chunks as they arrive
- `model.batch([input1, input2, ...])` → parallel calls over multiple inputs
- `model.ainvoke(input)` → async version of invoke

This consistency is what lets you later drop a model into a Chain, an Agent,
or a RAG pipeline without rewriting anything.

---

## 2.2 Code: connecting to Euri as a Chat Model

### Example 1 — Basic invoke

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),   # https://api.euron.one/api/v1/euri
    temperature=0.7,
)

response = llm.invoke("What is LangChain in one sentence?")
print(response.content)
```

Output (`response`) is an `AIMessage` object, not a plain string — that's
covered fully in Module 4 (Messages).

### Example 2 — Streaming a response

```python
for chunk in llm.stream("List 3 benefits of using LangChain."):
    print(chunk.content, end="", flush=True)
```

Streaming is essential for chat UIs — the user sees tokens appear live instead
of waiting for the full response.

### Example 3 — Batch calls (parallel requests)

```python
questions = [
    "What is a vector database?",
    "What is prompt engineering?",
    "What is RAG?",
]

responses = llm.batch(questions)

for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r.content}\n")
```

`.batch()` fires requests concurrently — much faster than looping `.invoke()`
manually.

### Example 4 — Controlling determinism with temperature

```python
creative_llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
    temperature=1.2,   # more random/creative
)

precise_llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
    temperature=0.0,   # deterministic, best for factual/coding tasks
)

print(creative_llm.invoke("Write a tagline for a coffee shop.").content)
print(precise_llm.invoke("What is 15 * 12?").content)
```

### Example 5 — Swapping providers with zero pipeline changes

This is the payoff of the Model abstraction — the exact same downstream code
works no matter which model object you plug in:

```python
def summarize(llm, text: str) -> str:
    return llm.invoke(f"Summarize this in one line:\n{text}").content

euri_llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

print(summarize(euri_llm, "LangChain is a framework for building LLM apps."))
# Later, swap in a different provider — `summarize()` never changes:
# openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
# print(summarize(openai_llm, "..."))
```

### Example 6 — Raw HTTP call (no LangChain, for understanding what's under the hood)

Useful to see that Euri is just a plain REST API — LangChain is only a
convenience wrapper on top:

```python
import requests

url = "https://api.euron.one/api/v1/euri/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('EURI_API_KEY')}",
}
payload = {
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gemini-2.5-flash",
}

resp = requests.post(url, headers=headers, json=payload)
print(resp.json()["choices"][0]["message"]["content"])
```

> **Key takeaway:** `ChatOpenAI(base_url=...)` works with Euri because Euri
> implements the OpenAI `/chat/completions` schema. Any framework that
> supports a custom `base_url` (LangChain, LlamaIndex, CrewAI, raw `openai`
> SDK) works with Euri out of the box.

---

# MODULE 3 — Prompts

## 3.1 Theory: what a Prompt is

A **Prompt** is the text (or template) you send to guide the model's output.
Writing prompts by hand as raw strings works for a demo, but breaks down fast
in real applications because you usually need to:

- Reuse the same instruction template with different variables
- Combine multiple pieces of context (system rules, user question, retrieved
  documents, chat history) into one structured input
- Validate that required inputs are present before calling an expensive API

LangChain solves this with **Prompt Templates** — objects that hold a
template string with `{placeholders}` and render it with real values at
runtime.

### 3.1.1 Core prompt classes

| Class | Purpose |
|---|---|
| `PromptTemplate` | Single string template → produces a plain string. Used for older LLM-style models. |
| `ChatPromptTemplate` | Template made of multiple **role-tagged** messages (system/human/ai) → produces a list of Messages. This is what you use with chat models (99% of the time). |
| `FewShotPromptTemplate` | Injects example input/output pairs into the prompt before the real question (few-shot learning). |
| `MessagesPlaceholder` | A slot inside a `ChatPromptTemplate` reserved for dynamically-inserted message history. |

### 3.1.2 Why templates instead of f-strings?

You *could* do `f"Translate this to French: {text}"` — but templates give you:

1. **Input validation** — LangChain errors clearly if a required variable is
   missing, instead of silently sending `{text}` literally to the model.
2. **Composability** — templates are `Runnable`s, so they chain directly into
   models with the `|` (pipe) operator (LCEL — covered in Module 5).
3. **Reusability** — one template, many calls, no repeated boilerplate.
4. **Separation of concerns** — prompt engineering lives in one place, not
   scattered through business logic.

---

## 3.2 Code: Prompt Templates with Euri

### Example 1 — Basic `PromptTemplate`

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in 2 sentences."
)

filled_prompt = template.invoke({"topic": "vector embeddings", "audience": "5-year-old"})
print(filled_prompt.text)
# -> "Explain vector embeddings to a 5-year-old in 2 sentences."
```

### Example 2 — Feeding a `PromptTemplate` into the Euri model

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in 2 sentences."
)

chain = template | llm    # LCEL: pipe the prompt straight into the model
result = chain.invoke({"topic": "vector embeddings", "audience": "5-year-old"})
print(result.content)
```

### Example 3 — `ChatPromptTemplate` (the one you'll use most)

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a concise programming tutor who always gives a code example."),
    ("human", "Explain {concept} in Python."),
])

messages = chat_template.invoke({"concept": "list comprehensions"})
print(messages.to_messages())
# -> [SystemMessage(...), HumanMessage(...)]
```

### Example 4 — Full chain: `ChatPromptTemplate` + Euri model

```python
chain = chat_template | llm

response = chain.invoke({"concept": "decorators"})
print(response.content)
```

### Example 5 — Few-shot prompting (teaching by example)

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"],
)

final_prompt = few_shot_prompt.invoke({"word": "generous"}).to_string()
print(llm.invoke(final_prompt).content)   # -> "stingy" (approximately)
```

### Example 6 — Reusable prompt library pattern (real-world structure)

```python
# prompts.py  (keep all prompt templates in one place)
from langchain_core.prompts import ChatPromptTemplate

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize input text in exactly 3 bullet points."),
    ("human", "{text}"),
])

TRANSLATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Only output the translation, nothing else."),
    ("human", "Translate to {language}:\n\n{text}"),
])

# main.py
from prompts import SUMMARIZER_PROMPT, TRANSLATOR_PROMPT

summary_chain = SUMMARIZER_PROMPT | llm
translation_chain = TRANSLATOR_PROMPT | llm

print(summary_chain.invoke({"text": "LangChain is a framework..."}).content)
print(translation_chain.invoke({"text": "Good morning!", "language": "Spanish"}).content)
```

---

# MODULE 4 — Messages

## 4.1 Theory: what Messages are and why they exist

Modern chat models don't take a single string — they take a **conversation**:
a list of role-tagged messages. This mirrors how chat APIs like OpenAI's or
Anthropic's actually work under the hood, and it's what lets a model
distinguish "instructions from the developer" vs. "what the user said" vs.
"what the model said earlier."

### 4.1.1 The message types

| Class | Role | Purpose |
|---|---|---|
| `SystemMessage` | `system` | Sets behavior/persona/rules for the whole conversation. Sent once, usually first. |
| `HumanMessage` | `user` | What the human/user says. |
| `AIMessage` | `assistant` | What the model generated. Returned by `.invoke()`. |
| `ToolMessage` | `tool` | The result of a tool/function call, sent back to the model (used heavily in Agents, Module 15). |
| `FunctionMessage` | `function` | Legacy predecessor to `ToolMessage` (mostly deprecated). |

All of these inherit from `BaseMessage` and share common fields:

```python
BaseMessage:
    content: str              # the text content
    additional_kwargs: dict   # provider-specific extras (e.g. tool_calls)
    response_metadata: dict   # token usage, finish_reason, model name, etc.
    id: str                   # optional unique id
```

### 4.1.2 Why not just pass strings?

Because a conversation has structure the model needs to reason about:

```
[system]  You are a helpful, terse assistant.
[human]   What's the capital of France?
[ai]      Paris.
[human]   What's its population?
```

Without roles, the model can't tell where one turn ends and another begins,
and it can't distinguish rules from questions. Messages are what make
**multi-turn memory** and **role separation** possible.

---

## 4.2 Code: working with Messages and Euri

### Example 1 — Building a message list manually

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

messages = [
    SystemMessage(content="You are a helpful assistant who replies in at most 2 lines."),
    HumanMessage(content="Why is the sky blue?"),
]

response = llm.invoke(messages)
print(response.content)
print(type(response))          # <class 'langchain_core.messages.ai.AIMessage'>
```

### Example 2 — Inspecting the `AIMessage` object

```python
response = llm.invoke(messages)

print(response.content)              # the actual text
print(response.response_metadata)    # token usage, model name, finish_reason
print(response.id)                   # unique message id
```

### Example 3 — Multi-turn conversation (manual memory)

```python
from langchain_core.messages import AIMessage

conversation = [
    SystemMessage(content="You are a friendly math tutor."),
    HumanMessage(content="What is 12 * 8?"),
]

first_reply = llm.invoke(conversation)
conversation.append(first_reply)                       # remember the AI's reply
conversation.append(HumanMessage(content="Now divide that by 4."))

second_reply = llm.invoke(conversation)
print(second_reply.content)   # correctly uses context from turn 1 -> 24
```

This is the manual foundation of what `ChatMessageHistory` and LangGraph
checkpoints automate later (Module 19 — Chat History).

### Example 4 — `ToolMessage` preview (used fully in Agents, Module 14–15)

```python
from langchain_core.messages import ToolMessage

# After a model requests a tool call (e.g. get_weather("Paris")),
# you run the real function yourself, then feed the result back like this:
tool_result = ToolMessage(
    content="18°C, partly cloudy",
    tool_call_id="call_abc123",   # must match the id the model generated
)
```

### Example 5 — Converting a `ChatPromptTemplate` into raw messages

Bridges Module 3 and Module 4 — this is literally what happens internally
before every model call:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a travel guide."),
    ("human", "Suggest 3 things to do in {city}."),
])

message_list = template.invoke({"city": "Kyoto"}).to_messages()
for m in message_list:
    print(type(m).__name__, "->", m.content)

# Feed straight into the model:
response = llm.invoke(message_list)
print(response.content)
```

### Example 6 — Streaming messages token by token

```python
full_reply = ""
for chunk in llm.stream(conversation):
    print(chunk.content, end="", flush=True)
    full_reply += chunk.content

# You still get a proper AIMessage-like chunk stream, aggregable with:
from langchain_core.messages import AIMessageChunk
```

### Example 7 — Putting it all together: Models + Prompts + Messages

A small, complete mini-app tying all three modules together:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1) MODEL — connect to Euri's free model
llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
    temperature=0.5,
)

# 2) PROMPT — reusable template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are CodeBuddy, a friendly Python mentor. Keep answers under 4 lines."),
    ("human", "{question}"),
])

# 3) MESSAGES — chat loop with memory
history = []

def ask(question: str) -> str:
    filled = chat_template.invoke({"question": question})
    history.extend(filled.to_messages()[-1:])   # add the new human turn
    full_context = filled.to_messages()[:-1] + history  # system + running history

    reply: AIMessage = llm.invoke(full_context)
    history.append(reply)
    return reply.content

print(ask("What does the `yield` keyword do?"))
print(ask("Give me a one-line example of it."))
```

---

## 5. Quick-reference cheat sheet

```python
# MODEL setup (Euri, OpenAI-compatible)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gemini-2.5-flash", api_key=EURI_API_KEY, base_url=EURI_BASE_URL)

# PROMPT
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{q}")])

# MESSAGES
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Combine (LCEL)
chain = prompt | llm
result = chain.invoke({"q": "your question"})
print(result.content)
```

| Concept | One-line memory hook |
|---|---|
| Model | The engine — swap providers via `base_url`, keep code identical |
| Prompt | The recipe — a reusable template with `{placeholders}` |
| Message | The ingredient — role-tagged text (`system`/`human`/`ai`/`tool`) the model actually consumes |

---

## 6. Recommended blogs & further reading

Curated, high-quality reads for each module — official docs first, then
community blogs that go deeper on specific angles.

### Models
- **LangChain official docs — Chat Models concept guide**: https://docs.langchain.com/oss/python/langchain/models — the authoritative reference for model init, parameters, and provider-agnostic patterns (`init_chat_model`, middleware, fallbacks).
- **JetBrains Blog — "LangChain Python Tutorial: A Complete Guide for 2026"**: https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/ — good coverage of static vs. dynamic model selection and model-fallback middleware for production apps.
- **Medium (Donato_TH) — "Exploring Chat Models with LangChain"**: https://medium.com/donato-story/exploring-chat-models-with-langchain-bfaa363f8edc — approachable walkthrough of chat model basics as part of a longer series.
- **Medium (Vinay Adatiya) — "ChatModels Mastery: How LLMs Actually Talk"**: https://medium.com/@adatiyavinayshaileshbhai/chatmodels-mastery-how-llms-actually-talk-38ba8c62a834 — explains why chat models use structured messages instead of raw text-in/text-out, and touches on LangSmith monitoring.

### Prompts
- **LangChain official docs — Prompt Templates concept guide**: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/prompt_templates.mdx — the canonical source; covers `PromptTemplate`, `ChatPromptTemplate`, and `MessagesPlaceholder` with minimal, precise examples.
- **Comet — "Introduction to Prompt Templates in LangChain"**: https://www.comet.com/site/blog/introduction-to-prompt-templates-in-langchain/ — frames prompts as Instructions + Context + User Input + Output Indicator, useful for structuring your own templates.
- **Latenode Blog — "LangChain Prompt Templates: Complete Guide with Examples"**: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples — good practical examples across single-message and multi-turn use cases, plus few-shot patterns.
- **Codecademy — "Getting Started with LangChain Prompt Templates"**: https://www.codecademy.com/article/getting-started-with-lang-chain-prompt-templates — beginner-friendly, pairs prompt templates with broader prompt-engineering technique explanations (zero/one/few-shot).

### Messages
- **LangChain official docs — Messages concept guide**: https://docs.langchain.com/oss/python/langchain/messages — up-to-date reference covering `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`, multimodal content, and the `name`/`id` fields.
- **Medium (Nazeer Syed) — "Human, AI, and System messages: Message Types in LangChain"**: https://medium.com/@nazeer.td/human-ai-and-system-messages-message-types-in-langchain-708ac2746daf — clear explanation of how each message type contributes to multi-turn memory, with a code-driven "CodeReviewAssistant" example.
- **DEV Community — "A Beginner's Guide to Getting Started with Messages in LangChain"**: https://dev.to/aiengineering/a-beginners-guide-to-getting-started-with-messages-in-langchain-4b6i — walks through a full message sequence (system → human → ai → tool) with JS examples, good for cross-referencing the Python patterns in this doc.
- **Telerik Blog — "Build an LLM Chat App Using LangGraph, OpenAI and Python — Understanding SystemMessage"**: https://www.telerik.com/blogs/build-llm-chat-app-using-langgraph-openai-python-part-2-understanding-systemmessage — focused deep dive specifically on `SystemMessage` and how it constrains model behavior.
- **Learnixo — "HumanMessage, AIMessage, SystemMessage"**: https://learnixo.io/blog/lc-message-types — concise mapping of every message class (including `ToolMessage` and `FunctionMessage`) to how each provider's API represents them under the hood.

> Tip: skim the **official docs guide** for each module first — the
> community blogs above are best used to reinforce specific angles (fallback
> middleware, few-shot patterns, multi-provider message mapping) once you
> have the fundamentals from this document.

---

## 7. Practice exercises

1. Create a `ChatPromptTemplate` for a "recipe generator" that takes
   `{ingredients}` and `{cuisine}` as variables, and run it against Euri.
2. Build a 5-turn conversation manually using `HumanMessage`/`AIMessage` lists
   and confirm the model correctly recalls earlier turns.
3. Write a function `translate(llm, text, language)` using a reusable
   `ChatPromptTemplate`, then call it with two different Euri free models
   (e.g. `gemini-2.5-flash` and `gpt-4.1-nano`) and compare outputs.
4. Modify Example 7 (mini-app) to persist `history` to a JSON file so the
   conversation survives a script restart.

---

## 8. Sources

- LangChain roadmap reference: raffeemdai/LangChain_LangGraph — `langchain topics.md`
- Euri AI free API: https://euron.one/euri (OpenAI-compatible endpoint, ~200K free tokens/day)
- LangChain core docs: https://python.langchain.com/docs/concepts/
- Full curated blog list per module: see **Section 6 — Recommended blogs & further reading** above
