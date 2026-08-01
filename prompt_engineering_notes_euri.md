# Prompt Engineering — Simple Notes (with Euri API Code Examples)

> Combined from: Gen-AI Developer Classroom notes (directai.blog), "3 Prompting
> Techniques for Reasoning in LLMs" (Daily Dose of Data Science), and the
> LangChain Models/Prompts/Messages notes (raffeemdai repos).
> All code examples use the **free Euri API** (OpenAI-compatible endpoint).
>
> **Note on this version:** the `ask()` convenience wrapper that used to live
> in `euri_client.py` has been removed. Every example below now calls
> `client.chat.completions.create(...)` directly, importing the shared
> `client` and `MODEL` objects instead.

---

## 0. Setting up Euri once (used in every example below)

[Euri](https://euron.one/euri) gives a free tier (~200,000 tokens/day, no
credit card) across models like Gemini, Llama, GPT-family mirrors — through a
single **OpenAI-compatible** endpoint. That means it works directly with the
official `openai` Python SDK, or with LangChain's `ChatOpenAI` class.

**Step 1 — Get a free key:** sign up at <https://euron.one/euri> and copy your API key.

**Step 2 — Install packages:**

```
pip install openai langchain langchain-openai python-dotenv
```

**Step 3 — Create a `.env` file** (never hardcode keys in code):

```
EURI_API_KEY=your_api_key_here
EURI_BASE_URL=https://api.euron.one/api/v1/euri
EURI_MODEL=gemini-2.5-flash
```

**Step 4 — A tiny helper you'll reuse everywhere:**

```
# euri_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),   # https://api.euron.one/api/v1/euri
)
MODEL = os.getenv("EURI_MODEL", "gemini-2.5-flash")
```

Every example below just imports `client` and `MODEL` from this file and
calls `client.chat.completions.create(...)` directly. (A LangChain version
using `ChatOpenAI` is shown in Section 7.)

---

## 1. What Is Prompt Engineering?

- A **prompt** = the text/instructions you send to an LLM.
- **Prompt Engineering** = the process of designing and refining prompts so
the model produces the output you actually want.

Think of it like giving instructions to a new employee — vague instructions
get vague results; clear, structured instructions get reliable results.

---

## 2. Prompt Design vs Prompt Engineering (simple distinction)

|                        | Focus                                                                                       | Example                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Prompt Design**      | Clarity & simplicity — just phrase the ask clearly                                          | "What is the capital of France?"                                  |
| **Prompt Engineering** | Sophisticated control — few-shot, chain-of-thought, system role, temperature, output format | "You are an expert nutritionist... respond in a friendly tone..." |

**Same task, two approaches:**

- *Design (simple):* "Give me a business idea for a tech startup."
- *Engineering (few-shot + reasoning):* "Here are 2 examples of good startup
ideas: [...]. Based on these, give me a new idea focused on remote team
collaboration."

Engineering-style prompts generally produce more specific, more controllable
output — at the cost of being longer to write.

---

## 3. 5 Basic Rules for Writing Good Prompts

1. **Assign a Persona** — "You are an expert personal finance advisor..."
2. **State the Core Task & Subtasks** — what exactly should it do?
3. **Provide Context** — background info the model needs to answer well.
4. **Explain the Expected Output** — format, length, tone, table vs prose.
5. **Describe Limitations** — what to avoid, what data/timeframe to focus on.

**Example (all 5 rules combined):**

```
You are an expert personal finance advisor. Explain different types of
investment options available in India. Assume the audience has basic
banking knowledge. Give the explanation as a table. Focus only on
data from the last 5 years.
```

**Euri code — sending this as a real prompt:**

```
from euri_client import client, MODEL

messages = [
    {"role": "user", "content": (
        "You are an expert personal finance advisor. Explain different "
        "types of investment options available in India. Assume the "
        "audience has basic banking knowledge. Give the explanation as "
        "a table. Focus only on data from the last 5 years."
    )}
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

---

## 4. Core Prompting Techniques

### 4.1 Zero-Shot Prompting

**Definition:** Ask the model to do a task with **no examples** — it relies
purely on what it already learned during training.

**Example prompt:** *"What is the capital of Japan?"* → **Expected:** "Tokyo"

**Euri code:**

```
from euri_client import client, MODEL

messages = [{"role": "user", "content": "What is the capital of Japan?"}]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
)
print(resp.choices[0].message.content)
# -> Tokyo
```

### 4.2 Few-Shot Prompting

**Definition:** Give the model a few input → output **examples** before
asking your real question. This teaches the pattern/format you want.

**Example prompt:**

```
Here are some translations from English to French:
English: "Good morning." -> French: "Bonjour."
English: "How are you?" -> French: "Comment ça va?"

Translate this sentence to French: "What is your name?"
```

**Euri code:**

```
from euri_client import client, MODEL

prompt = """Here are some translations from English to French:
English: "Good morning." -> French: "Bonjour."
English: "How are you?" -> French: "Comment ça va?"

Translate this sentence to French: "What is your name?"
"""

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
print(resp.choices[0].message.content)
# -> "Quel est ton nom?"
```

### 4.3 Chain-of-Thought (CoT) Prompting

**Definition:** Ask the model to reason **step by step** instead of jumping
straight to the final answer. This breaks a complex problem into smaller,
logical steps — and often improves accuracy on math/logic tasks.

**Example prompt:** *"Solve this step by step: What is 12 multiplied by 9?"*

**Euri code:**

```
from euri_client import client, MODEL

messages = [{
    "role": "user",
    "content": "Solve this step by step: What is 12 multiplied by 9?"
}]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0,   # deterministic/precise
)
print(resp.choices[0].message.content)
# -> Step 1: 12 x 9 means adding 12 nine times...
# -> Final answer: 108
```

### 4.4 System Prompting (Setting Behavior/Persona)

**Definition:** A **system message** sets the model's role/behavior for the
whole conversation — it's sent once, before the actual question.

**Euri code:**

```
from euri_client import client, MODEL

messages = [
    {"role": "system", "content": (
        "You are an expert nutritionist. Give healthy eating advice based "
        "on someone's age and lifestyle. Respond in a friendly, supportive tone."
    )},
    {"role": "user", "content": "I'm 28 and work a desk job. Any tips?"},
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

### 4.5 Temperature Control (Creativity vs Precision)

**Definition:** `temperature` controls randomness.

- **Low (0 – 0.3):** deterministic, focused, best for facts/code/math.
- **High (0.7 – 1.2):** creative, varied, best for stories/brainstorming.

**Euri code:**

```
from euri_client import client, MODEL

creative_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a tagline for a coffee shop."}],
    temperature=1.2,
)
precise_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 15 * 12?"}],
    temperature=0.0,
)

print("Creative:", creative_resp.choices[0].message.content)
print("Precise :", precise_resp.choices[0].message.content)
```

---

## 5. Advanced Reasoning Techniques (for hard problems)

These go one level deeper than plain Chain-of-Thought — useful when a task
needs strong logical/mathematical reasoning.

### 5.1 Chain of Thought (CoT) — recap

The simplest reasoning nudge: "let's think step by step" before the final
answer. Covered above in 4.3.

### 5.2 Self-Consistency (Majority Voting over CoT)

**Definition:** CoT with one call isn't always reliable — the same question
can get different answers on different runs (especially at higher
temperature). Self-consistency asks the model **multiple times**, generates
several independent reasoning paths, and picks the **most common final
answer** — like taking a vote.

**Simple explanation:** if you're unsure, ask 5 people the same question and
go with what most of them say.

**Euri code:**

```
from euri_client import client, MODEL
from collections import Counter

question = "If a train travels 60 km in 45 minutes, what is its speed in km/h?"
prompt = f"Solve step by step and give a final numeric answer only at the end:\n{question}"

answers = []
for _ in range(5):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    reply = resp.choices[0].message.content
    # crude extraction: take the last line as "the final answer"
    final_line = reply.strip().splitlines()[-1]
    answers.append(final_line)

most_common_answer, count = Counter(answers).most_common(1)[0]
print("All answers:", answers)
print(f"Majority answer ({count}/5 votes):", most_common_answer)
```

### 5.3 Tree of Thoughts (ToT)

**Definition:** Instead of voting on final answers (self-consistency), ToT
explores **multiple reasoning branches at each step**, like a search tree,
and evaluates which branch/path looks most promising before continuing. It's
more compute-intensive but often outperforms plain CoT on complex problems.

**Simple explanation:** think of it as a decision tree — at each step you
consider a few different next moves, judge which is best, and only keep
following the strongest path (instead of committing to the first idea).

**Simplified Euri code (manual 2-step tree, 2 branches per step):**

```
from euri_client import client, MODEL

problem = "You have 3 boxes weighing 4kg, 7kg, and 9kg. Split them into two groups with the smallest possible weight difference."

# Step 1: generate multiple candidate first moves (branches)
branch_prompt = f"""Problem: {problem}
Propose 2 different possible groupings (branches) as short bullet points,
without solving fully yet."""

branches_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": branch_prompt}],
    temperature=0.9,
)
branches = branches_resp.choices[0].message.content
print("Candidate branches:\n", branches)

# Step 2: ask the model to evaluate and pick the best branch, then solve it
evaluate_prompt = f"""Problem: {problem}
Here are candidate groupings:
{branches}

Evaluate each candidate's weight difference, pick the best one, and give
the final answer with the reasoning."""

final_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": evaluate_prompt}],
    temperature=0.2,
)
final_answer = final_resp.choices[0].message.content
print("\nFinal (best path):\n", final_answer)
```

### 5.4 Quick Comparison

| Technique            | What varies                                       | What it optimizes for                                              |
| --------------------- | -------------------------------------------------- | --------------------------------------------------------------------|
| **Chain of Thought** | Nothing — single reasoning pass                   | Making one answer more transparent/accurate                        |
| **Self-Consistency** | Runs the *same* prompt multiple times             | Reliability — majority vote on the final answer                    |
| **Tree of Thoughts** | Explores *different* reasoning paths at each step | Quality — searches for the best path, not just the most common one |

---

## 6. Prompt Templates (Reusable Prompts in Code)

Hardcoding prompts as raw strings gets messy fast. LangChain-style **templates** let you define a prompt once with `{placeholders}` and reuse it
with different inputs — cleanly, with input validation.

### 6.1 Plain Python (no library) — a simple template function

```
from euri_client import client, MODEL

def explain_template(topic: str, audience: str) -> str:
    return f"Explain {topic} to a {audience} in 2 sentences."

prompt = explain_template("vector embeddings", "5-year-old")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

### 6.2 LangChain `PromptTemplate` + Euri

```
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in 2 sentences."
)

chain = template | llm     # LCEL: pipe prompt straight into the model
result = chain.invoke({"topic": "vector embeddings", "audience": "5-year-old"})
print(result.content)
```

### 6.3 `ChatPromptTemplate` — system + user roles together

```
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a concise programming tutor who always gives a code example."),
    ("human", "Explain {concept} in Python."),
])

chain = chat_template | llm
response = chain.invoke({"concept": "decorators"})
print(response.content)
```

### 6.4 Few-Shot Template (built-in pattern)

```
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

### 6.5 Reusable Prompt Library Pattern (real projects)

```
# prompts.py — keep all templates in one place
from langchain_core.prompts import ChatPromptTemplate

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize input text in exactly 3 bullet points."),
    ("human", "{text}"),
])

TRANSLATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Only output the translation."),
    ("human", "Translate to {language}:\n\n{text}"),
])

# main.py
from prompts import SUMMARIZER_PROMPT, TRANSLATOR_PROMPT

summary_chain = SUMMARIZER_PROMPT | llm
translation_chain = TRANSLATOR_PROMPT | llm

print(summary_chain.invoke({"text": "LangChain is a framework for building LLM apps."}).content)
print(translation_chain.invoke({"text": "Good morning!", "language": "Spanish"}).content)
```

---

## 7. Messages — Organizing a Conversation

A real conversation is more than one prompt/response — LangChain represents
every turn as a typed **Message**:

| Type            | Role      | Simple meaning                                  |
| ---------------- | --------- | ------------------------------------------------ |
| `SystemMessage` | system    | Instructions given once, before the chat starts |
| `HumanMessage`  | user      | What the human typed                            |
| `AIMessage`     | assistant | What the model replied                          |
| `ToolMessage`   | tool      | Result of a tool call, sent back to the model   |

**Why not just plain strings?** Different providers expect conversation
history in slightly different formats. Message classes are a common,
provider-agnostic way to write it once.

**Euri code — multi-turn conversation with memory:**

```
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("EURI_MODEL", "gemini-2.5-flash"),
    api_key=os.getenv("EURI_API_KEY"),
    base_url=os.getenv("EURI_BASE_URL"),
)

conversation = [
    SystemMessage(content="You are a friendly math tutor."),
    HumanMessage(content="What is 12 * 8?"),
]

first_reply = llm.invoke(conversation)
conversation.append(first_reply)                      # remember the AI's reply
conversation.append(HumanMessage(content="Now divide that by 4."))

second_reply = llm.invoke(conversation)
print(second_reply.content)   # correctly uses context from turn 1 -> 24
```

---

## 8. Chaining (LCEL) — the assembly-line mental model

```
chain = prompt | llm | parser
```

Think of it like an assembly line: the prompt template fills in your
question → passes it to the LLM → the model's answer comes out → an optional
parser cleans the raw text into something usable (e.g. a Python dict).

```
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

---

## 9. Image Prompting (Bonus — Prompt Engineering for Images)

Image prompting guides an image-generation model. A typical structure:

```
[main subject, action, mood],
[art form, art style, artist references],
[additional settings: lighting, colors, framing]
```

**Example:**

```
Lion roaring in a dense forest, angry mood, other animals in the distant
background, realistic, colorful, 8K
```

Things to think about: art medium (painting, ink, 3D render), camera/lens,
resolution (4K/8K), lighting, and material (metal, cloth, glass, wood).

---

## 10. Quick-Reference Cheat Sheet

```
# Setup (once)
from openai import OpenAI
client = OpenAI(api_key=EURI_API_KEY, base_url=EURI_BASE_URL)

# Zero-shot
client.chat.completions.create(model=MODEL, messages=[
    {"role": "user", "content": "What is the capital of Japan?"}
])

# Few-shot -> add examples inside the user message before the real question

# Chain-of-thought -> add "Let's think step by step" / "Solve step by step"

# System prompt -> add {"role": "system", "content": "You are a ..."} first

# Temperature -> 0 = precise/factual, 1+ = creative

# Self-consistency -> call the same prompt N times, majority-vote the answers

# Tree of Thoughts -> generate branches, evaluate them, pick the best path
```

| Concept          | One-line memory hook                                      |
| ---------------- | ----------------------------------------------------------- |
| Prompt           | The instructions you send to the model                    |
| Zero-shot        | No examples — just ask                                    |
| Few-shot         | A few examples first, then the real ask                   |
| Chain-of-thought | "Think step by step" before the final answer              |
| Self-consistency | Ask N times, trust the majority answer                    |
| Tree of Thoughts | Explore multiple reasoning branches, pick the best        |
| System prompt    | Sets persona/behavior for the whole chat                  |
| Temperature      | 0 = precise, 1+ = creative                                 |
| Prompt Template  | Reusable prompt with `{placeholders}`                      |
| Message          | Role-tagged turn (system/human/ai/tool) in a conversation  |
| Chain (LCEL)     | `prompt \| llm \| parser` — assembly line for LLM calls    |

---

## 11. Sources

- Gen-AI Developer Classroom notes (26 Mar 2025) — directai.blog
- "3 Prompting Techniques for Reasoning in LLMs" — Daily Dose of Data Science (Avi Chawla)
- LangChain Models/Prompts/Messages notes with Euri examples — raffeemdai/LangChain_LangGraph
- LangChain Chaining/Prompts/Messages notes — raffeemdai/RAG_BY_ME
- Euri free API: <https://euron.one/euri>
