# LangChain v1 Crash Course — Complete Notes (with actual notebook code)

Notes built directly from your 7 uploaded notebooks, with the real code from each notebook plus a full explanation of every topic and concept.

**Notebooks covered:**
1. `1-langchainintro.ipynb` — Agents
2. `2-modelintegration.ipynb` — Model Integration (OpenAI, Gemini, Groq) + Streaming/Batch
3. `3-tools.ipynb` — Tools & Tool Execution Loops
4. `4-messages.ipynb` — Messages
5. `5-structuredoutput.ipynb` — Structured Output
6. `6-middleware.ipynb` — Middleware (Summarization + Human-in-the-Loop)
7. `langchain_guardrails_crash_course.ipynb` — Guardrails

---

# 1. LangChain Intro — Agents

### Version check

```python
import langchain
langchain.__version__
```

This just confirms which LangChain release you're on (the course targets v1, where agent-creation syntax changed significantly from earlier versions).

### Environment setup

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

`load_dotenv()` reads your `.env` file into the process environment; the line after it explicitly re-sets `OPENAI_API_KEY` in `os.environ` so any library that reads directly from `os.environ` (like the OpenAI SDK under the hood) can find it.

### Why agents exist

A plain LLM call only knows what was in its training data — it has no way to fetch live information. An **agent** solves this: it's an LLM wrapped so that, given an input, it can *decide* whether to answer directly or call an external **tool** to fetch more context first, then use that tool's result to produce the final answer.

### Creating an agent with a tool, using `create_agent`

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
agent
```

**Explanation of each argument:**
- `model="gpt-5"` — the model is specified as a plain string; `create_agent` resolves it internally.
- `tools=[get_weather]` — a list of plain Python functions. The **docstring** (`"""Get the weather for a city."""`) is what the LLM reads to decide *when* and *how* to call this function — LangChain turns this docstring + type hints (`city: str`) into the tool's schema automatically (you don't need the `@tool` decorator here because `create_agent` wraps bare functions for you).
- `system_prompt` — sets the agent's overall behavior/persona.

Printing `agent` shows the compiled internal graph object — conceptually it's **start → model → tools → model → end**: the model can loop back to itself after a tool call before producing the final answer.

### Running the agent

```python
### run the agent
response = agent.invoke({"messages":[{"role":"user","content":"What is the weather like in New York?"}]})
```

`create_agent`'s graph expects a **dictionary with a `messages` key**, where each message is a role/content dict (mirroring the OpenAI chat format). Internally:
1. Your `HumanMessage` is added to state.
2. The model reads it, recognizes it needs live weather data, and emits an `AIMessage` containing a **tool call** to `get_weather`.
3. LangChain automatically executes `get_weather("New York")` and appends the result as a `ToolMessage`.
4. The model is invoked again with the tool result now in context, and produces the final natural-language `AIMessage`.

```python
response["messages"]
```

This prints the **full message list** — you'll see, in order: the `HumanMessage` you sent, the `AIMessage` with the tool call, the `ToolMessage` with the raw tool output (`"The weather in New York is sunny."`), and the final `AIMessage` with the natural-language answer.

### Looser input format

```python
agent.invoke({"messages":"What is the weather in New Yourk"})
```

Note this passes a **bare string** instead of a list of role/content dicts — LangChain is flexible enough to interpret this as a single human message. It even tolerates the typo ("New Yourk") — the LLM still infers intent correctly and successfully calls `get_weather`.

**Key takeaway from this notebook:** the entire mechanics of a basic agent are: (1) attach tools as plain functions with descriptive docstrings, (2) call `create_agent(model=..., tools=[...], system_prompt=...)`, (3) invoke with `{"messages": [...]}`, and LangChain automatically handles the "decide → call tool → get context → answer" loop for you.

---

# 2. Model Integration — OpenAI, Google Gemini, and Groq

### Loading all three provider keys at once

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
```

This loads keys for all three providers up front so any of the model classes used later in the notebook can find their credentials.

### Method 1: `init_chat_model` (provider-agnostic)

```python
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4.1")
model
```

```python
## invoke the model
response = model.invoke("Hello How are you?")
response
```

`init_chat_model` is a single universal entry point — you just pass a model-name string and (optionally) a provider prefix, and it figures out which integration/class to instantiate under the hood. `response` here is a full `AIMessage` object (content + metadata), not just plain text.

### Method 2: Provider-specific class — `ChatOpenAI`

```python
### ChatOpenAI

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4.1")
response = model.invoke("Hello How are you?")
response
```

```python
response.content
```

Same result as `init_chat_model("gpt-4.1")`, but instantiated directly via the OpenAI-specific class. `.content` extracts just the text string from the `AIMessage`.

### Google Gemini Model Integration

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("google_genai:gemini-2.5-flash")
response = model.invoke("Why do parrots talk?")
response.content
```

Note the `"google_genai:gemini-2.5-flash"` string — the `google_genai:` prefix tells `init_chat_model` which provider integration to route to, followed by the specific Gemini model name.

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
response = model.invoke("Why do parrots talk?")
response
```

Same question asked via the dedicated `ChatGoogleGenerativeAI` class instead, using the lighter `gemini-2.5-flash-lite` variant this time.

### GROQ Model Integration

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

```python
import os
from langchain_groq import ChatGroq

model = ChatGroq(model="qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

Same pattern: `init_chat_model("groq:<model>")` vs. the dedicated `ChatGroq` class — both call the same Qwen 3 32B model hosted on Groq's fast inference infrastructure. Groq models like this one often return a `reasoning_content` field in `additional_kwargs` showing the model's internal chain-of-thought before its final answer (visible in the raw `AIMessage` repr).

**Takeaway across all three providers:** the pattern is identical regardless of provider — `init_chat_model("<provider>:<model>")` for a config-driven, swappable approach, or the provider's dedicated `Chat*` class for more direct control. `.invoke()` behaves the same way in all cases and always returns an `AIMessage`.

### Streaming

> *Most models can stream their output content while it is being generated. By displaying output progressively, streaming significantly improves user experience, particularly for longer responses. Calling `stream()` returns an iterator that yields output chunks as they are produced. You can use a loop to process each chunk in real-time.*

```python
model.invoke("Write me a 200 words paragraph on Artificial Intelligence")
```

This is the baseline — `.invoke()` blocks until the *entire* 200-word paragraph is generated before returning anything.

```python
for chunk in model.stream("Write me a 200 words paragraph on Artificial Intelligence"):
    print(chunk.text, end="|", flush=True)
```

`.stream()` instead returns an **iterator of chunks** as they're generated token-by-token (or in small groups). The `end="|"` here visually separates each chunk with a pipe character so you can see the granularity of streaming (in real UI code you'd normally use `end=""` for a smooth typing effect). `flush=True` forces each `print` to appear immediately rather than being buffered.

```python
model.invoke("Why do parrots have colorful feathers?")

for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

Same invoke-vs-stream comparison repeated on a second question, reinforcing that `.stream()` is a drop-in alternative to `.invoke()` whenever you want progressive output.

### Batch

> *Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel.*

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

`.batch()` takes a **list of independent prompts** and sends them concurrently rather than looping through `.invoke()` one at a time — faster and typically cheaper since providers can process them in parallel.

```python
model.batch(
    ["Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"],
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)
```

Passing `config={"max_concurrency": 5}` caps how many requests run in parallel at once — useful for staying under a provider's rate limit when batching a much larger list (e.g. 50 prompts processed 5 at a time instead of all 50 simultaneously).

---

# 3. Tools & Tool Execution Loops

> *Models can request to call tools that perform tasks such as fetching data from a database, searching the web, or running code. Tools are pairings of: (1) a schema, including the name of the tool, a description, and/or argument definitions (often a JSON schema), and (2) a function or coroutine to execute.*

### Setting up the base model

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

This is just establishing a working model (Groq's Qwen 3 32B) before attaching tools to it. Note the very long `AIMessage.content` you get back for a reasoning model like this — it includes a full `<think>...</think>` block where the model works through the reasoning before its final answer, plus `usage_metadata` showing `input_tokens: 14`, `output_tokens: 1384`, `total_tokens: 1398` for this single call.

### Defining a tool with `@tool` and binding it to the model

```python
from langchain.tools import tool

@tool
def get_weather(location:str)->str:
    """Get the weather at a location"""
    return f"It's sunny in {location}"


model_with_tools=model.bind_tools([get_weather])
```

- `@tool` — the decorator converts a plain function into a LangChain `Tool` object with a name, an argument schema (derived from the type hints `location: str`), and a description (derived from the docstring `"""Get the weather at a location"""`).
- `model.bind_tools([get_weather])` — returns a **new model object** that, on every `.invoke()` call, has access to this tool's schema and can choose to emit a tool call instead of (or alongside) plain text.

### Invoking the tool-bound model and inspecting the tool call

```python
response = model_with_tools.invoke("What's the weather like in Boston?")
print(response)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

Output (actual from the run):
```
content='' additional_kwargs={'reasoning_content': "Okay, the user is asking about the weather in Boston. I need to use the get_weather function. Let me check the function parameters. It requires a location, which is Boston here. I'll call the function with location set to Boston...", 'tool_calls': [{'id': 'c6smtcmmb', 'function': {'arguments': '{"location":"Boston"}', 'name': 'get_weather'}, 'type': 'function'}]} ...
Tool: get_weather
Args: {'location': 'Boston'}
```

Key things to notice:
- `content=''` — when the model decides to call a tool, it typically returns *empty* text content; the actual "answer" comes only after the tool result is fed back in a second call.
- `reasoning_content` (in `additional_kwargs`) shows the model's internal reasoning for *why* it chose this tool.
- `response.tool_calls` is a clean, structured list you can iterate over: each entry has `name`, `args`, and `id`.

### The Tool Execution Loop (the mechanics behind `create_agent`)

This is what `create_agent` does automatically — here it's done by hand to show exactly how it works:

```python
# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "The current weather in Boston is 72°F and sunny."
```

Actual output: `The weather in Boston is sunny.`

Step-by-step:
1. **Step 1** — send the user's question; the model responds with an `AIMessage` containing a tool call (no real answer yet). Append this `AIMessage` to the running `messages` list.
2. **Step 2** — for every tool call the model requested, call `<tool>.invoke(tool_call)`. Passing the whole `tool_call` dict (not just the args) lets LangChain automatically produce a properly-formed `ToolMessage` (with matching `tool_call_id`) that gets appended to `messages`.
3. **Step 3** — call the model *again*, now with the full conversation (human question → AI tool call → tool result) as context. This time the model has the actual weather data available and produces a real natural-language final answer.

### Inspecting the full message transcript

```python
messages
```

Actual output:
```python
[{'role': 'user', 'content': "What's the weather in Boston?"},
 AIMessage(content='', additional_kwargs={'reasoning_content': 'Okay, the user is asking for the weather in Boston...', 'tool_calls': [{'id': '20cfda4p0', 'function': {'arguments': '{"location":"Boston"}', 'name': 'get_weather'}, 'type': 'function'}]}, ..., tool_calls=[{'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': '20cfda4p0', 'type': 'tool_call'}], ...),
 ToolMessage(content="It's sunny in Boston", name='get_weather', tool_call_id='20cfda4p0')]
```

This confirms the exact structure: a plain dict for the human turn, an `AIMessage` object carrying the tool call, and a `ToolMessage` object carrying the tool's raw return value, matched by `tool_call_id`. (Note `final_response` — the model's actual answer — isn't shown here since it wasn't appended to `messages` in this cell, only computed and printed above.)

**Summary:** Two ways to work with tools — (a) `@tool` + `model.bind_tools([...])` for full manual control over the loop (shown here), or (b) `create_agent(model=..., tools=[...])` which runs this exact loop for you automatically. Understanding this manual loop is what makes `create_agent`'s behavior transparent rather than "magic."

---

# 4. Messages

> *Messages are the fundamental unit of context for models in LangChain. They represent the input and output of models, carrying both the content and metadata needed to represent the state of a conversation when interacting with an LLM. Messages are objects that contain:*
> - *Role — Identifies the message type (e.g. system, user)*
> - *Content — Represents the actual content of the message (like text, images, audio, documents, etc.)*
> - *Metadata — Optional fields such as response information, message IDs, and token usage*
>
> *LangChain provides a standard message type that works across all model providers, ensuring consistent behavior regardless of the model being called.*

### Setup

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
```

```python
model.invoke("Please tell what is artificial intelligence")
```

### Text Prompts

> *Text prompts are strings - ideal for straightforward generation tasks where you don't need to retain conversation history.*

```python
model.invoke("what is langchain")
```

> *Use text prompts when:*
> - *You have a single, standalone request*
> - *You don't need conversation history*
> - *You want minimal code complexity*

A bare string passed to `.invoke()` is the simplest possible call — LangChain implicitly treats it as a single `HumanMessage` under the hood. No system instruction, no history — just one question, one answer.

### Message Prompts

> *Alternatively, you can pass in a list of messages to the model by providing a list of message objects.*

This is the more powerful alternative to a text prompt: instead of a bare string, you build an explicit **list of message objects**, which lets you (a) set a system instruction, and (b) preserve multi-turn history.

### The Four Message Types

> **Message types**
> - **System message** — Tells the model how to behave and provide context for interactions
> - **Human message** — Represents user input and interactions with the model
> - **AI message** — Responses generated by the model, including text content, tool calls, and metadata
> - **Tool message** — Represents the outputs of tool calls

**System Message:** *A `SystemMessage` represents an initial set of instructions that primes the model's behavior. You can use a system message to set the tone, define the model's role, and establish guidelines for responses.*

**Human Message:** *A `HumanMessage` represents user input and interactions. They can contain text, images, audio, files, and any other amount of multimodal content.*

**AI Message:** *An `AIMessage` represents the output of a model invocation. They can include multimodal data, tool calls, and provider-specific metadata that you can later access.*

**Tool Message:** *For models that support tool calling, AI messages can contain tool calls. Tool messages are used to pass the results of a single tool execution back to the model.*

### Using a system message + human message together

```python
from langchain.messages import SystemMessage, HumanMessage,AIMessage

messages=[
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a poem on artificial intelligence")
]

response=model.invoke(messages)
response.content
```

The `SystemMessage` primes the model's *role* ("poetry expert") before the actual request arrives, shaping the style of the response.

```python
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
print(response.content)
```

A generic one-line system message ("helpful coding assistant") produces a fairly generic answer — no specific language, framework, or depth is implied.

### Detailed system prompts produce more targeted answers

```python
## Detailed info to the LLM through System message
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
print(response.content)
```

Compared to the generic version above, this system message specifies the exact persona ("senior Python developer with expertise in web frameworks") and explicit output requirements ("always provide code examples," "be concise but thorough"). The resulting answer is far more specific — walking through choosing a framework (e.g. Flask), installing it, a concrete code example, and even production concerns like disabling debug mode and adding input validation. **The more detail you put in the system message, the more tailored and useful the response.**

### Message Metadata

```python
## Message Metadata
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
```

```python
response = model.invoke([
  human_msg
])
response
```

- `name` — useful in multi-user applications to identify *who* sent a given message.
- `id` — a unique identifier, useful for tracing/debugging/logging a specific message through a pipeline.

### Manually constructing conversation history (including a fake `AIMessage`)

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# Add to conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
print(response.content)
```

This demonstrates that you don't need every `AIMessage` in your history to have actually come from a real model call — you can **construct fake conversation turns manually** (e.g. to seed a conversation, replay a saved session, or test specific scenarios). The model treats this exactly as if it had said "I'd be happy to help you with that question!" itself, and continues the conversation naturally from there.

### Inspecting token usage

```python
response.usage_metadata
```

Returns a dict with `input_tokens`, `output_tokens`, and `total_tokens` for the most recent call — useful for cost tracking.

### Tool Messages — manually constructing a tool round-trip

```python
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

# After a model makes a tool call
# (Here, we demonstrate manually creating the messages for brevity)
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# Execute tool and create result message
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)

# Continue conversation
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # Model's tool call
    tool_message,  # Tool execution result
]
response = model.invoke(messages)  # Model processes the result
```

```python
tool_message
```

```python
response
```

This is the same tool round-trip you saw automated in the Tools notebook — but built **entirely by hand** here to show exactly what each piece looks like:
- `ai_message` is manually built with an empty `content` and a `tool_calls` list — simulating what the model would have generated on its own.
- `tool_message` is manually built with the (fake, hardcoded) result `"Sunny, 72°F"`, tagged with `tool_call_id="call_123"` so it matches back to the exact tool call it's answering.
- Feeding all three messages (`HumanMessage` → `AIMessage` with tool call → `ToolMessage` with result) into `model.invoke()` lets the model produce a final natural-language answer, exactly as if this had all happened automatically via `bind_tools`.

**Summary of all four message types:** `SystemMessage` = instruction/persona, `HumanMessage` = user input, `AIMessage` = model output (text and/or tool calls), `ToolMessage` = a tool's result linked back by `tool_call_id`. Every multi-turn conversation or agent trace in LangChain is ultimately just a growing list of these four object types.

---

# 5. Structured Output

> *Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.*

### Pydantic

> *Pydantic models provide the richest feature set with field validation, descriptions, and nested structures.*

```python
import os
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
model=init_chat_model("groq:qwen/qwen3-32b")
model
```

```python
from pydantic import BaseModel,Field

class Movie(BaseModel):
    title:str=Field(description="The title of the movie")
    year:int=Field(description="This year the movie was released")
    director:str=Field(description="The director of the movie")
    rating:float=Field(description="The movies rating out of 10")
```

Each field's `Field(description=...)` doesn't just document the field for humans — it's actually sent to the model as part of the schema, telling it exactly what kind of value belongs in each slot.

```python
model_with_structure=model.with_structured_output(Movie)
model_with_structure
```

`.with_structured_output(Movie)` wraps the model so every response is coerced into (and validated against) the `Movie` schema rather than returned as free text.

```python
model.invoke("Provide details about the moview Inception")
```

Calling the **plain, unstructured** model on the same prompt returns a normal free-text `AIMessage` — a paragraph *about* Inception, not a structured object.

```python
response=model_with_structure.invoke("Provide details about the moview Inception")
response
```

Calling the **structured** model instead returns a validated `Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8, ...)` object — ready to use directly in code (e.g. `response.title`, `response.year`) rather than needing to parse text.

### Message output alongside parsed structure

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)  

response = model_with_structure.invoke("Provide details about the movie Inception")
response
```

`include_raw=True` changes the return value to a dict containing **both** the raw `AIMessage` (`response["raw"]`) *and* the parsed, validated object (`response["parsed"]`) — useful when you want the structured object but also want to inspect/log the model's original raw output (e.g. for debugging or token-usage tracking).

### Nested Structure

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Inception")
response
```

`Actor` is a *separate* Pydantic model nested inside `MovieDetails` via `cast: list[Actor]` — this shows Pydantic structured output isn't limited to flat key/value schemas; you can build arbitrarily nested object graphs (a movie with a list of actor objects, each having their own `name`/`role` fields) and the model will populate the entire nested structure correctly (e.g. `cast=[Actor(name='Leonardo DiCaprio', role='Dom Cobb'), Actor(name='Joseph Gordon-Levitt', role='Arthur'), ...]`, `budget=160.0`).

### TypedDict

> *TypedDict provides a simpler alternative using Python's built-in typing, ideal when you don't need runtime validation.*

```python
from typing_extensions import TypedDict,Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]


model_withtypedict=model.with_structured_output(MovieDict)
response=model_withtypedict.invoke("Please provide the details of the movie avengers")
response
```

`Annotated[str, ..., "description"]` is the TypedDict equivalent of Pydantic's `Field(description=...)` — the third element of `Annotated` is the field description passed to the model. Unlike `BaseModel`, though, **`TypedDict` performs no runtime validation** — if the model returns a value of the wrong type, Python won't complain (it's effectively just a typed dictionary literal at runtime).

```python
class Actor(TypedDict):
    name: str
    role: str

class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Avengers")
response
```

Same nested pattern as with Pydantic — `Actor` nested inside `MovieDetails` — but built with `TypedDict` instead. (Note: mixing `Field(...)` from Pydantic into a `TypedDict` class body, as shown here, is unconventional — normally you'd use `Annotated[float | None, ..., "Budget in millions USD"]` for a TypedDict field description, matching the `MovieDict` example above.)

```python
model.profile
```

`model.profile` is available on the **plain, non-structured** model object and reports metadata about the model itself — such as max input/output tokens, and boolean support flags for things like image input, audio input, reasoning output, and tool calling. This attribute is **not** available on a structured-output-wrapped model (e.g. `model_with_structure.profile` would fail), since that's a different kind of runnable object (a `RunnableSequence`).

### DataClasses

> *A data class is a class typically containing mainly data, although there aren't really any restrictions. You create it using the `@dataclass` decorator.*

```python
import os
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
```

Switches over to OpenAI for the remaining examples, since these use `create_agent` with `model="gpt-5"`.

### Structured output inside an agent — Pydantic

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result
```

```python
result["structured_response"]
```

`response_format=ContactInfo` tells `create_agent` to constrain the **final** agent output to this schema (LangChain "auto-selects" the appropriate underlying structured-output strategy for whichever provider you're using — noted in the comment as `ProviderStrategy`). The parsed object is available at `result["structured_response"]`, alongside the normal `result["messages"]` transcript.

### Structured output inside an agent — TypedDict

```python
## Typedict
from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

Identical usage pattern, just swapping the Pydantic `BaseModel` for a `TypedDict` — the agent-level API doesn't care which schema style you use; `result["structured_response"]` comes back as a plain dict instead of a validated Pydantic instance.

### Structured output inside an agent — Dataclass

```python
## Dataclass

from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person


agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
```

Same again with a plain `@dataclass` — the **exact same three lines of `create_agent`/`invoke`/`result["structured_response"]`** work regardless of which of the three schema styles (Pydantic, TypedDict, dataclass) you choose. This is the notebook's core point: `response_format` in `create_agent` is schema-agnostic.

**Summary — when to use which:**
- **Pydantic** — richest option: field validation, `Field(description=...)`, easy nesting of sub-models. Best when correctness matters.
- **TypedDict** — lightweight, no runtime validation, still supports descriptions via `Annotated[...]`.
- **Dataclass** — simplest, standard-library only, no validation, minimal boilerplate.

All three work identically with both `model.with_structured_output(...)` (on a plain model) and `create_agent(response_format=...)` (on a full tool-using agent).

---

# 6. Middleware

> *Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:*
> - *Tracking agent behavior with logging, analytics, and debugging.*
> - *Transforming prompts, tool selection, and output formatting.*
> - *Adding retries, fallbacks, and early termination logic.*
> - *Applying rate limits, guardrails, and PII detection.*

### Setup

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

## 6.1 Summarization Middleware

> *Automatically summarize conversation history when approaching token limits, preserving recent messages while compressing older context. Summarization is useful for the following:*
> - *Long-running conversations that exceed context windows.*
> - *Multi-turn dialogues with extensive history.*
> - *Applications where preserving full conversation context matters.*

### Trigger type 1 — message count

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

### Message based summarization
agent=create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages",10),
            keep=("messages",4)
        )
    ]
)
```

- `checkpointer=InMemorySaver()` — required so that conversation state persists across multiple `.invoke()` calls tied to the same `thread_id` (without it, each `.invoke()` would start fresh with no memory).
- `SummarizationMiddleware(model=..., trigger=("messages", 10), keep=("messages", 4))` — the `trigger` tuple says *"once the running message count hits 10, summarize"*; the `keep` tuple says *"but always leave the most recent 4 messages untouched, verbatim."* A cheaper model (`gpt-4o-mini`) is used to *perform* the summarization itself, separate from whatever model is answering the user.

```python
### Run with thread id
config={"configurable":{"thread_id":"test-1"}}
```

```python
# Alternative test data
questions = [
    "What is 2+2?",
    "What is 10*5?",
    "What is 100/4?",
    "What is 15-7?",
    "What is 3*3?",
    "What is 4*4?",
]

for q in questions:
    response=agent.invoke({"messages":[HumanMessage(content=q)]},config)
    print(f"Messages: {response}")
    print(f"Messages: {len(response['messages'])}")
```

Running six simple arithmetic questions through the *same* `thread_id` builds up conversation history turn by turn. Once the accumulated message count crosses the `trigger=("messages", 10)` threshold, the middleware automatically compresses the older messages into a generated summary (something like *"Here is a summary of the conversation to date: the user asked several arithmetic questions..."*), while the most recent `keep=("messages", 4)` messages remain exactly as they were. Printing `len(response["messages"])` on each turn is how you can visibly confirm this — the count grows normally, then **drops** right after a summarization event fires, because old individual turns just got folded into a single summary message.

### Trigger type 2 — token count

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_hotels(city: str) -> str:
    """Search hotels - returns long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


agent=create_agent(
    model="gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens",550),
            keep=("tokens",200),
        ),
    ]
)

config = {"configurable": {"thread_id": "test-1"}}

# Token counter (approximate)
def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4  # 4 chars ≈ 1 token
```

This time the tool (`search_hotels`) is designed to return a deliberately long response, so the conversation's token count grows quickly. `trigger=("tokens", 550)` fires summarization once the running token count crosses 550, keeping the most recent `("tokens", 200)` worth verbatim. The helper `count_tokens()` is a rough manual approximation (4 characters ≈ 1 token) used purely to observe the effect in this notebook — it's not an exact tokenizer.

```python
# Run test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config
    )
    
    tokens = count_tokens(response["messages"])
    print(f"{city}: ~{tokens} tokens, {len(response['messages'])} messages")
    print(f"{(response['messages'])}")
```

Looping through six cities, each call triggers the `search_hotels` tool and accumulates its long response into history. Watching the printed `~{tokens} tokens` values across iterations, you'll see the count climb, then **drop noticeably** the moment it crosses the 550-token threshold — direct evidence the middleware compressed history down to a summary + the most recent ~200 tokens.

### Trigger type 3 — fraction of context window

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_hotels(city: str) -> str:
    """Search hotels."""
    return f"Hotels in {city}: Grand Hotel $350, City Inn $180, Budget Stay $75"

# LOW fraction for testing!
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("fraction", 0.005),  # 0.5% = ~640 tokens
            keep=("fraction", 0.002),     # 0.2% = ~256 tokens
        ),
    ],
)

config = {"configurable": {"thread_id": "test-1"}}

# Token counter
def count_tokens(messages):
    return sum(len(str(m.content)) for m in messages) // 4

# Test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Hotels in {city}")]},
        config=config
    )
    tokens = count_tokens(response["messages"])
    fraction = tokens / 128000  # gpt-4o-mini context
    print(f"{city}: ~{tokens} tokens ({fraction:.4%}), {len(response['messages'])} msgs")
    print(response['messages'])
```

Instead of a hardcoded token number, `trigger=("fraction", 0.005)` expresses the threshold as a **fraction of the model's total context window** (here, `gpt-4o-mini`'s ~128,000-token window, so 0.5% ≈ 640 tokens — as the comment notes). This is convenient because the same middleware config auto-scales correctly if you later swap in a model with a much larger or smaller context window, without needing to recompute an absolute token number. The comment "LOW fraction for testing!" flags that 0.005 (0.5%) is deliberately small here just so the summarization behavior triggers quickly and is easy to observe within a short demo loop — in production you'd likely use a much larger fraction (e.g. 0.7–0.8) so summarization only kicks in when genuinely close to the context limit.

## 6.2 Human-in-the-Loop Middleware

### `### Human In the Loop MiddleWare`

> *Pause agent execution for human approval, editing, or rejection of tool calls before they execute. Human-in-the-loop is useful for the following:*
> - *High-stakes operations requiring human approval (e.g. database writes, financial transactions).*
> - *Compliance workflows where human oversight is mandatory.*
> - *Long-running conversations where human feedback guides the agent.*

### Setup — mock email tools

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"
```

```python
agent=create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool":{
                    "allowed_decisions":["approve","edit","reject"]
                },
                "read_email_tool":False,

            }
        )
    ]
)
```

`interrupt_on` is a dict keyed by tool name:
- `"send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]}` — every call to this tool **pauses execution** and requires a human to explicitly approve, edit, or reject it before it actually runs.
- `"read_email_tool": False` — this tool runs freely with **no** human gate (it's read-only/low-risk).

`checkpointer=InMemorySaver()` is required here too — it's what lets the paused/interrupted state survive between the first `.invoke()` (which pauses) and the second `.invoke()` (which resumes with the human's decision), as long as both share the same `thread_id`.

### Approve flow

```python
config = {"configurable": {"thread_id": "test-approve"}}
# Step 1: Request
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config
)
```

```python
result
```

Because the user's request will require calling `send_email_tool`, the run **pauses** here — `result` contains an `"__interrupt__"` entry rather than a completed response.

```python
from langgraph.types import Command
# Step 2: Approve
if "__interrupt__" in result:
    print("⏸️ Paused! Approving...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config
    )
    
    print(f"✅ Result: {result['messages'][-1].content}")
```

```python
result
```

- `Command(resume={"decisions": [{"type": "approve"}]})` is how you resume a paused agent run — it's passed *instead of* a normal `{"messages": [...]}` dict.
- Using the **same `config`** (same `thread_id`) is what lets LangChain correctly resume the exact paused execution rather than starting a new one.
- After approving, `send_email_tool` actually executes and the agent produces its final confirmation message ("Email sent to john@test.com with subject 'Hello'").

### Reject flow

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
```

```python
config = {"configurable": {"thread_id": "test-reject"}}
# Step 1: Request
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config)
```

```python
# Step 2: Reject
if "__interrupt__" in result:
    print("⏸️ Paused! Approving...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "reject"}
                ]
            }
        ),
        config=config
    )
    
    print(f"✅ Result: {result['messages'][-1].content}")
```

```python
result
```

Identical setup to the approve example (rebuilt with a fresh `thread_id`, `"test-reject"`), but this time `Command(resume={"decisions": [{"type": "reject"}]})` is sent instead. `send_email_tool` never actually executes — the agent instead reports that the action was declined (e.g. *"It seems that you have decided not to proceed..."*).

### Edit flow

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
```

```python
config = {"configurable": {"thread_id": "test-edit"}}

# Step 1: Request (with wrong info)
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to wrong@email.com with subject 'Test' and body 'Hello'")]},
    config=config
)
```

```python
result
```

This time the user's own request has an intentionally wrong recipient (`wrong@email.com`), simulating a human catching and correcting a mistake at approval time.

```python
# Step 2: Edit and approve
if "__interrupt__" in result:
    print("⏸️ Paused! Editing...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email_tool",      # Tool name
                            "args": {                   # New arguments
                                "recipient": "correct@email.com",
                                "subject": "Corrected Subject",
                                "body": "This was edited by human before sending"
                            }
                        }
                    }
                ]
            }
        ),
        config=config
    )
    
    print(f"✏️ Result: {result['messages'][-1].content}")
```

```python
result
```

The `"type": "edit"` decision includes an `"edited_action"` dict specifying the **corrected** tool name and args — here the recipient, subject, and body are all overwritten by the human before the tool actually runs. `send_email_tool` then executes with the *edited* arguments, not the model's original (wrong) ones — proving a human can intercept and fix an agent's proposed action rather than just binary approve/reject it.

**Summary of Human-in-the-Loop:** Requires a `checkpointer` + consistent `thread_id`. Configure per-tool via `interrupt_on={"tool_name": {"allowed_decisions": [...]}}` (or `False` to skip the gate entirely). When paused, `result` contains `"__interrupt__"`; resume with `agent.invoke(Command(resume={"decisions": [...]}), config)` using `{"type": "approve"}`, `{"type": "reject"}`, or `{"type": "edit", "edited_action": {...}}`.

---

# 7. Guardrails

**By Krish Naik | KRISHAI Technologies** — *This notebook covers everything you need to know about implementing Guardrails in LangChain agents using the middleware system.*

**Topics covered:** what guardrails are & why they matter; deterministic vs. model-based approaches; built-in PII detection middleware; built-in human-in-the-loop middleware; custom before-agent guardrail (input filtering); custom after-agent guardrail (output safety); layered/combined guardrails; a real-world healthcare-chatbot use case.

> Docs reference: https://docs.langchain.com/oss/python/langchain/guardrails

### Setup

```python
from dotenv import load_dotenv
load_dotenv()
```

```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

## 7.1 What Are Guardrails?

> *Guardrails help you build **safe, compliant AI applications** by validating and filtering content at key points in your agent's execution. They are implemented as **middleware** that intercepts execution:*
> - *Before the agent starts (input guardrails)*
> - *After it completes (output guardrails)*
> - *Around model and tool calls*

**Common use cases:**

| Use Case | Example |
|---|---|
| PII leakage prevention | Redact emails/credit cards before logging |
| Prompt injection blocking | Detect adversarial inputs |
| Harmful content filtering | Block dangerous requests |
| Business rule enforcement | Require approval for financial ops |
| Output quality validation | Ensure response meets safety standards |

## 7.2 Two Approaches to Guardrails

**Deterministic Guardrails** — rule-based (regex, keyword matching, explicit checks). ✅ Fast, predictable, cost-effective. ❌ May miss nuanced violations.

**Model-Based Guardrails** — uses LLMs/classifiers for semantic understanding. ✅ Catches subtle/nuanced issues. ❌ Slower and more expensive.

```python
# Quick illustration of the two approaches

import re

# --- Deterministic approach ---
def deterministic_guardrail(text: str) -> bool:
    """Returns True if content is blocked."""
    banned_keywords = ["hack", "exploit", "malware", "bomb"]
    return any(kw in text.lower() for kw in banned_keywords)

test_inputs = [
    "How do I hack into a database?",
    "What is the capital of France?",
    "Explain how malware spreads",
]

print("=== Deterministic Guardrail Demo ===")
for inp in test_inputs:
    blocked = deterministic_guardrail(inp)
    status = "🚫 BLOCKED" if blocked else "✅ ALLOWED"
    print(f"{status}: {inp}")
```

Note the third test case: *"Explain how malware spreads"* is a **legitimate educational question**, but the naive keyword check blocks it anyway (it contains "malware") — this is exactly the "may miss nuance / false positives" weakness called out above.

```python
from langchain_openai import ChatOpenAI

# --- Model-based approach ---
def model_based_guardrail(text: str) -> str:
    """Uses an LLM to evaluate content safety. Returns SAFE or UNSAFE."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Is the following user input safe to process? 
Reply with only 'SAFE' or 'UNSAFE'.

Input: {text}"""
    result = model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()

print("=== Model-Based Guardrail Demo ===")
for inp in test_inputs:
    verdict = model_based_guardrail(inp)
    status = "🚫 UNSAFE" if "UNSAFE" in verdict else "✅ SAFE"
    print(f"{status}: {inp}")
```

Run against the same three inputs, the model-based check correctly distinguishes the genuinely malicious request ("how do I hack into a database?") from the educational one ("explain how malware spreads") — because it understands *intent*, not just keyword presence. `temperature=0` is used to make the safe/unsafe classification as deterministic and repeatable as possible.

## 7.3 Built-in Guardrail — PII Detection Middleware

LangChain's built-in `PIIMiddleware` detects and handles **Personally Identifiable Information**.

**Supported PII types:** `email` (`user@example.com`), `credit_card` (`5105-1051-0510-5100`), `ip` (`192.168.1.1`), `mac_address` (`00:1A:2B:3C:4D:5E`), `url` (`https://secret-site.com`).

**Strategies:** `redact` → `[REDACTED_EMAIL]`; `mask` → `****-****-****-1234`; `hash` → `a8f5f167...`; `block` → raises an exception.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define a simple dummy tool
@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"

# Create agent with PII Middleware
agent = create_agent(
    model="gpt-4o",
    tools=[customer_lookup],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

print("Agent with PII middleware created successfully!")
```

Three `PIIMiddleware` instances are stacked, each targeting a different PII type with a **different strategy**: emails get *redacted* (replaced with a placeholder before the model ever sees the real value), credit card numbers get *masked* (partially hidden), and anything matching a custom API-key-shaped regex (`detector=r"sk-[a-zA-Z0-9]{32}"`) gets *blocked* outright (raises an exception rather than continuing).

```python
# Test PII Redaction
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is john.doe@example.com and my card is 5105-1051-0510-5100. Can you help me?"
    }]
})

print("=== Agent Response ===")
print(result["messages"][-1].content)
```

```python
result
```

The email and credit card in the user's message are redacted/masked **before** the underlying model ever processes the request — inspecting the full `result` shows the actual `HumanMessage` content the model received had the sensitive values already replaced.

```python
# Test API Key Blocking
try:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here is my key: sk-abcdefghijklmnopqrstuvwxyz123456"
        }]
    })
    
except Exception as e:
    print(f"🚫 Blocked as expected: {e}")
```

```python
result
```

Because `"api_key"` uses `strategy="block"`, sending a string matching the `sk-[a-zA-Z0-9]{32}` pattern raises an exception rather than silently continuing — the run is stopped immediately, and the `try/except` catches and reports the block.

## 7.4 Built-in Guardrail — Human-in-the-Loop Middleware

*Pauses agent execution before sensitive operations and waits for human approval.* Best for: financial transactions, sending emails to external parties, deleting production data, any operation with significant business impact. **Key requirement:** a `checkpointer` for state persistence across interrupts.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"

# Create agent with HITL middleware
hitl_agent = create_agent(
    model="gpt-4o",
    tools=[search_web, send_email, delete_records],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,       # Require approval
                "delete_records": True,   # Require approval
                "search_web": False,      # Auto-approve
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # Required for state persistence
)

print("Human-in-the-Loop agent created!")
```

Note this example uses the simpler boolean form (`"send_email": True`) rather than the `{"allowed_decisions": [...]}` dict form seen in the middleware notebook — both are valid: `True` gates the tool behind human approval with the default set of decisions, while the dict form lets you restrict exactly which decision types (approve/edit/reject) are allowed for that specific tool.

```python
# Step 1: Invoke — agent will pause before send_email
config = {"configurable": {"thread_id": "session_001"}}

result = hitl_agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to team@company.com about the Q4 results"}]},
    config=config
)

print("=== Agent paused — awaiting human approval ===")
print(result)
```

```python
# Step 2: Human reviews and APPROVES
approved_result = hitl_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config   # Same thread_id resumes the paused session
)

print("=== Approved! Final response ===")
print(approved_result["messages"][-1].content)
```

```python
# Step 3: Alternative — Human REJECTS
config2 = {"configurable": {"thread_id": "session_002"}}

hitl_agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from the users table where active=false"}]},
    config=config2
)

rejected_result = hitl_agent.invoke(
    Command(resume={"decisions": [{"type": "reject", "reason": "Too risky, needs DBA review"}]}),
    config=config2
)

print("=== Rejected! Final response ===")
print(rejected_result["messages"][-1].content)
```

Two independent `thread_id`s (`"session_001"`, `"session_002"`) demonstrate the approve and reject paths side-by-side in the same run. Note the reject `Command` here also carries an optional `"reason"` field, giving the model context on *why* it was rejected — useful so the agent's follow-up response can acknowledge the specific concern rather than just a generic "declined" message.

## 7.5 Custom Guardrail — Before-Agent Hook (Input Filter)

Use `before_agent()` to validate or block requests **before any LLM processing begins**. Best for: keyword/content filtering, authentication checks, rate limiting, blocking specific categories of requests.

```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool

class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"🚫 Blocked — keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# Create agent with content filter
filtered_agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass"]
        ),
    ],
)

print("Content filter agent created!")
```

Key implementation details of a **custom** middleware class:
- Subclass `AgentMiddleware` and override the `before_agent(self, state, runtime)` method.
- `@hook_config(can_jump_to=["end"])` — declares up front which parts of the graph this hook is allowed to short-circuit to; here it can jump straight to `"end"`.
- `state["messages"]` gives you the current conversation state — the code checks the *first* message, confirms it's a human message, then lowercases and scans it against the banned keyword list.
- If a match is found, it returns a dict with a canned `"assistant"` refusal message **and** `"jump_to": "end"` — this skips the model call entirely (hence "zero LLM cost for blocked requests," as the docstring notes) and routes straight to the end of the graph.
- If nothing matches, returning `None` means "no objection, let execution continue normally."

```python
# Test 1: Safe request — should pass through
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": "What is machine learning?"}]
})
print("✅ Safe request response:")
print(result["messages"][-1].content)
```

```python
# Test 2: Unsafe request — should be blocked
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a server?"}]
})
print("🚫 Unsafe request response:")
print(result["messages"][-1].content)
```

The safe question flows through normally and the model actually answers it; the unsafe one never reaches the model at all — it's intercepted by `before_agent` and immediately returns the canned refusal message.

## 7.6 Custom Guardrail — After-Agent Hook (Output Safety)

Use `after_agent()` to validate the final agent response **before the user sees it**. Best for: model-based safety evaluation of outputs, compliance scanning (legal/medical/financial disclaimers), quality validation, removing sensitive info that slipped through.

```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        # Use a smaller, cheaper model for the safety check
        self.safety_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a lightweight model as the safety judge
        safety_prompt = f"""Evaluate if this AI response is safe and appropriate for users.
Respond with only 'SAFE' or 'UNSAFE'.

Response to evaluate:
{last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in result.content.upper():
            print("⚠️  Output flagged as UNSAFE — replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"


safe_agent = create_agent(
    model="gpt-4o",
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

print("Output safety agent created!")
```

This mirrors the input-side `ContentFilterMiddleware`, but implements `after_agent` instead: it inspects the *last* message in state, confirms it's an `AIMessage` (i.e. the model actually produced a final answer, as opposed to some other state), and asks a cheap judge model (`gpt-4o-mini`) whether that response is safe. Note it **mutates `last_message.content` directly** and returns `None` (rather than constructing a new dict + `jump_to`) — a slightly different pattern than the before-agent example, since here it's *editing the existing final message in place* rather than short-circuiting the graph.

```python
# Test output safety check
result = safe_agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like today?"}]
})
print("Response:")
print(result["messages"][-1].content)
```

## 7.7 Layered / Combined Guardrails

Stack multiple guardrails in the `middleware=[]` array. They execute **in order**, building layered protection:

```
User Input
    ↓
[Layer 1] ContentFilterMiddleware    ← Deterministic input filter
    ↓
[Layer 2] PIIMiddleware (input)      ← PII redaction on input
    ↓
[Layer 3] HumanInTheLoopMiddleware   ← Approval for sensitive tools
    ↓
[Layer 4] PIIMiddleware (output)     ← PII redaction on output
    ↓
[Layer 5] SafetyGuardrailMiddleware  ← Model-based output safety
    ↓
User Response
```

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Full layered guardrail stack
production_agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

        # Layer 2: PII redaction on input
       
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email_tool": True, "search_tool": False}
        ),

        # Layer 4: PII redaction on output
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 5: Model-based output safety
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=InMemorySaver(),
)

print("🏭 Production-grade agent with 5-layer guardrails created!")
```

This is the notebook's flagship example of composing everything covered so far into one agent: custom keyword filter → PII masking on the way in → human approval gating the risky `send_email_tool` → PII redaction on the way *out* (`apply_to_output=True`, different from the input-side redaction used earlier) → a final model-based safety check on the finished response. Each middleware only concerns itself with its own narrow job; stacking them is what builds "defense in depth."

## 7.8 Real-World Use Case — Healthcare Chatbot

A healthcare chatbot that: (1) **blocks** off-topic or harmful requests, (2) **redacts** patient PII (emails, credit card numbers), (3) **requires human approval** before booking appointments, (4) **validates** that outputs are medically appropriate.

```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# --- Healthcare-specific content filter ---
class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    BLOCKED_TOPICS = ["drug synthesis", "self-harm", "suicide method", "weapon", "hack"]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_msg = state["messages"][0]
        if first_msg.type != "human":
            return None

        content = first_msg.content.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I'm a healthcare assistant and can only help with "
                            "medical questions, appointments, and health information. "
                            "If you're in crisis, please call 112 or your local emergency number."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


# --- Medical output validator ---
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = "\n\n⚕️ *This is general health information, not medical advice. Please consult a qualified healthcare professional.*"

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Add disclaimer if not already present
        if "medical advice" not in last_message.content.lower():
            last_message.content += self.DISCLAIMER

        return None


# --- Healthcare tools ---
@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return f"Symptom information for: {symptoms}. Please consult a doctor for diagnosis."

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return f"Appointment booked for {patient_name} with Dr. {doctor} on {date}"

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return f"General info about {medication}. Always follow your doctor's prescription."


# --- Build the healthcare chatbot ---
healthcare_bot = create_agent(
    model="gpt-4o",
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # Guardrail 1: Block harmful/off-topic requests
        HealthcareSafetyFilter(),

        # Guardrail 2: Redact patient PII from inputs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Guardrail 3: Require approval before booking appointments
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,
            }
        ),

        # Guardrail 4: Add medical disclaimer to all outputs
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, and help book appointments. "
        "Always be empathetic and remind users to consult a doctor for diagnosis."
    )
)

print("🏥 Healthcare chatbot with full guardrail stack created!")
```

This ties together **every technique from the notebook** into one realistic system: a domain-specific `before_agent` filter (`HealthcareSafetyFilter`, blocking topics like drug synthesis or self-harm and pointing users to an emergency number instead), two `PIIMiddleware` layers protecting patient privacy, `HumanInTheLoopMiddleware` gating only the one genuinely consequential tool (`book_appointment`) while leaving the read-only tools (`search_symptoms`, `get_medication_info`) unrestricted, and a custom `after_agent` hook (`MedicalOutputValidator`) that appends a medical disclaimer to every response that doesn't already mention "medical advice."

```python
# Test 1: Safe medical query
config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
    config=config_t1
)

result
```

```python
# Test 2: Query with PII (email gets redacted)
result = healthcare_bot.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is patient123@gmail.com. What can I take for a headache?"
    }]},
    config=config_t1
)
print("=== PII Redaction Test ===")
print(result["messages"][-1].content)
```

```python
# Test 3: Off-topic / harmful request — gets blocked
result = healthcare_bot.invoke({
    "messages": [{"role": "user", "content": "How do I synthesize drugs at home?"}]
},
 config=config_t1)
print("=== Blocked Request ===")
print(result["messages"][-1].content)
```

```python
# Test 4: Appointment booking — requires human approval
config = {"configurable": {"thread_id": "healthcare_session_001"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": "Book me an appointment with Dr. Sharma on March 15"}]},
    config=config
)
print("=== Appointment Booking — Awaiting Approval ===")
print(result)

# Approve
from langgraph.types import Command
approved = healthcare_bot.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
print("\n=== After Approval ===")
print(approved["messages"][-1].content)
```

Four end-to-end scenarios, in order:
1. **Safe query** ("symptoms of Type 2 Diabetes") — flows straight through, `search_symptoms` tool is auto-approved, and the final response gets the medical disclaimer appended.
2. **PII in the query** — the patient's email is redacted before the model sees it, but the (unrelated) medical question still gets answered normally.
3. **Blocked topic** ("synthesize drugs") — matches `HealthcareSafetyFilter.BLOCKED_TOPICS`, so it's intercepted by `before_agent` and never reaches the model — the canned crisis-redirect message is returned instead.
4. **Appointment booking** — because `book_appointment` requires approval, the first call pauses (`"__interrupt__"` present); a second call with `Command(resume={"decisions": [{"type": "approve"}]})` on the same `thread_id` then lets the booking actually complete.

## Summary

| Guardrail Type | Hook | When it Runs | Best For |
|---|---|---|---|
| PII Middleware | Input/Output | Around model calls | Data privacy, compliance |
| Human-in-the-Loop | Tool level | Before sensitive tools | High-stakes decisions |
| Content Filter | `before_agent` | Start of invocation | Blocking bad inputs early |
| Safety Validator | `after_agent` | End of invocation | Output quality/safety |
| Custom Logic | Any hook | Anywhere | Any business rule |

**Key takeaways:**
1. **Guardrails = Middleware** — implement them via the `middleware=[]` parameter in `create_agent()`.
2. **Layer your guardrails** — defense in depth is best practice.
3. **Deterministic first, model-based second** — use cheap rule-based checks early to avoid expensive LLM calls.
4. **Human-in-the-Loop requires a checkpointer** — use `InMemorySaver` for dev, a persistent store for production.
5. **Custom middleware** gives you full control via `before_agent()` and `after_agent()` hooks.

**Additional resources referenced in the notebook:**
- [LangChain Guardrails Docs](https://docs.langchain.com/oss/python/langchain/guardrails)
- [Middleware Docs](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [Human-in-the-Loop Docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangSmith for Observability](https://docs.langchain.com/oss/python/langchain/observability)

---

# Quick Reference — All Imports Used Across the 7 Notebooks

```python
# Environment
import os
from dotenv import load_dotenv

# Agents
from langchain.agents import create_agent

# Models
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Tools
from langchain.tools import tool
from langchain_core.tools import tool   # equivalent alt import

# Messages
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage   # equivalent alt import

# Structured output
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass

# Middleware (built-in)
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware import PIIMiddleware

# Middleware (custom)
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

# Memory / checkpointing
from langgraph.checkpoint.memory import InMemorySaver

# Human-in-the-loop resume
from langgraph.types import Command
```

---

# Master Summary Table

| Notebook | Core Concept | What the code actually does |
|---|---|---|
| `1-langchainintro.ipynb` | Agents | `create_agent(model, tools, system_prompt)`; invoke with `{"messages": [...]}`; inspect `response["messages"]` |
| `2-modelintegration.ipynb` | Model loading, streaming, batch | `init_chat_model` vs. `ChatOpenAI`/`ChatGoogleGenerativeAI`/`ChatGroq`; `.stream()` with `end="|"`; `.batch()` with `max_concurrency` |
| `3-tools.ipynb` | Tools & execution loop | `@tool` decorator; `model.bind_tools()`; `response.tool_calls`; manual 3-step tool-execution loop |
| `4-messages.ipynb` | Message types | Text prompt vs. message list; `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`; `name`/`id` metadata; manual `AIMessage` construction; `usage_metadata` |
| `5-structuredoutput.ipynb` | Structured output | Pydantic `Movie`/`MovieDetails` (nested); `include_raw=True`; `TypedDict` with `Annotated`; `model.profile`; `dataclass`; `create_agent(response_format=...)` for all three schema types |
| `6-middleware.ipynb` | Middleware hooks | `SummarizationMiddleware` with `trigger`/`keep` as `("messages", n)`, `("tokens", n)`, `("fraction", n)`; `HumanInTheLoopMiddleware` approve/reject/edit via `Command(resume=...)` |
| `langchain_guardrails_crash_course.ipynb` | Guardrails | Deterministic vs. model-based checks; `PIIMiddleware` (redact/mask/block); custom `AgentMiddleware` subclasses with `before_agent`/`after_agent` + `hook_config`; layered 5-guardrail stack; full healthcare chatbot example |
