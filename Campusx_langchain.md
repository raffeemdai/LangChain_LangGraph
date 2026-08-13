# LangChain Notes

refer : https://www.youtube.com/watch?v=-xSJA8-o6Eg&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=4

These notes are converted from a two-part Hindi video transcript ("Introduction to LangChain" and "LangChain Components") into detailed English notes, covering all the content discussed in the videos.

---

# Video 1: Introduction to LangChain

## What is LangChain?

LangChain is an **open-source framework for developing applications powered by LLMs (Large Language Models)**.

In simple words: if you want to build any application based on an LLM, the framework that helps you build that application is LangChain.

However, this one-line definition alone does not make the importance of LangChain clear. To truly understand *what* something is, it is important to first understand *why* it was needed in the first place. So before defining LangChain in depth, the video explains **why LangChain was needed**.

---

## Why do we need LangChain?

To explain the need for LangChain, the instructor shares a personal example from his own experience.

### Personal Example — the PDF Reading App Idea

Around **2014**, when AI was not yet a big buzzword (at least in India), but smartphones had become more affordable and people had started reading more PDFs (instead of physical books), the instructor had an idea:

> What if I build an application where anyone can upload their PDFs and then not only read the PDF, but also **chat with it**?

For example, if someone uploads a Machine Learning textbook, then instead of just reading it, the user could:
- Ask: "Explain page 5 as if I am a 5-year-old" → get a simplified summary.
- Ask: "Generate some true/false questions on Linear Regression" → for practice.
- Ask: "Generate notes on Decision Trees from this book."

This kind of application would be extremely useful because the user can both **read** and **converse with** their document.

The instructor decided to work on this idea and next explains how he planned to build it — i.e., the system design.

---

## High-Level Discussion of the App (System Design)

### Step-by-step flow:

1. **User uploads a PDF** → the PDF is picked up and stored in a database.
2. **User opens the PDF and asks a query**, e.g., *"What are the assumptions of Linear Regression?"*
3. The system must **find out where in the book this topic is discussed** (which pages). This requires a **search operation**.

### Two types of search:

**1. Keyword Search**
- Directly searches for the exact words used in the query (e.g., "assumption", "linear regression") across the whole document.
- Problem: This is **inefficient** — many irrelevant pages may get returned just because a word appears there, even if not contextually relevant.

**2. Semantic Search**
- Instead of matching exact words, semantic search tries to understand the **meaning** of the query.
- It searches for content that is *semantically* related to "assumptions of linear regression" rather than literal word matches.
- Result: Fewer but **more meaningful/relevant** results.

### Continuing the flow:

- Semantic search returns, say, **2 relevant pages** (e.g., page 372 and page 461) where "assumptions of linear regression" is discussed.
- These pages + the user's original query are combined to form a **"system query"**.
- This system query is sent to the most important component of the application, which the instructor calls the **"Brain"**.

### The "Brain" Component

The Brain has **two purposes**:
1. **Understand the query** — it needs **NLU (Natural Language Understanding)** capability, so it can understand the query whether asked in English, Hindi, or any language.
2. **Generate context-aware text** — after understanding the query, it needs to search through the given pages and generate a relevant answer. This is called **context-aware text generation**.

So the Brain:
- Reads the given pages (the 2 retrieved pages).
- Extracts the 5 assumptions of Linear Regression from them.
- Generates the final text answer, which is shown to the user as the final output.

### Why not just send the entire book to the Brain instead of doing semantic search first?

This is a natural question: if the Brain (LLM) can understand a query and search for relevant content, why not just give it the entire 1000-page book directly instead of doing all this semantic search work?

**Analogy:** Imagine you're a student with a doubt in Algebra.
- **Scenario 1:** You go to your teacher and hand over the entire book, saying "I have a doubt in Algebra."
- **Scenario 2:** You go to your teacher and say "I have a doubt specifically on page 155."

Obviously, in Scenario 2, the teacher can respond faster and better because you gave a **specific, focused context**.

Similarly:
- Sending the whole book is **computationally expensive**.
- The results may also be **less accurate**, because the model has too much irrelevant content to sift through.

This is exactly why semantic search is implemented — to give the Brain a **small, relevant, focused context** rather than the entire document.

---

## Understanding Semantic Search in Detail

To understand the low-level system design, we must first understand how semantic search works.

### Example: 3 paragraphs about 3 cricketers

Suppose you have 3 paragraphs about Virat Kohli, Jasprit Bumrah, and Rohit Sharma. A question is asked, e.g., *"How many runs has Virat scored?"* We know intuitively the answer lies in the Virat Kohli paragraph — but how does code figure this out?

### The process:

1. **Convert all text into embeddings.**
   - "Embedding" means converting text into a **vector** (a set of numbers).
   - Techniques used: Word2Vec, Doc2Vec, BERT embeddings, etc.
   - The idea: represent the **semantic meaning** of a paragraph as numbers.
   - Suppose each vector has 100 dimensions.

2. Now we have 3 vectors (one for each paragraph) in a 100-dimensional space.

3. When a **query** comes in, it is also converted into a vector (embedding) in the same 100-dimensional space.

4. Now we have **4 vectors total** in the 100-dimensional space (3 paragraph vectors + 1 query vector).

5. **Compute similarity** between the query vector and each of the 3 paragraph vectors.

6. Whichever paragraph vector has the **strongest similarity** with the query vector is identified as the relevant paragraph, and that paragraph is used to generate the answer.

This is how semantic search works, and this same technique is applied on PDFs in our system.

---

## Detailed (Low-Level) System Design

Now that semantic search is understood, here is the complete detailed flow:

1. **User uploads a PDF** → stored in the cloud (example: AWS S3).
2. **Document Loader**: Loads the PDF into the system from cloud storage.
3. **Chunking**: The entire PDF is broken into smaller chunks. Chunking can be based on:
   - Chapters
   - Pages
   - Paragraphs
   - Example: A 1000-page PDF is split into 1000 chunks (page-wise), using a **Text Splitter**.
4. **Embedding Generation**: Each chunk (page) is passed through an **embedding model**, which generates a vector (embedding) for it. Now we have 1000 vectors for 1000 pages.
5. **Vector Storage**: These embeddings are stored in a **database** (specifically, a vector database) so that they can be queried in the future.
6. **User Query**: When the user opens the PDF and asks a query (a piece of text), the query is also converted into an embedding using the **same embedding model**.
7. **Similarity Search**: The query vector is compared against all 1000 stored vectors in the database — distances are calculated, and the **most similar vectors** (e.g., top 5) are retrieved.
8. **Extract Pages**: The pages corresponding to these top similar vectors are extracted.
9. **System Query Formation**: The original user query + the retrieved relevant pages are combined into a **system query**.
10. **Send to Brain (LLM)**: This system query goes to the app's Brain, where **NLU + context-aware text generation** happens.
11. **Final Output**: The generated answer is shown to the user.

---

## Challenges in Building This System

### Challenge 1: Building the "Brain"

The Brain component needs to:
1. Fully understand any given query (NLU).
2. Generate relevant, context-aware text as a response.

This was a huge research challenge in NLP for a long time. The real breakthrough came in **2017** with the **Transformers paper**, followed by **BERT** and **GPT** papers, which finally solved this problem. 

**Solution:** We don't need to build this Brain from scratch — **LLMs already exist** in the market with both NLU and context-aware text generation capabilities. We simply need to **use an LLM** as the Brain.

### Challenge 2: Computation

LLMs are massive — often **more than 100GB** in size (billions of parameters), trained on huge internet-scale data. Hosting and running such large models on your own server is:
- Extremely engineering-intensive.
- Very costly (in terms of compute + cloud costs).

**Solution:** Big companies like **OpenAI** and **Anthropic** have already hosted these LLMs on their own servers and exposed them via **APIs**. This means:
- You don't need to host the LLM yourself.
- You simply call the API with your query, and the LLM returns a reply through the API.
- You pay only for what you use (usage-based billing).

So instead of using "an LLM" directly, in practice we use an **"LLM API."**

### Challenge 3: Orchestration

The third big challenge is **orchestrating this entire system** — connecting and managing all the different components together.

Components involved (5 total):
1. AWS S3 (document storage)
2. Text Splitter
3. Embedding model
4. Vector Database
5. LLM

Tasks performed (multiple):
- Loading the document
- Splitting text
- Generating embeddings
- Managing the database
- Retrieval
- Talking to the LLM

Writing all this code from scratch is **very difficult**, and it becomes even harder to maintain. For example, if you decide to switch from OpenAI's API to another provider because of cost, or switch cloud providers, or switch embedding models — you'd have to rewrite large chunks of your hand-written integration code. There are many moving parts with a lot of interaction between them, making hand-coding this system extremely challenging.

**This is exactly where LangChain comes into the picture.**

LangChain provides **built-in functionalities** that let you plug-and-play connect all these components with each other. If tomorrow you want to switch from OpenAI to Google's models, it becomes a matter of changing very little code — because LangChain handles all the boilerplate code and integration behind the scenes.

### Summary of the Core Idea

If you want to build an LLM-powered application:
- The LLM itself does most of the "heavy lifting" (understanding + generating text).
- But running the **entire end-to-end application** with all its moving components is very difficult — especially because this technology is still new.
- **LangChain lets you focus on your idea**, while it handles the orchestration and interfacing work for you.

---

## Benefits of LangChain

1. **Chains**
   - This is such a central concept that LangChain gets its name from it.
   - Chains let you connect different components/tasks into a **pipeline**.
   - The biggest benefit of chains: **the output of one component automatically becomes the input of the next component** — no need to manually write this connecting code.
   - You can build complex chains: **sequential chains, parallel chains, conditional chains**, etc., in a highly expressive way.

2. **Model-Agnostic Development**
   - You can use *any* LLM provider (OpenAI, Google, etc.) — switching providers typically requires changing just 1–2 lines of code.
   - You can focus on your core business logic; the underlying model/component can be swapped easily.

3. **Complete Ecosystem**
   - LangChain provides interfaces for almost every type of component you might need:
     - Document loaders (cloud files, local files, PDFs, etc.)
     - ~50 types of text splitters
     - Many embedding models
     - Many types of vector databases
   - Whatever product/component your company wants to work with, LangChain likely has an interface for it.

4. **Memory and State Handling**
   - Example: User asks "What are the assumptions of Linear Regression?" → gets an answer. Then asks, "Also give me a few interview questions on this ML algorithm" — without memory, the system wouldn't know "this algorithm" refers to Linear Regression.
   - LangChain solves this via its **memory** concept — enabling conversational memory, so the model understands references to earlier parts of the conversation without them being restated.

**Summary:** LangChain is a very good tool/library that greatly helps in building LLM-powered applications.

---

## What Can You Build With LangChain?

Several major use cases:

1. **Conversational Chatbots**
   - Very popular use case — especially for internet-based companies (e.g., Uber, Swiggy-like companies) that deal with **scale** (many customers at once).
   - Instead of large call centers, companies build chatbots that act like a call center executive — understanding queries and providing solutions.
   - The chatbot handles the first layer of communication; unresolved queries are forwarded to a human.

2. **AI Knowledge Assistants**
   - Similar to a chatbot, but trained/connected to your own specific data.
   - Example: A chatbot integrated into an online course website, where a student watching a lecture video can ask doubts specific to that lecture's content, and the chatbot (which has knowledge of that lecture) can answer.

3. **AI Agents**
   - A very popular term in the last year. 
   - Agents are like "**chatbots on steroids**" — they don't just talk, they can also **perform actions/tasks**.
   - Example: On a travel booking site like MakeMyTrip, an AI agent could not only converse (helpful for less tech-savvy/older users) but also **actually book** a flight ticket on the user's behalf — not just talk about it.
   - AI agents are considered "the next big thing" in AI, and LangChain makes it possible to build agents (a basic agent will be demonstrated later in the playlist).

4. **Workflow Automation**
   - LLMs (and hence LangChain) can be used to automate workflows at a personal, professional, or company level.

5. **Summarization and Research Helpers**
   - Useful because tools like ChatGPT have limitations on how much you can upload (context length problems) and companies often restrict uploading private/sensitive data to third-party tools like ChatGPT.
   - Using LangChain, a company can build its own "ChatGPT-like" tool that:
     - Can process very large documents.
     - Can be trained/connected to the company's own private data securely.

The instructor notes that the future looks bright for LLM-based applications — similar to how websites and apps had their boom eras, LLM-based applications are expected to have a similar boom, with LangChain playing a key role.

---

## Alternatives to LangChain

LangChain is **not the only framework** for building LLM applications. Two other popular frameworks mentioned:

1. **LlamaIndex** — Slightly more popular; there's a dedicated course on it on the instructor's platform.
2. **Haystack** — A similar kind of library/framework/platform for building LLM-based applications easily.

The choice between LangChain, LlamaIndex, and Haystack depends on factors like pricing and which tool best suits the specific need. A detailed comparative study between these three was deferred to a future video, since LangChain itself hadn't been studied in depth yet at that point.

---
---

# Video 2: LangChain Components

## Recap of Previous Video

The previous video covered:
- **Definition:** LangChain is an open-source framework to help build LLM-powered applications.
- **Why LangChain is needed:** Using the PDF chat application example and its system design, showing how many components and interactions are involved, and how difficult it would be to hand-code such a system from scratch.
- **Key Benefit:** LangChain lets you orchestrate all components efficiently, letting you build a pipeline with minimal code for maximum output.
- **Chains concept**: Where the output of one component automatically becomes the input to the next.
- **Model-agnostic nature**: Switching between LLM providers (e.g., OpenAI ↔ Google) requires very little code change.
- **Use cases discussed**: Conversational chatbots, AI knowledge assistants, and AI agents.

(Note: If you haven't watched the first video, it's recommended to watch it first for better context.)

## Introduction to This Video

This video focuses on understanding the **6 core components of LangChain**. No coding/projects are done in this video — the goal is purely conceptual/theoretical, since the instructor believes it's important to build a strong conceptual foundation before jumping into code (unlike many online resources that jump directly into projects without first explaining the fundamentals).

---

## Overview: The 6 Components of LangChain

LangChain consists of **6 main components**:

1. **Models**
2. **Prompts**
3. **Chains**
4. **Memory**
5. **Indexes**
6. **Agents**

Understanding these 6 components gives you the majority of LangChain's conceptual foundation. The rest of the playlist is structured around deep-diving into each of these.

---

## 1. Models

> "In LangChain, models are core interfaces through which you interact with AI models."

### Backstory / Why This Component Exists

In the history of NLP, everyone wanted to build the most popular NLP application: the **chatbot**. But building chatbots had **two big problems** (which are largely solved today):

1. **Understanding the user's query** (NLU — Natural Language Understanding).
   - E.g., understanding what "Hi, can you check my email" means.
2. **Generating a relevant reply** even after understanding the query — this is **context-aware text generation**.

**LLMs solved both problems simultaneously.** Because LLMs were trained on almost the entire internet's data, they developed both:
- A strong understanding of natural language.
- The capability for context-aware text generation.

### New Problem After LLMs: Size

Since LLMs are trained on massive datasets, they have **billions of parameters**, making their file sizes huge — often **greater than 100GB**. This is far too large for:
- A normal individual to run on their personal computer.
- Small/medium companies to host on their own servers (cloud costs would be too high).

**Solution: APIs**
Big companies (OpenAI, Anthropic, Google, etc.) hosted these LLMs on their own servers and exposed **APIs**. Now anyone can:
- Send a query by hitting the API.
- The API talks to the LLM.
- The LLM's response comes back through the API to the user.

Benefit: You don't need to host or run the LLM yourself — you only pay for what you use.

### The Third Problem: Standardization

Different LLM providers built their APIs in **different ways**. This means:
- If an application developer wants to use **two different LLM APIs** in one application, they need to write **different types of code** for each.
- Example shown: The code to talk to OpenAI's GPT models is different from the code to talk to Anthropic's Claude Sonnet API — even though both are conceptually doing the same thing.
- If a developer built their app initially with OpenAI's API and later wants to switch to Claude's API (e.g., due to cost), they'd have to **rewrite a significant part of their codebase**.

**This is the "Models" component's core purpose**: LangChain **standardizes** the interface for talking to different AI model providers. This means:
- Minimal code changes are needed to switch between providers.
- With LangChain, switching from OpenAI to Claude, for example, requires changing only about **1–2 lines of code** (mainly the import/package statement) — the rest of the calling code and how you process the output remains largely the same.

### Two Types of Models in LangChain

1. **Language Models**
   - These are LLMs that take **text as input** and give **text as output** (text-in, text-out philosophy).
   - Example: Input "How are you today?" → Output "I am good, how about you?"
   - Used to build chatbots, AI agents, and most text-based applications.

2. **Embedding Models**
   - These take **text as input** but return a **vector** as output (instead of text).
   - Their main use case is **semantic search** (as discussed in the previous video).

LangChain supports communication with both types of models.

### Available Providers (from documentation)

Under LangChain's documentation, in the "Chat Models" section, you can find providers such as:
- Anthropic (Claude)
- Mistral AI
- Azure
- OpenAI
- Vertex AI (Google)
- Bedrock (AWS)
- Hugging Face
- ...and many more.

The documentation also shows what features are available for each model, such as:
- Tool calling support (important for building agents).
- Structured output (e.g., JSON mode).
- Whether the model can be run locally.
- Whether multimodal input is supported.

There's a similar list/page for **embedding models** as well (OpenAI, Mistral AI, IBM, LLaMA, etc.).

### Summary

The Models component is an **interface** for communicating with any AI model. Its core value: it **standardizes** the previously fragmented landscape of different provider APIs, so you can switch between LLM providers with minimal code change.

---

## 2. Prompts

### What is a Prompt?

A **prompt** is the input you provide to an LLM.

Example: If you ask ChatGPT "What is Campus X?" — the string "What is Campus X?" is the prompt.

### Why Prompts Matter

Prompts are **extremely important** in the world of LLMs. LLM outputs are highly dependent on — in fact, very sensitive to — the exact prompt used. Even a small change in wording can drastically change the output.

**Example**: 
- "Explain linear regression in academic tone" 
- vs. "Explain linear regression in fun tone"

Just one word changed ("academic" → "fun"), but the output will be very different.

Because of this sensitivity, an entire field of study has emerged around prompts in the last two years: **Prompt Engineering**, with a corresponding job role called **Prompt Engineer** (though the term has become somewhat of a meme on social media, it remains an important field of study around LLMs).

Since prompts are so crucial, LangChain developed a dedicated **Prompts component** to handle them, offering a lot of flexibility for creating different and powerful types of prompts.

### Types of Prompts You Can Create in LangChain

1. **Dynamic and Reusable Prompts**
   - You may not know in advance what topic or tone a user will ask for a summary in.
   - You can create a prompt template with **placeholders**, e.g.:
     - `"Summarize {topic} in {tone} tone"`
   - At runtime, placeholders get filled in based on user input:
     - E.g., User: "Tell me about cricket in a fun tone" → placeholders filled with "cricket" and "fun."
   - This same template can be reused later for different topics/tones (e.g., "biology" in a "serious" tone) by different users.

2. **Role-Based Prompts**
   - You define a **system-level prompt** with a role placeholder, e.g.:
     - `"Hi, you are an experienced {profession}"`
   - Then a **user-level prompt**, e.g.:
     - `"Tell me about {topic}"`
   - At runtime: "You are an experienced doctor" + "Tell me about viral fever" → the LLM responds as an experienced doctor would.
   - Another user could later use: "You are an experienced engineer" + "Tell me about developing bridges."

3. **Few-Shot Prompts**
   - You show the LLM a few **examples** first, then ask it a new question based on the pattern from those examples.
   - Example scenario: Building a customer support chatbot that classifies tickets into categories.
   - Example training data given to the LLM:
     - "I was charged twice for my subscription this month" → **Billing Issue**
     - "The app crashes every time I try to log in" → **Technical Issue**
     - "Can you explain how to upgrade my plan?" → **General Inquiry**
   - A template is created for how the ticket/query and category should be formatted.
   - A **Few-Shot Prompt Template** is then built by combining:
     - The example prompt/template.
     - Multiple such examples.
     - A new query at the end.
   - The prompt effectively asks: *"Based on the previous examples you've seen, classify the following customer support ticket into one of the following categories: Billing Issue, Technical Problem, or General Inquiry."*
   - The LLM then returns the category for the new query, based on the pattern shown in the examples.

This is a very powerful component, and future videos in the playlist will dive deeper into it with actual code.

---

## 3. Chains

Chains are such a central concept that **LangChain is literally named after it**.

### What Are Chains?

Chains are a component that let you **build pipelines** in LangChain. Essentially, any LLM application you build can be structured as a **pipeline**, and that pipeline can be built using chains.

### Example: English-to-Hindi Summary App

**Goal:** Build an app where:
- Input: A large English text (~1000 words).
- Output: A Hindi summary in less than 100 words.

**Flow:**
1. Input text is sent to **LLM 1**, whose job is to **translate** the input into Hindi.
2. The translated (Hindi) text is then sent to **LLM 2**, whose job is to **generate a summary** in Hindi, under 100 words.

### Without Chains (Manual Approach)

Without using chains, you'd have to manually:
- Take user input.
- Call LLM 1, feed it the input, instruct it to translate to Hindi.
- Receive the Hindi translation.
- Manually take that output and feed it into LLM 2, instructing it to summarize.
- Manually retrieve the final output.

This means manually taking the output of every stage and manually feeding it as input to the next stage.

### With Chains

**Chains solve this problem** — they **automatically** make the output of one stage become the input of the next stage, with **no manual code required** for this connection.

So with chains, you simply:
- Provide the English text as input.
- Call the chain.
- Behind the scenes, the entire task executes automatically, and you directly get the final result — no need to manually manage intermediate LLM calls.

### Types of Chains

1. **Sequential Chain**
   - The simple example above (Translate → Summarize) is a sequential chain — one stage follows another.

2. **Parallel Chain**
   - Example: Building an app where the user gives an input (e.g., "911 Incident") and the app returns a **detailed report** by combining outputs from **multiple LLMs**.
   - Flow:
     - Input is sent **simultaneously** to LLM 1 and LLM 2.
     - LLM 1 generates a report on the topic.
     - LLM 2 also generates a report on the topic (perhaps differently).
     - Both outputs are then sent to a **third LLM**, whose job is to **combine** both reports into one.
     - The combined result is shown to the user as the final output.
   - This entire flow can easily be executed using chains.

3. **Conditional Chain**
   - Example: Building an AI agent that receives user feedback.
     - The user is asked: "How did you find our service?"
     - The user's feedback is processed by an LLM.
     - **If the feedback is good** → the agent thanks the user, and the task ends there.
     - **If the feedback is bad** → the agent immediately sends an email to the customer support team.
   - Here, processing branches based on a **condition** — and this too can be easily implemented using chains.

### Summary

Chains let you build **many types of complex pipelines** very elegantly and efficiently, drastically minimizing manual hard work. This component will be studied in more depth later in the playlist.

---

## 4. Indexes

> "Indexes connect your application to external knowledge, such as PDFs, websites, and databases."

### Sub-components of Indexes

Indexes consist of **4 main parts**:
1. **Document Loader**
2. **Text Splitter**
3. **Vector Store**
4. **Retriever**

### Why Are Indexes Needed?

ChatGPT (and similar tools) can answer most general queries because it was trained on a huge amount of internet data. However, there are certain scenarios where ChatGPT **cannot** answer — particularly questions about **private/specific data** it was never trained on.

**Example:**
- If you work at a company "XYZ" and ask ChatGPT:
  - *"What is the leave policy of my company XYZ?"*
  - *"What is the notice period policy of my company XYZ?"*
- ChatGPT **cannot answer** these, because this is private company data that it never saw during training.

### The Solution: External Knowledge Connection

You can connect an LLM to an **external knowledge source**. For example:
- Take an LLM and provide it with your company's complete rulebook/policy document.
- Now:
  - General questions (e.g., "Who is the Prime Minister of India?") — the LLM can answer directly from its training.
  - Company-specific questions (e.g., "What is the leave policy of XYZ?") — the LLM can now also answer, because it has access to this **external knowledge source**.

This is exactly what the **Indexes** component enables — and its 4 sub-components handle this process.

### How This System Is Implemented (Same as Video 1's System Design)

1. **Document Loader**: Load the source document (e.g., rulebook PDF) from wherever it is stored (e.g., Google Drive).
2. **Text Splitter**: Break the document into small chunks (pages, paragraphs, chapters) — e.g., a 1000-page rulebook becomes 1000 chunks.
3. **Vector Store (Embedding + Storage)**: Each chunk is converted into an embedding (vector) using an embedding model, and stored in a special database called a **vector database / vector store**.
4. **Retriever**: When a user query comes in (e.g., "What is the leave policy of XYZ?"):
   - The retriever generates an embedding for the query using the same embedding model.
   - It performs a semantic search against the vector store.
   - Relevant results are retrieved.
   - These results + the original query are sent to the LLM, which generates the final reply.

### Summary

Indexes provide a way to build LLM applications that have access to **external knowledge sources** — which can be PDFs, websites, or any company database. This is highly flexible, and practical projects using this will be built later in the playlist.

---

## 5. Memory

> "LLM API calls are stateless, and this is a big problem."

### The Problem: Statelessness

When you make API calls to an LLM, **each call is independent (stateless)** — meaning the LLM has **no memory of previous requests**.

**Example demonstrating this:**
1. Query 1 to the API: *"Who is Narendra Modi?"*
   - Response: "Narendra Modi is an Indian politician who is the current Prime Minister of India."
2. Query 2 (in the same "conversation" from the user's perspective): *"How old is he?"*
   - Response: "As an AI, I don't have access to personal data about individuals unless it has been shared with me."
   - This shows the model has **no memory** of the previous question about Narendra Modi — it doesn't know who "he" refers to.

Every request is treated **independently**, with no memory of prior requests. This is a **huge problem** for building conversational applications — imagine how frustrating it would be to talk to a chatbot that has no memory of the ongoing conversation, forcing you to repeat context every time.

### The Solution: LangChain's Memory Component

The **Memory** component solves exactly this problem — it lets you add memory features to your entire conversation flow.

### Types of Memory in LangChain

1. **Conversation Buffer Memory**
   - Stores the **entire conversation history** so far.
   - On every new API call, the full chat history is sent along with the new query, so the model understands the context.
   - **Drawback**: If the chat becomes very long, the chat history becomes huge too, leading to higher processing costs (more text = more money).

2. **Conversation Buffer Window Memory**
   - Stores only the **last N interactions** (e.g., last 100 messages).
   - This list constantly updates, and only the most recent N interactions are sent with each new API call.

3. **Summarizer-Based Memory**
   - Instead of storing the full chat history, a **summary** of the conversation so far is generated and sent with each API call.
   - This helps save on text/tokens and reduces cost.

4. **Custom Memory**
   - Used for more advanced use cases — stores specialized pieces of information such as user preferences, specific facts/figures, etc., which help make future conversations smoother.

This is described as a very interesting and highly practical topic that will be covered in more depth later in the playlist.

---

## 6. Agents

Agents are the component that lets you easily build **AI agents** in LangChain.

### Why Agents Are a Big Deal

"AI agents" have been a hot topic in AI circles over the last 6 months, with many people saying agents are going to be "the next big thing" in AI.

### From Chatbots to Agents

LLMs have two key strengths:
1. **NLU (Natural Language Understanding)**
2. **Text Generation**

Since LLMs understand language and can generate correct/relevant responses, the most obvious first application was **chatbots** — and today's most popular AI application, ChatGPT, is itself a chatbot.

Eventually, people realized: if a chatbot can understand and reply well, could it also **perform actions**?

### Chatbot vs. AI Agent — Example

**Scenario: Talking to a travel website's chatbot (e.g., MakeMyTrip)**

1. **As a chatbot**: 
   - You ask: "What is the best travel destination in India during summer?"
   - Since the chatbot is based on an LLM trained on internet data, it knows hill stations are typically good summer spots, and replies: "You can go to Shimla" or "You can go to Manali."

2. **As an AI agent** (with tool access):
   - You ask: "What is the cheapest flight from Delhi to Shimla on 24th January?"
   - The agent hits a relevant API and fetches the actual answer: e.g., "IndiGo has the cheapest flight on 24th January from Delhi to Shimla."
   - You can go further: "Can you book the flight?" — and since the agent has this capability, it will actually go and **book the flight** on the website.

**This is the core difference between a chatbot and an AI agent.**

> "An AI agent is like a chatbot with superpowers." A chatbot can only talk; an AI agent can also **perform tasks/actions** for you.

### What Makes an AI Agent Different?

An AI agent has **two capabilities** that a regular chatbot doesn't have:
1. **Reasoning capability**
2. **Access to tools**

### Example: How an AI Agent Works

Suppose we build an AI agent and give it **two tools**:
1. A **calculator** (for any mathematical calculations).
2. A **weather API** (to fetch weather conditions for any city on any date).

**User Query:** *"Can you multiply the temperature of Delhi today with 3?"*

Here's how the agent handles this, using its **reasoning capability** (a popular technique here is **Chain of Thought prompting**, where the agent breaks the query down step by step):

1. The agent breaks down the query: "I need to multiply today's Delhi temperature by 3. First, I need today's Delhi temperature."
2. It checks its available tools and finds it has a **weather API**.
3. It hits the weather API with input "Delhi," and gets back, e.g., "25°C."
4. Now it reasons: "I have the temperature (25°C). Now I need to multiply this number by 3 — but for this operation, I need a calculator."
5. It checks its tools again, finds the calculator, and calls it with inputs: 25, 3, and the "multiply" operation.
6. The calculator returns **75**, which becomes the final output.

### Summary

The key difference between an AI agent and a chatbot: an AI agent has **reasoning capability** and **access to tools**, enabling it to actually perform actions — not just converse. An AI agent is essentially an **evolved form of a chatbot** that can perform actions with the help of reasoning and tool access.

This is considered a very hot area of AI research right now, and significant progress is expected in the next 1–1.5 years. LangChain makes it quite easy to build AI agents, and a hands-on AI agent build will be shown later in the playlist.

---

## Conclusion of Video 2

This video gave an introduction to all 6 components of LangChain:
1. Models
2. Prompts
3. Chains
4. Memory
5. Indexes
6. Agents

The next video in the playlist will do a deep dive into the **first component: Models**.

---

*End of notes.*



# LangChain Models — Detailed Notes (with Code Explanation)

These notes are converted from the Hindi video transcript **"Introduction to LangChain Models"** (Video 3 of the CampusX LangChain playlist) into detailed English notes, including full explanations of every piece of Python code discussed, plus a walkthrough of the accompanying GitHub repository: **[campusx-official/langchain-models](https://github.com/campusx-official/langchain-models)**.

---

## Recap of Previous Videos

Before this video, two videos were already covered in the playlist:

1. **Video 1 – Introduction to LangChain:** What LangChain is, why it is needed, what kind of applications can be built with it, and its alternatives.
2. **Video 2 – LangChain Components:** An overview of the 6 core components of LangChain — Models, Prompts, Chains, Memory, Indexes, and Agents.

This video (Video 3) is a **deep, hands-on dive into the first component: Models**. Unlike the previous two videos, this one is fully code-based — actual code is written and run against real models.

---

## What Are Models? (Quick Recap)

> "The Model component in LangChain is a crucial part of the framework designed to facilitate interactions with various language models and embedding models."

In simple words: many different types of AI models exist in the world today, built by different companies, and each behaves slightly differently when you try to interact with it in code. LangChain's **Model component** provides a **common/standard interface** so you can connect to *any* AI model very easily, regardless of which company built it.

**In short:** The Model component in LangChain is an **interface that helps you connect with different kinds of AI models.**

### Two Types of Models in LangChain

1. **Language Models**
   - Input: text (e.g., "What is the capital of India?")
   - Output: text (e.g., "New Delhi")
   - Used to build applications like **chatbots**.

2. **Embedding Models**
   - Input: text (e.g., "What is the capital of India?")
   - Output: a **series of numbers** (a vector) — NOT text.
   - These numbers are called **embeddings** — essentially vectors that represent the **contextual/semantic meaning** of the text.
   - Used for **semantic search**, and hence for building **RAG (Retrieval-Augmented Generation)** based applications — e.g., "chat with your PDF" type apps discussed in Video 1.

---

## Plan of Action for This Video

The video is structured into two major parts:

### Part 1: Language Models
- Work with **closed-source (paid) models**:
  - OpenAI's GPT models
  - Anthropic's Claude models
  - Google's Gemini models
- Work with **open-source models** (via Hugging Face):
  - Using Hugging Face's **Inference API**
  - Downloading and running a model **locally** on your own machine

### Part 2: Embedding Models
- Closed-source: OpenAI's embedding models
- Open-source: A Hugging Face embedding model, downloaded and run locally
- Finally, build a **small Document Similarity application** using embeddings and cosine similarity — figuring out which document in a set is most relevant to a user's query.

---

## Language Models — LLMs vs Chat Models

Language Models (models that take text in, and give text out) are further divided into **two types**:

### 1. LLMs (older-style / legacy)
- **General-purpose models** — can be used for almost any NLP task: text generation, text summarization, code generation, question answering, etc.
- Take a **plain string** as input and return a **plain string** as output.
- **Important note from the video:** LLMs (in this legacy sense) are becoming outdated. Newer versions of LangChain are actively discouraging their use for new projects, because they are being replaced by Chat Models.

### 2. Chat Models
- **Specialized for conversational tasks.**
- Take a **sequence of messages** as input, and return **chat messages** as output (not just plain text).
- This means a chat model can understand the full flow of a conversation and reply accordingly (a series of messages back and forth).
- Chat models are the recommended choice today for building chatbots, AI agents, coding assistants, etc.

### Comparison Table (LLM vs Chat Model)

| Aspect | LLMs | Chat Models |
|---|---|---|
| **Purpose** | Free-form text generation (Q&A, summarization, translation, etc.) | Multi-turn conversations between user and AI |
| **Training Data** | General text: books, articles, Wikipedia, etc. | Trained generally, then **fine-tuned on chat datasets** (conversations between multiple people) |
| **Memory** | No memory concept — doesn't remember previous interactions | Supports conversation history — can "remember" what was discussed earlier in the same session |
| **Role Awareness** | No concept of roles | Understands roles — e.g., you can assign a "system" role like *"You are a very knowledgeable doctor"* |
| **Example Models** | GPT-3, Instruct-tuned base models | GPT-4, Claude, etc. |
| **Best Used For** | Text generation, summarization, translation, code generation | Conversational AI, chatbots, virtual assistants, customer support bots, AI tutors |

**Key takeaway:** Most modern GenAI applications fall into the "conversational" category, which is why the industry (and LangChain itself) is shifting focus toward **Chat Models**, while support for legacy LLMs is gradually being phased out.

The video's primary focus is on Chat Models, but LLMs are covered briefly first (since they need less explanation) before moving to Chat Models for the rest of the video.

---

## Setup (Before Writing Any Code)

Steps followed to set up the coding environment:

1. **Create a project folder** (e.g., `langchain-models`) and open it in VS Code.

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment (Windows):**
   ```bash
   venv\Scripts\activate
   ```

4. **Create a `requirements.txt` file** listing all the libraries needed for this tutorial (LangChain, provider-specific integration packages, dotenv, huggingface libraries, scikit-learn, numpy, etc.)

5. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify installation** by creating a small test file (`test.py`):
   ```python
   import langchain
   print(langchain.__version__)
   ```
   ```bash
   python test.py
   ```
   If this prints a version number, LangChain has been installed correctly.

7. **Create 3 folders** to organize the code for this lecture:
   - `1.LLMs`
   - `2.ChatModels`
   - `3.EmbeddingModels`

   (This mirrors the exact folder structure used in the actual GitHub repository for this video, which is explained in detail later in these notes.)

---

## Part 1: Language Models

### 1.1 Working with LLMs (OpenAI)

**Getting an OpenAI API Key:**
1. Go to [platform.openai.com](https://platform.openai.com).
2. Create an account (if you don't already have one).
3. **Important:** OpenAI's API keys are usable only if there is some **credit balance** in your account (minimum around $5). Unlike a year ago, OpenAI no longer offers free credits to new users, so you may need to add funds. The instructor recommends this because many companies still use OpenAI's API in production, so getting hands-on practice with it is valuable.
4. Go to **Settings → API Keys**, click **Create new secret key**, give it a name, and copy the generated key.

**Storing the API key securely:**
- Never hard-code the API key directly into your Python file.
- Instead, create a `.env` file in your project, and store the key as an environment variable:
  ```
  OPENAI_API_KEY="your_key_here"
  ```

**Code: `1.LLMs/llm_demo.py`**

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of India")

print(result)
```

**Explanation, line by line:**

1. `from langchain_openai import OpenAI` — imports the `OpenAI` class from the `langchain_openai` package. This package is the **integration package** between LangChain and OpenAI — it contains the code LangChain needs to know how to talk to OpenAI's API.
2. `from dotenv import load_dotenv` — imports the function that loads environment variables (like your API key) from the `.env` file into your current Python program.
3. `load_dotenv()` — actually executes the loading process, so your `OPENAI_API_KEY` becomes available to the program.
4. `llm = OpenAI(model='gpt-3.5-turbo-instruct')` — creates an **object** of the `OpenAI` class, and this object is stored in a variable called `llm`. Here, you specify **which OpenAI model** you want to talk to — in this case, `gpt-3.5-turbo-instruct`.
5. `result = llm.invoke("What is the capital of India")` — calls the **`invoke()`** method, which is one of the most important methods in LangChain. Almost every core LangChain component (models, chains, prompts) has this `invoke()` method. Behind the scenes, this hits the OpenAI API, sends the prompt, waits for the model to process and generate a reply, and returns the response.
6. `print(result)` — prints the result. Since this is an **LLM** (not a Chat Model), notice that the output here is a **plain string** — you send a string in, you get a string back. This is the defining trait of the LLM interface (as opposed to Chat Models, explained next).

**Output:** `The capital of India is New Delhi.`

---

### 1.2 Working with Chat Models

#### a) Chat Model — OpenAI (GPT-4)

**Code: `2.ChatModels/chatmodel_openai.py`**

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("What is the capital of India")

print(result)
```

**Explanation:**

- The **only real structural difference** from the LLM code is:
  - You import `ChatOpenAI` instead of `OpenAI`.
  - You create an instance of `ChatOpenAI` instead of `OpenAI`.
- This is a great demonstration of how **consistent LangChain's interface is** — very little needs to change to switch from an LLM to a Chat Model.
- **Technical note (from the source code):** If you inspect LangChain's source code, `OpenAI` (LLM class) inherits from a **Base LLM class**, while `ChatOpenAI` inherits from a **Base Chat Model class**. This is *the* fundamental distinction between LLMs and Chat Models in LangChain's internal design — all LLM classes inherit from the base LLM class, and all chat model classes inherit from the base chat model class.
- **Choosing the model:** You can check available OpenAI models (and their context window / max output tokens) on OpenAI's website under the "Models" section. In this example, `gpt-4` is used.
- `model.invoke("What is the capital of India")` — again uses the `invoke()` method, sending the prompt to the chat model.

**Important difference in output:** Unlike the LLM's plain string output, a Chat Model's result is a **rich object**, not a simple string. If you print `result` directly, you'll see something like:

```
content='The capital of India is New Delhi.' additional_kwargs={} response_metadata={'token_usage': {...}, 'model_name': 'gpt-4', ...} id='...'
```

This includes:
- `content` — the actual answer text (what you usually care about).
- Extra metadata: **completion tokens**, **prompt tokens**, **total tokens**, model name, etc.

**To get just the answer text**, you access the `.content` attribute:

```python
print(result.content)
```

This is the recommended pattern anytime you use a Chat Model in LangChain.

#### b) Important Chat Model Parameters: `temperature` and `max_completion_tokens`

While creating a `ChatOpenAI` object, you can pass extra parameters:

```python
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
```

**`temperature`** — a "creativity" parameter, ranging roughly from `0` to `2`.
> "Temperature is a parameter that controls the randomness of a language model's output. It affects how creative and deterministic the responses are."

| Use Case | Recommended Temperature |
|---|---|
| Factual answers (math, code) | 0 – 0.3 |
| General Q&A / explanations | 0.5 – 0.7 |
| Creative writing (stories, jokes) | 0.9 – 1.2 |
| Maximum randomness / brainstorming | 1.5+ |

- Lower temperature (closer to 0) → more **deterministic and predictable** output (good for facts, code).
- Higher temperature (closer to 1.5–2) → more **random, creative, diverse** output (good for storytelling, poems, jokes).

The video demonstrates this by asking for "5 Indian male names" and "a 5-line poem on cricket" at different temperature values (0, 1.8, 1.5), showing how outputs shift from more standard/predictable to more varied and creative as temperature increases.

**`max_completion_tokens`** (also seen as `max_tokens` in some versions) — controls the **maximum number of tokens** the model is allowed to return in its response.

- A "token" can be thought of (roughly) as a word, though technically tokenization is more nuanced than that (a topic for a future lecture).
- This is useful because **paid LLM APIs charge you per token** (both input/prompt tokens and output/completion tokens). By restricting `max_completion_tokens`, you control cost.
- Example: Setting `max_completion_tokens=10` restricts the model's response to a maximum of 10 tokens — you'll notice the output gets cut off.

**Repo file matching this section — `2.ChatModels/1_chatmodel_openai.py`** (actual code from the GitHub repository):

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
```

This matches exactly what's demonstrated in the video: using `gpt-4`, a high temperature (1.5) for a creative task (poem writing), and limiting the output to 10 tokens.

#### c) Chat Model — Anthropic (Claude)

Claude is another very popular chat model (from the company **Anthropic**), known for performing extremely well — in some cases even outperforming GPT models. Since companies may use Claude's API instead of OpenAI's, it's important to know how to work with it too.

**Getting an Anthropic API key:** Similar process — go to the Anthropic Console website, top up credits (a paid service, just like OpenAI), go to **Get API Keys**, create a new key, and copy it.

**Storing it in `.env`:**
```
ANTHROPIC_API_KEY="your_key_here"
```
**Important:** The variable name must be written exactly this way (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) — LangChain's `load_dotenv()` mechanism looks for these exact names to correctly detect the keys. Renaming them will break the code.

**Code: `2.ChatModels/chatmodel_anthropic.py`**

```python
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke("What is the capital of India")

print(result.content)
```

**Explanation:**
- Import `ChatAnthropic` from `langchain_anthropic` (the integration package for Anthropic).
- Load environment variables as before.
- Create a `ChatAnthropic` object, specifying the Claude model to use (the video uses the latest Claude 3.5 model available at the time).
- Call `.invoke()` with the prompt, and print `.content` to get just the answer text.

**Key takeaway:** The code is **almost identical** to the OpenAI version — this is the entire point of LangChain's Model component: it **standardizes** the interface across providers, so switching between OpenAI and Claude requires minimal code change (just the import and the class name essentially).

#### d) Chat Model — Google (Gemini)

Similarly, for Google's Gemini models:
1. Go to Google's Gemini API documentation page.
2. Click "Get a Gemini API key" and generate one.
3. Store it in `.env`:
   ```
   GOOGLE_API_KEY="your_key_here"
   ```

**Code: `2.ChatModels/chatmodel_google.py`**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

result = model.invoke("What is the capital of India")

print(result.content)
```

**Explanation:**
- Import `ChatGoogleGenerativeAI` from `langchain_google_genai`.
- Load the environment variables.
- Create the model object with `gemini-1.5-pro` as the chosen model.
- Call `.invoke()` and print `.content`.

**Summary so far:** Using three different, extremely popular closed-source language models (OpenAI GPT, Anthropic Claude, Google Gemini), the same overall pattern of code was used each time — only the import statement and class name change. This is the core value proposition of LangChain's Models component.

---

### 1.3 Open Source Models

#### Why Open Source Models?

> "Open-source models are freely available AI models that can be downloaded, modified, fine-tuned, and deployed without restrictions from a central provider. Unlike closed-source models such as GPT, Claude, and Gemini, open-source models allow full control and customization."

**Two big drawbacks of closed-source (proprietary) models:**
1. **Cost** — you must pay to use their APIs.
2. **Lack of control** — since the model sits on someone else's server and you can only interact via API, you have no ability to modify or customize it.

**How open-source models solve this:**
- A company/organization trains a model and releases the **trained model itself** publicly on the internet.
- You, as a user, can **download this model to your own machine**.
- Once downloaded, you have full freedom: no more API costs, you can fine-tune it, modify it, and deploy it wherever you like (even fully offline).

**Side-by-side comparison:**

| Aspect | Open Source Models | Closed Source Models |
|---|---|---|
| **Cost** | Free to use, run locally without an API | Must pay to use the API |
| **Control** | Full control — fine-tune, modify, deploy anywhere | Only use the provider's infrastructure; zero control |
| **Data Privacy** | Your data stays on your own machine — safe for confidential documents | Your data is sent to the provider's servers (a concern for private/confidential data) |
| **Customization** | Can be fine-tuned on your own datasets | Some providers offer limited fine-tuning options |
| **Deployment** | Can deploy on your own servers/cloud | No option to self-host |

**Popular open-source language models today:** LLaMA (Meta), Mistral, Falcon, and domain-specific models like BLOOM.

**Where to find open-source models:** **Hugging Face** — described as the **largest repository of open-source LLMs**. Under the "Models" section on Hugging Face's website, you'll find thousands of models across categories: multimodal models (text, audio, video, speech input), computer vision models (image classification, object detection), and NLP models (including text generation — the category relevant here). Popular text-generation models found there include DeepSeek, LLaMA, and Qwen (from China), among many others.

**Two main ways to use open-source models via Hugging Face:**
1. **Download and run locally** on your own machine.
2. **Use Hugging Face's Inference API** — similar concept to OpenAI's API, but for open-source models hosted on Hugging Face's servers. There's a usage limit before you need to pay, but for students/small projects this is often more than enough, and it gives access to thousands of hosted models without needing heavy local hardware.

**Disadvantages of open-source models:**
1. **Hardware requirements** — running large models locally requires solid hardware (often expensive GPUs), which most individuals don't have.
2. **Setup complexity** — more configuration work is required to get things running.
3. **Less refinement** — technically, open-source models often have less fine-tuning done via **RLHF (Reinforcement Learning from Human Feedback)** compared to closed-source models, so responses may feel comparatively less polished (though this can be improved via your own fine-tuning).
4. **Limited multimodal abilities** — at least at this point in time, most freely available open-source models are primarily text-focused; fewer options exist for image/audio multimodal tasks compared to closed-source offerings.

#### a) Open Source via Hugging Face Inference API

**Getting a Hugging Face access token:**
1. Create an account on [huggingface.co](https://huggingface.co).
2. Go to your account → **Access Tokens**.
3. Click **Create new token**, choose "Read" access (since we're only reading/using models, not writing), name it, and create it.
4. Store it in `.env`:
   ```
   HUGGINGFACEHUB_API_TOKEN="your_token_here"
   ```
   (This exact variable name must be used.)

**The model used in the demo:** **TinyLlama** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) — a small, fine-tuned version of LLaMA with only 1.1 billion parameters (much smaller than typical production LLMs), chosen specifically so it's light enough to demonstrate easily.

**Code: `2.ChatModels/chatmodel_hf_api.py`**

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

**Explanation:**

1. `from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint` — imports **two classes**:
   - `ChatHuggingFace` — analogous to `ChatOpenAI`/`ChatAnthropic`, this is the actual chat model wrapper.
   - `HuggingFaceEndpoint` — used specifically when you want to use a model via Hugging Face's **hosted Inference API** (i.e., without downloading it locally).
2. `load_dotenv()` — loads your Hugging Face access token from `.env`.
3. `HuggingFaceEndpoint(repo_id="...", task="text-generation")` — this creates the underlying LLM configuration:
   - `repo_id` — tells it exactly which model on Hugging Face's hub to use (copied directly from the model's page URL/path, e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
   - `task` — tells it what kind of task to perform with this model; here, `"text-generation"`.
4. `model = ChatHuggingFace(llm=llm)` — wraps the configured `llm` object inside `ChatHuggingFace`, giving you the standard chat-model interface (with `.invoke()`, etc.) that you've used with all the other providers.
5. `model.invoke("What is the capital of India")` — sends the query, same as before.
6. `print(result.content)` — prints just the text answer.

**Key point:** The model here is *not* downloaded to your machine — it lives on Hugging Face's servers, and you interact with it purely through their API, just like you did with OpenAI/Anthropic/Google.

#### b) Open Source — Downloading and Running Locally

Now the same TinyLlama model is downloaded and run **entirely on the local machine** (no API call at all).

**Code: `2.ChatModels/chatmodel_hf_local.py`**

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

**Explanation:**

1. This time, instead of `HuggingFaceEndpoint`, you import **`HuggingFacePipeline`** — this class is used specifically for running a model **locally** on your own machine (via Hugging Face's `transformers` library under the hood).
2. `HuggingFacePipeline.from_model_id(...)` — a class method used to configure the local pipeline:
   - `model_id` — again, the Hugging Face repo ID of the model to use (same `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
   - `task="text-generation"` — the task to perform.
   - `pipeline_kwargs` — a dictionary of extra keyword arguments passed to the underlying pipeline, such as:
     - `temperature` — same creativity-control concept as before.
     - `max_new_tokens` — limits how many new tokens the model can generate in its response (similar purpose to `max_completion_tokens` seen earlier).
3. `model = ChatHuggingFace(llm=llm)` — again wraps it into the standard chat interface.
4. `model.invoke(...)` and `print(result.content)` — same pattern as always.

**What happens when you run this code:**
- The very first time you run it, the model's files (weights, tokenizer files, configuration files — roughly 300–500MB for TinyLlama) get **downloaded** to your machine and cached locally.
- After that, the model is **loaded into RAM**, and all inference (processing) happens **on your own machine's CPU/GPU** — no internet/API call is needed once the model is downloaded.
- Having a **GPU** significantly speeds up inference; running purely on **CPU** (as in the instructor's demo) can be quite slow, especially on machines with limited RAM (the instructor notes his machine has 8GB RAM and struggled — it took around 10 minutes and even caused the machine to become unresponsive at one point).
- **Optional: Controlling the download location.** By default, Hugging Face downloads model files to your C-drive cache folder (on Windows). If you want to change this (e.g., because your C-drive is full), you can set an environment variable before loading the model:
  ```python
  import os
  os.environ['HF_HOME'] = 'D:/huggingface_cache'
  ```
  This tells Hugging Face to use a different drive/folder for its cache.

**Summary of Section 1.3:** The exact same local-download approach shown here for TinyLlama can be used to download **any** other model from Hugging Face — just change the `model_id`.

---

### Summary of Part 1 (Language Models)

By this point, the video has covered:
- **LLMs** — using OpenAI's `gpt-3.5-turbo-instruct` model.
- **Chat Models** — both:
  - **Closed-source / proprietary**: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini)
  - **Open-source**: TinyLlama, via both the Hugging Face Inference API and a local download

---

## Part 2: Embedding Models

**Reminder:** Embedding models are used to convert a piece of text into a **vector**, so that the contextual/semantic understanding of that text is captured numerically.

### 2.1 Embedding Models — OpenAI (Single Query)

**Code: `3.EmbeddingModels/embedding_openai_query.py`**

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))
```

**Explanation:**

1. `from langchain_openai import OpenAIEmbeddings` — imports the `OpenAIEmbeddings` class (as opposed to `ChatOpenAI`/`OpenAI` used for language models).
2. `load_dotenv()` — loads your OpenAI API key as usual.
3. `OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)` — creates the embedding model object:
   - `model` — which OpenAI embedding model to use. You can find the full list of available embedding models on OpenAI's website; `text-embedding-3-large` is used here.
   - `dimensions` — how many numbers (dimensions) you want in the output vector. Here, `32` is used (a small number, for demonstration). 
     - **Larger vectors** → capture **more contextual meaning**, but **cost more** (since cost is generally tied to the size of processing/output).
     - **Smaller vectors** → capture **less context**, but are **cheaper**.
     - Note: by default (without specifying `dimensions`), OpenAI's embedding models return much larger vectors — **1536 dimensions** for the "small" model, and **3072 dimensions** for the "large" model.
4. `embedding.embed_query("Delhi is the capital of India")` — this is the key method for generating an embedding for a **single piece of text (a query)**. It sends the text to the embedding model, which processes it and returns a vector (in this case, 32 numbers).
5. `print(str(result))` — the result is converted to a string just so it prints cleanly/readably.

**Output:** A list of 32 floating-point numbers representing the semantic embedding of the input sentence.

### 2.2 Embedding Models — OpenAI (Multiple Documents)

**Code: `3.EmbeddingModels/embedding_openai_docs.py`**

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))
```

**Explanation:**

- The setup (imports, `.env` loading, creating the `embedding` object) is **identical** to the single-query example.
- The difference: instead of a single string, you create a **list** of multiple text strings — `documents` — here, 3 separate statements.
- Instead of `embed_query()`, you use **`embed_documents()`** — this function is specifically designed to generate embeddings for **multiple pieces of text at once**, in a single call.
- The result is a **2D list** — a list containing 3 separate lists, where each inner list is the 32-dimension embedding vector for the corresponding document.

**This is how you generate embeddings for any set of documents** — in future videos (when building RAG applications), this same idea is applied to much larger, real paragraphs of text.

### 2.3 Embedding Models — Open Source (Hugging Face, Local)

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`
> "This is a sentence-transformer model. It maps sentences and paragraphs to a 384-dimensional dense vector space and can be used for tasks like clustering and semantic search."

This is a small model (~90MB), so it's practical to download and run locally.

**Code: `3.EmbeddingModels/embedding_hf_local.py`**

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

result = embedding.embed_query(text)

print(str(result))
```

**Explanation:**

1. `from langchain_huggingface import HuggingFaceEmbeddings` — imports the class used for open-source Hugging Face embedding models.
2. `HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')` — creates the embedding object; you just need to specify the `model_name` (the Hugging Face repo ID of the model).
3. From here, the usage pattern is **exactly the same** as with OpenAI's embeddings: `embed_query()` for a single string.
4. The first time you run this, the ~90MB model gets downloaded and cached locally; subsequent runs use the cached version and are faster.

**Output:** A 384-dimensional vector (as specified by the model itself — this model always outputs 384-dimension vectors, since that's how it was trained/designed).

**For multiple documents**, exactly the same substitution applies as with OpenAI: replace the single `text` with a `documents` list, and call **`embed_documents(documents)`** instead of `embed_query(text)`.

### A Note on Cost (OpenAI Embeddings)

The instructor clarifies that OpenAI's embedding models are actually **very cheap** — around **$0.13 per 1 million tokens** processed (roughly, "cost is very low"), because these models only output numbers, not lengthy text. Because of this low cost:
- Paid (OpenAI) embeddings are often worth using, since they generally produce **better/more accurate contextual embeddings**.
- Free (open-source) embeddings are a valid option too but, in the instructor's experience, tend to be **slightly less accurate**, with slightly lower-quality results.

---

## 2.4 Mini-Project: Document Similarity Search

**Goal:** Build a simple application where:
- You have a **set of documents** (5 short documents, each about a different cricketer, in the video's example).
- A user asks a **question** related to one of these documents.
- The application must find out **which document is most relevant** to the question.

### How It Works (Conceptually)

1. Generate embeddings for all documents in advance → this gives you a set of vectors (e.g., 5 vectors, each of some fixed dimension, say 300).
2. When a query comes in, generate its embedding too → this gives you 1 more vector, in the same dimensional space.
3. Compute the **cosine similarity** between the query vector and each of the document vectors.
4. Whichever document has the **highest similarity score** is the most relevant one — that's your answer.

This is the exact same core idea used later when building **RAG-based applications**.

### Code: `3.EmbeddingModels/document_similarity.py`

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting style and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'Tell me about Bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is:", score)
```

**Explanation, step by step:**

1. **Imports:**
   - `OpenAIEmbeddings` — to generate embeddings (as covered above).
   - `load_dotenv` — for the API key.
   - `cosine_similarity` from `sklearn.metrics.pairwise` — a ready-made function from **scikit-learn** that calculates the cosine similarity between vectors (essentially, the angle/closeness between them).
   - `numpy as np` — for numerical/array operations (may be needed for supporting computations).

2. **Create the embedding model** — same as before (`text-embedding-3-large`, this time with `dimensions=300`).

3. **`documents`** — a list of 5 strings, each describing a different cricketer.

4. **`query`** — the user's question, e.g., `"Tell me about Bumrah"`.

5. **`doc_embeddings = embedding.embed_documents(documents)`** — generates embeddings for **all 5 documents at once**, giving a 2D list of 5 vectors (each 300-dimensional).

6. **`query_embedding = embedding.embed_query(query)`** — generates a **single embedding vector** for the user's query.

7. **`cosine_similarity([query_embedding], doc_embeddings)[0]`** — this is the core similarity computation:
   - `cosine_similarity()` expects **both arguments to be 2D lists**. Since `query_embedding` is just a single 1D vector, it's wrapped in an extra list (`[query_embedding]`) to make it 2D.
   - `doc_embeddings` is already a 2D list (list of 5 vectors), so it's passed as-is.
   - This function returns a 2D result (a matrix of similarity scores); `[0]` extracts the first (and only) row — i.e., the similarity scores between the query and each of the 5 documents.
   - `scores` now holds 5 numbers — one similarity score per document.

8. **Finding the best match — using `enumerate()` and `sorted()`:**
   - `enumerate(scores)` pairs up each score with its **index position** (0, 1, 2, 3, 4) — this is crucial because you want to know *which* document had the highest score, not just what the highest score value is.
   - `sorted(list(enumerate(scores)), key=lambda x: x[1])` sorts this list of (index, score) pairs based on the **score** (the second element, `x[1]`), in **ascending order** by default.
   - `[-1]` grabs the **last item** in this sorted list — i.e., the pair with the **highest similarity score**.
   - This is unpacked into two variables: `index` (the position of the most similar document) and `score` (its similarity value).

9. **Printing the results:**
   - The original query.
   - `documents[index]` — the actual text of the most similar document, retrieved using the index found above.
   - The similarity score itself, for reference.

**Example output (for the query "Tell me about Bumrah"):**
```
Tell me about Bumrah
Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.
Similarity score is: 0.71 (example value)
```

**Important caveat mentioned in the video:** In this simple demo, the document embeddings are **regenerated every single time** the script runs — this is wasteful and costly, since you're calling the embedding model repeatedly for the same unchanging documents. In a real-world system, you would:
- Generate the document embeddings **once**.
- Store them permanently in a special kind of database called a **vector database** (a topic to be covered later in the playlist).
- Then, whenever a **new query** comes in, generate its embedding on the fly and compare it against the already-stored document embeddings.

This entire process of finding and pulling out the most relevant document(s) for a query is called **retrieval**, and it's a foundational step in building RAG-based applications later in the playlist.

---

## Summary of the Whole Video

By the end of this video, the following were covered hands-on with real, working code:

**Language Models:**
- LLMs (OpenAI `gpt-3.5-turbo-instruct`)
- Chat Models — Closed-source: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini)
- Chat Models — Open-source: TinyLlama, via both the Hugging Face Inference API and local download
- Key parameters: `temperature`, `max_completion_tokens` / `max_new_tokens`

**Embedding Models:**
- OpenAI embeddings — for a single query (`embed_query`) and for multiple documents (`embed_documents`)
- Open-source embeddings — Hugging Face's `all-MiniLM-L6-v2` model, run locally
- A working **Document Similarity Search** mini-application using cosine similarity

The next video in the playlist covers **Prompts** in LangChain, which will then be used to build a small chatbot application.

---
---

# GitHub Repository Walkthrough: `campusx-official/langchain-models`

Repository link: **[https://github.com/campusx-official/langchain-models](https://github.com/campusx-official/langchain-models)**

> "Codes related to the model component in LangChain."

This is the official code repository that accompanies this video. Below is a simple explanation of its structure and what each part of the code does.

## Repository Structure

```
langchain-models/
├── 1.LLMs/                 → Code for working with (legacy) LLMs
├── 2.ChatModels/            → Code for working with Chat Models
│   ├── 1_chatmodel_openai.py
│   ├── 2_chatmodel_claude.py    (Anthropic's Claude)
│   ├── 3_chatmodel_google.py
│   ├── 4_chatmodel_hf_api.py
│   └── 5_chatmodel_hf_local.py
├── 3.EmbeddingModels/        → Code for working with Embedding Models
├── .gitignore
├── requirements.txt          → All Python dependencies needed for this playlist section
└── test.py                   → Simple script to verify LangChain is installed correctly
```

This mirrors **exactly** the folder structure the instructor builds in the video (`1.LLMs`, `2.ChatModels`, `3.EmbeddingModels`), and each file corresponds to one of the code demos explained step by step above.

## File-by-File Explanation (Simple Terms)

### `test.py`
A minimal sanity-check script:
```python
import langchain
print(langchain.__version__)
```
Just confirms LangChain is installed properly and prints its version — nothing more.

### `requirements.txt`
Lists every Python package needed to run all the demos in this repo — LangChain core, the provider-specific integration packages (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface`), `python-dotenv` (for reading the `.env` file), `scikit-learn` (for `cosine_similarity`), and `numpy`. You install them all in one go with:
```bash
pip install -r requirements.txt
```

### `1.LLMs/` folder
Contains the demo for the **legacy LLM interface** — using `langchain_openai.OpenAI` with the `gpt-3.5-turbo-instruct` model. This is the "plain text-in, plain text-out" style of interacting with a model, as explained in detail earlier in these notes.

### `2.ChatModels/` folder
Contains 5 separate, self-contained scripts — one for each chat-model provider/approach demonstrated in the video:

| File | What it demonstrates |
|---|---|
| `1_chatmodel_openai.py` | Talking to OpenAI's `gpt-4` via `ChatOpenAI`, including the `temperature` and `max_completion_tokens` parameters. Confirmed actual code:<br>`from langchain_openai import ChatOpenAI`<br>`from dotenv import load_dotenv`<br>`load_dotenv()`<br>`model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)`<br>`result = model.invoke("Write a 5 line poem on cricket")`<br>`print(result.content)` |
| `2_chatmodel_claude.py` | Talking to Anthropic's Claude model via `ChatAnthropic`, following the exact same pattern (just a different import/class). |
| `3_chatmodel_google.py` | Talking to Google's Gemini model. Confirmed import from the repo: `from langchain_google_genai import ChatGoogleGenerativeAI` |
| `4_chatmodel_hf_api.py` | Using an **open-source** model through **Hugging Face's hosted Inference API** (no local download). Confirmed import: `from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint` |
| `5_chatmodel_hf_local.py` | Using an **open-source** model **downloaded and run locally**. Confirmed import: `from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline` |

Every single file follows the **same 4-step pattern**, which is really the central lesson of this whole video:
1. Import the right provider-specific class.
2. Load your API key from `.env` (skipped only for the fully-local model, since no API key is needed there).
3. Create a model object, specifying the model name/ID and any parameters (`temperature`, token limits, etc.).
4. Call `.invoke("your prompt")` and read `.content` from the result.

### `3.EmbeddingModels/` folder
Contains the embedding-related demos:
- Generating an embedding for a **single query** using OpenAI (`embed_query`).
- Generating embeddings for **multiple documents at once** using OpenAI (`embed_documents`).
- Generating embeddings **locally** with an open-source Hugging Face sentence-transformer model.
- The **document similarity search** mini-application, which ties everything together using `cosine_similarity` from scikit-learn to find which document best matches a user's query.

## Why This Repo Structure Matters

The repo is intentionally organized to mirror the **conceptual structure** taught in the video:
- **Folder 1 (`1.LLMs`)** → the older, simpler text-in/text-out interface.
- **Folder 2 (`2.ChatModels`)** → the modern, recommended interface, shown across 3 closed-source providers and 2 open-source approaches — proving how *consistent* LangChain's API design is across completely different companies' models.
- **Folder 3 (`3.EmbeddingModels`)** → the second major model type in LangChain, used for semantic search and (eventually) RAG applications, culminating in a small working similarity-search demo.

This structure is a very practical, hands-on reference: if you want to see exactly how to connect to any of OpenAI, Anthropic, Google, or an open-source Hugging Face model — for either **chat/generation** or **embeddings** — this single small repository has a ready, minimal, working example for each case.

---

*End of notes.*
