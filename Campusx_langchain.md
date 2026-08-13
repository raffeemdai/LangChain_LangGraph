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
