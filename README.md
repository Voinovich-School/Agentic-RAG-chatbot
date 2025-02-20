# Agentic RAG chatbot

## What is it?

The embedding and LLM model used in this example is OpenAI. Cloud database used is Supabase. PydanticAI is used to make building the chatbot much easier. 

chatbot for specialized information of a site or a local database.

### Getting the knowledge base first
ONLY HAVE TO DO THIS **ONCE**.

1. Crawl all the web pages of a site using [crawl4ai](https://docs.crawl4ai.com/).
2. Feed the *markdown* content of each page to an embedding model API (e.g. OpenAI).
3. Store each embedding along with its metadata into a database. The database used in this repo is Supabase, which has a PostgreSQL backend. You can change this to any (local/remote) database you want.

### Then start the chatbot
Only do this **AFTER** getting all the knowledge base as described above.

1. Feed the user's question into the same embedding model API used to create the knowledge base.
2. Call PydanticAI's agent to process our query and retrieve the results. It does this by querying the LLM (OpenAI in our case) with a prompt and we can also define functions (specified by `@pydantic_ai_expert.tool`) to help the LLM (We won't call them anywhere in our code, the LLM will if needed).

## How to use it

### STEP 1: Crawl
ONLY DO THIS **ONCE**.
1. First, get the URL of the root website you want to crawl.
2. Edit `crawl4ai_test.py` and set the root website to that URL.
3. Run `python3 crawl4ai_test.py` to crawl. The duration depends on how big that site is.

### STEP 2: Chatbot
Run `streamlit run streamlit_ui.py`.