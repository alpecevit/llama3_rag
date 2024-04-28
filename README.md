# Llama 3 LangChain RAG

## Overview
LangChain Chatbot is a conversational AI system designed for question-answering tasks. It leverages various NLP techniques and libraries to understand user questions, retrieve relevant information from a database of embeddings, and provide detailed answers.

## Components
### 1. Embeddings
 RAG pipeline utilizes Hugging Face's Transformers library for generating embeddings. It employs the "sentence-transformers
 all-MiniLM-l6-v2" model to convert text inputs into numerical representations.

### 2. Vector Store
The system uses FAISS (Facebook AI Similarity Search) for efficient similarity search over the embeddings space. The FAISS index is pre-computed on [finance news](https://finance.yahoo.com/news/) and loaded locally to enable fast retrieval of relevant documents.

### 3. Language Model (LLM)
RAG pipeline uses Llama 3 as its LLM wrapped into Ollama framework.

## Usage
1. Install the required dependencies mentioned in `requirements.txt`.
2. Load the FAISS index using the `rag_index.py` script.
3. Interact with the chatbot by running the `rag_bot.py` script.

## Example

Initial question:

```python
chat_history = []
question = "What are the recent news on US mortage rates?"

if retriever.invoke(question):
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    print(ai_msg_1["answer"])
    chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
else:
    default_answer = "I am sorry, I don't have relevant information about your question."
    print(default_answer)
    chat_history.extend([HumanMessage(content=question), default_answer])
```

*Output:*

```
According to recent reports, US mortgage rates have been surging past 7% for the first time this year. According to Freddie Mac, the daily rate on the popular 30-year fixed loan was 7.52% on Thursday, the highest reading since November 2023. This uptick in rates is a result of inflation remaining hotter than expected and has forced some first-time and repeat buyers back on the sidelines.

The recent increases have brought mortgage rates to their highest level since November 2023, with the average rate on a 30-year mortgage climbing for the fourth consecutive week to its highest level since then. This surge in rates is attributed to inflation remaining hotter than forecast, which has stoked doubts over how soon the Federal Reserve might decide to start lowering its benchmark interest rate.

In fact, some experts believe that the rising mortgage rates will continue to influence the housing market, potentially giving pause to prospective homebuyers as they weigh their purchasing decisions. According to Jiayu Xu, an economist at Realtor.com, "The uncertainty has pushed up bond yields, and unfortunately, the rising mortgage rates occurred during what is typically a busy time in the housing market, potentially giving pause to prospective homebuyers as they weigh their purchasing decisions."

Overall, the recent news suggests that US mortgage rates are trending upward, reaching levels not seen since November 2023. This development has significant implications for the housing market and may impact the purchasing power of potential buyers.
```

Follow-up question:

```python
second_question = "Can you explain the factors affected this?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])
```

*Output:*

```
The recent surge in US mortgage rates is influenced by several factors, including:

1. **Inflation**: Inflation remaining hotter than expected has led to a surge in bond yields, which lenders use as a guide to pricing home loans. This has pushed up mortgage rates.

2. **Federal Reserve's interest rate policy**: The uncertainty surrounding the Fed's decision on when to start cutting its benchmark interest rate has also contributed to the rise in mortgage rates. Top Fed officials have said they could hold interest rates high for a while before getting full confidence inflation is heading down toward their target of 2%.

3. **Bond market reaction**: The bond market has been reacting to the Fed's interest rate policy, with yields rising as investors price in potential future rate hikes.

4. **10-year Treasury yield**: The 10-year Treasury yield, which lenders use as a guide to pricing home loans, has also risen significantly, contributing to the increase in mortgage rates.

5. **Economic reports and data**: The release of economic reports showing inflation remaining hot has stoked doubts over how soon the Fed might decide to start lowering its benchmark interest rate, leading to an uptick in bond yields and mortgage rates.

These factors have combined to push US mortgage rates higher, with the average rate on a 30-year mortgage reaching its highest level since November 20,23. This development has significant implications for the housing market and may impact the purchasing power of potential buyers.
```