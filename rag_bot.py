from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import warnings

warnings.filterwarnings("ignore")

# sentence transformers
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# load faiss index
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# load llm
llm = Ollama(model="llama3")

# retriever
retriever = db.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k": 4,
                                           "score_threshold": 0.2
                                           })

# Prompt for chatbot memory
contextualize_q_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.<|eot_id|>
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Question prompt
qa_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
If the context is irrelevant, just say that you don't know the answer. Use five sentences minimum and give detailed answers.
Don't mention you got the knowledge from context.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {input} 
Context: {context} 
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# chains
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# chat history list for saving the conversation
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

# question with chat history passed on to the retriever
second_question = "Can you explain the factors affected this?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])
