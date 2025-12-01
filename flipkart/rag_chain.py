from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # ------------------------------
        # STEP 1: REWRITE QUESTION USING HISTORY
        # ------------------------------
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the user query as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")  # placeholder for user input
        ])

        # Lambda receives user_input and chat_history object
        rewrite_chain = RunnableLambda(
            lambda user_input, chat_history=None: rewrite_prompt.format_messages(
                input=user_input,
                chat_history=chat_history.messages if chat_history else []
            )
        ) | self.model | RunnableLambda(lambda x: x.content)

        # ------------------------------
        # STEP 2: RAG – retrieve documents
        # ------------------------------
        def retrieve_fn(inputs):
            standalone_q = inputs["standalone_question"]
            docs = retriever.invoke(standalone_q)
            return {"docs": docs, **inputs}

        retrieve_chain = RunnableLambda(retrieve_fn)

        # ------------------------------
        # STEP 3: QA PROMPT
        # ------------------------------
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an e-commerce bot. Use the context to answer clearly and concisely.\n\n"
             "CONTEXT:\n{context}\n\nQUESTION: {input}"),
            MessagesPlaceholder("chat_history"),
        ])

        # ------------------------------
        # STEP 4: Combine docs + question
        # ------------------------------
        def format_docs(inputs):
            docs_text = "\n\n".join([d.page_content for d in inputs["docs"]])
            return {
                "context": docs_text,
                "input": inputs["standalone_question"],
                "chat_history": inputs["chat_history"].messages if inputs.get("chat_history") else []
            }

        format_chain = RunnableLambda(format_docs)

        # ------------------------------
        # STEP 5: Final LLM Answer
        # ------------------------------
        answer_chain = qa_prompt | self.model

        # ------------------------------
        # COMPLETE PIPELINE
        # ------------------------------
        rag_chain = (
            {
                "standalone_question": rewrite_chain,
                "input": RunnablePassthrough(),       # user input
                "chat_history": RunnablePassthrough() # chat history object
            }
            | retrieve_chain
            | format_chain
            | answer_chain
        )

        # Wrap with message history
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
