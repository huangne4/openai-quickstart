import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate


def initialize_sales_bot(vector_store_dir: str="real_car_sale", base_url: str=None, api_key: str=None):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(openai_api_base=base_url, openai_api_key=api_key),allow_dangerous_deserialization='True')
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_base=base_url, openai_api_key=api_key)
    
    # 构建提示
    template = """Use the following pieces of context to answer the question at the end.
    You are a professional car salesperson, and you need to answer customers' questions smoothly without revealing that you are an AI assistant.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}),
                                                                     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    base_url = os.environ.get("BASE_URL").strip()
    api_key = os.environ.get("API_KEY").strip()
    # 初始化房产销售机器人
    initialize_sales_bot(base_url=base_url, api_key=api_key)
    # 启动 Gradio 服务
    launch_gradio()
