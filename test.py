from src.factory.graph_factory import create_chatbot_default_workflow

if __name__ == "__main__":
    graph = create_chatbot_default_workflow()
    rp = graph.invoke({"question": "Where is Perry?"})
    print(rp)
    rp = graph.invoke({"question": "Tell me about his new invention?", "img_path": "imexample.jpg"})
    print(rp)
    print("Done")
