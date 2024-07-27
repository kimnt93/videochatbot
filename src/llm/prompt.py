DEFAULT_SYS_PROMPT = """You are a helpful assistant. Please answer the user's question. Today is {today}.
- Question: {question}
- Answer: """


CORRECT_SYS_PROMPT = "You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."

QUERY_ROUTING_PROMPT = """You are an expert at routing a user question to a datastore, vectorstore or no relevant.
The datastore contains information about video, including transcripts, subtitles, and metadata.
Use the vectorstore for semantic search and retrieval of relevant information.

Input 1: What is the best way to train a chatbot?
Output 1: datastore

Input 2: Who is the CEO of OpenAI?
Output 2: web-search

"""

HYDE_GENERATE_PROMPT = """"""

SUMMARY_TRANSCRIPT_PROMPT = """Your task is to summarize a text, which is a subtitle of a video. Please give a short summarize without any additional preamble. 
Subtitle: {subtitle}
Summary: """

SUMMARY_CONVERSATION_PROMPT = """Here is a chat between a user (User) and a chatbot (AI). Please summarize the Conversation in a few sentences without any preamble. Today is {today}.
Conversation: {conversation}
Summary: """


QUESTION_RE_WRITER_PROMPT = """Your task is to rewrite the following question to enhance its effectiveness in semantic search for vectorstore retrieval, focusing on optimizing the underlying semantic intent or meaning. Please provide the revised question without any additional preamble. Today's date is {today}.
Question: {question}
Improved Question: """

RAG_GENERATION_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Avoid prefaces such as "based on information", "according to the provided data", etc. Today's date is {today}.
- Question: {question}
- Context: {context}
- Answer: """


GRADE_DOCUMENT_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question. If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. d1, d2,... stands for document 1, document 2,...
Below is an example of the grading format. The Documents must be a list of dictionaries, and Grades must be a dictionary.

Example:
- Question: <example question>
- Documents: [{"d1": <example document 1>, "d2": <example document 2>, "d3": <example document 3>,...}]
- Grades: {"d1": "yes", "d2": "no", "d3": "yes", "d4": "yes", "d5": "no", "d6": "yes"}

Here is your task. Please follow the same format as the example. Today's date is {today}.
- Question: {question}
- Documents: {documents}
- Grades: 
"""