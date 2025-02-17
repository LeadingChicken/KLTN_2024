from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from wikipedia_pull_data import get_wikipedia_summary
import pandas as pd

# Load api key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = API_KEY
# Initialize the LLM
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini")
# load 

# Create a custom prompt template for RAG
rag_template = """
Ngữ cảnh: Bạn là một người được thuê để đánh nhãn dữ liệu. Bạn sẽ được cung cấp thông tin về một nhân vật được trích ra từ wikipedia. Tiếp theo đó bạn cần phải đánh nhãn cho 1 atomic facts do 1 LLM sinh ra. Lưu ý rằng tất cả thông tin bạn có chỉ có ở trong đoạn văn bản thông tin được cung cấp và không có thông tin nào khác. Có 3 loại nhãn: Supported, Unsupported, Irrelevant. Định nghĩa của các nhãn được hiểu như sau:

* Supported
Định nghĩa: Thông tin mang tính chất thực tế được xác nhận là đúng dựa trên nguồn tham khảo là Wikipedia.

* Unsupported
Định nghĩa: Thông tin là sai, không thể xác minh, hoặc mâu thuẫn với nguồn kiến thức tham khảo.

* Irrelevant 
Định nghĩa: Irrelevant chỉ ra rằng thông tin không liên quan đến lời nhắc đầu vào và có thể được chia thành hai trường hợp: (1) Thông tin đó phụ thuộc vào các thông tin khác vì nó mở rộng những thông tin trước đó trong quá trình tạo ra, và các thông tin đó là Unsupported , và (2) Cả câu hoàn toàn không liên quan đến lời nhắc, độc lập với các thông tin khác trong quá trình tạo ra.

Ví dụ (được giới hạn bởi XML tag <example></example>):
<example>
Prompt: Hãy cho tôi tiểu sử của Ylona Garcia.
Thông tin: [Ylona Garcia] đã xuất hiện trong nhiều chương trình truyền hình khác nhau như ASAP (All-Star Sunday Afternoon Party), Wansapanataym Presents: Annika PINTAsera và Maalaala Mo Kaya.

- Ylona Garcia đã xuất hiện trong nhiều chương trình truyền hình. (Supported)
- Cô ấy đã xuất hiện trong ASAP. (Supported)
- ASAP là viết tắt của All-Star Sunday Afternoon Party. (Supported)
- ASAP là một chương trình truyền hình. (Supported)
- Cô ấy đã xuất hiện trong Wansapanataym Presents: Annika PINTAsera. (Unsupported)
- Wansapanataym Presents: Annika PINTAsera là một chương trình truyền hình. (Irrelevant)
- Cô ấy đã xuất hiện trong Maalaala Mo Kaya. (Unsupported)
- Maalaala Mo Kaya là một chương trình truyền hình. (Irrelevant)
</example>

Sau đây là phần đánh nhãn của bạn:
Prompt: Cho tôi tiểu sử của {character}
Thông tin về nhân vật trên wikipedia (được giới hạn bởi XML tag <context></context>): <context>{context}</context>

Câu hỏi: Hãy đánh nhãn cho atomic fact sau (được giới hạn bởi XML tag <question></question>): <question>{question}</question>

Hãy trả về kết quả và đưa ra lí do tại sao bạn đã đánh nhãn như vậy và chỉ ra đoạn thông tin phản ánh điều đó bằng cách trích dẫn (có thể có nhiều đoạn thông tin, hãy trích dẫn tất cả).Lưu ý đoạn thông tin trích dẫn không phải là atomic fact được đưa vào Trả lời theo định dạng như sau:

Nhãn: <Nhãn>
Lí do: <Lí do>
Đoạn thông tin: <Đoạn thông tin>
"""

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["character","context", "question"],
    template=rag_template
)

# Create the chain using the new pipe syntax
rag_chain = prompt | llm | StrOutputParser()

# Add text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=50
)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

def get_rag_response(character, query, topic):
    # Get context from Wikipedia
    wiki_text = get_wikipedia_summary(topic)
    
    # Use the entire wiki_text as context
    context = wiki_text
    
    # Get response using RAG chain
    response = rag_chain.invoke({
        "character": character,
        "context": context,
        "question": query
    })
    # Extract label from response
    try:
        label = response.split("Nhãn:")[1].split("\n")[0].strip()
    except:
        label = "Error parsing label"
    return {"full_response": response, "label": label}

def factscore(entity_name):
    # Load atomic facts and append into array
    atomic_facts = []
    with open('atomic_facts.txt', 'r', encoding='utf-8') as file:
        for line in file:
            atomic_facts.append(line.strip())

    labels = []
    reasons = []
    # Process each atomic fact
    for atomic_fact in atomic_facts:
        topic = entity_name
        question = atomic_fact
        result = get_rag_response(entity_name, question, topic)
        print(f"Atomic fact: {atomic_fact}")
        print(f"Label: {result['label']}")
        print(f"Full response:\n{result['full_response']}")
        labels.append(result['label'])
        reasons.append(result['full_response'])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'Atomic Fact': atomic_facts,
        'Label': labels,
        'Reason': reasons
    })
    df.to_csv("Biographies of " + entity_name + '.csv', index=False, encoding='utf-8-sig')

    # Write labels to a text file
    with open("labels.txt", "w", encoding='utf-8') as label_file:
        for label in labels:
            label_file.write(label + "\n")

    # Count labels and create result map
    result_map = {
        'total_facts': len(atomic_facts),
        'supported_count': labels.count('Supported'),
        'unsupported_count': labels.count('Unsupported'),
        'irrelevant_count': labels.count('Irrelevant'),
        'factscore': labels.count('Supported') / len(atomic_facts)
    }

    return result_map
