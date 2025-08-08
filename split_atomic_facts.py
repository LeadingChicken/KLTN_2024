from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List
from dotenv import load_dotenv
import os
# import nltk
# from nltk.tokenize import sent_tokenize
from generate_biographies import  generate_biography

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# Load api key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
print(API_KEY)


def split_into_atomic_facts(biography_text: str) -> List[str]:
    """
    Split a biography text into atomic facts by first splitting into sentences,
    then extracting atomic facts from each sentence.
    
    Args:
        biography_text (str): The biography text to split
        
    Returns:
        List[str]: List of atomic facts
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini")
    
    # Split the biography text into sentences
    # sentences = sent_tokenize(biography_text)
    
    all_atomic_facts = []
    
    # Process each sentence to extract atomic facts
    # for sentence in sentences:
        # if sentence.strip():  # Skip empty sentences
            # Create the message with your custom prompt for each sentence
    messages = [
        HumanMessage(content=f"""
        Hãy phân tích câu sau thành các sự thật độc lập: Anh ấy ra mắt với vai trò diễn viên trong bộ phim The Moon is the Sun’s Dream (1992), và tiếp tục xuất hiện trong các vai nhỏ và vai phụ xuyên suốt những năm 1990.

        - Anh ấy ra mắt với vai trò diễn viên trong một bộ phim.
        - Anh ấy ra mắt với vai trò diễn viên trong The Moon is the Sun’s Dream.
        - The Moon is the Sun’s Dream là một bộ phim.
        - The Moon is the Sun’s Dream được phát hành vào năm 1992.
        - Sau khi ra mắt với vai trò diễn viên, anh ấy xuất hiện trong các vai nhỏ và vai phụ.
        - Sau khi ra mắt với vai trò diễn viên, anh ấy xuất hiện trong các vai nhỏ và vai phụ xuyên suốt những năm 1990.

        Hãy phân tích câu sau thành các sự thật độc lập: Anh ấy cũng là một nhà sản xuất và kỹ sư thành công, đã làm việc với nhiều nghệ sĩ khác nhau, bao gồm Willie Nelson, Tim McGraw và Taylor Swift.

        - Anh ấy thành công.
        - Anh ấy là một nhà sản xuất.
        - Anh ấy là một kỹ sư.
        - Anh ấy đã làm việc với nhiều nghệ sĩ khác nhau.
        - Willie Nelson là một nghệ sĩ.
        - Anh ấy đã làm việc với Willie Nelson.
        - Tim McGraw là một nghệ sĩ.
        - Anh ấy đã làm việc với Tim McGraw.
        - Taylor Swift là một nghệ sĩ.
        - Anh ấy đã làm việc với Taylor Swift.

        Hãy phân tích câu sau thành các sự thật độc lập: Năm 1963, Collins trở thành một trong những phi hành gia thuộc nhóm thứ ba được NASA tuyển chọn và ông là phi công mô-đun chỉ huy dự bị cho sứ mệnh Gemini 7.

        - Collins trở thành một phi hành gia.
        - Collins trở thành một trong những phi hành gia thuộc nhóm thứ ba.
        - Collins trở thành một trong những phi hành gia thuộc nhóm thứ ba được tuyển chọn.
        - Collins trở thành một trong những phi hành gia thuộc nhóm thứ ba được NASA tuyển chọn.
        - Collins trở thành một trong những phi hành gia thuộc nhóm thứ ba được NASA tuyển chọn vào năm 1963.
        - Ông ấy là phi công mô-đun chỉ huy.
        - Ông ấy là phi công mô-đun chỉ huy dự bị.
        - Ông ấy là phi công mô-đun chỉ huy cho sứ mệnh Gemini 7.

        Hãy phân tích câu sau thành các sự thật độc lập: Ngoài vai trò diễn xuất, Bateman đã viết và đạo diễn hai phim ngắn và hiện đang phát triển phim truyện đầu tay của mình.

        - Bateman có vai diễn.
        - Bateman đã viết hai phim ngắn.
        - Bateman đã đạo diễn hai phim ngắn.
        - Bateman đã viết và đạo diễn hai phim ngắn.
        - Bateman hiện đang phát triển phim truyện đầu tay của mình.

        Hãy phân tích câu sau thành các sự thật độc lập: Michael Collins (sinh ngày 31 tháng 10 năm 1930) là một phi hành gia và phi công thử nghiệm người Mỹ đã nghỉ hưu, người từng là phi công mô-đun chỉ huy cho sứ mệnh Apollo 11 vào năm 1969.

        - Michael Collins sinh ngày 31 tháng 10 năm 1930.
        - Michael Collins đã nghỉ hưu.
        - Michael Collins là người Mỹ.
        - Michael Collins từng là một phi hành gia.
        - Michael Collins từng là một phi công thử nghiệm.
        - Michael Collins từng là phi công mô-đun chỉ huy.
        - Michael Collins từng là phi công mô-đun chỉ huy cho sứ mệnh Apollo 11.
        - Michael Collins từng là phi công mô-đun chỉ huy cho sứ mệnh Apollo 11 vào năm 1969.

        Hãy phân tích câu sau thành các sự thật độc lập: Ông ấy là một nhà soạn nhạc, nhạc trưởng và giám đốc âm nhạc người Mỹ.

        - Ông ấy là người Mỹ.
        - Ông ấy là một nhà soạn nhạc.
        - Ông ấy là một nhạc trưởng.
        - Ông ấy là một giám đốc âm nhạc.

        Hãy phân tích câu sau thành các sự thật độc lập: Cô ấy hiện đang đóng vai chính trong bộ phim hài lãng mạn Love and Destiny, được công chiếu vào năm 2019.

        - Cô ấy hiện đang đóng vai chính trong Love and Destiny.
        - Love and Destiny là một bộ phim hài lãng mạn.
        - Love and Destiny được công chiếu vào năm 2019.

        Hãy phân tích câu sau thành các sự thật độc lập: Trong sự nghiệp chuyên nghiệp của mình, McCoy đã chơi cho Broncos, San Diego Chargers, Minnesota Vikings và Jacksonville Jaguars.

        - McCoy đã chơi cho Broncos.
        - McCoy đã chơi cho Broncos trong sự nghiệp chuyên nghiệp của mình.
        - McCoy đã chơi cho San Diego Chargers.
        - McCoy đã chơi cho San Diego Chargers trong sự nghiệp chuyên nghiệp của mình.
        - McCoy đã chơi cho Minnesota Vikings.
        - McCoy đã chơi cho Minnesota Vikings trong sự nghiệp chuyên nghiệp của mình.
        - McCoy đã chơi cho Jacksonville Jaguars.
        - McCoy đã chơi cho Jacksonville Jaguars trong sự nghiệp chuyên nghiệp của mình.
        Hãy phân tích câu sau thành các sự thật độc lập: {biography_text}
        """)
    ]
    
    # Get the response from the model
    response = llm.invoke(messages)
    # Split the response into individual facts
    sentence_atomic_facts = [fact.strip()[2:] for fact in response.content.split('\n') if fact.strip()]
    
    # Append the atomic facts from this sentence to the overall list
    all_atomic_facts.extend(sentence_atomic_facts)
    
    return all_atomic_facts

