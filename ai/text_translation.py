import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from divide_into_five import divide_into_five

load_dotenv()

# Google Generative AI 모델 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    api_key=os.getenv("GEMINI_API")
)

system_prompt_str = """
You are a professional translator.
You need to translate words naturally into {target_language}.
You must understand the characteristics, context, and nuances of the text you are translating,
and then render it appropriately in the {target_language}.
Do not translate words that are inappropriate for translation (such as proper nouns or technical terms). 
Translate all words naturally into the {target_language}. However, for proper nouns, technical terms, or words that should not be translated, translate the word and then place the original word in parentheses immediately after.
Example: Correct translation: 구글 클라우드(Google Cloud), 코드(code)
Do not translate any code, technical terms, or proper nouns.
Do not alter any punctuation marks, and preserve any HTML tags such as <span> exactly as they are.
**Do not insert any additional characters like \\n that are not present in the original text.**
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

chain = RunnablePassthrough() | prompt | llm

# 각 word_dict를 번역하는 함수
def translate_dict(word_dict, target_language):
    # word_dict 내의 단어들을 줄바꿈으로 결합하여 입력 준비
    combined_input = "\n\n".join(word_dict.word_dict.values())
    
    try:
        # 번역 요청 수행
        response = chain.invoke({"input": combined_input, "target_language": target_language})
        
        translated_texts = response.content.split("\n\n")
        
        # 마지막 텍스트에서 불필요한 공백 제거
        if translated_texts:
            translated_texts[-1] = translated_texts[-1].rstrip()

        return translated_texts
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

# 전체 텍스트를 번역하는 함수
def translate_texts_parallel(texts, target_language):
    # 텍스트 리스트를 5개의 그룹으로 나눔 - 지금은 3개 그룹
    cut_box = divide_into_five(texts)
    
    translated_texts = []
    # 각 그룹에 대해 번역 수행
    for word_dict in cut_box.values():
        translated_list = translate_dict(word_dict, target_language)
        translated_texts.extend(translated_list)
    
    return translated_texts

def main(texts):
    # 주어진 텍스트 리스트를 번역
    translated_texts = translate_texts_parallel(texts, "Korean")
    # 번역 결과를 딕셔너리 형태로 반환
    result = {"strs": translated_texts, "language": "ko"}
    return result

if __name__ == "__main__":
    texts_to_translate = [
        "one", "two", "three", "four", "five", 
        "hello", "world", "apple", "banana", 
        "cherry", "date", "elephant", "lion", 
        "tiger", "bear", "red", "blue", 
        "green", "yellow", "purple", "ipad"
    ]

    translated_result = main(texts_to_translate)
    print(translated_result)
