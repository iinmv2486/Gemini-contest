import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import RootModel, Field
from typing import Dict
import os
from dotenv import load_dotenv
import json

from divide_into_five import divide_into_five
from divide_into_five import WordDict

class TranslatedDictionary(RootModel):
    root: Dict[str, str] = Field(default_factory=dict, description="The translated phrases")


def create_translator():
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GEMINI_API"),
        temperature=0.3
    )

    system_prompt_str = """
    You are a professional translator.
    Translate the following numbered phrases into {target_language}.
    Only translate the text after the colon (:) in each line. Do not translate or modify the numbers or any JSON syntax.
    Return your translations in the same numbered format, enclosed in a JSON object like this:
    {{"0": "translated text 0", "1": "translated text 1", ...}}
    Each phrase must be translated exactly as it is provided, without any additional interpretation, context, or meaning.
    Your translation should be literal, preserving the exact words and structure of the original text.
    Do not change the meaning of the phrases, infer additional information, or attempt to create a context.
    Translate only what is explicitly written.
    These phrases are independent of each other, so treat each one as a standalone translation.
    Only use parentheses to include the original text when translating proper nouns, names, technical terms, or specific words that should not be translated.
    Use parentheses sparingly and only when absolutely necessary.
    Preserve any HTML tags such as <span> exactly as they are. Do not alter, add, or remove any characters, words, or line breaks that are not present in the original text.

    Here are examples of correct translations:
    
    Example 1:
    - Original: 네이버 클라우드
    - Correct translation: Naver Cloud
    
    Example 2:
    - Original: 이전
    - Correct translation: Previous
    
    Example 3:
    - Original: 다음
    - Correct translation: Next
    
    Example 4:
    - Original: LIVE
    - Correct translation: LIVE
    
    Example 5:
    - Original: 연합뉴스
    - Correct translation: Yonhap News
    
    Example 6:
    - Original: <a>배드민턴협회, 진상조사위 구성…'부상 관리 소홀'엔 적극 반박</a>
    - Correct translation: <a>Badminton Association forms fact-finding committee... strongly refutes 'negligence in injury management'</a>
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_str),
        HumanMessagePromptTemplate.from_template(
            "Translate the following phrases:\n\n{numbered_texts}"
        )
    ])

    parser = PydanticOutputParser(pydantic_object=TranslatedDictionary)

    chain = RunnablePassthrough() | prompt | llm | parser

    async def translate(input_dict, target_language):
        if isinstance(input_dict, WordDict):
            input_dict = input_dict.to_dict()
        
        # 번역할 텍스트 값만 추출
        texts_to_translate = list(input_dict.values())
        
        # LLM을 위한 간단한 번호 목록 생성
        numbered_texts = "\n".join(f"{i}: {text}" for i, text in enumerate(texts_to_translate))
        
        try:
            result = await asyncio.to_thread(chain.invoke, {
                "numbered_texts": numbered_texts,
                "target_language": target_language
            })
            
            if result is None or not isinstance(result.root, dict):
                raise ValueError("LLM에서 예상치 못한 출력값을 받았습니다.")
            
            # 결과를 후처리하여 원래 구조를 복원
            translated_dict = {}
            for key, value in zip(input_dict.keys(), result.root.values()):
                translated_dict[key] = value.strip()
            
            return translated_dict
        except Exception as e:
            print(f"번역 중 오류 발생: {e}")
            # 오류 발생 시 번역되지 않은 원본 딕셔너리를 반환
            return input_dict

    return translate  # translate 함수 반환


async def translate_text(input_dict: dict) -> dict:
    translator = create_translator()

    texts = input_dict["strs"]
    target_language = input_dict["language"]

    cut_box = divide_into_five(texts)

    # Translate all dictionaries concurrently
    tasks = [translator(d, target_language) for d in cut_box.values()]
    translated_dicts = await asyncio.gather(*tasks)

    translated_texts = []
    for translated_dict in translated_dicts:
        translated_texts.extend(translated_dict.values())

    result = {"strs": translated_texts, "language": target_language}
    return result

if __name__ == "__main__":
    input_dict = {
        "strs": ["봄 봄 봄 봄이 왔네요 우리가 처음 만났던  그때의 향기 그대로 그대가 앉아 있었던  그 벤치 옆에 나무도  아직도 남아있네요 살아가다 보면  잊혀질 거라 했지만 그 말을 하며  안될거란걸 알고 있었소", "그대여 너를 처음 본 순간  나는 바로 알았지 그대여 나와 함께 해주오  이 봄이 가기 전에", "다시 봄 봄 봄 봄이 왔네요 그대 없었던 내 가슴  시렸던 겨울을 지나 또 벚꽃 잎이 피어나듯이  다시 이 벤치에 앉아  추억을 그려 보네요 사랑하다 보면  무뎌질 때도 있지만 그 시간 마저  사랑이란 걸 이제 알았소", "그대여 너를 처음 본 순간  나는 바로 알았지 그대여 나와 함께 해주오  이 봄이 가기 전에 우리 그만 참아요 이제  더 이상은 망설이지 마요 아팠던 날들은 이제  뒤로하고 말할 거예요 그대여 너를 처음 본 순간  나는 바로 알았지 그대여 나와 함께 해 주오  이 봄이 가기 전에"],
        "language": "en"
    }

    print(asyncio.run(translate_text(input_dict)))
