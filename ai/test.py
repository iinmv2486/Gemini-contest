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
from langchain_openai import ChatOpenAI

from divide_into_five import divide_into_five
from divide_into_five import WordDict


# json output parser implementation
class TranslatedDictionary(RootModel):
    root: Dict[str, str] = Field(description="The translated dictionary")

def create_translator():
    load_dotenv()

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_str),
        HumanMessagePromptTemplate.from_template(
            "Translate the following dictionary values into {target_language}:\n\n{input_dict}"
        )
    ])

    parser = PydanticOutputParser(pydantic_object=TranslatedDictionary)

    chain = RunnablePassthrough() | prompt | llm | parser

    async def translate(input_dict, target_language):
        # input_dict가 WordDict 타입인 경우
        if isinstance(input_dict, WordDict):
            input_dict = input_dict.to_dict()
        result = await asyncio.to_thread(chain.invoke, {"input_dict": json.dumps(input_dict), "target_language": target_language})
        return result.root

    return translate

async def main():
    translator = create_translator()

    texts = [
    "When and Why you should apply Tensor Parallel",
    "The PyTorch Fully Sharded Data Parallel (FSDP) already has the capability to scale model training to a specific number of GPUs. However, when it comes to further scale the model training in terms of model size and GPU quantity, many additional challenges arise that may require combining Tensor Parallel with FSDP.",
    "As the world size (number of GPUs) is becoming excessively large (exceeding 128/256 GPUs), the FSDP collectives (such as allgather) are being dominated by ring latency. By implementing TP/SP on top of FSDP, the FSDP world size could be reduced by 8 by applying FSDP to be inter-host only, consequently decreasing the latency costs by the same amount.",
    "Hit data parallelism limit where you can not raise the global batch size to be above the number of GPUs due to both convergence and GPU memory limitations, Tensor/Sequence Parallel is the only known way to \"ballpark\" the global batch size and continue scaling with more GPUs. This means both model size and number of GPUs could continue to scale.",
    "For certain types of models, when local batch size becomes smaller, TP/SP can yield matrix multiplication shapes that are more optimized for floating point operations (FLOPS).",
    "So, when pre-training, how easy is it to hit those limits? As of now, pre-training a Large Language Model (LLM) with billions or trillions of tokens could take months, even when using thousands of GPUs.",
    "How to apply Tensor Parallel",
    "PyTorch Tensor Parallel APIs offers a set of module level primitives (ParallelStyle) to configure the sharding for each individual layers of the model, including:",
    "ColwiseParallel and RowwiseParallel: Shard the nn.Linear and nn.Embedding in the column or row fashion.",
    "SequenceParallel: Perform sharded computations on nn.LayerNorm, nn.Dropout, RMSNormPython, etc.",
    "PrepareModuleInput and PrepareModuleOutput: Configure the module inputs/outputs sharding layouts with proper communication operations.",
    "To demonstrate how to use the PyTorch native Tensor Parallel APIs, let us look at a common Transformer model. In this tutorial, we use the most recent Llama2 model as a reference Transformer model implementation, as it is also widely used in the community.",
    "Since Tensor Parallel shard individual tensors over a set of devices, we would need to set up the distributed environment (such as NCCL communicators) first. Tensor Parallelism is a Single-Program Multiple-Data (SPMD) sharding algorithm similar to PyTorch DDP/FSDP, and it under the hood leverages the PyTorch DTensor to perform sharding. It also utilizes the DeviceMesh abstraction (which under the hood manages ProcessGroups) for device management and sharding. To see how to utilize DeviceMesh to set up multi-dimensional parallelisms, please refer to this tutorial. Tensor Parallel usually works within each host, so let us first initialize a DeviceMesh that connects 8 GPUs within a host."
    ]
    
    target_language = "Korean"
    
    cut_box = divide_into_five(texts)
    
    # Translate all dictionaries concurrently
    tasks = [translator(d, target_language) for d in cut_box.values()]
    translated_dicts = await asyncio.gather(*tasks)
    
    translated_texts = []
    # translated_dicts는 [{'0': '...', '1': '...'}, {'0': '...', '1': '...'}, ...] 형태입니다.
    for translated_dict in translated_dicts:
        # 각 딕셔너리의 값을 순서대로 리스트에 추가
        for key in translated_dict.keys():
            translated_texts.append(translated_dict[key])
    
    print(translated_texts)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())



