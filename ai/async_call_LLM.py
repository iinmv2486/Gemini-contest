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
    input_dict = {"strs":["Join us in Silicon Valley September 18-19 at the 2024 PyTorch Conference.<a>Learn more</a>.","Learn","Get Started","Run PyTorch locally or get started quickly with one of the supported cloud platforms","Tutorials","Whats new in PyTorch tutorials","Learn the Basics","Familiarize yourself with PyTorch concepts and modules","PyTorch Recipes","Bite-size, ready-to-deploy PyTorch code examples","Intro to PyTorch - YouTube Series","Master PyTorch basics with our engaging YouTube tutorial series","Ecosystem","Tools","Learn about the tools and frameworks in the PyTorch Ecosystem","Community","Join the PyTorch developer community to contribute, learn, and get your questions answered.","Forums","A place to discuss PyTorch code, issues, install, research","Developer Resources","Find resources and get questions answered","Contributor Awards - 2023","Award winners announced at this year's PyTorch Conference","Edge","About PyTorch Edge","Build innovative and privacy-aware AI experiences for edge devices","ExecuTorch","End-to-end solution for enabling on-device inference capabilities across mobile and edge devices","Docs","PyTorch","Explore the documentation for comprehensive guidance on how to use PyTorch.","PyTorch Domains","Read the PyTorch Domains documentation to learn more about domain-specific libraries.","Blog &amp; News","PyTorch Blog","Catch up on the latest technical news and happenings","Community Blog","Stories from the PyTorch ecosystem","Videos","Learn about the latest PyTorch tutorials, new, and more","Community Stories","Learn how our community solves real, everyday machine learning problems with PyTorch","Events","Find events, webinars, and podcasts","About","PyTorch Foundation","Learn more about the PyTorch Foundation.","<span>Governing Board</span>","<a>Become a Member</a>","X","Get Started","Select preferences and run the command to install PyTorch locally, or\n          get started quickly with one of the supported cloud platforms.","Start Locally","PyTorch 2.0","Start via Cloud Partners","Previous PyTorch Versions","ExecuTorch","Shortcuts","<a>Prerequisites</a>","<a>macOS Version</a>","<a>Python</a>","<a>Package Manager</a>","<a>Installation</a>","<a>Anaconda</a>","<a>pip</a>","<a>Verification</a>","<a>Building from source</a>","<a>Prerequisites</a>","<a>Prerequisites</a>","<a>Supported Linux Distributions</a>","<a>Python</a>","<a>Package Manager</a>","<a>Installation</a>","<a>Anaconda</a>","<a>pip</a>","<a>Verification</a>","<a>Building from source</a>","<a>Prerequisites</a>","<a>Prerequisites</a>","<a>Supported Windows Distributions</a>","<a>Python</a>","<a>Package Manager</a>","<a>Installation</a>","<a>Anaconda</a>","<a>pip</a>","<a>Verification</a>","<a>Building from source</a>","<a>Prerequisites</a>","Start Locally","Select your preferences and run the install command. Stable represents the most currently tested and supported version of PyTorch. This should\n   be suitable for many users. Preview is available if you want the latest, not fully tested and supported, builds that are generated nightly.\n   Please ensure that you have<b>met the prerequisites below (e.g., numpy)</b>,  depending on your package manager. Anaconda is our recommended\n   package manager since it installs all dependencies. You can also<a>install previous versions of PyTorch</a>. Note that LibTorch is only available for C++.","<b>NOTE:</b>Latest PyTorch requires Python 3.8 or later.","PyTorch Build","Your OS","Package","Language","Compute Platform","Run this Command:","PyTorch Build","Stable (2.4.0)","Preview (Nightly)","Your OS","Linux","Mac","Windows","Package","Conda","Pip","LibTorch","Source","Language","Python","C++ / Java","Compute Platform","CUDA 11.8","CUDA 12.1","CUDA 12.4","ROCm 6.1","CPU","Run this Command:","pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118","","Installing on macOS","PyTorch can be installed and used on macOS. Depending on your system and GPU capabilities, your experience with PyTorch on a Mac may vary in terms of processing time.","Prerequisites","macOS Version","PyTorch is supported on macOS 10.15 (Catalina) or above.","Python","It is recommended that you use Python 3.8 - 3.11.\nYou can install Python either through the Anaconda\npackage manager (see<a>below</a>),<a>Homebrew</a>, or\nthe<a>Python website</a>.","In one of the upcoming PyTorch releases, support for Python 3.8 will be deprecated.","Package Manager","To install the PyTorch binaries, you will need to use one of two supported package managers:<a>Anaconda</a>or<a>pip</a>. Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.","Anaconda","To install Anaconda, you can<a>download graphical installer</a>or use the command-line installer. If you use the command-line installer, you can right-click on the installer link, select<code>Copy Link Address</code>, or use the following commands on Intel Mac:","<code>&lt;span&gt;# The version of Anaconda may be different depending on when you are installing`&lt;/span&gt;curl&lt;span&gt;-O&lt;/span&gt;https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh<br>sh Miniconda3-latest-MacOSX-x86_64.sh&lt;span&gt;# and follow the prompts. The defaults are generally good.`&lt;/span&gt;</code>","or following commands on M1 Mac:","<code>&lt;span&gt;# The version of Anaconda may be different depending on when you are installing`&lt;/span&gt;curl&lt;span&gt;-O&lt;/span&gt;https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh<br>sh Miniconda3-latest-MacOSX-arm64.sh&lt;span&gt;# and follow the prompts. The defaults are generally good.`&lt;/span&gt;</code>","pip","<em>Python 3</em>","If you installed Python via Homebrew or the Python website,<code>pip</code>was installed with it. If you installed Python 3.x, then you will be using the command<code>pip3</code>.","Tip: If you want to use just the command<code>pip</code>, instead of<code>pip3</code>, you can symlink<code>pip</code>to the<code>pip3</code>binary.","Installation","Anaconda","To install PyTorch via Anaconda, use the following conda command:","<code>conda&lt;span&gt;install&lt;/span&gt;pytorch torchvision&lt;span&gt;-c&lt;/span&gt;pytorch</code>","pip","To install PyTorch via pip, use one of the following two commands, depending on your Python version:","<code>&lt;span&gt;# Python 3.x&lt;/span&gt;pip3&lt;span&gt;install&lt;/span&gt;torch torchvision</code>","Verification","To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.","<code>&lt;span&gt;import&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;=&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;rand&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;5&lt;/span&gt;&lt;span&gt;,&lt;/span&gt;&lt;span&gt;3&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;&lt;span&gt;print&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;</code>","The output should be something similar to:","<code>tensor([[0.3380, 0.3845, 0.3217],<br>        [0.8337, 0.9050, 0.2650],<br>        [0.2979, 0.7141, 0.9069],<br>        [0.1449, 0.1132, 0.1375],<br>        [0.4675, 0.3947, 0.1426]])</code>","Building from source","For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to<a>build PyTorch from source</a>.","Prerequisites","[Optional] Install<a>Anaconda</a>","Follow the steps described here:<a>https://github.com/pytorch/pytorch#from-source</a>","You can verify the installation as described<a>above</a>.","Installing on Linux","PyTorch can be installed and used on various Linux distributions. Depending on your system and compute requirements, your experience with PyTorch on Linux may vary in terms of processing time. It is recommended, but not required, that your Linux system has an NVIDIA or AMD GPU in order to harness the full power of PyTorch’s<a>CUDA</a><a>support</a>or<a>ROCm</a>support.","Prerequisites","Supported Linux Distributions","PyTorch is supported on Linux distributions that use<a>glibc</a>&gt;= v2.17, which include the following:","<a>Arch Linux</a>, minimum version 2012-07-15","<a>CentOS</a>, minimum version 7.3-1611","<a>Debian</a>, minimum version 8.0","<a>Fedora</a>, minimum version 24","<a>Mint</a>, minimum version 14","<a>OpenSUSE</a>, minimum version 42.1","<a>PCLinuxOS</a>, minimum version 2014.7","<a>Slackware</a>, minimum version 14.2","<a>Ubuntu</a>, minimum version 13.04","The install instructions here will generally apply to all supported Linux distributions. An example difference is that your distribution may support<code>yum</code>instead of<code>apt</code>. The specific examples shown were run on an Ubuntu 18.04 machine.","Python","Python 3.8-3.11 is generally installed by default on any of our supported Linux distributions, which meets our recommendation.","Tip: By default, you will have to use the command<code>python3</code>to run Python. If you want to use just the command<code>python</code>, instead of<code>python3</code>, you can symlink<code>python</code>to the<code>python3</code>binary.","However, if you want to install another version, there are multiple ways:","APT","<a>Python website</a>","If you decide to use APT, you can run the following command to install it:","<code>&lt;span&gt;sudo&lt;/span&gt;apt&lt;span&gt;install&lt;/span&gt;python</code>","If you use<a>Anaconda</a>to install PyTorch, it will install a sandboxed version of Python that will be used for running PyTorch applications.","Package Manager","To install the PyTorch binaries, you will need to use one of two supported package managers:<a>Anaconda</a>or<a>pip</a>. Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.","Anaconda","To install Anaconda, you will use the<a>command-line installer</a>. Right-click on the 64-bit installer link, select<code>Copy Link Location</code>, and then use the following commands:","<code>&lt;span&gt;# The version of Anaconda may be different depending on when you are installing`&lt;/span&gt;curl&lt;span&gt;-O&lt;/span&gt;https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh<br>sh Miniconda3-latest-Linux-x86_64.sh&lt;span&gt;# and follow the prompts. The defaults are generally good.`&lt;/span&gt;</code>","You may have to open a new terminal or re-source your<code>~/.bashrc</code>to get access to the<code>conda</code>command.","pip","<em>Python 3</em>","While Python 3.x is installed by default on Linux,<code>pip</code>is not installed by default.","<code>&lt;span&gt;sudo&lt;/span&gt;apt&lt;span&gt;install&lt;/span&gt;python3-pip</code>","Tip: If you want to use just the command<code>pip</code>, instead of<code>pip3</code>, you can symlink<code>pip</code>to the<code>pip3</code>binary.","Installation","Anaconda","No CUDA/ROCm","To install PyTorch via Anaconda, and do not have a<a>CUDA-capable</a>or<a>ROCm-capable</a>system or do not require CUDA/ROCm (i.e. GPU support), in the above selector, choose OS: Linux, Package: Conda, Language: Python and Compute Platform: CPU.\nThen, run the command that is presented to you.","With CUDA","To install PyTorch via Anaconda, and you do have a<a>CUDA-capable</a>system, in the above selector, choose OS: Linux, Package: Conda and the CUDA version suited to your machine. Often, the latest CUDA version is better.\nThen, run the command that is presented to you.","With ROCm","PyTorch via Anaconda is not supported on ROCm currently. Please use pip instead.","pip","No CUDA","To install PyTorch via pip, and do not have a<a>CUDA-capable</a>or<a>ROCm-capable</a>system or do not require CUDA/ROCm (i.e. GPU support), in the above selector, choose OS: Linux, Package: Pip, Language: Python and Compute Platform: CPU.\nThen, run the command that is presented to you.","With CUDA","To install PyTorch via pip, and do have a<a>CUDA-capable</a>system, in the above selector, choose OS: Linux, Package: Pip, Language: Python and the CUDA version suited to your machine. Often, the latest CUDA version is better.\nThen, run the command that is presented to you.","With ROCm","To install PyTorch via pip, and do have a<a>ROCm-capable</a>system, in the above selector, choose OS: Linux, Package: Pip, Language: Python and the ROCm version supported.\nThen, run the command that is presented to you.","Verification","To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.","<code>&lt;span&gt;import&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;=&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;rand&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;5&lt;/span&gt;&lt;span&gt;,&lt;/span&gt;&lt;span&gt;3&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;&lt;span&gt;print&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;</code>","The output should be something similar to:","<code>tensor([[0.3380, 0.3845, 0.3217],<br>        [0.8337, 0.9050, 0.2650],<br>        [0.2979, 0.7141, 0.9069],<br>        [0.1449, 0.1132, 0.1375],<br>        [0.4675, 0.3947, 0.1426]])</code>","Additionally, to check if your GPU driver and CUDA/ROCm is enabled and accessible by PyTorch, run the following commands to return whether or not the GPU driver is enabled (the ROCm build of PyTorch uses the same semantics at the python API level<a>link</a>, so the below commands should also work for ROCm):","<code>&lt;span&gt;import&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;cuda&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;is_available&lt;/span&gt;&lt;span&gt;()&lt;/span&gt;</code>","Building from source","For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to<a>build PyTorch from source</a>.","Prerequisites","Install<a>Anaconda</a>or<a>Pip</a>","If you need to build PyTorch with GPU support\na. for NVIDIA GPUs, install<a>CUDA</a>, if your machine has a<a>CUDA-enabled GPU</a>.\nb. for AMD GPUs, install<a>ROCm</a>, if your machine has a<a>ROCm-enabled GPU</a>","Follow the steps described here:<a>https://github.com/pytorch/pytorch#from-source</a>","You can verify the installation as described<a>above</a>.","Installing on Windows","PyTorch can be installed and used on various Windows distributions. Depending on your system and compute requirements, your experience with PyTorch on Windows may vary in terms of processing time. It is recommended, but not required, that your Windows system has an NVIDIA GPU in order to harness the full power of PyTorch’s<a>CUDA</a><a>support</a>.","Prerequisites","Supported Windows Distributions","PyTorch is supported on the following Windows distributions:","<a>Windows</a>7 and greater;<a>Windows 10</a>or greater recommended.","<a>Windows Server 2008</a>r2 and greater","The install instructions here will generally apply to all supported Windows distributions. The specific examples shown will be run on a Windows 10 Enterprise machine","Python","Currently, PyTorch on Windows only supports Python 3.8-3.11; Python 2.x is not supported.","As it is not installed by default on Windows, there are multiple ways to install Python:","<a>Chocolatey</a>","<a>Python website</a>","<a>Anaconda</a>","If you use Anaconda to install PyTorch, it will install a sandboxed version of Python that will be used for running PyTorch applications.","If you decide to use Chocolatey, and haven’t installed Chocolatey yet, ensure that you are running your command prompt as an administrator.","For a Chocolatey-based install, run the following command in an administrative command prompt:","<code>choco&lt;span&gt;install&lt;/span&gt;python</code>","Package Manager","To install the PyTorch binaries, you will need to use at least one of two supported package managers:<a>Anaconda</a>and<a>pip</a>. Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python and<code>pip.</code>","Anaconda","To install Anaconda, you will use the<a>64-bit graphical installer</a>for PyTorch 3.x. Click on the installer link and select<code>Run</code>. Anaconda will download and the installer prompt will be presented to you. The default options are generally sane.","pip","If you installed Python by any of the recommended ways<a>above</a>,<a>pip</a>will have already been installed for you.","Installation","Anaconda","To install PyTorch with Anaconda, you will need to open an Anaconda prompt via<code>Start | Anaconda3 | Anaconda Prompt</code>.","No CUDA","To install PyTorch via Anaconda, and do not have a<a>CUDA-capable</a>system or do not require CUDA, in the above selector, choose OS: Windows, Package: Conda and CUDA: None.\nThen, run the command that is presented to you.","With CUDA","To install PyTorch via Anaconda, and you do have a<a>CUDA-capable</a>system, in the above selector, choose OS: Windows, Package: Conda and the CUDA version suited to your machine. Often, the latest CUDA version is better.\nThen, run the command that is presented to you.","pip","No CUDA","To install PyTorch via pip, and do not have a<a>CUDA-capable</a>system or do not require CUDA, in the above selector, choose OS: Windows, Package: Pip and CUDA: None.\nThen, run the command that is presented to you.","With CUDA","To install PyTorch via pip, and do have a<a>CUDA-capable</a>system, in the above selector, choose OS: Windows, Package: Pip and the CUDA version suited to your machine. Often, the latest CUDA version is better.\nThen, run the command that is presented to you.","Verification","To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.","From the command line, type:","<code>python</code>","then enter the following code:","<code>&lt;span&gt;import&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;=&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;rand&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;5&lt;/span&gt;&lt;span&gt;,&lt;/span&gt;&lt;span&gt;3&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;&lt;span&gt;print&lt;/span&gt;&lt;span&gt;(&lt;/span&gt;&lt;span&gt;x&lt;/span&gt;&lt;span&gt;)&lt;/span&gt;</code>","The output should be something similar to:","<code>tensor([[0.3380, 0.3845, 0.3217],<br>        [0.8337, 0.9050, 0.2650],<br>        [0.2979, 0.7141, 0.9069],<br>        [0.1449, 0.1132, 0.1375],<br>        [0.4675, 0.3947, 0.1426]])</code>","Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:","<code>&lt;span&gt;import&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;torch&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;cuda&lt;/span&gt;&lt;span&gt;.&lt;/span&gt;&lt;span&gt;is_available&lt;/span&gt;&lt;span&gt;()&lt;/span&gt;</code>","Building from source","For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to<a>build PyTorch from source</a>.","Prerequisites","Install<a>Anaconda</a>","Install<a>CUDA</a>, if your machine has a<a>CUDA-enabled GPU</a>.","If you want to build on Windows, Visual Studio with MSVC toolset, and NVTX are also needed. The exact requirements of those dependencies could be found out<a>here</a>.","Follow the steps described here:<a>https://github.com/pytorch/pytorch#from-source</a>","You can verify the installation as described<a>above</a>.","Docs","Access comprehensive developer documentation for PyTorch","View Docs","Tutorials","Get in-depth tutorials for beginners and advanced developers","View Tutorials","Resources","Find development resources and get your questions answered","View Resources","© Copyright The Linux Foundation. The PyTorch Foundation is a project of The Linux Foundation. \n          For web site terms of use, trademark policy and other policies applicable to The PyTorch Foundation please see<a>Linux Foundation Policies</a>. The PyTorch Foundation supports the PyTorch open source \n          project, which has been established as PyTorch Project a Series of LF Projects, LLC. For policies applicable to the PyTorch Project a Series of LF Projects, LLC, \n          please see<a>LF Projects, LLC Policies</a>.<a>Privacy Policy</a>and<a>Terms of Use</a>.","<a>Learn</a>","<a>Get Started</a>","<a>Tutorials</a>","<a>Learn the Basics</a>","<a>PyTorch Recipes</a>","<a>Introduction to PyTorch - YouTube Series</a>","<a>Ecosystem</a>","<a>Tools</a>","<a>Community</a>","<a>Forums</a>","<a>Developer Resources</a>","<a>Contributor Awards - 2023</a>","<a>Edge</a>","<a>About PyTorch Edge</a>","<a>ExecuTorch</a>","<a>Docs</a>","<a>PyTorch</a>","<a>PyTorch Domains</a>","<a>Blog &amp;amp; News</a>","<a>PyTorch Blog</a>","<a>Community Blog</a>","<a>Videos</a>","<a>Community Stories</a>","<a>Events</a>","<a>About</a>","<a>PyTorch Foundation</a>","<a>Governing Board</a>","<a>Become a Member</a>","To analyze traffic and optimize your experience, we serve cookies on this site. By clicking or navigating, you agree to allow our usage of cookies. As the current maintainers of this site, Facebook’s Cookies Policy applies. Learn more, including about available controls:<a>Cookies Policy</a>."],"language":"ko"}

    print(asyncio.run(translate_text(input_dict)))