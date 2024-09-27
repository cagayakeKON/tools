import os
import subprocess
import json
from langchain.agents import tool,initialize_agent,load_tools
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain.prompts import ChatPromptTemplate  
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import LLMChain  
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables import RunnableLambda
from langchain.chains import SequentialChain
object_name = ""
def extract_subtitles(video_path):
    global object_name
    object_name = os.path.basename(video_path).split('.')[0]
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    streams = json.loads(result.stdout)
    subtitle_stream_index = None
    for stream in streams['streams']:
        if stream['codec_type'] == 'subtitle':
            subtitle_stream_index = stream['index']
            break

    if subtitle_stream_index is None:
        print("not found subtitle stream.")
    cmd = ['ffmpeg', '-y', '-i', video_path, '-map', '0:2', '-c:s', 'srt', f"./sub{object_name}.srt"]
    subprocess.run(cmd, check=True)
    print(f"sub will save to {object_name}.srt")

def split_subtitles(subtitle_file: str, max_number_per_part: int = 15):
    lines = subtitle_file.strip().split('\n')
    parts = []
    current_part = []
    current_index = 0
    
    for line in lines:
        if line.strip().isdigit():
            if current_index >= max_number_per_part:
                parts.append('\n'.join(current_part))
                current_part = []
                current_index = 0
            current_index += 1
        current_part.append(line)   
    # Append the last part if it has any content
    if current_part:
        parts.append('\n'.join(current_part))
    return parts

llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key='sk-b5c554e608a74e18b902583f98547f94',
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024*4, # the deepseek-chat api can handle 4096 tokens
    temperature=0
)

# convert Subtitles to sentences chain

sub_to_sentence_chain = LLMChain(
    llm=llm,
    prompt = ChatPromptTemplate.from_template(
    """C
    私は現在英語を学んでいます。あなたは英語の専門家として、以下のことを行う必要があります。これは英語の字幕です。
    今、あなたが行うべきことは以下の通りです。
    1. 各字幕セグメントを抽出し、それらを適切な完全な文章にまとめ、改行を加えてください。文章以外の出力は不要です。
    例えば：
    10
    00:01:44,880 --> 00:01:46,000
    <font face="Noto Sans" size="55">were you two able to get done</font>

    11
    00:01:46,080 --> 00:01:47,750
    <font face="Noto Sans" size="55">with all your
    homework and stuff?</font>
    
    出力:
    were you two able to get done with all your homework and stuff?\n
    
    テキストは以下の通り:
    ```text
    {subtitle}
    ```
    """
    ),
    output_key="sentences"
)
# explain the sentences chain
explain_chain = LLMChain(
    llm=llm,
    prompt = ChatPromptTemplate.from_template(
    """
    あなたは英語の教師で、私は日本の学生で英語レベルは低い。対話はできるだけ日本語で行い、私の入力した内容を使って英語の教育を行ってください。また、小説の内容の紹介であろうと私が直接入力したものであろうと、一文一文を日本語で解説し、文法構造を付け加えてください。これには文の成分や品詞、その他の難しい点も含まれます。
    不規則な名詞や動詞の変形がある場合は、原形を示し、それが不規則な変形であることを指摘してください。他の語彙についても、原形を使って説明してください。
    解説以外の出力は行わないでください。以下の例に厳密に従って、一文一文を解説してください。

    例:
    It might be hard to play games then. But it is a good idea to have something to distract us..

    - It might be hard to play games then.
      * it /ɪt/ pron. それ
      * might /maɪt/ aux. かもしれない
      * be /biː/ v. である
      * hard /hɑːrd/ adj. 難しい
      * to play games ゲームをする（不定詞が述語名詞）
      * then /ðen/ adv. その時
      日本語訳: その時、ゲームをするのは難しいかもしれません。
      文法: 主語は 'it'、述語は 'might be hard'、補語は形容詞の 'hard'、目的を表す不定詞 'to play games' が述語に続いています。'then' は副詞で、いつゲームが難しくなるかを示しています。

    - But it is a good idea to have something to distract us.
      * but /bʌt/ conj. しかし
      * it /ɪt/ pron. それ
      * is /ɪz/ v. である
      * a good idea 良い考え
      * to have 持つ（不定詞が述語名詞）
      * something /'sʌmθɪŋ/ pron. 何か
      * to distract 注意を散らす（不定詞が形容詞）
      * us /ʌs/ pron. 私たち
      日本語訳: しかし、私たちの注意を散らす何かを持つことは良い考えです。
      文法: 主語は 'it'、述語は 'is a good idea'、補語は名詞句の 'a good idea'。'to have something to distract us' は目的を表す不定詞で、'something' を持って 'us' の注意を 'distract' する方法を述べています。
    
    テキストは以下の通り:
    ```text
    {sentences}
    ```
    """),
    output_key="explanations"
)


sequence_chain = SequentialChain(
    chains=[sub_to_sentence_chain, explain_chain],
    input_variables=['subtitle'],
    output_variables=['sentences', 'explanations'],
    verbose=True
)


def main():
    extract_subtitles(os.path.abspath(f"D:/MyData/study/New folder/01(1)/english_study_agent/video/06.mkv"))
    with open(f"./sub{object_name}.srt") as f:
        subtitle = f.read()
    parts = split_subtitles(subtitle)
    runnable = RunnableLambda(
        lambda x: sequence_chain.invoke(x),
    )
    for index, part in enumerate(parts):
        print(f"part: {index+1}, handle start,total {len(parts)} parts")
        result = runnable.with_retry(
            stop_after_attempt=10
        ).invoke(part)
        print(result['sentences'])
        print(result['explanations'])
        print(f"part: {index+1}, handle end")
        with open(f"{object_name}.md", "a", encoding='utf-8') as f:
            f.write(result['explanations']+ '\n\n')
            
main()