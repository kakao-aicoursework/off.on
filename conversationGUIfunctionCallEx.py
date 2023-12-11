import json
import openai
import tkinter as tk
import chromadb
import os
from tkinter import scrolledtext

openai.api_key = os.environ.get("API_KEY")


def make_vector_db():
    data = read_to_txt('./project_data_카카오톡채널.txt')
    json = []
    ids = []
    documents = []

    for idx in range(1, len(data)):
        if data[idx].startswith('#'):
            json.append({
                "title": data[idx].replace('#', ''),
                "content": ""
            })
            ids.append(data[idx].replace('#', ''))
        else:
            json[len(json) - 1]["content"] += data[idx]

    for idx in range(len(json)):
        document = f"{json[idx]['title']}-{json[idx]['content']}"
        documents.append(document)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_talk_channel_db",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=ids,
        documents=documents
    )


def get_kakao_talk_channel_info(question: str):
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="kakao_talk_channel_db",
        metadata={"hnsw:space": "cosine"}
    )
    results = collection.query(
        query_texts=question,
        n_results=3
    )

    data = []

    for document in results["documents"][0]:
        splits = document.split('-')
        data.append(
            {
                "title": splits[0],
                "content": splits[1]
            }
        )

    return json.dumps(data, ensure_ascii=False)


def read_to_txt(file_path):
    filedata = []
    file = open(file_path, 'r', encoding='utf-8')
    while True:
        line = file.readline()
        if not line: break
        line = line.replace('\n', '')
        if line != '': filedata.append(line)
    file.close()
    return filedata


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "get_kakao_talk_channel_info": get_kakao_talk_channel_info,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def main():
    message_log = [  # System Property
        {
            "role": "system",
            "content": '''
            You are a chatbot that informs you of the features supported by KakaoTalk. Your user will be Korean, so you need to communicate in Korean. 
            First of all, you need to ask the user what he or she has any questions about KakaoTalk.
            '''
        }
    ]

    functions = [
        {
            "name": "get_kakao_talk_channel_info",
            "description": "Get information about KakaoTalk Channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "This is the question value entered by the user about what you are curious about KakaoTalk.",
                    },
                },
                "required": ["question"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    make_vector_db()
    main()
