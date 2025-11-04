import uuid
import gradio as gr
import re
from diffusers.utils import load_image
import requests
from awesome_chat import chat_huggingface

all_messages = []
QWEN_ENDPOINT = "http://localhost:8006"

def add_message(content, role):
    message = {"role":role, "content":content}
    all_messages.append(message)

def extract_medias(message):
    image_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png)")
    image_urls = []
    for match in image_pattern.finditer(message):
        if match.group(0) not in image_urls:
            image_urls.append(match.group(0))

    audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav)")
    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    video_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(mp4)")
    video_urls = []
    for match in video_pattern.finditer(message):
        if match.group(0) not in video_urls:
            video_urls.append(match.group(0))

    return image_urls, audio_urls, video_urls

def set_qwen_endpoint(qwen_endpoint):
    global QWEN_ENDPOINT
    QWEN_ENDPOINT = qwen_endpoint
    return QWEN_ENDPOINT

def add_text(messages, message):
    if not QWEN_ENDPOINT or "localhost" not in QWEN_ENDPOINT:
        return messages, "Please ensure Qwen server is running on localhost:8006"
    
    add_message(message, "user")
    messages = messages + [(message, None)]
    image_urls, audio_urls, video_urls = extract_medias(message)

    for image_url in image_urls:
        if not image_url.startswith("http"):
            image_url = "public/" + image_url
        image = load_image(image_url)
        name = f"public/images/{str(uuid.uuid4())[:4]}.jpg" 
        image.save(name)
        messages = messages + [((f"{name}",), None)]
    for audio_url in audio_urls:
        if not audio_url.startswith("http"):
            audio_url = "public/" + audio_url
        ext = audio_url.split(".")[-1]
        name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
        response = requests.get(audio_url)
        with open(name, "wb") as f:
            f.write(response.content)
        messages = messages + [((f"{name}",), None)]
    for video_url in video_urls:
        if not video_url.startswith("http"):
            video_url = "public/" + video_url
        ext = video_url.split(".")[-1]
        name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
        response = requests.get(video_url)
        with open(name, "wb") as f:
            f.write(response.content)
        messages = messages + [((f"{name}",), None)]
    return messages, ""

def bot(messages):
    if not QWEN_ENDPOINT or "localhost" not in QWEN_ENDPOINT:
        return messages
    
    # Use Qwen endpoint instead of OpenAI
    message = chat_huggingface(all_messages, "", "qwen", QWEN_ENDPOINT)["message"]
    image_urls, audio_urls, video_urls = extract_medias(message)
    add_message(message, "assistant")
    messages[-1][1] = message
    for image_url in image_urls:
        if not image_url.startswith("http"):
            image_url = image_url.replace("public/", "")
            messages = messages + [((None, (f"public/{image_url}",)))]
    for audio_url in audio_urls:
        if not audio_url.startswith("http"):
            audio_url = audio_url.replace("public/", "")
            messages = messages + [((None, (f"public/{audio_url}",)))]
    for video_url in video_urls:
        if not video_url.startswith("http"):
            video_url = video_url.replace("public/", "")
            messages = messages + [((None, (f"public/{video_url}",)))]
    return messages

with gr.Blocks() as demo:
    gr.Markdown("<h2><center>JARVIS with Qwen 2.5 7B (Local)</center></h2>")
    with gr.Row():
        qwen_endpoint = gr.Textbox(
            show_label=False,
            placeholder="Qwen server endpoint (default: http://localhost:8006)",
            lines=1,
            value="http://localhost:8006"
        )

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter. The url of the multimedia resource must contain the extension name.",
        ).style(container=False)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    qwen_endpoint.submit(set_qwen_endpoint, [qwen_endpoint], [qwen_endpoint])

    gr.Examples(
        examples=[
            "Please identify named entities in this text: 'John Smith works at Microsoft in Seattle since 2020.'",
            "Please summarize this text: 'The pandemic has hit the global economy hard, causing significant disruptions...'",
            "Please translate this to Spanish: 'Hello, how are you today? I hope you are doing well.'",
            "Answer this question: 'What is the main function of mitochondria?' Context: 'Mitochondria are membrane-bound cell organelles...'",
            "Classify this image: /examples/a.jpg",
            "Describe what you see in this image: /examples/a.jpg",
            "What objects are in this image: /examples/a.jpg?",
            "First, identify the named entities in this text: 'Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.' Then, translate the identified entities to Spanish and create a summary of what you found."
        ],
        inputs=txt
    )

demo.launch()
