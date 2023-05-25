import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, LlamaTokenizer
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread


path_bisandbytes = "/home/demo/venv/lib/python3.8/site-packages/bitsandbytes"
if os.path.exists(path_bisandbytes):
    command = "sudo cp " + path_bisandbytes + "/libbitsandbytes_cuda118.so " + path_bisandbytes + "/libbitsandbytes_cpu.so"
    p = os.system(command)

model_name = "eachadea/vicuna-13b-1.1"

print(f"Starting to load the model to memory")
# m = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.float16).cuda()
# tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# generator = pipeline('text-generation', model=m, tokenizer=tok, device=0)

m = AutoModelForCausalLM.from_pretrained(
    model_name, device_map='auto', load_in_8bit=True)
tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
generator = pipeline('text-generation', model=m, tokenizer=tok)


print(f"Sucessfully loaded the model to the memory")

start_message = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""


def check_stopwords(list_key, input_str):
    input_str_lower = input_str.lower()
    matched_indices = []
    for keyword in list_key:
        keyword_lower = keyword.lower()
        start = 0
        while start < len(input_str_lower):
            index = input_str_lower.find(keyword_lower, start)
            if index == -1:
                break
            end = index + len(keyword_lower) - 1
            matched_indices.append((index, end))
            start = end + 1
    return len(matched_indices) > 0, matched_indices

class StopOnWords(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

# stop_words = ["<|USER|>", "<|ASSISTANT|>", "<|SYSTEM|>", "</s>"]
stop_words = ["<|", "</s>"]

# Have to manually hardcode the ids (instead of tok.encode) because
# 1. to skip the bos token (1)
# 2. eachadea/vicuna-7b-1.1 has two ids (529 and 29966) for <  and that causes headaches for executing the StopOnWords function
# stop_words_ids = [tok(stop_word, return_tensors='pt')['input_ids'].squeeze()[1:] for stop_word in stop_words]
# stop_words_ids = [[ 29966, 29989, 11889, 29989, 29958], [ 29966, 29989, 22933,  9047, 13566, 29989, 29958], [  29966, 29989, 14816,  1254, 12665, 29989, 29958], [2]]

stop_words_ids = [[ 29966, 29989], [ 529, 29989], [2]]


stop_words_ids_pt = [torch.tensor(x) for x in stop_words_ids]

stopping_criteria = StoppingCriteriaList([StopOnWords(stops=stop_words_ids_pt)])

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def chat(curr_system_message, history, temperature):

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = curr_system_message + \
        "".join(["".join(["<|USER|>"+item[0], "<|ASSISTANT|>"+item[1]])
                for item in history])

    # Tokenize the messages string
    model_inputs = tok([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        tok, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=temperature,
        num_beams=1,
        stopping_criteria=stopping_criteria
    )
    t = Thread(target=m.generate, kwargs=generate_kwargs)
    t.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:

        found_keyword, matched_indices = check_stopwords(stop_words, new_text)
        if not found_keyword:
            pass
        else:
            matched_indices.sort(key=lambda x: x[0], reverse=True)
            for start, end in matched_indices:
                new_text = new_text[:start] + new_text[end+1:]

        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history

    return partial_text


with gr.Blocks() as demo:
    # history = gr.State([])
    # gr.Markdown("## Vicuna Chat")

    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div>
                <img class="logo" src="https://lambdalabs.com/hubfs/logos/lambda-logo.svg" alt="Lambda Logo"
                    style="margin: auto; max-width: 7rem;">
                <h1 style="font-weight: 900; font-size: 3rem;">
                  Chat With Vicuna 13B
                </h1>
              </div>
            </div>
        """
    )

    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        temperature = gr.Slider(
            label="Temperature",
            value=0.1,
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            interactive=True,
            info="Higher values produce more diverse outputs",
        )
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")

    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True)
    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[
               submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=False)

    gr.HTML(
        """
            <div class="footer">
                <p> A chatbot tries to give helpful, detailed, and polite answers to the user's questions. Gradio Demo created by <a href="https://lambdalabs.com/">Lambda</a>.</p>
            </div>
            <div class="acknowledgments">
                <p> It is based on <a href="https://huggingface.co/eachadea/vicuna-13b-1.1">vicuna-13b-1.1</a>, an open-source LLM trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. More information can be found <a href="https://vicuna.lmsys.org/">here</a>.</p>
            </div>
        """
    )
    
demo.queue(max_size=32, concurrency_count=2)
demo.launch()
