import gradio as gr
import spaces
import json
import re
import random
import numpy as np
from gradio_client import Client

MAX_SEED = np.iinfo(np.int32).max

def check_api(model_name):
    if model_name == "MAGNet":
        try :
            client = Client("https://fffiloni-magnet.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "AudioLDM-2":
        try :
            client = Client("https://haoheliu-audioldm2-text2audio-text2music.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "Riffusion":
        try :
            client = Client("https://fffiloni-spectrogram-to-music.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "Mustango":
        try :
            client = Client("https://declare-lab-mustango.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "MusicGen":
        try :
            client = Client("https://facebook-musicgen.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
        
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioClip

def extract_audio(video_in):
    input_video = video_in
    output_audio = 'audio.wav'
    
    # Open the video file and extract the audio
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    
    # Save the audio as a .wav file
    audio_clip.write_audiofile(output_audio, fps=44100)  # Use 44100 Hz as the sample rate for .wav files  
    print("Audio extraction complete.")

    return 'audio.wav'



def get_caption(image_in):
    kosmos2_client = Client("https://ydshieh-kosmos-2.hf.space/")
    kosmos2_result = kosmos2_client.predict(
        image_in,	# str (filepath or URL to image) in 'Test Image' Image component
        "Detailed",	# str in 'Description Type' Radio component
        fn_index=4
    )

    print(f"KOSMOS2 RETURNS: {kosmos2_result}")

    with open(kosmos2_result[1], 'r') as f:
        data = json.load(f)
    
    reconstructed_sentence = []
    for sublist in data:
        reconstructed_sentence.append(sublist[0])

    full_sentence = ' '.join(reconstructed_sentence)
    #print(full_sentence)

    # Find the pattern matching the expected format ("Describe this image in detail:" followed by optional space and then the rest)...
    pattern = r'^Describe this image in detail:\s*(.*)$'
    # Apply the regex pattern to extract the description text.
    match = re.search(pattern, full_sentence)
    if match:
        description = match.group(1)
        print(description)
    else:
        print("Unable to locate valid description.")

    # Find the last occurrence of "."
    #last_period_index = full_sentence.rfind('.')

    # Truncate the string up to the last period
    #truncated_caption = full_sentence[:last_period_index + 1]

    # print(truncated_caption)
    #print(f"\n—\nIMAGE CAPTION: {truncated_caption}")
    
    return description

def get_caption_from_MD(image_in):
    client = Client("https://vikhyatk-moondream1.hf.space/")
    result = client.predict(
		image_in,	# filepath  in 'image' Image component
		"Describe precisely the image.",	# str  in 'Question' Textbox component
		api_name="/answer_question"
    )
    print(result)
    return result

def get_magnet(prompt):

    client = Client("https://fffiloni-magnet.hf.space/")
    result = client.predict(
        "facebook/magnet-small-10secs",	# Literal['facebook/magnet-small-10secs', 'facebook/magnet-medium-10secs', 'facebook/magnet-small-30secs', 'facebook/magnet-medium-30secs', 'facebook/audio-magnet-small', 'facebook/audio-magnet-medium']  in 'Model' Radio component
        "",	# str  in 'Model Path (custom models)' Textbox component
        prompt,	# str  in 'Input Text' Textbox component
        3,	# float  in 'Temperature' Number component
        0.9,	# float  in 'Top-p' Number component
        10,	# float  in 'Max CFG coefficient' Number component
        1,	# float  in 'Min CFG coefficient' Number component
        20,	# float  in 'Decoding Steps (stage 1)' Number component
        10,	# float  in 'Decoding Steps (stage 2)' Number component
        10,	# float  in 'Decoding Steps (stage 3)' Number component
        10,	# float  in 'Decoding Steps (stage 4)' Number component
        "prod-stride1 (new!)",	# Literal['max-nonoverlap', 'prod-stride1 (new!)']  in 'Span Scoring' Radio component
        api_name="/predict_full"
    )
    print(result)
    return result[1]

def get_audioldm(prompt):
    client = Client("https://haoheliu-audioldm2-text2audio-text2music.hf.space/")
    seed = random.randint(0, MAX_SEED)
    result = client.predict(
        prompt,	# str in 'Input text' Textbox component
        "Low quality.",	# str in 'Negative prompt' Textbox component
        10,	# int | float (numeric value between 5 and 15) in 'Duration (seconds)' Slider component
        6.5,	# int | float (numeric value between 0 and 7) in 'Guidance scale' Slider component
        seed,	# int | float in 'Seed' Number component
        3,	# int | float (numeric value between 1 and 5) in 'Number waveforms to generate' Slider component
        fn_index=1
    )
    print(result)
    audio_result = extract_audio(result)
    return audio_result

def get_riffusion(prompt):
    client = Client("https://fffiloni-spectrogram-to-music.hf.space/")
    result = client.predict(
		prompt,	# str  in 'Musical prompt' Textbox component
		"",	# str  in 'Negative prompt' Textbox component
		None,	# filepath  in 'parameter_4' Audio component
		10,	# float (numeric value between 5 and 10) in 'Duration in seconds' Slider component
		api_name="/predict"
    )
    print(result)
    return result[1]

def get_mustango(prompt):
    client = Client("https://declare-lab-mustango.hf.space/")
    result = client.predict(
		prompt,	# str  in 'Prompt' Textbox component
		200,	# float (numeric value between 100 and 200) in 'Steps' Slider component
		6,	# float (numeric value between 1 and 10) in 'Guidance Scale' Slider component
							api_name="/predict"
    )
    print(result)
    return result

def get_musicgen(prompt):
    client = Client("https://facebook-musicgen.hf.space/")
    result = client.predict(
        prompt,	# str  in 'Describe your music' Textbox component
        None,	# str (filepath or URL to file) in 'File' Audio component
        fn_index=0
    )
    print(result)
    return result[1]
    
import re
import torch
from transformers import pipeline

zephyr_model = "HuggingFaceH4/zephyr-7b-beta"
mixtral_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

pipe = pipeline("text-generation", model=zephyr_model, torch_dtype=torch.bfloat16, device_map="auto")

standard_sys = f"""
You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users.
In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model.

For example, if a user says, "a picture of a man in a black suit and tie riding a black dragon", provide immediately a musical prompt corresponding to the image description. 
Immediately STOP after that. It should be EXACTLY in this format:
"A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle"
"""

mustango_sys = f"""
You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users.
In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model, you MUST include chords progression.

For example, if a user says, "a painting of three old women having tea party", provide immediately a musical prompt corresponding to the image description. 
Immediately STOP after that. It should be EXACTLY in this format:
"The song is an instrumental. The song is in medium tempo with a classical guitar playing a lilting melody in accompaniment style. The song is emotional and romantic. The song is a romantic instrumental song. The chord sequence is Gm, F6, Ebm. The time signature is 4/4. This song is in Adagio. The key of this song is G minor."
"""

@spaces.GPU(enable_queue=True)
def get_musical_prompt(user_prompt, chosen_model):

    """
    if chosen_model == "Mustango" :
        agent_maker_sys = standard_sys
    else :
        agent_maker_sys = standard_sys
    """
    agent_maker_sys = standard_sys
    
    instruction = f"""
<|system|>
{agent_maker_sys}</s>
<|user|>
"""
    
    prompt = f"{instruction.strip()}\n{user_prompt}</s>"    
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    pattern = r'\<\|system\|\>(.*?)\<\|assistant\|\>'
    cleaned_text = re.sub(pattern, '', outputs[0]["generated_text"], flags=re.DOTALL)
    
    print(f"SUGGESTED Musical prompt: {cleaned_text}")
    return cleaned_text.lstrip("\n")

def infer(image_in, chosen_model, api_status):
    if image_in == None :
        raise gr.Error("Please provide an image input")

    if chosen_model == [] :
        raise gr.Error("Please pick a model")

    if api_status == "api not ready yet" :
        raise gr.Error("This model is not ready yet, you can pick another one instead :)")
    
    gr.Info("Getting image caption with Kosmos2...")
    user_prompt = get_caption(image_in)
    
    gr.Info("Building a musical prompt according to the image caption ...")
    musical_prompt = get_musical_prompt(user_prompt, chosen_model)

    if chosen_model == "MAGNet" :
        gr.Info("Now calling MAGNet for music...")
        music_o = get_magnet(musical_prompt)
    elif chosen_model == "AudioLDM-2" :
        gr.Info("Now calling AudioLDM-2 for music...")
        music_o = get_audioldm(musical_prompt)
    elif chosen_model == "Riffusion" :
        gr.Info("Now calling Riffusion for music...")
        music_o = get_riffusion(musical_prompt)
    elif chosen_model == "Mustango" :
        gr.Info("Now calling Mustango for music...")
        music_o = get_mustango(musical_prompt)
    elif chosen_model == "MusicGen" :
        gr.Info("Now calling MusicGen for music...")
        music_o = get_musicgen(musical_prompt)
    
    return gr.update(value=musical_prompt, interactive=True), gr.update(visible=True), music_o

def retry(chosen_model, caption):
    musical_prompt = caption

    if chosen_model == "MAGNet" :
        gr.Info("Now calling MAGNet for music...")
        music_o = get_magnet(musical_prompt)
    elif chosen_model == "AudioLDM-2" :
        gr.Info("Now calling AudioLDM-2 for music...")
        music_o = get_audioldm(musical_prompt)
    elif chosen_model == "Riffusion" :
        gr.Info("Now calling Riffusion for music...")
        music_o = get_riffusion(musical_prompt)
    elif chosen_model == "Mustango" :
        gr.Info("Now calling Mustango for music...")
        music_o = get_mustango(musical_prompt)
    elif chosen_model == "MusicGen" :
        gr.Info("Now calling MusicGen for music...")
        music_o = get_musicgen(musical_prompt)

    return music_o

demo_title = "Image to Music V2"
description = "Get music from a picture, compare text-to-music models"

css = """
#col-container {
    margin: 0 auto;
    max-width: 980px;
    text-align: left;
}
#inspi-prompt textarea {
    font-size: 20px;
    line-height: 24px;
    font-weight: 600;
}
/* fix examples gallery width on mobile */
div#component-11 > .gallery > .gallery-item > .container > img {
    width: auto!important;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
    
        gr.HTML(f"""
        <h2 style="text-align: center;">{demo_title}</h2>
        <p style="text-align: center;">{description}</p>
        """)
        
        with gr.Row():
            
            with gr.Column():
                image_in = gr.Image(
                    label = "Image reference",
                    type = "filepath",
                    elem_id = "image-in"
                )
                
                with gr.Row():
                    
                    chosen_model = gr.Dropdown(
                        label = "Choose a model",
                        choices = [
                            "MAGNet",
                            "AudioLDM-2",
                            "Riffusion",
                            "Mustango",
                            "MusicGen"
                        ],
                        value = None,
                        filterable = False
                    )
                    
                    check_status = gr.Textbox(
                        label="API status",
                        interactive=False
                    )
                
                submit_btn = gr.Button("Make music from my pic !")

                gr.Examples(
                    examples = [
                        ["examples/ocean_poet.jpeg"],
                        ["examples/jasper_horace.jpeg"],
                        ["examples/summer.jpeg"],
                        ["examples/mona_diner.png"],
                        ["examples/monalisa.png"],
                        ["examples/santa.png"],
                        ["examples/winter_hiking.png"],
                        ["examples/teatime.jpeg"],
                        ["examples/news_experts.jpeg"]
                    ],
                    fn = infer,
                    inputs = [image_in, chosen_model],
                    examples_per_page = 4
                )
            
            with gr.Column():
            
                caption = gr.Textbox(
                    label = "Inspirational musical prompt",
                    interactive = False,
                    elem_id = "inspi-prompt"
                )
                
                retry_btn = gr.Button("Retry with edited prompt", visible=False)
                
                result = gr.Audio(
                    label = "Music"
                )
        

    chosen_model.change(
        fn = check_api,
        inputs = chosen_model,
        outputs = check_status,
        queue = False
    )

    retry_btn.click(
        fn = retry,
        inputs = [chosen_model, caption],
        outputs = [result]
    )
    
    submit_btn.click(
        fn = infer,
        inputs = [
            image_in,
            chosen_model,
            check_status
        ],
        outputs =[
            caption,
            retry_btn,
            result
        ],
        concurrency_limit = 4
    )

demo.queue(max_size=16).launch(show_api=False)