import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io
import librosa, librosa.display
import soundfile as sf
import torch
import pickle
import urllib.request

from tqdm import tqdm
import json

import sys
sys.path.insert(0, '../')
from diffusers import DDPMScheduler
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
from tango import Tango
from tqdm.auto import tqdm, trange

from accelerate.utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

model_name = "declare-lab/tango-full-ft-audiocaps"
sample_rate=16000
duration = 10.0 #Duration is minimum 10 secs. The generated sounds are weird for <10secs
guidance_scale = 3
random_seed = 42
n_candidates = 1
batch_size = 1
diffusion_steps = 20

latent_diffusion = None


def get_config(filepath='config/config.json'):
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def populate_prompts():
    prompts_map = {}
    prompts = get_config()
    for prompt in prompts:
        prompts_map[prompt['text']] = {'id':prompt['id'], 'slider_words': prompt['slider_words']}        
    return prompts_map

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def get_model():
    print('Loading model')
    
    latent_diffusion = Tango(model_name)

    print('Model loaded')
    return latent_diffusion


def generate(latent_diffusion, prompt, steps, seed, disable_progress):
    print('In generate')
    set_seed(seed)

    attention_weights = get_attention_weights(latent_diffusion)

    wav = latent_diffusion.generate(prompt=prompt, steps=steps, disable_progress=disable_progress, \
                         attention_weights=attention_weights).astype(np.float16)
    
    fig =plt.figure(figsize=(10, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=512)),ref=np.max)
    librosa.display.specshow(D, y_axis='linear', sr=16000, hop_length=512, x_axis='time')
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    return wav, img_arr


def get_words_token_mapping(latent_diffusion, prompt):
    word_token_map = {}
    
    batch = latent_diffusion.model.tokenizer(prompt)
    desired_output = []
    print(batch)
    for word_id in batch.word_ids():
        if word_id is not None:
            start, end = batch.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = [start, end-1]
            if len(desired_output) == 0 or desired_output[-1] != tokens:
                desired_output.append(tokens)
    for ind, word in enumerate([i for i in prompt.split(' ')]):
        print(word, desired_output[ind])
        word_token_map[word] = desired_output[ind]

    return word_token_map, len(batch['input_ids'])

def get_attention_weights(latent_diffusion):
    print('In get_attention_weights')
    prompt = st.session_state['prompt_selected']

    word_token_map, len_batch = get_words_token_mapping(latent_diffusion, prompt)

    attention_weights = torch.from_numpy(np.array([1.0 for i in range(len_batch)])).float().cuda()
    
    print(word_token_map)
    for ind, word in enumerate(word_token_map.keys()):
        print(word, word_token_map[word])

        if 'slider_'+word in st.session_state:
            print(st.session_state['slider_'+word], word_token_map[word][0], word_token_map[word][-1]+1)
            attention_weights[word_token_map[word][0]:word_token_map[word][-1]+1] = st.session_state['slider_'+word]
    
    print('******', attention_weights)

    return attention_weights

def main():

    print('before get model')
    latent_diffusion = get_model()
    print('after get model')

    prompts_map = populate_prompts()


    st.markdown("<h2 style='text-align: center;'>'Text-to-Continuous' Semantic Control For Audio</h2>", unsafe_allow_html=True)

    prompt_selected =  st.selectbox('Select a prompt', sorted(prompts_map.keys()), key='prompt_selected')
    slider_words = prompts_map[prompt_selected]['slider_words']
    slider_id = str(prompts_map[prompt_selected]['id'])

    
    
    s_wav, s_spec = generate(latent_diffusion, prompt_selected, diffusion_steps, random_seed, disable_progress=False)
    # s_wav = st.session_state['wav']
    # s_spec = st.session_state['spectrogram']
    print(s_wav)

    col1, col2, col3, col4 = st.columns((0.3,0.1,0.4,0.2))

    with col1:
        st.markdown("<br/>", unsafe_allow_html=True)
        display_text = prompt_selected
        for word in slider_words:
            display_text = display_text.replace(word, "<span style='background-color: yellow; color:black;'>"+word+"</span>")
        st.markdown("<div style='text-align: left;'>"+display_text+"</div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        for word in slider_words:
            slider_position=st.slider(word, min_value=-5.0, max_value=5.0, value=1.0, step=0.1,  format=None, key='slider_'+word, help=None, args=None, kwargs=None, disabled=False)
    with col2:
        vert_space = '<div style="padding: 25%;">&nbsp;</div>'
        st.markdown(vert_space, unsafe_allow_html=True)
        # st.button("**Generate** =>", on_click=sample_diffusion(latent_diffusion), type='primary')
    with col3:
        st.image(s_spec)
        st.audio(s_wav, format="audio/wav", start_time=0, sample_rate=16000)

    st.markdown('<div style="text-align:center;color:white"><i>All audio samples on this page are generated with a sampling rate of 16kHz.</i></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()