import gradio as gr

from model import RandomCNN, run_transfer
from utils import read_audio_spectrum, spectrum_to_audio

example_audios = [
    ["wavs/voices/boy.wav",
    "wavs/voices/girl.wav"],

    ["wavs/corpus/johntejada-1.wav",
     "wavs/target/beat-box-2.wav"],

    ["wavs/songs/imperial.mp3",
     "wavs/songs/usa.mp3"],

    ["wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3",
     "wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3"],

    ["wavs/birds/MEX_ALTAMIRA_ORIOLE/MEX_Altamira_Oriole-ACelisM_01.mp3",
    "wavs/birds/BR_ALAGOAS_FOLIAGE/BR_AL_XC181063-PHINOV36_0101_LIMPO.mp3"],
]


def do_transfer(content_path, style_path):
    content_spectrum, content_sr, content_p = read_audio_spectrum(content_path)
    style_spectrum, style_sr, style_p = read_audio_spectrum(style_path)

    mcnn = RandomCNN(out_channels=16, kernel=(11, 3))
    result = run_transfer(mcnn,
                          content_spectrum, style_spectrum,
                          num_steps=2000,
                          content_weight=1e-1, style_weight=1e2
                         )

    result_s = result.cpu().data.numpy().squeeze()
    gen_wav = spectrum_to_audio(result_s, p=content_p, rounds=150)

    content_slug = content_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
    style_slug = style_path.split("/")[-1].split(" ")[0].split(".")[0][:32].split("-0-")[0]
    filename = f"c-{content_slug}_s-{style_slug}.wav"

    return (content_sr, gen_wav), filename

demo = gr.Interface(
    title="Audio Style Transfer",
    description="Combine style and content from two different audio files",

    fn=do_transfer,
    inputs=[
        gr.Audio(type="filepath", sources=["upload"], label="Content"),
        gr.Audio(type="filepath", sources=["upload"], label="Style")
    ],
    outputs=[
        gr.Audio(label="Output"),
        gr.Textbox(label="filename")
    ],

    examples=example_audios,
    cache_examples=True,

    allow_flagging="never",
    analytics_enabled=None
)

demo.launch(show_api=False, server_name="0.0.0.0")
