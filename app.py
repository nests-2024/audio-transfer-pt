import gradio as gr
import numpy as np

NUM_INPUTS = 4

def update_inputs(*minputs):
    mouts = []
    for ma,mi in zip(minputs[0::2], minputs[1::2]):
        if ma is not None and mi is None:
            print("render spec for", ma)
            img = np.array([[0]])
            mouts.append(gr.Audio.update(visible=True))
            mouts.append(gr.Image.update(visible=True, value=img))
        elif ma is None and mi is not None:
            mouts.append(gr.Audio.update(visible=True))
            mouts.append(gr.Image.update(visible=True, value=None))
        else:
            mouts.append(gr.Audio.update(visible=True))
            mouts.append(gr.Image.update(visible=True))
    return [*mouts]

def clicked(*file_paths):
    print("click")
    for fp in file_paths:
        print(fp)
    return [gr.Audio.update(visible=True, value=None),
            gr.Image.update(visible=True, value=None)]

with gr.Blocks() as demo:
    gr.Markdown("Audio Style Transfer")
    minputs = []
    for i in range(NUM_INPUTS):
        with gr.Row():
            ma = gr.Audio(source="upload", type="filepath", label="Source", visible=True)
            mi = gr.Image(visible=True, interactive=False)
            minputs.append(ma)
            minputs.append(mi)

    for ma in minputs[0::2]:
        ma.change(update_inputs, inputs=[*minputs], outputs=[*minputs])

    
    result_but = gr.Button(value="Generate")
    with gr.Row():
        result_wav = gr.Audio(visible=False)
        result_img = gr.Image(visible=False)
    
    result_but.click(clicked, inputs=[*minputs], outputs=[result_wav, result_img])


# examples=example_audios,
# cache_examples=True,

# allow_flagging="never",
# analytics_enabled=None

# demo.launch(show_api=False, server_name="0.0.0.0")

if __name__ == "__main__":
   demo.launch()
