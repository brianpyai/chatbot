import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import inspect
import time
times=40
angle =3
delay=0.3
def yieldText(text,times=times,delay =0):
    step=len (text ) /times
    position =0
    while position< len(text):
        yield text[:int(position +step )]
        position=int(position+step)
        time.sleep(delay)
def rotateaimage(image,angle=angle,times=times,delay =delay):
    #print(image)
    if isinstance(image,dict):
        image=image['composite']
    for i in range(times):
        image1=image.rotate (angle)
        angle+=angle
        yield image1
        time.sleep(delay)
def zipFunctions(image,text,imgFunc=rotateaimage, textFunc=yieldText):
    for img , txt in zip( imgFunc(image) , textFunc(text) ): yield img , txt

def process_image_and_text(image, text, code):
    if isinstance(image,dict):
        image=image['composite']
    local_vars = dict(image=image, text=text, images=[] ,zipFunctions=zipFunctions , yieldText =yieldText , rotateaimage= rotateaimage )
    
    try:
        exec(code, local_vars, local_vars)
        edited_image = local_vars.get("image", image)
        edited_images = local_vars.get("images", [])
        edited_text = local_vars.get("text", text)
    except Exception as e:
        yield Image.fromarray(np.zeros_like(np.array(image))), f"Error: {str(e)}"

    if inspect.isgenerator(edited_images):
        for img, txt in edited_images:
            yield img, txt
    else:
        yield edited_image, edited_text

with gr.Blocks() as demo:
    gr.Markdown("# Python Image Editor")

    image_editor = gr.ImageEditor(type="pil", label="Input Image")
    text_editor = gr.Textbox(label="Text Editor", show_copy_button=True)
    python_code = gr.Code(label="Python Code (local_vars: image, text)", interactive=True, language="python")

    output_image = gr.Image(type="pil", label="Output Image")
    output_text = gr.Textbox(label="Output Text", show_copy_button=True)
    submit_button = gr.Button("Submit")
    submit_button.click(process_image_and_text, inputs=[image_editor, text_editor, python_code], outputs=[output_image, output_text])

demo.launch()