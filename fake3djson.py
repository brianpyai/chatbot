import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fileDict3 import FileDict
import json
import numpy as np
from PIL import Image
import rembg
import base64,time
import io

app = FastAPI()
model_db = FileDict("./3d0.sql")
Names = []
Updated = False
CurrentName=None 
def loadList():
    global Updated, Names, model_db
    if not Updated:
        model_db._commit()
        model_db.close()
        model_db = FileDict("./3d0.sql")
        Names1 = list(model_db)
        if Names1 != Names:
            Updated = True
            Names = Names1
    return Names

def loadListApp():
    names=loadList()
    name=CurrentName 
    return gr.update(choices =names ,value =name )


def remove_background(input_image):
    input_image = Image.fromarray(input_image)
    output = rembg.remove(input_image)
    return np.array(output)

def process_and_create_3d(name,input_image, thickness, size):
    processed_image = remove_background(input_image)
    
    buffered = io.BytesIO()
    Image.fromarray(processed_image).save(buffered, format="webp")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    model_data = {
        "kind": 1,
        "name": name,
        "pngUrl": f"data:image/webp;base64,{img_str}",
        "thickness": thickness,
        "size": size
    }
    
    return processed_image, json.dumps(model_data)

def save_model(name, json_data, input_image=None, thickness=0.05, size=4):
    global model_db,Updated,CurrentName
    try:
        if input_image is not None:
            _, json_data = process_and_create_3d(name,input_image, thickness, size)
            model_db[name] = json_data
        else:
            model_db[name] = json_data
        model_db._commit()
        Updated=True 
        names=loadList()
        CurrentName=name
        
        return  name,json_data,render_model(name),gr.update(choices =names ,value =name )
    except json.JSONDecodeError:
        return "Error: Invalid JSON data.", Names, gr.update()

def load_model(name):
    global CurrentName
    CurrentName=name
    if name in model_db:
        return name, model_db[name], render_model(name)
    else:
        return f"Error: Model '{name}' not found.", "", ""

def generate_html(model_data,name=None ):
    if name==None :name =json.loads(model_data)["name"]
    pos=5
    if name=="SolarSystem":pos=100
    if name=="Windmill" : pos=40
    html_template = open("00base.hrml","r").read()
    js=open ("three.min.js" ,"r").read()
    return html_template % (js,pos ,model_data)
def render_model(name):
    
    timestamp = int(time.time())
    return f'<div class="iframe-container"><iframe src="/model/{name}/?t={timestamp}"></iframe></div>'


css = '''
div {
  overflow: hidden;
  resize: both;
}
iframe {
  height: 648px;
  width: 432px;
  border: none;
}
'''
css = '''
.iframe-container {
  position: relative;
  width: 100%;
  padding-bottom: 75%; /* 4:3 Aspect Ratio */  height: 648px;
  border: none;
  overflow: hidden;
  resize: both;
}
.iframe-container iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: none;
}
'''


 



@app.get("/model/{name}/", response_class=HTMLResponse)
async def get_model(name: str):
    if name in model_db:
        return generate_html(model_db[name],names_list)
    return HTMLResponse(content="Model not found", status_code=404)



with gr.Blocks(css=css) as demo:
    
    
    with gr.Row():
        name_input = gr.Textbox(label="Model Name")
    
    with gr.Row():
        with gr.Column():
            json_input = gr.Code(label="JSON Data", language="json", lines=10)
        with gr.Column():
            image_input = gr.Image(label="Input Image")
            thickness_slider = gr.Slider(minimum=0.001, maximum=0.2, step=0.001, value=0.05, label="Thickness")
            size_slider = gr.Slider(minimum=1, maximum=5, step=0.1, value=3, label="Size")
    
    
    
    with gr.Row():
        save_btn = gr.Button("Save Model")
        load_btn = gr.Button("Load Model")
        process_btn= gr.Button("Process and Save")     
    
    with gr.Row():
        render_output = gr.HTML()
        
    with gr.Row():
        names_list = gr.Radio(label="Models", choices=[] )
    
    

    save_btn.click(save_model, inputs=[name_input, json_input], outputs=[name_input , json_input, render_output,names_list])
    load_btn.click(load_model, inputs=name_input, outputs=[name_input, json_input, render_output,names_list])
    names_list.change(load_model, inputs=names_list, outputs=[name_input, json_input, render_output])
    process_btn.click(save_model, inputs=[name_input, json_input, image_input, thickness_slider, size_slider], outputs=[name_input , json_input, render_output,names_list])
    demo.load(fn=loadListApp,outputs=[names_list])
    

    

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
