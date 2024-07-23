import gradio as gr
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

# Create FastAPI app
app = FastAPI()

# Cool text animation function
async def cool_text_animation(hello="hello world"):
    az = [chr(x) for x in range(ord('a'), ord('z')+1)]
    lines = []

    for x in range(len(hello)):
        line = hello[:x]
        lines.append(hello + hello[x])
        if hello[x] == " ":
            continue
        for c in az:
            yield "".join([line+c+"\n" for _ in range(len(hello))])
            await asyncio.sleep(0.05)
            if c == hello[x]:
                break

    yield "\n".join(lines)

# Create Gradio interface
iface = gr.Interface(
    fn=cool_text_animation,
    inputs=gr.Textbox(label="Input Text", value="hello world"),
    outputs=gr.Textbox(label="Animated Output")
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# Add custom FastAPI route with animation
@app.get("/hello")
async def hello():
    return StreamingResponse(cool_text_animation(), media_type="text/plain")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)