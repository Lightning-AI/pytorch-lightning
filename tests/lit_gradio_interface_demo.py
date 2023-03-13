from lightning.app.components.serve import ServeGradio
import gradio as gr
import lightning as L

class LitGradio(ServeGradio):
    inputs = [
      gr.components.Textbox(label="print your prompt here", elem_id="label"),
      gr.components.Dropdown(label="choose you model",
        choices=[
          "PulpSciFiDiffusion",
          "pixel_art_diffusion_hard_256",
          "pixel_art_diffusion_soft_256",
        ]),
      gr.components.Number(value=250, label="number of steps"),
      gr.components.Audio(label="Audio"),
      gr.components.Image(label="Image"),
      gr.components.Video(label="Video"),
      gr.components.HTML(label="HTML"),
      gr.components.BarPlot(label="Bar Plot"),
      gr.components.Button(value="Hi"),
      gr.components.Chatbot(label="Bot"),
      gr.components.Checkbox(label="Checkbox"),
      gr.components.ColorPicker(label="Color Picker"),
      gr.components.File(label="File"),
      gr.components.Dataframe(label="Data Frame"),
      gr.components.Gallery(label="Gallery"),
      gr.components.HighlightedText(label="Highlighted Text", ),
      gr.components.Image(label="Image"),
      gr.components.JSON(label="JSON"),
      gr.components.Label(label="Label"),
      gr.components.LinePlot(label="Line Plot"),
      gr.components.Markdown(label="Markdown"),
      gr.components.Model3D(label="Model 3D"),
      gr.components.Plot(label="Plot"),
      gr.components.Radio(label="Radio"),
      gr.components.ScatterPlot(label="Scatter Plot"),
      gr.components.Slider(label="Slider"),
      gr.components.Timeseries(label="Time Series"),
      gr.components.UploadButton(label="Upload Button"),
      gr.components.Video(label="Video"),
    ]
    outputs = gr.outputs.Image(type="pil", label="Output Image")

    def predict(self, label, model, steps):
      return 0

    
    def build_model(self):
      return None


app = L.LightningApp(LitGradio())
