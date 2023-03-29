# Customize Gradio components

If you want to customize the Gradio component by yourself, you can pass the `css` argument to `ServeGradio` like this:

```py
class AnimeGANv2UI(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    ...
    css="""
      .gradio-container button.primary, .gradio-container button.plain {
        background: red;
        color: white;
        font-size: 15;
        font-weight: 500;
    }
    """

    def __init__(self):
        super().__init__(css=self.css)
        self.ready = False
```

You can also pass a path to a css file like this:

```py
class AnimeGANv2UI(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    ...

    def __init__(self):
        super().__init__(css="path/to/styles.css")
        self.ready = False
```
