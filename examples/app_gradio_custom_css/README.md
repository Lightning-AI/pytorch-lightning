# Customize Gradio components

If you want to customize gradio component by yourself, you need to pass `css` argument to ServeGradio like this:

```py
class AnimeGANv2UI(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    ...
    css='''
      .gradio-container button.primary, .gradio-container button.plain {
        background: red;
        color: white;
        font-size: 15;
        font-weight: 500;
    }
    '''

    def __init__(self):
        super().__init__(css=self.css)
        self.ready = False
```

You can also pass path to css file like this:

```py
class AnimeGANv2UI(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    ...
    css="styles.css"

    def __init__(self):
        super().__init__(css=self.css)
        self.ready = False
```
