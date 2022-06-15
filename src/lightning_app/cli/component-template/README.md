# placeholdername component

This ⚡ [Lightning component](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init component placeholdername
```

## To run placeholdername

First, install placeholdername (warning: this app has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/placeholdername
```

Once the app is installed, use it in an app:

```python
from placeholdername import TemplateComponent
import lightning_app as la


class LitApp(lapp.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.placeholdername = TemplateComponent()

    def run(self):
        print(
            "this is a simple Lightning app to verify your component is working as expected"
        )
        self.placeholdername.run()


app = lapp.LightningApp(LitApp())
```
