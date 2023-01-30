# placeholdername component

This ⚡ [Lightning component](https://lightning.ai/) ⚡ was generated automatically with:

```bash
lightning init component placeholdername
```

## To run placeholdername

First, install placeholdername (warning: this component has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/placeholdername
```

Once the app is installed, use it in an app:

```python
from placeholdername import TemplateComponent
import lightning as L


class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.placeholdername = TemplateComponent()

    def run(self):
        print("this is a simple Lightning app to verify your component is working as expected")
        self.placeholdername.run()


app = L.LightningApp(LitApp())
```
