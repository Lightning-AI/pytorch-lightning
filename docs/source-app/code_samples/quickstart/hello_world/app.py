from lightning_app import LightningApp, LightningFlow


# Step 1: Subclass LightningFlow component to define the app flow.
class HelloWorld(LightningFlow):

    # Step 2: Add the app logic to the LightningFlow run method to
    # ``print("Hello World!")`.
    # The LightningApp executes the run method of the main LightningFlow
    # within an infinite loop.
    def run(self):
        print("Hello World!")


# Step 3: Initalize a LightningApp with the LightningFlow you defined (in step 1)
app = LightningApp(HelloWorld())
