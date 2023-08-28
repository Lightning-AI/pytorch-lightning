from lightning.app import LightningWork


class ExampleWork(LightningWork):
    def __init__(self):
        super().__init__(cache_calls=False)

    def run(self, *args, **kwargs):
        print(f"I received the following props: args: {args} kwargs: {kwargs}")


work = ExampleWork()
work.run(value=1)

# Providing the same value. This won't run as already cached.
work.run(value=1)
work.run(value=1)
work.run(value=1)
work.run(value=1)

# Changing the provided value. This isn't cached and will run again.
work.run(value=10)
