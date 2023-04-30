import random
import time
from time import sleep
from rich.progress import Progress
from lightning.pytorch.callbacks.progress import RichProgressBarTheme

custom_theme = RichProgressBarTheme(
    description="yellow",
    progress_bar="green",
    batch_progress="red",
    time="blue",
    processing_speed="magenta",
    metrics="cyan"
)

# Parameters for the simulation
num_epochs = 1
num_batches = 860
batch_size = 32

with Progress(theme=custom_theme) as progress:
    for epoch in range(num_epochs):
        task_id = progress.add_task(f"Epoch {epoch + 1}/{num_epochs}", total=num_batches)

        for batch in range(num_batches):
            sleep(0.01)
            progress.update(task_id, advance=1)

            elapsed_time = time.time() - progress.tasks[task_id].start_time
            remaining_time = (num_batches - batch - 1) * elapsed_time / (batch + 1)
            speed = random.uniform(75, 85)

            progress.update(
                task_id,
                description=(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"{batch + 1}/{num_batches} "
                    f"{elapsed_time:.0f}s . {remaining_time:.0f}s "
                    f"{speed:.2f}it/s"
                ),
            )

    print("[bold green]Training complete.")
