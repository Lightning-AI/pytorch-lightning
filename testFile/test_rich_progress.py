import random
from time import sleep
from rich.progress import Progress

# Parameters for the simulation
num_epochs = 1
num_batches = 860
batch_size = 32

with Progress() as progress:
    for epoch in range(num_epochs):
        # Add a task for the current epoch
        task_id = progress.add_task(f"Epoch {epoch + 1}/{num_epochs}", total=num_batches)

        # Simulate the training process for each batch
        for batch in range(num_batches):
            sleep(0.01)  # Simulate time taken to process one batch
            progress.update(task_id, advance=1)

            # Calculate elapsed time and remaining time (for illustration purposes)
            elapsed_time = progress.tasks[task_id].elapsed_time
            remaining_time = (num_batches - batch - 1) * elapsed_time / (batch + 1)

            # Randomly generate processing speed (items per second)
            speed = random.uniform(75, 85)

            # Update the description with the current batch's information
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
