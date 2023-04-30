import random
from time import sleep
from rich.progress import Progress

# Parameters for the simulation
num_epochs = 5
num_batches = 100
batch_size = 32

with Progress() as progress:
    for epoch in range(num_epochs):
        # Add a task for the current epoch
        task_id = progress.add_task(f"[cyan]Epoch {epoch + 1}", total=num_batches)

        # Simulate the training process for each batch
        for _ in range(num_batches):
            sleep(0.02)  # Simulate time taken to process one batch
            progress.update(task_id, advance=1)

            # Randomly generate loss and accuracy for the current batch
            loss = random.uniform(0.1, 1.5)
            accuracy = random.uniform(0.5, 1.0)

            # Update the description with the current batch's loss and accuracy
            progress.update(
                task_id,
                description=f"[cyan]Epoch {epoch + 1} [bold red]Loss: {loss:.2f} [bold green]Accuracy: {accuracy:.2f}",
            )

        # Mark the epoch task as completed
        progress.stop()
        progress.remove_task(task_id)

    print("[bold green]Training complete.")
