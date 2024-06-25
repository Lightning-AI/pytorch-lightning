import subprocess

with open("test_list_new2.txt", "r") as file:
    tests = [line.strip() for line in file.readlines()]

subprocess.run(["pytest", "-v"] + tests)
