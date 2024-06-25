import subprocess

with open("test_list_new.txt") as file:
    tests = [line.strip() for line in file.readlines()]

subprocess.run(["pytest", "-v"] + tests[-100:])
