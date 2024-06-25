import subprocess

with open("test_list_new.txt", "r") as file:
    tests = file.readlines()


subprocess.run(["pytest", "-v"] + tests)
