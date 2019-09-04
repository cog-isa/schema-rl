import os

top = os.getcwd()
print(top)

exclude = set(['.git', '.idea', 'venv', '__pycache__'])

for root, dirs, files in os.walk(top, topdown=True):
    dirs[:] = [d for d in dirs if d not in exclude]
    #print(root)
    for name in files:
        print(os.path.join(root, name))

#with open("homework.py", 'r') as infile:
#    hw = infile.readlines().replace(chr(0), '')