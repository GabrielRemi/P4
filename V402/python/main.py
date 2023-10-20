""" Hier wird der code von beiden Aufgaben ausgeführt, 
Code für den photoeffekt ist in photozelle.py
code für die Balmer Serie ist in balmer.py"""

import os, sys
import subprocess

os.chdir(os.path.dirname(__file__))
if len(sys.argv) == 1:
    print("--------------PHOTOZELLE-------------------------")
    subprocess.run(["python3", "photozelle.py"])
    print("Fertig")

    print("--------------BALMER-SERIE-----------------------")
    subprocess.run(["python3", "balmer.py"])
    print("Fertig")
    exit(0)

if "photozelle" in sys.argv or "photo" in sys.argv: 
    subprocess.run(["python3", "photozelle.py"])
elif "balmer" in sys.argv: 
    subprocess.run(["python3", "balmer.py"])