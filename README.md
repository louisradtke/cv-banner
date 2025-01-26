# cv-banner

Create a nice banner for my CV.

This repo contains a Python script (main.py) that generates randomized points and connects them using delaunay
triangulation to form nice simplices.

## Setup:

```bash
# optional: create venv
python3 -m venv .venv
. .venv/bin/activate

pyhton3 -m pip install -r requirements.txt
```

**The script also uses the `incscape` command, to create a PDF. You may install it as well.**

## Execution:

Just execute the `main.py`. Inside the `main` func, you will find variables for adding polygons and circles.

The color map can be edited in the static files in the top of the `main.py`. The script will also generate a square
with the mapping of the colors. Have a look at `output/`

```bash
python3 main.py
```
