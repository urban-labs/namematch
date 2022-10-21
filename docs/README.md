# Using sphinx to make docs

Make changes in `.rst` files stored in `source`. 

```
pip install sphinx
pip install sphinx_rtd_theme
python3 -m sphinx docs/source docs/build
```

To see changes locally in a live web browser,

```
pip install sphinx-autobuild
python3 -m sphinx_autobuild docs/source docs/build
```

To serve the doc site in `/build` folder without re-rendering the `/build` folder
```
cd docs/build
python3 -m http.server PORT
```
then you can see the static site on your browser (http://0.0.0.o:PORT/)


To add a new file:

- Create a rst file in `/source`
- Add file name to toctree in `/source/index.rst`
