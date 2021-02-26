# Using sphinx to make docs

Make changes in `.rst` files stored in `source`. 

```
pip install sphinx
pip install karma_sphinx_theme
python3 -m sphinx docs/source docs/build
```

To see changes locally in a live web browser,

```
pip install sphinx-autobuild
python3 -m sphinx_autobuild docs/source docs/build
```

To add a new file:

- Create a rst file in `/source`
- Add file name to toctree in `/source/index.rst`
