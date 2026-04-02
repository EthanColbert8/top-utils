# top-utils
A mono-repo for some Python tools common to top quark physics workflows at Purdue.

### Notebooks for scratch work
It is highly discouraged, but not forbidden, to commit Jupyter notebooks to the git repository.
To use notebooks for scratch work, you should give the `.ipynb` files names beginning with `Scratch` or `scratch`, and they will automatically be ignored by git.

Other similar types of "notebook" files (Mathematica notebooks, etc.) are currently not ignored by git.
If you use one, you should name it similarly and add lines to the `.gitignore` to have git ignore them in the same way as Jupyter notebooks.
