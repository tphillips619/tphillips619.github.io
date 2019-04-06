---
layout: page 
title: About

---

# About Me

I'm a graduate student with an interest in machine learning. I enjoy sharing what I find useful or interesting and am often impressed in how a handful of open source libraries can be combined in interesting and powerful ways to make sense of prodigious amounts of data.

# Jekyll Template

Credit to [Pavel Makhov](http://pavelmakhov.com) and his theme [Jekyll Clean Dark](https://github.com/streetturtle/jekyll-clean-dark) 

There are a few collisions in style elements that prevent users from getting both the beauty of Jupyter Notebook's web based content and the ease of use/user extensibility of Jekyll static web page deployment using Github Pages.  The best approach I've found so far is to export Jupyter Notebooks without the ~10,000 lines of inline style code by using nbconvert and the 'basic' flag.
 
> jupyter nbconvert --to html --template basic 2018-01-01-my-notebook.ipynb

Then add style elements present in theme.scss and syntax.scss in the /assets/css folder.  Add the front matter into the resulting html file and make sure that any meaningful tag you use to categorize your post is also present in the /tags/ folder.
