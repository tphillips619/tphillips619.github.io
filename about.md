---
layout: page 
title: About

---

# About

I wanted a space to post projects I've worked on. 


This page was built using the theme [Jekyll Clean Dark](https://github.com/streetturtle/jekyll-clean-dark) &copy; by [Pavel Makhov](http://pavelmakhov.com)

There are a few collisions in style elements that prevent users from getting both the beauty of Jupyter Notebook's web based content and the ease of use/user extensibility of Jekyll static web page deployment using Github Pages.  The best approach I've found so far is to export Jupyter Notebooks without the ~10,000 lines of inline style code by using nbconvert and the 'basic' flag.
 
> jupyter nbconvert --to html --template basic 2018-01-01-my-notebook.ipynb

And then adding style elements present in theme.scss and syntax.scss in the /assets/css folder.

The notebook source and data files are copied in their respective project repository on my [GitHub page](https://github.com/tphillips619/)
