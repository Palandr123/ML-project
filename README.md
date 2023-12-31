# ML project

## Table of contents
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Project objectives](#project-objectives)
- [Examples](#examples)
- [Discussion](#discussion)
- [Acknowledgments](#acknowledgments)

## Installation
1. Clone the repository
2. Install poetry using https://python-poetry.org/docs/
3. ```poetry install```

## Project objectives
* Develop a method for image editing with diffusion models using guidance based on cross-attention maps that is able to move and resize the objects
* Analyze the method with respect to the following properties:
    * Number of objects in the original image
    * Overlap between objects in the original image
    * The amount of clutter in the background


## Examples
"A polygonal illustration of a cat and a bunny" (move bunny to (0.9, 0.9))

<img src="imgs/bunny_orig.jpeg" width="300"/> <img src="imgs/bunny_move1.jpeg" width="300"/> 

"A photo of sharks in the ocean" (move sharks to (0.1, 0.1))

<img src="imgs/sharks_orig.jpeg" width="300"/> <img src="imgs/sharks_move2.jpeg" width="300"/> 

"A photo of a bronze horse in a museum" (size(bronze horse)*2.0)

<img src="imgs/horse_orig.jpeg" width="300"/> <img src="imgs/horse_size1.jpeg" width="300"/> 

"A photo of rubber ducks walking on street" (size(rubber ducks)*0.5)

<img src="imgs/ducks_orig.jpeg" width="300"/> <img src="imgs/ducks_size2.jpeg" width="300"/> 

## Discussion
* The method preserves image layout and object appearances
* The method does not preserve the image details including the clutter in the background
* It may create new objects instead of moving existing ones
* It may not preserve the number of objects when there are many of them
* Fails to move the objects that take a significant portion of the image
* May fail when the cross-attention maps are inaccurate
* Although preserves the poses and layout of overlapped objects, their details (and sometimes appearances) may change


## Acknowledgments
The Demo.ipynb is based on the following notebook: https://colab.research.google.com/drive/1SEM1R9mI9cF-aFpqg3NqHP8gN8irHuJi?usp=sharing

Some code will be rewritten and some methods will be added in the future.
