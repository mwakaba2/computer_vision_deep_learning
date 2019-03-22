## Question 1: Is this Photo Blurry?
Write a function that calculates a bluriness score for each of the 3 photos and rank them in terms of the score. Confirm visually that your ranking of the photos make sense. For instance, a photo that is very sharp could have a score of 5, while a photo that is very blurry has a score of 1.

### Sample images
Sample images are located in the `../images/` folder

#### Stanford image examples
* Image path: `../images/stanford`

#### Photos with bokeh effect
I added some photos with bokeh effects in the test sample to ensure that the blurriness function doesn't penalize focused images with deliberate blurry backgrounds.
* `../images/cab_bokeh.jpg`
* `../images/rocks.jpg`
* `../images/turtle.jpeg`
* `../images/school_bus.jpeg`

### How to run the code.
```bash
python calculate_blurriness.py --img-path ../images/stanford/sharp.png

Outputs -->
Image: ../images/stanford/sharp.png
Blurriness Score: 5.000
```

```bash
python calculate_blurriness.py --img-path ../images/stanford/blurry.png

Outputs -->
Image: ../images/stanford/blurry.png
Blurriness Score: 2.016
```

```bash
python calculate_blurriness.py --img-path ../images/stanford/very_blurry.png

Outputs -->
Image: ../images/stanford/very_blurry.png
Blurriness Score: 1.619
```

```bash
python calculate_blurriness.py --img-path ../images/rocks.jpg

Outputs -->
Image: ../images/rocks.jpg
Blurriness Score: 4.576
```
