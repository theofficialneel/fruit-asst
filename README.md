# Fruit Assistant
A simple image processing assistant to detect the fruit shown and maintaining a calorie / nutrient chart

## Training Set : (4800, 2000)
- Contains X, test and dev
- Data present as .npy bitmaps
- Bitmaps created using gen_data.py
View the image back using the X_view script
`````
	python3 X_view.py <number>
`````
- where number lies from 0 to 1599

## Requirements :
- Python 3
- Scipy
- Numpy
- Pillow

## Setup :
Set a virtual env and install requirements
````
	python3 -m venv venv
	. venv/bin/activate
	pip3 install -r requirements.txt
````
Initiate parameters of the model
````
	python3 init_params.py
````
Train the model to gains params
````
	python3 fttrain_nn.py <number>
````
- where number represents iter count
- A script is present to create 3000 iterations
````
	python3 train_script.py
````
Predict an image
````
	python3 predict.py
	Enter the fruit-image file location: data/orange.jpg
	You have consumed 1 Orange
	Added new fruit to chart on 2020-04-16
	{
		"calories (kJ)": 354,
		"carbs (g)": 21,
		"fat (g)": 0.2,
		"mineral (mg)": {
			"calcium": 72,
			"iron": 0.2,
			"magnesium": 18
		},
		"name": "Orange",
		"protein (g)": 1.7,
		"vitamin": {
			"A (IU)": 405,
			"C (mg)": 95.8,
			"E (mg)": 0.3
		}
	}
````

## Accuracy Results :
Run
````
	python3 model_result
````
After 3000 iterations
````
	Accuracy with training set, 85.640625%
	Accuracy with dev set, 84.25%
	Accuracy in recognising mango: 79.3333333333
	Accuracy in recognising apple: 90.5555555556
	Accuracy in recognising orange: 94.3333333333
	Accuracy in recognising pear: 68.4444444444
````
