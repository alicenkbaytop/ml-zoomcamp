# Session 1
## MLZ - 1.1 - Introduction to Machine Learning
?
## MLZ - 1.2: Machine Learning vs. Rule-Based Systems
Example: Mail Spam Detection

In traditional rule-based systems, we create explicit rules to classify emails as spam or not. For instance:

- If the sender is promotions@com, classify as "spam."
- If the title contains "tax review," classify as "spam."
While this approach can work for obvious cases, it becomes complex and difficult to maintain for more nuanced situations. Not every email with the word "promotion" is spam.

### Rule-Based Approach:
- **Data + Code** → Software → Outcome
### Machine Learning Approach:
1. **Gather Data**: Collect a dataset labeled as spam or not spam.
2. **Feature Engineering**: Define and calculate features, such as the length of the title, domain, and description.
3. **Model Training**: Fit the model to the data.
4. **Prediction**: Use the model to predict outcomes. For example, if the model outputs 0.8, and we set a threshold of 0.5, classify as "spam."

#### Outcome:
- **Data** + **Outcome** → ML Model → Predictions


## MLZ - 1.3: Supervised Machine Learning
Supervised learning involves using labeled data to train models. Here's a breakdown:

- **Data**: Feature matrix 𝑋 (2D array) where each column represents a feature.
- **Target**: Desired output 𝑦 (1D array) corresponding to each row.

### Model Representation:
𝑔(𝑋)≈𝑦

- **Regression Problems**: e.g., predicting car prices.
- **Classification Problems**: e.g., distinguishing between cats and dogs.
- **Ranking**: e.g., recommender systems like Google search or e-commerce platforms.

## MLZ - 1.4: CRISP-DM (Cross-Industry Standard Process for Data Mining)

Developed by IBM, CRISP-DM is a widely used methodology for organizing ML projects. It consists of six key phases:

1. **Business Understanding**: Determine if machine learning is needed for the project.
2. **Data Understanding**: Identify data sources and gather initial data.
3. **Data Preparation**: Clean the data and build pipelines, converting it into a usable format.
4. **Modeling**: Train models on the prepared data.
5. **Evaluation**: Assess the model's performance.
6. **Deployment**: Implement the model into production.

Other methodologies include SEMMA, KDD, and Agile.

## MLZ - 1.5: Model Selection Process
In the model selection process, we typically split the data into training and testing sets. Here’s a common approach:

- **Data Split**: Train (60%) + Validation (20%) + Test (20%)
- **Model Evaluation**: 
    - (Logistic Regression) = 66%
    - (Decision Tree) = 60%
    - (Neural Network) = 80%

Choose the model with the highest accuracy for predictions.

## MLZ - 1.6: GitHub Codespaces 2024 Edition
?

## MLZ - 1.7: Introduction to NumPy
NumPy is a fundamental library for numerical computing in Python. Here’s how to get started:

```
import numpy as np
```

### Basic Operations:
```
# Creating arrays
np.zeros(5)  # Output: [0. 0. 0. 0. 0.]
np.ones(10)   # Output: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
np.full(10, 2.5)  # Output: [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
arr = np.array([1, 2, 3, 4, 5])  # Output: [1, 2, 3, 4, 5]
arr[2] = 9  # Output: [1, 2, 9, 4, 5]
```

### Multi-Dimensional Arrays:
```
# Creating a 2D array
arr_2d = np.zeros((5, 2))  # Output: [[0. 0.], [0. 0.], [0. 0.], [0. 0.], [0. 0.]]
````

### Randomly Generated Arrays:
```
np.random.seed(2)
np.random.rand(5, 2)  # Random array
np.random.randint(low=0, high=100, size=(5, 2))  # Random integers
```

### Element-Wise Operations:
```
arr_a = np.arange(5)
arr_b = arr_a + 1
```

### Summary Operations:
```
arr.max()   # Maximum value
arr.sum()   # Sum of elements
arr.mean()  # Mean value
```

## MLZ - 1.8: Linear Algebra Refresher
?

## MLZ - 1.9: Introduction to Pandas
?

## MLZ - 1.10: Summary of Session 1
?