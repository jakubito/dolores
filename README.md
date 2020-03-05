# Dolores

Dolores is a simple [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) implementation written in R. It uses backpropagation as learning mechanism.

## Installation

```R
devtools::install_github('jakubito/dolores')
```

## Basic example

```R
library(dolores)

# load data
data <- read.csv('iris.csv')
data <- data[sample(nrow(data)),] # shuffle rows
data_train <- data[1:100,]
data_test <- data[101:150,]

# define network layer structure
layers <- list(
  # input layer
  layer(
    nodes = 4
  ),
  # hidden layer
  layer(
    nodes = 6,
    activation = Activation$RELU
  ),
  # output layer
  layer(
    nodes = 3,
    activation = Activation$SOFTMAX
  )
)

# create new instance
dolores <- Dolores$new(
  layers,
  learning_rate = .00001,
  batch_size = 2,
  epochs = 100,
  cost = Cost$CATEGORICAL_CROSS_ENTROPY
)

# train using training data
dolores$train(data_train)

# validate using test data
dolores$validate(data_test)
```

### Output

```
Training in progress...
Epoch: 100 / 100
$total_error
[1] 3.195128

$mean_error
[1] 0.06390255

$accuracy
[1] 1
```

## How to build fresh docs

```R
devtools::document()
pkgdown::build_site()
```

## License

ISC
