#' @import purrr zeallot R6
#' @include enums.R helpers.R
NULL

#' Dolores R6 Class
#'
#' Dolores is a simple feedforward neural network implementation.
#' It uses backpropagation as learning mechanism.
#' @export
Dolores <- R6::R6Class('Dolores',
  public = list(
    #' @field config List of configuration values
    config = NULL,

    #' @field weights List of weights matrices
    weights = NULL,

    #' @description
    #' Create a new Dolores object.
    #' @param layers List of layers configurations created by [layer()]
    #' @param learning_rate Learning rate. Defaults to `.01`
    #' @param batch_size Batch size. Defaults to `1`
    #' @param epochs Number of epochs. Defaults to `10`
    #' @param cost Cost function vector defined as `[function, function_derivative]`
    #'
    #' Check out [`Cost`] enum for available functions. Defaults to `Cost$QUADRATIC`.
    #' @return A new `Dolores` object.
    initialize = function(
      layers,
      learning_rate = .01,
      batch_size = 1,
      epochs = 10,
      cost = Cost$QUADRATIC
    ) {
      if (length(layers) < 2) {
        stop('At least 2 layers are required')
      }

      self$config <- named_list(layers, learning_rate, batch_size, epochs, cost)
      self$weights <- weights(layers, runif) 
    },

    #' @description
    #' Train neural network.
    #' @param data Training data in matrix or data frame format.
    #' @return Current `Dolores` instance.
    train = function(data) {
      cat('Training in progress...\n')

      input_nodes <- first(self$config$layers)$nodes
      output_nodes <- last(self$config$layers)$nodes

      for (epoch in 1:self$config$epochs) {
        cat(sprintf('\rEpoch: %i / %i', epoch, self$config$epochs))

        data %>% shuffle_rows %>% chunks(self$config$batch_size) %>% walk(function(batch) {
          c(batch_values, batch_target) %<-% split_data(batch, input_nodes, output_nodes)
          private$feed_forward(batch_values)
          private$propagate_error(batch_target)
        })
      }

      cat('\n')
      invisible(self)
    },

    #' @description
    #' Validate neural network.
    #' @param data Test data in matrix or data frame format.
    #' @return Returns output of cost function. It's usually a list of total and mean errors.
    validate = function(data) {
      input_nodes <- first(self$config$layers)$nodes
      output_nodes <- last(self$config$layers)$nodes
      c(data_values, data_target) %<-% split_data(data %>% as.matrix, input_nodes, output_nodes)
      c(cost, cost_derivative) %<-% self$config$cost

      private$feed_forward(data_values)
      cost(last(private$outputs), data_target)
    },

    #' @description
    #' Feed data into network and calculate output.
    #' @param data Input data in matrix or data frame format.
    #' @return Calculated output values.
    calculate = function(data) {
      private$feed_forward(data)
      last(private$outputs)
    }
  ),

  private = list(
    # A list of matrices containing output (activated) values for each layer.
    outputs = list(),

    # A list of matrices containing gradient values for each layer. Used in back propagation.
    gradients = list(),

    # A list of matrices containing delta values for each layer. Used in back propagation.
    deltas = list(),

    # Internal function for forward pass.
    feed_forward = function(initial_values) {
      private$outputs[[1]] <- initial_values %>% as.matrix %>% cbind(., nrow(.) %>% rep(1, .))

      for (i in 1:length(self$weights)) {
        c(activation, activation_derivative) %<-% self$config$layers[[i + 1]]$activation
        input <- private$outputs[[i]] %*% self$weights[[i]]
        output <- activation(input)
        gradient <- activation_derivative(input, output)

        if (i < length(self$weights)) {
          output[, ncol(output)] <- output %>% nrow %>% rep(1, .)
        }

        private$outputs[[i + 1]] <- output
        private$gradients[[i + 1]] <- gradient
      }
    },

    # Internal function for backward pass.
    propagate_error = function(target) {
      c(cost, cost_derivative) %<-% self$config$cost
      error <- cost_derivative(last(private$outputs), target)
      weights_length <- length(self$weights)

      for (i in weights_length:1) {
        if (i < weights_length) {
          error <- (error * private$gradients[[i + 2]]) %*% t(self$weights[[i + 1]])
        }

        delta <- t(private$outputs[[i]]) %*% (error * private$gradients[[i + 1]])
        private$deltas[[i]] <- self$config$learning_rate * delta
      }

      for (i in 1:weights_length) {
        self$weights[[i]] <- self$weights[[i]] + private$deltas[[i]]
      }
    }
  )
)
