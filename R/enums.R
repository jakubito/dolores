#' Activation functions enum
#' 
#' Available activation functions:
#' * `Activation$IDENTITY`
#' * `Activation$SIGMOID`
#' * `Activation$TANH`
#' * `Activation$SOFTMAX`
#' * `Activation$RELU`
#' @export
Activation <- list(
  IDENTITY = c(
    function(input) {
      input
    },
    function(input, output) {
      1
    }
  ),

  SIGMOID = c(
    function(input) {
      1 / (1 + exp(-input))
    },
    function(input, output) {
      output * (1 - output)
    }
  ),

  TANH = c(
    function(input) {
      (exp(input) - exp(-input)) / (exp(input) + exp(-input))
    },
    function(input, output) {
      1 - output^2
    }
  ),

  SOFTMAX = c(
    function(input) {
      apply(input, 1, function(row) {
        exp(row - max(row)) %>% {. / sum(.)}
      }) %>% t
    },
    function(input, output) {
      apply(input, 1, function(row) {
        row * (1 - row)
      }) %>% t
    }
  ),

  RELU = c(
    function(input) {
      apply(input, 1:2, function(value) {
        max(0, value)
      })
    },
    function(input, output) {
      apply(input, 1:2, function(value) {
        as.numeric(value > 0)
      })
    }
  )
)

#' Cost functions enum
#' 
#' Available cost functions:
#' * `Cost$QUADRATIC`
#' * `Cost$BINARY_CROSS_ENTROPY`
#' * `Cost$CATEGORICAL_CROSS_ENTROPY`
#' @export
Cost <- list(
  QUADRATIC = c(
    function(output, target) {
      total_error <- sum(.5 * (output - target)^2)
      mean_error <- total_error / nrow(output)
      named_list(total_error, mean_error)
    },
    function(output, target) {
      output - target
    }
  ),

  BINARY_CROSS_ENTROPY = c(
    function(output, target) {
      rows <- nrow(output)
      total_error <- 0

      for (i in 1:rows) {
        output_row <- output[i,]
        target_row <- target[i,]
        error <- sum(target_row * log(output_row) + (1 - target_row) * log(1 - output_row))
        total_error <- total_error - error
      }

      mean_error <- total_error / rows
      accuracy <- sum(round(output) + target > 1) / rows
      named_list(total_error, mean_error, accuracy)
    },
    function(output, target) {
      output - target
    }
  ),

  CATEGORICAL_CROSS_ENTROPY = c(
    function(output, target) {
      rows <- nrow(output)
      total_error <- 0

      for (i in 1:rows) {
        error <- sum(target[i,] * log(output[i,]))
        total_error <- total_error - error
      }

      mean_error <- total_error / rows
      accuracy <- sum(round(output) + target > 1) / rows
      named_list(total_error, mean_error, accuracy)
    },
    function(output, target) {
      output - target
    }
  )
)
