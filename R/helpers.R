#' Create layer configuration list
#' @param nodes Number of nodes
#' @param activation Optional activation function vector defined as `[function, function_derivative]`
#'
#' This parameter is required for all hidden and final layers.
#' 
#' Check out [`Activation`] enum for available functions.
#' @return Layer configuration list.
#' @examples
#' input_layer <- layer(nodes = 4)
#' 
#' hidden_layer <- layer(
#'   nodes = 6,
#'   activation = Activation$SIGMOID
#' )
#' 
#' layer_with_custom_activation <- layer(
#'   nodes = 6,
#'   activation = c(
#'    function(input) {
#'      1 / (1 + exp(-input))
#'    },
#'    function(input, output) {
#'      output * (1 - output)
#'    }
#'   )
#' )
#' @export
layer <- function(nodes, activation = NULL) {
  named_list(nodes, activation)
}

weights <- function(layers, data_type = numeric) {
  last_layer <- length(layers) - 1
  last_layer %>% seq_len %>% map(function(i) {
    rows <- layers[[i]]$nodes + 1
    cols <- layers[[i + 1]]$nodes %>% ifelse(i < last_layer, . + 1, .)
    (rows * cols) %>% data_type %>% matrix(nrow = rows, ncol = cols)
  })
}

chunks <- function(data, size) {
  data %>% as.data.frame %>% split((nrow(.) %>% seq - 1) %/% size) %>% map(as.matrix)
}

first <- function(value) {
  value[[1]]
}

last <- function(value) {
  value[[length(value)]]
}

contains <- function(list, key) {
  !is.null(list[[key]])
}

named_list <- function(...) {
  result <- list(...)
  names(result) <- substitute(list(...)) %>% as.list %>% .[-1]
  result
}

split_data <- function(data, input_nodes, output_nodes) {
  values <- data[, 1:input_nodes] %>%
    matrix(nrow = nrow(data), ncol = input_nodes)
  target <- data[, (input_nodes + 1):(input_nodes + output_nodes)] %>%
    matrix(nrow = nrow(data), ncol = output_nodes)
  list(values, target)
}

shuffle_rows <- function(data) {
  data[sample(nrow(data)),]
}
