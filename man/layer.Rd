% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helpers.R
\name{layer}
\alias{layer}
\title{Create layer configuration list}
\usage{
layer(nodes, activation = NULL)
}
\arguments{
\item{nodes}{Number of nodes}

\item{activation}{Optional activation function vector defined as \verb{[function, function_derivative]}

This parameter is required for all hidden and final layers.

Check out \code{\link{Activation}} enum for available functions.}
}
\value{
Layer configuration list.
}
\description{
Create layer configuration list
}
\examples{
input_layer <- layer(nodes = 4)

hidden_layer <- layer(
  nodes = 6,
  activation = Activation$SIGMOID
)

layer_with_custom_activation <- layer(
  nodes = 6,
  activation = c(
   function(input) {
     1 / (1 + exp(-input))
   },
   function(input, output) {
     output * (1 - output)
   }
  )
)
}
