import numpy as np


class NeuralNetwork(object):
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# Set number of nodes in input, hidden and output layers.
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

		# Initialize weights
		self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
									   (self.input_nodes, self.hidden_nodes))

		self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
									   (self.hidden_nodes, self.output_nodes))
		self.lr = learning_rate
		
		#### TODO: Set self.activation_function to your implemented sigmoid function ####
		#
		# Note: in Python, you can define a function with a lambda expression,
		# as shown below.
		# self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
		
		### If the lambda code above is not something you're familiar with,
		# You can uncomment out the following three lines and put your 
		# implementation there instead.
		#
		def sigmoid(x):
		   return 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here
		self.activation_function = sigmoid
					

	def train(self, features, targets):
		
		''' Train the network on batch of features and targets. 
		
			Arguments
			---------
			
			features: 2D array, each row is one data record, each column is a feature
			targets: 1D array of target values
		
		'''
		n_records = features.shape[0]
		delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
		delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
		for X, y in zip(features, targets):
			# print("X", X)
			final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
			# Implement the backproagation function below
			delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
																		delta_weights_i_h, delta_weights_h_o)
		self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


	def forward_pass_train(self, X):
		''' Implement forward pass here 
		 
			Arguments
			---------
			X: features batch
		
		'''
		#### Implement the forward pass here ####
		### Forward pass ###
		# TODO: Hidden layer - Replace these values with your calculations.
		# print("forward X.shape", X.shape)
		hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
		# print("hidden_inputs.shape", hidden_inputs.shape)
		hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
		# print("hidden_outputs.shape", hidden_outputs.shape)
		# TODO: Output layer - Replace these values with your calculations.
		final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
		final_outputs = final_inputs # signals from final output layer
		# print("final_inputs.shape", final_inputs.shape)
		# print("final_outputs.shape", final_outputs.shape)

		return final_outputs, hidden_outputs

	def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
		''' Implement backpropagation
		 
			Arguments
			---------
			final_outputs: output from forward pass
			y: target (i.e. label) batch
			delta_weights_i_h: change in weights from input to hidden layers
			delta_weights_h_o: change in weights from hidden to output layers

		'''
		#### Implement the backward pass here ####
		### Backward pass ###

		# TODO: Output error - Replace this value with your calculations.
		error = y - final_outputs # Output layer error is the difference between desired target and actual output.
		
		# TODO: Calculate the hidden layer's contribution to the error
		hidden_error = np.dot(error, self.weights_hidden_to_output.T)
		# print("hidden_error.shape", hidden_error.shape)
		# TODO: Backpropagated error terms - Replace these values with your calculations.
		output_error_term = error
		# print("output_error_term.shape", output_error_term.shape)
		hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
		# print("hidden_error_term.shape", hidden_error_term.shape)
		# print("backward X.shape", X.shape)
		# Weight step (input to hidden)
		delta_weights_i_h += hidden_error_term * X[:, None]
		# print("delta_weights_i_h.shape", delta_weights_i_h.shape)
		# Weight step (hidden to output)
		# print("delta_weights_h_o.shape", delta_weights_h_o.shape)
		# print("(output_error_term * hidden_outputs).shape", (output_error_term * hidden_outputs).shape)
		delta_weights_h_o += (output_error_term * hidden_outputs[:, None])
		
		return delta_weights_i_h, delta_weights_h_o

	def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
		''' Update weights on gradient descent step
		 
			Arguments
			---------
			delta_weights_i_h: change in weights from input to hidden layers
			delta_weights_h_o: change in weights from hidden to output layers
			n_records: number of records

		'''
		self.weights_hidden_to_output += ((self.lr / n_records) * delta_weights_h_o)  # update hidden-to-output weights with gradient descent step
		self.weights_input_to_hidden += ((self.lr / n_records) * delta_weights_i_h) # update input-to-hidden weights with gradient descent step

	def run(self, features):
		''' Run a forward pass through the network with input features 
		
			Arguments
			---------
			features: 1D array of feature values
		'''
		
		#### Implement the forward pass here ####
		# TODO: Hidden layer - replace these values with the appropriate calculations.
		hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hihiddendden layer
		hidden_outputs = self.activation_function(hidden_inputs) # signals from  layer
		
		# TODO: Output layer - Replace these values with the appropriate calculations.
		final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
		final_outputs = final_inputs # signals from final output layer 
		
		return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10500
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1
