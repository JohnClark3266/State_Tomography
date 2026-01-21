import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights and biases
        # W1: (input_size x hidden_size)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # b1: (1 x hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        # W2: (hidden_size x output_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        # b2: (1 x output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        # Backward pass
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.hidden_error = self.output_delta.dot(self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W1 += X.T.dot(self.hidden_delta) * learning_rate
        self.b1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * learning_rate
        self.W2 += self.a1.T.dot(self.output_delta) * learning_rate
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

if __name__ == "__main__":
    # Example: XOR Problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    print("Training Neural Network on XOR problem...")
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    print("\nPredictions after training:")
    print(nn.forward(X))
