import math
import random

def random_matrix(rows, cols, lower=-0.5, higher=0.5):
    # Randomly initialize a matrix
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(lower + (higher - lower) * random.random())
        matrix.append(row)
    return matrix

def init_vector(size, value=0.0):
    return [value for _ in range(size)]

def dot_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def matrix_multiply(matrix, vector):
    result = []
    for row in matrix:
        result.append(dot_product(row, vector))
    return result

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def xor_dataset():
    dataset = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    return dataset

def two_bit_adder_dataset():
    dataset = []

    for a0 in [0, 1]:
        for b0 in [0, 1]:
            for c0 in [0, 1]:
                for a1 in [0, 1]:
                    for b1 in [0, 1]:
                        s0 = (a0 + b0 + c0) % 2
                        carry1 = (a0 + b0 + c0) // 2
                        s1 = (a1 + b1 + carry1) % 2
                        c2 = (a1 + b1 + carry1) // 2

                        inputs = [a0, b0, c0, a1, b1]
                        outputs = [s0, s1, c2]
                        dataset.append((inputs, outputs))

    return dataset

def accuracy(model, X, Y):
    correct = 0
    for x, y in zip(X, Y):
        pred = model.predict(x)
        pred_binary = [1 if p >= 0.5 else 0 for p in pred] # Use 0.5 as threshold

        if pred_binary == y:
            correct += 1

    return correct / len(X) 

class MLP:
    def __init__(self, input_dim, output_dim, hidden_layers, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layers = [input_dim] + hidden_layers + [output_dim]
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(random_matrix(self.layers[i + 1], self.layers[i]))
            self.biases.append(init_vector(self.layers[i + 1], 0.0))

    def forward(self, x):
        activations = [x]
        weighted_sums = []

        # z = Wx + b
        for W, b in zip(self.weights, self.biases):
            z = []
            temp = matrix_multiply(W, activations[-1])
            for i in range(len(temp)):
                z.append(temp[i] + b[i])
            weighted_sums.append(z)

            # Sigmoid activation
            a = [sigmoid(val) for val in z]
            activations.append(a)

        return activations, weighted_sums
    
    def backward(self, activations, weighted_sums, y):
        deltas = [None] * len(self.weights)

        # Calculate output layer error
        output = activations[-1]
        error = [output[i] - y[i] for i in range(len(y))]
        deltas[-1] = [error[i] * sigmoid_derivative(output[i]) for i in range(len(output))]

        # Backpropagate
        for l in range(len(deltas) - 2, -1, -1):
            next_delta = deltas[l + 1]
            W_next = self.weights[l + 1]
            z = activations[l + 1]

            deltas[l] = []
            for i in range(len(z)):
                backprop_error = 0.0
                for j in range(len(next_delta)):
                    backprop_error += next_delta[j] * W_next[j][i]
                deltas[l].append(backprop_error * sigmoid_derivative(z[i]))
        
        # Update weights and biases
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])): # Neuron in next layer
                for j in range(len(self.weights[l][i])): # Neuron in current layer
                    self.weights[l][i][j] -= self.learning_rate * deltas[l][i] * activations[l][j]
                self.biases[l][i] -= self.learning_rate * deltas[l][i]

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X, Y):
                activations, weighted_sums = self.forward(x)
                output = activations[-1]

                # Loss (Mean Squared Error)
                loss = 0.0
                for i in range(len(y)):
                    loss += (output[i] - y[i]) ** 2
                total_loss += loss / len(y)

                self.backward(activations, weighted_sums, y)
            
            if epoch % 100 == 0:
                acc = accuracy(self, X, Y)
                print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}, Accuracy: {acc:.2f}")

    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]    
    
if __name__ == "__main__":
    dataset = xor_dataset()
    # dataset = two_bit_adder_dataset()

    X = [x for x, y in dataset]
    Y = [y for x, y in dataset]

    # Initialize MLP
    if dataset == xor_dataset():
        mlp = MLP(input_dim=2, output_dim=1, hidden_layers=[3], learning_rate=0.5)
    else:
        mlp = MLP(input_dim=5, output_dim=3, hidden_layers=[6, 4], learning_rate=0.3)

    # Train
    mlp.train(X, Y, epochs=10000)

    # Test
    print("\nTesting:")
    for x, y in dataset:
        pred = mlp.predict(x)
        print(f"Input: {x}, Predicted: {[round(p) for p in pred]}, Actual: {y}")
