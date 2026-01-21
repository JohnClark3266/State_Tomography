import torch

# 设置随机种子以便复现
torch.manual_seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        使用 PyTorch 张量从头构建全连接神经网络
        不使用 nn.Module，手动管理权重矩阵
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置 (需要梯度)
        # W1: (input_size, hidden_size)
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        # b1: (1, hidden_size)
        self.b1 = torch.zeros(1, hidden_size, requires_grad=True)
        # W2: (hidden_size, output_size)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        # b2: (1, output_size)
        self.b2 = torch.zeros(1, output_size, requires_grad=True)
    
    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + torch.exp(-x))
    
    def forward(self, X):
        """前向传播"""
        # 隐藏层: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # 输出层: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def train(self, X, y, epochs=10000, learning_rate=0.5):
        """训练网络"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失 (均方误差)
            loss = torch.mean((y - output) ** 2)
            
            # 反向传播 (自动计算梯度)
            loss.backward()
            
            # 手动更新权重 (梯度下降)
            with torch.no_grad():
                self.W1 -= learning_rate * self.W1.grad
                self.b1 -= learning_rate * self.b1.grad
                self.W2 -= learning_rate * self.W2.grad
                self.b2 -= learning_rate * self.b2.grad
                
                # 清零梯度
                self.W1.grad.zero_()
                self.b1.grad.zero_()
                self.W2.grad.zero_()
                self.b2.grad.zero_()
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

if __name__ == "__main__":
    # 示例: XOR 问题
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]])
    
    y = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]])
    
    print("使用 PyTorch 张量训练神经网络 (XOR 问题)...")
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.5)
    
    print("\n训练后的预测结果:")
    with torch.no_grad():
        predictions = nn.forward(X)
        print(predictions)
