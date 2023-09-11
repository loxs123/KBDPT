import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        
        # 定义查询向量、键向量和值向量的全连接层
        self.query = nn.Linear(input_size, input_size, bias=False)
        self.key = nn.Linear(input_size, input_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)
        
        # 定义多头注意力的线性变换层
        self.linear = nn.Linear(input_size, input_size, bias=False)
        
    def forward(self, inputs):
        # inputs: [batch_size, sequence_length, input_size]
        batch_size, sequence_length, input_size = inputs.size()
        
        # 将输入张量变换为多头注意力的形状
        queries = self.query(inputs).view(batch_size, sequence_length, self.num_heads, input_size // self.num_heads)
        keys = self.key(inputs).view(batch_size, sequence_length, self.num_heads, input_size // self.num_heads)
        values = self.value(inputs).view(batch_size, sequence_length, self.num_heads, input_size // self.num_heads)
        
        # 计算注意力权重
        energy = torch.matmul(queries, keys.transpose(2, 3)) / (input_size // self.num_heads)**0.5
        attention_weights = torch.softmax(energy, dim=-1)
        
        # 计算加权的值向量
        context_vector = torch.matmul(attention_weights, values).view(batch_size, sequence_length, input_size)
        
        # 对加权的值向量进行线性变换
        output = self.linear(context_vector)
        
        return output


if __name__ == '__main__':
    attn = SelfAttention(100,1)
    feats = torch.zeros(1,34,100)
    out = attn(feats)
    print(out.size())
