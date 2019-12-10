# 实例1:利用lstm预测时间序列

这里选择的seq_len，其实是选择通过读取几个数据去预测下一个。例如：序列1，2，3，4，5，如果设定步长为2，那么输入train_x=[[[1,2],[2,3],[3,4],[4,5]]],train_y=[[[2],[3],[4],[5]]].通过输入1，2来训练预测2，通过rnn的时间顺序来阅读输入序列1，2。

	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt

读入数据

	data_csv = pd.read_csv("./data/airline-passengers.csv",usecols=[1])
	plt.plot(data_csv)

数据预处理

	data_csv = data_csv.dropna()  # 滤除缺失数据
	dataset = data_csv.values   # 获得csv的值
	dataset = dataset.astype('float32')
	max_value = np.max(dataset)  # 获得最大值
	min_value = np.min(dataset)  # 获得最小值
	scalar = max_value - min_value  # 获得间隔数量
	dataset = list(map(lambda x: x / scalar, dataset)) # 归一化

创建数据集

	def create_dataset(dataset, look_back=5):
	    dataX, dataY = [], []
	    for i in range(len(dataset) - look_back):
	        a = dataset[i:(i + look_back)]
	        dataX.append(a)
	        dataY.append(dataset[i + look_back])
	    return np.array(dataX), np.array(dataY)

生成输入输出

	data_X, data_Y = create_dataset(dataset)


划分训练集和测试集，70% 作为训练集

	train_size = int(len(data_X) * 0.7)
	test_size = len(data_X) - train_size
	train_X = data_X[:train_size]
	train_Y = data_Y[:train_size]
	test_X = data_X[train_size:]
	test_Y = data_Y[train_size:]


将输入转化为tensor格式向量

	import torch
	from torch.autograd import Variable

	train_X = train_X.reshape(-1, 1, 5)
	train_Y = train_Y.reshape(-1, 1, 1)
	test_X = test_X.reshape(-1, 1, 5)

	train_x = torch.from_numpy(train_X)
	train_y = torch.from_numpy(train_Y)
	test_x = torch.from_numpy(test_X)

构建lstm模型

	from torch import nn
	from torch.autograd import Variable

batch_first=True 这个设置确保输入数据的格式为input的size为(batch,seq_length input_size),output的size为(batch_size, seq_length, hidden_size)

尤其需要注意，每一个lstm单元里面会含有多个隐藏层。所以对隐藏层的数量要做一个全连接层，对数量进行控制。

	class lstm(nn.Module):
	    def __init__(self,input_size=2,hidden_size=10,output_size=1,num_layer=2):
	        super(lstm,self).__init__()
	        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
	        self.layer2 = nn.Linear(hidden_size,output_size)
	    
	    def forward(self,x):
	        # out: tensor of shape (batch_size, seq_length, hidden_size)
	        x,_ = self.layer1(x)
	        #s,b,h = x.size()
	        #x = x.view(s*b,h)
	        x = self.layer2(x)
	        #x = x.view(s,b,-1)
	        return x

	model = lstm(5, 4,1,2)


创建损失函数和优化器


	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

开始训练

	for e in range(1000):
	    var_x = train_x.reshape(1, -1, 5)
	    var_y = train_y.reshape(1, -1, 1)
	    # 前向传播
	    out = model(var_x)
	    loss = criterion(out, var_y)
	    
	    # 反向传播
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()

对训练结果进行测试

	model = model.eval() # 转换成测试模式

	data_X = data_X.reshape(1, -1, 5)
	data_X = torch.from_numpy(data_X)
	var_data = Variable(data_X)
	pred_test = model(var_data) # 测试集的预测结果
	# 改变输出的格式
	pred_test = pred_test.view(-1).data.numpy()

显示测试结果

	# 画出实际结果和预测的结果
	plt.plot(pred_test, 'r', label='prediction')
	plt.plot(dataset, 'b', label='real')
	plt.legend(loc='best')# 画出实际结果和预测的结果
	plt.plot(pred_test, 'r', label='prediction')
	plt.plot(dataset, 'b', label='real')
	plt.legend(loc='best')