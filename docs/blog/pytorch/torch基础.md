# torch重点相关概念

## 构建数据集
Dataset是一个抽象类, 为了能够方便的读取，需要将要使用的数据包装为Dataset类。 自定义的Dataset需要继承它并且实现两个成员方法：

__getitem__() 该方法定义用索引(0 到 len(self))获取一条数据或一个样本
__len__() 该方法返回数据集的总长度

	#引用
	from torch.utils.data import Dataset
	import pandas as pd
	#定义一个数据集
	class BulldozerDataset(Dataset):
	    """ 数据集演示 """
	    def __init__(self, csv_file):
	        """实现初始化方法，在初始化的时候将数据读载入"""
	        self.df=pd.read_csv(csv_file)
	    def __len__(self):
	        '''
	        返回df的长度
	        '''
	        return len(self.df)
	    def __getitem__(self, idx):
	        '''
	        根据 idx 返回一行数据
	        '''
	        return self.df.iloc[idx].SalePrice



	ds_demo= BulldozerDataset('median_benchmark.csv')

## 读取数据集

DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)，下面做一个简单的操作


	dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)
DataLoader返回的是一个可迭代对象，我们可以使用迭代器分次获取数据


	idata=iter(dl)
	print(next(idata))
	tensor([24000., 24000., 24000., 24000., 24000., 24000., 24000., 24000., 24000.,
	        24000.], dtype=torch.float64)
常见的用法是使用for循环对其进行遍历


	for i, data in enumerate(dl):
	    print(i,data)
	    # 为了节约空间, 这里只循环一遍
	    break
	0 tensor([24000., 24000., 24000., 24000., 24000., 24000., 24000., 24000., 24000.,
	        24000.], dtype=torch.float64)
我们已经可以通过dataset定义数据集，并使用Datalorder载入和遍历数据集。




## autograd相关
### grad属性 
官方文档提到的grad相关
> This attribute is **None** by default and becomes a Tensor the first time a call to backward() computes gradients for self. The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.

> By default, gradients are only retained for leaf variables. non-leaf variables’ gradients are not retained to be inspected later. **This was
done by design, to save memory.**

根据官方文档，torch会自动保留计算图叶子结点的grad，而计算图的非叶子则不会自动保留。但是，可以通过二种方法：

+ 调用retain_grad()函数来保存计算图中非叶子结点的grad值；

+ 设定hook来保存。

****

## 继承构造模型

逻辑回归模型

	import torch
	import torch.nn as nn
	import torch.nn.functional as func

	class LogisticModel(nn.Module):
	    def __init__(self, in_dim, out_dim):
	        super(LogisticModel, self).__init__()
	        self.linear = nn.Linear(in_dim, out_dim)
	        
	    def forward(self,x):
	        out = func.sigmoid(self.linear(x))
	        return out
构造模型

	model = LogisticModel(1, 1)
	
设定epoch的总数

	epochs = 1000 

每一个epoch
	
	for epoch in range(epochs):
	    inputs = x_train
	    labels = y_train
	    #注意对比model的输出和model的forward方法的输出
	    out = model(inputs)    
	    #梯度清零
	    optimiser.zero_grad()    
	    loss = criterion(out, labels)   
	    #误差逆向传导 
	    loss.backward()    
	    optimiser.step()
	    predicted = model.forward(x_train)    
	    print('epoch{}, loss {}'.format(epoch, loss.item()))




