## 数据的输入

### Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

### Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

### Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

### When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

### Mini-batch images and labels.
images, labels = data_iter.next()

### Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass


### 使用pandas作为数据源


出现RuntimeError: Expected object of scalar type Double but got scalar type Float for argument #2 'mat2'

但是修改了images=images.float()即可


## 模型的保存

	# Save and load the entire model.
	torch.save(resnet, 'model.ckpt')
	model = torch.load('model.ckpt')

	# Save and load only the model parameters (recommended).
	torch.save(resnet.state_dict(), 'params.ckpt')
	resnet.load_state_dict(torch.load('params.ckpt'))