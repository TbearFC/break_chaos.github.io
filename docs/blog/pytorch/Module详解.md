## nn.Module详解

### 继承nn.Module会发生什么

一个Net，也就是继承自nn.Module的类，当实例化后，本质上就是维护了以下8个字典(OrderedDict)：

	_parameters
	_buffers
	_backward_hooks
	_forward_hooks
	_forward_pre_hooks
	_state_dict_hooks
	_load_state_dict_pre_hooks
	_modules
	
源码中初始化生成8个字典

	class Module(object):
	
	def __init__(self):
	        self._construct()
	        # initialize self.training separately from the rest of the internal
	        # state, as it is managed differently by nn.Module and ScriptModule
	        self.training = True

    def _construct(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        

_parameters：字典，保存用户直接设置的parameter，self.param1 = nn.Parameter(t.randn(3, 3))会被检测到，在字典中加入一个key为'param'，value为对应parameter的item。而self.submodule = nn.Linear(3, 4)中的parameter则不会存于此。
_modules：子module，通过self.submodel = nn.Linear(3, 4)指定的子module会保存于此。
_buffers：缓存。如batchnorm使用momentum机制，每次前向传播需用到上一次前向传播的结果。
_backward_hooks与_forward_hooks：钩子技术，用来提取中间变量，类似variable的hook。
training：BatchNorm与Dropout层在训练阶段和测试阶段中采取的策略不同，通过判断training值来决定前向传播策略。
上述几个属性中，_parameters、_modules和_buffers这三个字典中的键值，都可以通过self.key方式获得，效果等价于self._parameters['key'].

这8个字典用于网络的前向、反向、序列化、反序列化中。

因此，当实例化你定义的Net(nn.Module的子类)时，要确保父类的构造函数首先被调用，这样才能确保上述8个OrderedDict被create出来，否则，后续任何的初始化操作将抛出类似这样的异常：cannot assign module before Module.__init__() call。

对于前述的CivilNet而言，当CivilNet被实例化后，CivilNet本身维护了这8个OrderedDict，更重要的是，CivilNet中的conv1和conv2(类型为nn.modules.conv.Conv2d）、pool（类型为nn.modules.pooling.MaxPool2d）、fc1、fc2、fc3（类型为torch.nn.modules.linear.Linear）均维护了8个OrderedDict，因为它们的父类都是nn.Module，而gemfield（类型为str）、syszux（类型为torch.Tensor)则没有这8个OrderedDict。

也因此，在你定义的网络投入运行前，必然要确保和上面一样——构造出那8个OrderedDict，这个构造，就在nn.Module的构造函数中。如此以来，你定义的Net就必须继承自nn.Module；如果你的Net定义了__init__()方法，则必须在你的__init__方法中调用nn.Module的构造函数，比如super(your_class).__init__() ，注意，如果你的子类没有定义__init__()方法，则在实例化的时候会默认用nn.Module的，这种情况也对。

## 数据是如何被划到这8个字典的


nn.Module通过使用__setattr__机制，使得定义在类中（不一定要定义在构造函数里）的成员（比如各种layer），被有序归属到_parameters、_modules、_buffers或者普通的attribute里；那具体怎么归属呢？很简单，当类成员的type 派生于Parameter类时（比如conv的weight，在CivilNet类中，就是self.conv1中的weight属性），该属性就会被划归为_parameters；当类成员的type派生于Module时（比如CivilNet中的self.conv1，其实除了gemfield和syszux外都是），该成员就会划归为_modules。

##  model(x)和model.forward(x)区别

网络的前向需要通过诸如CivilNet(input)这样的形式来调用，而非CivilNet.forward(input)，是因为前者实现了额外的功能：

1，先执行完所有的_forward_pre_hooks里的hooks
2, 再调用CivilNet的forward函数
3, 再执行完所有的_forward_hooks中的hooks
4, 再执行完所有的_backward_hooks中的hooks
可以看到:

1，_forward_pre_hooks是在网络的forward之前执行的。这些hooks通过网络的register_forward_pre_hook() API来完成注册，通常只有一些Norm操作会定义_forward_pre_hooks。这种hook不能改变input的内容。

2，_forward_hooks是通过register_forward_hook来完成注册的。这些hooks是在forward完之后被调用的，并且不应该改变input和output。目前就是方便自己测试的时候可以用下。

3，_backward_hooks和_forward_hooks类似。

所以总结起来就是，如果你的网络中没有Norm操作，那么使用CivilNet(input)和CivilNet.forward(input)是等价的。

另外，你必须使用CivilNet.eval()操作来将dropout和BN这些op设置为eval模式，否则你将得到不一致的前向返回值。eval()调用会将Net的实例中的training成员设置为False。