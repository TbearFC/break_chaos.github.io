# 有关pytorch的小tips


## 简单设置training属性的技巧

对于batchnorm、dropout、instancenorm等在训练和测试阶段行为差距巨大的层，如果在测试时不将其training值设为True，则可能会有很大影响，这在实际使用中要千万注意。虽然可通过直接设置training属性，来将子module设为train和eval模式，但这种方式较为繁琐，因如果一个模型具有多个dropout层，就需要为每个dropout层指定training属性。更为推荐的做法是调用model.train()函数，它会将当前module及其子module中的所有training属性都设为True，相应的，model.eval()函数会把training属性都设为False。

## batch first

这个参数会确保输入输出，batch_size会在第一个维度上面