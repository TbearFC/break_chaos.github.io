#torch重点相关概念
##autograd相关
###grad属性 
官方文档提到的grad相关
>This attribute is **None** by default and becomes a Tensor the first time a call to backward() computes gradients for self. The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it.

>By default, gradients are only retained for leaf variables. non-leaf variables’ gradients are not retained to be inspected later. **This was
done by design, to save memory.**

根据官方文档，torch会自动保留计算图叶子结点的grad，而计算图的非叶子则不会自动保留。但是，可以通过二种方法：

+ 调用retain_grad()函数来保存计算图中非叶子结点的grad值；

+ 设定hook来保存。

****




