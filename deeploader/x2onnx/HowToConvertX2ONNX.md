#  X2ONNX

# ONNX的前世今生

深度学习的框架五花八门，为AI研究人员提供了丰富的选择，但是到了部署的时候，花样繁多的框架就成了麻烦。我们需要在docker环境中把各种框架都打包进去，体积会变得很大， 而且框架都是为训练设计的， 部署的效率不一定高， 这就促成了江湖中一门新的武术的诞生——模型转换之神功。

早期一些个人开发者和少数框架，提供点对点的模型转换技法，用起来很多坑，十分不便，一时间江湖上对此怨声载道， 但框架都系大佬们创立，信众们无能为力。

## 统一文件格式

江湖老牌大佬微软老大， 觉得这样也蛮生长也不是办法（关键自家的框架CNTK太不受武林人士待见，所以要想一统江湖收复失地，就得想点法子），于是就联合几个小弟推出了一个**专门用来描述深度学习模型的文件格式**，叫做**[ONNX]( https://github.com/onnx/onnx )**

## 模型转换大法

仅仅搞画个大饼，立个flag，是远远不够的，谷哥有tensorflow，脸谱哥有pytorch，还有亚马逊背书的mxnet，个个都身手不凡，后面跟着一堆信众。

深度学习发展多年， 各种武功秘籍层出不穷， 家家都有自己的藏书阁，取其名曰**ModelZoo**

于是大佬就想， 要想一统武林， 必须要能翻译各家经典， 这样不同教派的秘籍就能知识共享。随后大佬找了几个马仔， 搞出一个模型转换大法，叫做 [MMdnn]( https://github.com/microsoft/MMdnn )。这个功法有点厉害，有了它，普通马仔敲敲键盘就能把武功秘籍进行翻译。



## 统一秘籍的修炼大法

好了， 到这里大佬已经规定了统一的武功秘籍写法， 又帮着翻译了各家的典藏秘籍， 是不是就大功告成了。

非也非也， 还有很重要的一部， 你拿到统一的武功秘籍， 还不知道该怎么修炼，一部能读懂却不能修炼的秘籍， 我要它何用， 一帮观望的看客正准备散去。

欲练神功还需独门心法。

少侠留步，少侠留步， 大佬发出了诚挚的挽留， 下面我们开放统一秘籍的独门心法， 人人皆可修炼， 取其名曰[onnxruntime ]( https://github.com/microsoft/onnxruntime )

有了这个心法， 各个教派的秘籍， 只要能翻译成统一格式， 就能被修炼， 发挥实效， 实在是妙哉妙哉。

大批信众开始投入怀抱， 大佬的教派规模越来越大， 影响力与日俱增， 到后来其它教派的大佬也不能忽视， 纷纷表示支持，并提供自己秘籍到统一秘籍的翻译官。



# ONNX 格式

格式的详细描述在这里[Open Neural Network Exchange - ONNX]( https://github.com/onnx/onnx/blob/master/docs/IR.md )，写得很好，就是有点长。我们长话短说，ONNX使用来自谷哥的[protobuf]( https://github.com/protocolbuffers/protobuf )来描述和存储， 转换后的模型实际上是一个二进制的 protobuf 文件，但是后缀改成“.onnx”。

## Model

来看一下这个文件的大致格式：

| Name             | Type               | Description                                               |
| ---------------- | ------------------ | --------------------------------------------------------- |
| ir_version       | int64              | 版本号                                                    |
| opset_import     | OperatorSetId      | 当前模型用到的操作符的所有版本， 不同版本支持的运算符不同 |
| producer_name    | string             | 是哪个工具转的                                            |
| producer_version | string             | 上面那个工具的版本                                        |
| domain           | string             | 给模型取个包名,比如 'org.onnx'                            |
| model_version    | int64              | 模型版本                                                  |
| doc_string       | string             | 模型的描述文档， 说是支持Markdown                         |
| graph            | Graph              | **唯一重要的东东——计算图**                                |
| metadata_props   | map<string,string> | 一些用kv对存储的元数据                                    |



## Graph

好的， 一个模型里面最重要的东西就是计算图`Graph`， 计算图定义了模型怎么样运算得到想要的结果，接下来看看它的定义：

| Name        | Type        | Description                                                  |
| ----------- | ----------- | ------------------------------------------------------------ |
| name        | string      | 随便取个名字                                                 |
| node        | Node[]      | **图上所有节点的列表， 注意是部分有序的**                    |
| initializer | Tensor[]    | 是一个列表， 每个元素有名字， 要么是模型的参数， 要么是输入数据的默认值 |
| doc_string  | string      | 计算图的描述文档， 说是支持Markdown                          |
| input       | ValueInfo[] | 输入列表， 参数的名字， 和维度信息                           |
| output      | ValueInfo[] | 输出列表                                                     |
| value_info  | ValueInfo[] | 参数的信息                                                   |

可以看出， 图实际上是通过定义一堆节点`Node`来表达的， 节点直接通过输入输出关系， 连接成一个有向无环图。

在ONNX里面， 所有的数据都是有名字的， 每个节点的输入数据、参数、输出都有名字（因此模型转换器转换的时候要给输出命名，**尤其是有些节点可能有多个输出**， 比如`Slice`）。因此， 不存在不能用名字找到的数据， 整个图的建立没有任何障碍。

总结一个， 一个计算图包含了如下信息：

- 输入信息列表
- 用节点列表表示的图结构
- 模型的参数数据
- 输出信息列表



## Node

图里面最要的当然就是节点列表， 下面来看一下节点长啥样

| Name       | Type        | Description                  |
| ---------- | ----------- | ---------------------------- |
| name       | string      | 可选的，取个名字，只用来调试 |
| input      | string[]    | 节点的输入名字列表           |
| output     | string[]    | 输出名字列表                 |
| op_type    | string      | **运算符的类型**             |
| domain     | string      | 可选的，域名                 |
| attribute  | Attribute[] | 基本用不到                   |
| doc_string | string      | 文字描述                     |

这个结构很熟悉有没有， 打开一个caffe的 prototxt看看， 是不是很layer的结构很像， 或者打开一个mxnet的 json文件， 也差不多。

一个节点就是要进行某种操作， 当然要有输入（包括输入数据和模型的参数），进行指定的运算，产生期望的输出。



## Operators

接下来，看看ONNX里面都规定了哪些可用的运算符，当前支持的列表在这里[Operator Schemas]( https://github.com/onnx/onnx/blob/master/docs/Operators.md )

基本上常见的运算都有了， 如果没有的话， 呵呵，你有的忙了。



# 模型转换

我没有用过[MMdnn]( https://github.com/microsoft/MMdnn )， 手里的模型只有mxnet和pytorch的需要转， 这两个框架都提供了内置转换的支持， 还是相信官方的比野生的好。

但是，模型转换定律

> "只要进行模型转换， 必然会有很多坑。"  —— 模型转换定律

从未曾改变。

## MxNet

用这个框架，是因为要做点人脸识别相关的工作，有个很好的开源代码[insightface]( https://github.com/deepinsight/insightface )是用MxNet做的，那就拿来参考参考。

高版本的MxNet已经自带了支持， 我用的V1.4.1，但是还是遇到了不少问题，分享一下填坑心得。

### PRelu

我们晓得，这个PRelu其实就是带参数的Relu，在MxNet里面，都叫做泄露的ReLu（“LeakyReLU”是也），多了一个参数而已，看上去一切都OK。

但是， 当前版本onnx规定了参数广播到输入张量的方式里面， 对于PRelu有特殊照顾，请看文档[Broadcasting in ONNX](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md)，太长了，我们抄一段重要的

> ## Unidirectional Broadcasting
>
> In ONNX, tensor B is unidirectional broadcastable to tensor A
> if one of the following is true:
>
> - Tensor A and B both have exactly the same shape.
> - Tensor A and B all have the same number of dimensions and the length of
>   each dimensions is either a common length or B's length is 1.
> - Tensor B has too few dimensions, and B can have its shapes prepended
>   with a dimension of length 1 to satisfy property 2.
>
> When unidirectional broadcasting happens, the output's shape is the same as 
> the shape of A (i.e., the larger shape of two input tensors).
>
> In the following examples, tensor B is unidirectional broadcastable to tensor A:
>
> - shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
> - shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
> - shape(A) = (2, 3, 4, 5), shape(B) = (2, 1, 1, 5), ==> shape(result) = (2, 3, 4, 5)
> - shape(A) = (2, 3, 4, 5), shape(B) = (1, 3, 1, 5), ==> shape(result) = (2, 3, 4, 5)
>
> Unidirectional broadcasting is supported by the following operators in ONNX:
>
> - [Gemm](Operators.md#Gemm)
> - [PRelu]( Operators.md#PRelu)

就是说，广播可以，维度上要保证一致，我们看一下代码，在`mx2onn/_op_translation.py`里面，找到`LeakyReLU`的实现：

这里把参数也就是input_nodes[1]， 进行reshape操作生成一个新的Tensor，然后传递给ONNX的PRelu op。注意这里的reshape操作， 直接把维度变成了四维（NCHW)，把参数都放到通道C上面，其它维度指定为1去进行对应的广播操作。

如果输入是四维向量， 当然没问题， 但是如果输入是其它维度，比如经过了全连接， 根据广播机制， 会把输入数据进行维度扩展， 变成四维， GAME OVER  !

```
# _op_translation.py
...
@mx_op.register("LeakyReLU")
def convert_leakyrelu(node, **kwargs):
    if act_type == "prelu" or act_type == "selu":
        reshape_value = np.array([1, -1, 1, 1], dtype='int64')
        dims = np.shape(reshape_value)

        shape_node = onnx.helper.make_tensor_value_info(reshape_val_name, input_type, dims)
        initializer.append(
            onnx.helper.make_tensor(
                name=reshape_val_name,
                data_type=input_type,
                dims=dims,
                vals=reshape_value,
                raw=False,
            )
        )

        slope_op_name = 'slope' + str(kwargs["idx"])

        reshape_slope_node = onnx.helper.make_node(
            'Reshape',
            inputs=[input_nodes[1], reshape_val_name],
            outputs=[slope_op_name],
            name=slope_op_name
        )

        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=[input_nodes[0], slope_op_name],
            outputs=[name],
            name=name)

        lr_node.append(shape_node)
        lr_node.append(reshape_slope_node)
        lr_node.append(node)
```

要解决这个问题， 我们需要知道输入的维度，然而不幸的是，当前版本官方code里面，参数的信息可以获取，从initializer里面取到，但是没有传入input_nodes的shape信息，so，GAME OVER !
来看一下转换的入口方法位于`mx2onnx/export_onnx.py`

```
# MXNetGraph.create_onnx_graph_proto
converted = MXNetGraph.convert_layer(
                    node,
                    is_input=False,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape,
                    in_type=in_type,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup,
                    idx=idx
            )
```

那我们只能往外层找，想办法。想一下， 现在onnx的模型没有生成， 所以没有办法用ONNX提供的`infer_shape`方法，那就只能寄希望于MxNet了。确实有这个方法， 好的， 来做个接口。

事实上， 找到了一个现成的获取一个symbol输出shape的接口，那就好办了，记得symbol有个`get_internals`的方法能获取所有依赖节点的， 遍历一下收集起来就搞定。

```
# MXNetGraph.get_outputs
# def get_outputs(sym, params, in_shape, in_label)

@staticmethod
def get_all_outputs(sym, params, in_shape, in_label):
    nodes = sym.get_internals()
    all_outputs = {}
    for node in nodes:
        if node.name in sym.list_inputs():
            continue
        graph_outputs = MXNetGraph.get_outputs(node, params, in_shape, in_label)
        all_outputs.update(graph_outputs)
    return all_outputs
```

剩下的就是把结果传进去了， easy

```
# MXNetGraph.create_onnx_graph_proto
all_outputs = MXNetGraph.get_all_outputs(sym, params, in_shape, output_label)  # <- 看这里

converted = MXNetGraph.convert_layer(
                    node,
                    is_input=False,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape,
                    in_type=in_type,
                    out_shape=all_outputs,   # <- 看这里
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup,
                    idx=idx
            )
```

### SliceChannel

现在这个层已经被弃用了， 但是老一点模型还有这个操作， 没办法， 谁让我倒霉碰上了。

在MxNet里面，一个op产生多个输出， 后面的节点可以用下标来引用输出结果， 来举个栗子：

```
{
  "op": "Convolution", 
  "param": {
    "cudnn_off": "False", 
    "cudnn_tune": "off", 
    "dilate": "(1,1)", 
    "kernel": "(3,3)", 
    "no_bias": "False", 
    "num_filter": "28", 
    "num_group": "1", 
    "pad": "(0,0)", 
    "stride": "(1,1)", 
    "workspace": "1024"
  }, 
  "name": "conv1_2", 
  "inputs": [[1, 1], [19, 0], [20, 0]],  # <- 看这里 [1, 1]
  "backward_source_id": -1
}, 
```

翻译成ONNX的卷积， 看一下代码：

```

@mx_op.register("Convolution")
def convert_convolution(node, **kwargs):
    """Map MXNet's convolution operator attributes to onnx's Conv operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)  # <- 看这里
	...
    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=input_nodes,  # <- 看这里
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims,
        dilations=dilations,
        pads=pad_dims,
        group=num_group,
        name=name
    )

    return [conv_node]
```

注意到这里获取的输入input_nodes，实际上都是节点（或者tensor）的名字，特别的，如果输入是SliceChannel这样的多输出层， 这里的输入名字都会是**节点的名字**，而**不是输出的tensor的名字**。 这样输入维度就对不上，GAME OVER !

要解决这个问题， 我们需要修改`get_inputs`这个函数，获取当前节点正确的输入tensor名字

```
input_nodes = []
for ip in inputs:
    input_node_id = index_lookup[ip[0]]
    input_nodes.append(proc_nodes[input_node_id].name)
```

修改后

```
input_nodes = []
for ip in inputs:
    input_node_id = index_lookup[ip[0]]
    input_node = proc_nodes[input_node_id]
    input_outputs = []
    if hasattr(input_node, 'output'):
        input_outputs = input_node.output
    if len(input_outputs) <= 1:
        input_nodes.append(proc_nodes[input_node_id].name)
    else:
        input_nodes.append(input_outputs[ip[1]])
```

### MaxPool

增加对`pooling_convention`转换的支持，更新onnx库的版本，增加 `ceil_mode`参数。

```
@mx_op.register("Pooling")
def convert_pooling(node, **kwargs):
    ...
    ceil_mode = False
    pooling_convention = attrs.get('pooling_convention', 'valid')
    if pooling_convention == 'full':
        pooling_warning = "Pooling: ONNX currently doesn't support pooling_convention. " \
                          "This might lead to shape or accuracy issues. " \
                          "https://github.com/onnx/onnx/issues/549"

        logging.warning(pooling_warning)
        ceil_mode = True
    ...         
    node = onnx.helper.make_node(
        pool_types[pool_type],
        input_nodes,  # input
        [name],
        kernel_shape=kernel,
        pads=pad_dims,
        strides=stride,
        name=name,
        ceil_mode=ceil_mode
    )
```






# 参考文献

- [ONNX]( https://github.com/onnx/onnx )
-  [MMdnn]( https://github.com/microsoft/MMdnn )
- [onnxruntime ]( https://github.com/microsoft/onnxruntime )
- [protobuf]( https://github.com/protocolbuffers/protobuf )
- [onnx runtime API Summary]( https://microsoft.github.io/onnxruntime/python/api_summary.html )