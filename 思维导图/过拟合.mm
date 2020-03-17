
<map version="0.9.0">
    <node CREATED="0" ID="80526354dcb24e06834fbdc323edcd4e" MODIFIED="0" TEXT="过拟合">
        <node CREATED="0" ID="8fd4c247ce134bc9b581354cff5b24e0" MODIFIED="0" POSITION="right" TEXT="​模型容量">
            <node CREATED="0" ID="5eae31cdd5e140c08438da7a548cf6f5" MODIFIED="0" POSITION="right" TEXT="​模型拟合复杂函数的能力"/>
            <node CREATED="0" ID="562b75d85b85422abf5f692637c9d579" MODIFIED="0" POSITION="right" TEXT="​指标: 模型的假设空间(Hypothesis Space)大小，即模型可以表示的函数集的大小"/>
        </node>
        <node CREATED="0" ID="0526b2e7df2040b49fe152268e3e8b93" MODIFIED="0" POSITION="left" TEXT="​抑制过拟合的方法">
            <node CREATED="0" ID="9d85cef00ae8413780beeed4eceb35e6" MODIFIED="0" POSITION="left" TEXT="​1. 数据集划分: 训练集、验证集、测试集"/>
            <node CREATED="0" ID="862b5320b4454d0b987ea5b531668b3f" MODIFIED="0" POSITION="left" TEXT="​2. 提前停止early stopping：验证准确率连续𝑛个Epoch 没有下降时， 停止"/>
            <node CREATED="0" ID="18376be00afe40d2a5aaa4842d04cddc" MODIFIED="0" POSITION="left" TEXT="​3. 修改模型参数： 减少网络层数，减少每层中网络的参数规模"/>
            <node CREATED="0" ID="ba3dcdc3627e47e6ae48c6f38d03f976" MODIFIED="0" POSITION="left" TEXT="​4. 添加正则项： L1或L2"/>
            <node CREATED="0" ID="7d30314365454d6aa678458cdc5b150c" MODIFIED="0" POSITION="left" TEXT="​5. 集成学习： Bagging、模型平均"/>
            <node CREATED="0" ID="835e5ae3d31244868e08759b496b5a63" MODIFIED="0" POSITION="left" TEXT="​6. Dropout：随机断开神经网络的连接"/>
            <node CREATED="0" ID="7af67fc2d3a649f0957b33f3c5a841c0" MODIFIED="0" POSITION="left" TEXT="​8. 数据增强：维持样本标签不变的条件下，根据先验知识改变样本的特征，使得新产生的样本也符合或者近似符合数据的真实分布">
                <node CREATED="0" ID="bfcfe64a6c5e4d59bd6175671ef4e39c" MODIFIED="0" POSITION="left" TEXT="​对于图片数据：添加噪声、旋转、缩放、平移、裁剪、改变视角、遮挡某局部"/>
            </node>
        </node>
        <node CREATED="0" ID="9bb1e6757cb4412b9e9a0a9d1b9b871e" MODIFIED="0" POSITION="right" TEXT="​欠拟合与过拟合">
            <node CREATED="0" ID="e511b54b2f3340b3b27ee974f2781d9e" MODIFIED="0" POSITION="right" TEXT="​欠拟合: 训练集高误差, 测试集表现不佳"/>
            <node CREATED="0" ID="bbead0a6b5a445ff8f011b5018cee160" MODIFIED="0" POSITION="right" TEXT="​过拟合: 训练集低误差, 测试集高误差"/>
        </node>
    </node>
</map>
