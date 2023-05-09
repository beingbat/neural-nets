from common import *


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        assert stride > 0, "Stride must be greater than 0"
        assert pad >= 0, "Pad must be zero or positive"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad


class ActLayer:
    def __init__(self, type="relu"):
        self.type = type


# default stride for pool is kernel size
class PoolLayer:
    def __init__(self, size, type="max"):
        assert size > 0, "Size must be greater than zero"
        self.size = size
        self.type = type


class ConvNetwork:
    def __init__(self, net_config, lr=0.01, loss_fn="quadratic", weight_init="uniform"):
        # net_config: its a multiple of 3 (args for conv layer, args for preceding activation, args for preceding pooling layer)
        # conv args: out channels [all in channels become input of each out channel], kernel size, stride, pad
        # activation args: type
        # pooling args: type, size
        self.layer_type = []
        self.in_channels = []
        self.out_channels = []
        self.kernel_size = []
        self.stride_size = []
        self.pad_size = []
        self.activation_type = []
        self.pool_type = []
        self.pool_size = []

        self.intermediate_results = []
        self.intermediate_gradients = []

        self.chain_grad = []

        for module_id in range(len(net_config) // 3):
            # Convolution
            conv_layer = net_config[module_id * 3]
            activation_layer = net_config[module_id * 3 + 1]
            pooling_layer = net_config[module_id * 3 + 2]

            if conv_layer != None:
                self.in_channels.append(conv_layer.in_channels)
                self.out_channels.append(conv_layer.out_channels)
                self.kernel_size.append(conv_layer.kernel_size)
                self.stride_size.append(conv_layer.stride)
                self.pad_size.append(conv_layer.pad)
                self.layer_type.append("conv")

            if activation_layer != None:
                self.activation_type.append(activation_layer.type)
                self.layer_type.append("act")

            if pooling_layer != None:
                self.pool_type.append(pooling_layer.type)
                self.pool_size.append(pooling_layer.size)
                self.layer_type.append("pool")

        self.kernels = [
            weights_initialization(weight_init)(
                shape=(
                    self.out_channels[i],
                    self.in_channels[i],
                    self.kernel_size[i],
                    self.kernel_size[i],
                )
            )
            for i in range(len(self.out_channels))
        ]

        # Here the bias is Tied Bias
        self.biases = [
            weights_initialization("uniform")(shape=(1, self.out_channels[i]))
            for i in range(len(self.out_channels))
        ]

        self.LR = lr

    # Convolution is usually implemented using unfold feature map -> dot product with kernel -> fold feature map. This allows for efficient use of memory.

    # feature_map and kernel channel dimension must be same. Format: CHW
    def full_convolution(self, feature_map, kernel, stride, pad):
        k_h = kernel.shape[1]
        k_w = kernel.shape[2]
        # Pad required
        if pad:
            feature_map = np.pad(feature_map, ((0, 0), (pad, pad), (pad, pad)))
        # Pad for making convolution 'full'
        padded_feature_map = np.pad(feature_map, ((0, 0), (k_h, k_h), (k_w, k_w)))
        return self.convolution(padded_feature_map, kernel, stride, 0)

    def convolution(self, feature_map, kernel, stride, pad):
        assert (
            feature_map.shape[0] == kernel.shape[0]
        ), "Feature map and kernel channel dimensions don't match"
        assert stride > 0, "stride must be greater than zero"

        if pad:
            feature_map = np.pad(feature_map, ((0, 0), (pad, pad), (pad, pad)))

        k_h = kernel.shape[1]
        k_w = kernel.shape[2]
        f_h = feature_map.shape[1]
        f_w = feature_map.shape[2]

        # out_size = floor(((in_size + 2 * pad + kernel_size - 1) - 1 / stride) + 1)
        out_h = int(np.floor(((f_h + 2 * pad - (k_h - 1) - 1) / stride) + 1))
        out_w = int(np.floor(((f_w + 2 * pad - (k_w - 1) - 1) / stride) + 1))
        output_feature_map = np.zeros((out_h, out_w), dtype="float")

        for o_h, h in enumerate(range(0, f_h - k_h + 1, stride)):
            for o_w, w in enumerate(range(0, f_w - k_w + 1, stride)):
                output_feature_map[o_h, o_w] = np.sum(
                    feature_map[
                        :,
                        h : h + k_h,
                        w : w + k_w,
                    ]
                    * kernel
                )
        return output_feature_map

    def pool(self, X, type, size):
        op = np.max

        assert (
            X.shape[1] % size == 0
        ), "size must be divideable by the feature map height"
        assert (
            X.shape[2] % size == 0
        ), "size must be divideable by the feature map width"

        out_X = np.zeros((len(X), X.shape[1] // size, X.shape[2] // size))

        for c in range(X.shape[0]):
            h_out = 0
            for h in range(0, X.shape[1], size):
                w_out = 0
                for w in range(0, X.shape[2], size):
                    out_X[c, h_out, w_out] = op(X[c, h : h + size, w : w + size])
                    w_out += 1
                h_out += 1

        return out_X

    def activation(self, X, type):
        if type == "relu":
            X = relu(X)
        return X

    # Return pred, loss
    def forward(self, X, y):
        conv_counts = 0
        pool_count = 0
        act_count = 0

        for l in self.layer_type:
            self.intermediate_results.append(X)
            if l == "conv":
                output = []
                kernel = self.kernels[conv_counts]
                stride = self.stride_size[conv_counts]
                pad = self.pad_size[conv_counts]

                for k in kernel:
                    output.append(self.convolution(X, k, stride, pad))
                X = np.array(output)
                conv_counts += 1
            elif l == "act":
                X = self.activation(X, self.activation_type[act_count])
                act_count += 1

            elif l == "pool":
                X = self.pool(X, self.pool_type[pool_count], self.pool_size[pool_count])
                pool_count += 1

            loss = quadratic_loss(np.squeeze(X), y)
            #
            self.chain_grad = quadratic_loss_d(np.squeeze(X), y)

        return X, loss

    def backward(self):
        return None


# net_config=[conv, act, pool, conv, act, pool, ...]
### conv: in_channel, out_channel, kernel_size, stride, pad
conv1 = ConvLayer(in_channels=1, out_channels=8, kernel_size=7, stride=1, pad=0)
### act: type
act1 = ActLayer()
### pool: type, size
pool1 = PoolLayer(size=2)

conv2 = ConvLayer(in_channels=8, out_channels=1, kernel_size=5, stride=1, pad=0)
act2 = ActLayer()
pool2 = PoolLayer(size=7)

net = ConvNetwork(net_config=[conv1, act1, pool1, conv2, act2, pool2])
input_img = np.ones((1, 28, 28), dtype="float")
output = net.forward(input_img, None)
print(output.shape)
print(np.squeeze(output))
