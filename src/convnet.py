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
    def __init__(
        self, net_config, lr=0.00001, loss_fn="quadratic", weight_init="uniform"
    ):
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

        # Here the bias is Tied Bias, But ignoring bias for now
        # self.biases = [
        #     weights_initialization("uniform")(shape=(1, self.out_channels[i]))
        #     for i in range(len(self.out_channels))
        # ]

        self.LR = lr

    # Convolution is usually implemented using unfold feature map -> dot product with kernel -> fold feature map. This allows for efficient use of memory.

    # feature_map and kernel channel dimension must be same. Format: CHW
    def full_convolution(self, feature_map, kernel, stride, pad):
        k_h = kernel.shape[1] - 1
        k_w = kernel.shape[2] - 1
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
        maskX = np.zeros(X.shape)

        for c in range(X.shape[0]):
            h_out = 0
            for h in range(0, X.shape[1], size):
                w_out = 0
                for w in range(0, X.shape[2], size):
                    input_patch = X[c, h : h + size, w : w + size]
                    out_X[c, h_out, w_out] = op(input_patch)
                    max2d_index = np.unravel_index(
                        input_patch.argmax(), input_patch.shape
                    )
                    maskX[c, h + max2d_index[0], w + max2d_index[1]] = 1
                    w_out += 1
                h_out += 1

        return out_X, maskX

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
                outX, maskX = self.pool(
                    X, self.pool_type[pool_count], self.pool_size[pool_count]
                )
                self.intermediate_results[-1] = maskX
                pool_count += 1
                X = outX

        loss = quadratic_loss(np.squeeze(X), y)
        #
        self.chain_grad = quadratic_loss_d(X, [[[y]]])

        return X, loss

    def backward(self):
        pool_index = -1
        act_index = -1
        conv_index = -1
        for index in range(len(self.layer_type) - 1, -1, -1):
            current_layer = self.layer_type[index]
            input_to_current_layer = self.intermediate_results[index]
            if current_layer == "act":
                # if self.activation_type[act_index] == "relu":
                self.chain_grad *= relu_prime(input_to_current_layer)
                act_index -= 1
            elif current_layer == "pool":
                # if self.pool_type[pool_index] == "max":
                maskX = input_to_current_layer
                pool_size = self.pool_size[pool_index]
                self.chain_grad = maskX * self.expand_grid(self.chain_grad, pool_size)
                pool_index -= 1
            elif current_layer == "conv":
                stride = self.stride_size[conv_index]
                pad = self.pad_size[conv_index]
                in_channels = self.in_channels[conv_index]
                out_channels = self.out_channels[conv_index]
                kernel_size = self.kernel_size[conv_index]
                # we know that: dL/dA * dA/dK = dL/dK = conv(X, dL/dA)
                # Need to review the args (stride and pad)
                K_derivative = np.zeros(
                    (out_channels, in_channels, kernel_size, kernel_size)
                )
                for idx, output_channel in enumerate(self.chain_grad):
                    for idy, input_channel in enumerate(input_to_current_layer):
                        out_ch = np.expand_dims(output_channel, axis=0)
                        in_ch = np.expand_dims(input_channel, axis=0)
                        K_derivative[idx, idy] = self.convolution(
                            in_ch, out_ch, stride, pad
                        )
                # (output_channels = number of kernels in current layer)
                previous_kernels = self.kernels[conv_index].copy()
                # find dL/dKs and apply it to Ks
                # all K_i have same derivative conv(X, dL/dA)
                self.kernels[conv_index] -= self.LR * K_derivative
                # find dL/dX and add it to chain_grad

                # flipping kernels for full convolution operation
                previous_kernels = np.flip(previous_kernels, -1)
                previous_kernels = np.flip(previous_kernels, -2)
                input_dims = input_to_current_layer.shape
                # (in_channels, in_H, in_W)
                X_derivative = np.zeros(input_dims, dtype="float")
                # (out_channel, in_channel, kH, kW)
                for idx, flipped_kernel in enumerate(previous_kernels):
                    # select output channel which is created from the current kernel
                    output_channel = np.expand_dims(self.chain_grad[idx], axis=0)
                    # and foreach kernel channel of that kernel, perform convolution op
                    for i in range(len(flipped_kernel)):
                        kernel_channel = np.expand_dims(flipped_kernel[i], axis=0)
                        # adding in X_derivative to accumulate derivatives from all differnet output channels + weights which created those channels
                        X_derivative[i] += self.full_convolution(
                            output_channel, kernel_channel, stride, pad
                        )
                self.chain_grad = X_derivative
                conv_index -= 1

    # Expands X by tiling each element of X size times vertically and horizontally
    def expand_grid(self, X, size):
        outputX = np.zeros((X.shape[0], X.shape[1] * size, X.shape[2] * size))
        for c in range(X.shape[0]):
            for h in range(X.shape[1]):
                for w in range(X.shape[2]):
                    h_o = h * size
                    w_o = w * size
                    outputX[c, h_o : h_o + size, w_o : w_o + size] = X[c, h, w]

        return outputX


# net_config=[conv, act, pool, conv, act, pool, ...]
### conv: in_channel, out_channel, kernel_size, stride, pad
conv1 = ConvLayer(in_channels=1, out_channels=8, kernel_size=5, stride=1, pad=0)
### act: type
act1 = ActLayer()
### pool: type, size
pool1 = PoolLayer(size=3)

conv2 = ConvLayer(in_channels=8, out_channels=1, kernel_size=3, stride=1, pad=0)
act2 = ActLayer()
pool2 = PoolLayer(size=6)


net = ConvNetwork(net_config=[conv1, act1, pool1, conv2, act2, pool2])

for i in range(500):
    input_img1 = np.ones((1, 28, 28), dtype="float")
    input_img2 = np.ones((1, 28, 28), dtype="float") * 2
    for id, img in enumerate([input_img1, input_img2]):
        pred, closs = net.forward(img, id)
        print(
            "pred: ",
            np.round(np.squeeze(pred), 2),
            "loss: ",
            np.round(closs, 2),
            "gt: ",
            id,
        )
        net.backward()
