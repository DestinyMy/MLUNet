from Search_space import *
from Search_space import Operations


class Node(nn.Module):
    def __init__(self, prev_channel, channel, op_id, stride):
        super(Node, self).__init__()
        self.op = Operations[op_id](prev_channel, channel, stride, None)

        self.out_channel = self.op.out_channel

    def forward(self, x):
        x = self.op(x)
        return x


class Stage(nn.Module):
    def __init__(self, op_list, prev_channel, channel, stride):
        super(Stage, self).__init__()
        self.num_node = len(op_list)
        self.ops = nn.ModuleList()

        for i in range(self.num_node):
            if i == 0 and stride != 1:
                node = Node(prev_channel, channel, op_list[i], stride)
            else:
                node = Node(prev_channel, channel, op_list[i], 1)
            self.ops.append(node)
            prev_channel = node.out_channel

        self.out_channel = prev_channel

    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, args, classes, init_channel, stages, pools, use_aux_head, keep_prob):
        super(NetworkCIFAR, self).__init__()
        self.args = args
        self.classes = classes
        self.init_channel = init_channel
        self.stages = stages
        self.pools = pools

        self.total_blocks = len(stages) + len(pools)  # 5
        self.pool_layer = [1, 3]

        self.use_aux_head = use_aux_head
        self.keep_prob = keep_prob

        if self.use_aux_head:
            self.aux_head_index = self.pool_layer[-1]

        self.stem1 = ConvBNReLU(3, 32, 3, 1, 1)

        prev_channel = 32
        channel = self.init_channel

        self.features = nn.ModuleList()
        for i in range(self.total_blocks):
            if i in self.pool_layer:
                channel = 2 * prev_channel
                cell = Node(prev_channel, channel, pools[(i - 1) // 2], 2)
            else:
                cell = Stage(self.stages[i // 2], prev_channel, channel, 1)
            self.features.append(cell)
            prev_channel = cell.out_channel

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(prev_channel, classes)

        self.stem2 = ConvBNReLU(prev_channel, self.args.search_last_channel, 3, 1, 1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(self.args.search_last_channel, classes)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        aux_logits = None
        out = self.stem1(x)
        for i, feature in enumerate(self.features):
            out = feature(out)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(out)
        out = self.stem2(out)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.use_aux_head:
            return logits, aux_logits
        else:
            return logits


class NetworkImageNet(nn.Module):
    def __init__(self, args, classes, init_channel, stages, pools, use_aux_head, keep_prob):
        super(NetworkImageNet, self).__init__()
        self.args = args
        self.classes = classes
        self.init_channel = init_channel
        self.stages = stages
        self.pools = pools

        self.total_blocks = len(stages) + len(pools)  # 5
        self.pool_layer = [1, 3]

        self.use_aux_head = use_aux_head
        self.keep_prob = keep_prob

        if self.use_aux_head:
            self.aux_head_index = self.pool_layer[-1]

        self.stem0 = nn.Sequential(
            ConvBNReLU(3, 16, 3, 2, 1),
            ConvBNReLU(16, 32, 3, 2, 1)
        )
        self.stem1 = ConvBNReLU(32, 32, 3, 2, 1)
        prev_channel = 32
        channel = self.init_channel

        self.features = nn.ModuleList()
        for i in range(self.total_blocks):
            if i in self.pool_layer:
                channel = 2 * prev_channel
                cell = Node(prev_channel, channel, pools[(i - 1) // 2], 2)
            else:
                cell = Stage(self.stages[i // 2], prev_channel, channel, 1)
            self.features.append(cell)
            prev_channel = cell.out_channel

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(prev_channel, classes)

        # search_last_channel = 1280
        self.stem2 = ConvBNReLU(prev_channel, self.args.search_last_channel, 3, 1, 1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(self.args.search_last_channel, classes)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        aux_logits = None
        out = self.stem0(x)
        out = self.stem1(out)
        for i, feature in enumerate(self.features):
            out = feature(out)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(out)
        out = self.stem2(out)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.use_aux_head:
            return logits, aux_logits
        else:
            return logits
