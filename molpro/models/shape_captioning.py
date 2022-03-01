import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'sigmoid':
            layers += [nn.Sigmoid()]

        self.conv_block = nn.Sequential(* layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, o_p=1, norm=True, non_linear='relu'):
        super(DeconvBlock, self).__init__()
        layers = []

        # Transpose Convolution Layer
        layers += [nn.ConvTranspose3d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, output_padding=o_p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-Linearity Layer
        if non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'sigmoid':
            layers += [nn.Sigmoid()]
        elif non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.deconv_block = nn.Sequential(* layers)

    def forward(self, x):
        out = self.deconv_block(x)
        return out


class VAE(nn.Module):
    """
    Variational autoencoder for ligand shapes.
    This network is used only in training of the shape decoder.
    """
    def __init__(self, nc=34, ndf=128, latent_variable_size=512, device='cpu'):
        super(VAE, self).__init__()
        self.device = device
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = ConvBlock(self.nc, 32, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')

        self.e2 = ConvBlock(32, 32, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')

        self.e3 = ConvBlock(32, 64, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')

        self.e4 = ConvBlock(64, ndf*4, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')

        self.e5 = ConvBlock(ndf*4, ndf*4, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')

        self.fc1 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)
        self.fc2 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 512 * 3 * 3 * 3)

        # up5
        self.d2 = DeconvBlock(ndf * 4, ndf * 4, k=3, s=2, p=1, o_p=1, norm=True, non_linear='leaky_relu')

        # up 4
        self.d3 = DeconvBlock(ndf * 4, ndf * 2, k=3, s=2, p=1, o_p=1, norm=True, non_linear='leaky_relu')

        # up3 12 -> 12
        self.d4 = ConvBlock(ndf*2, ndf, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')

        # up2 12 -> 24
        self.d5 = DeconvBlock(ndf+32, 32, k=3, s=2, p=1, o_p=1, norm=True, non_linear='leaky_relu')

        # Output layer
        self.d6 = ConvBlock(64, nc, k=3, s=1, p=1, norm=False, non_linear='sigmoid')

        # Condtional encoding
        self.ce1 = ConvBlock(3, 32, 3, 1, 1)
        self.ce2 = ConvBlock(32, 32, 3, 2, 1)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        h5 = self.e5(h4)
        h5 = h5.view(-1, 512 * 3 * 3 * 3)
        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar,factor):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size(), dtype=torch.float32, device=self.device)
        return eps.mul(std*factor).add_(mu)

    def decode(self, z,cond_x):
        cc1 = self.relu(self.ce1(cond_x))
        cc2 = self.relu(self.ce2(cc1))

        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ndf * 4, 3, 3, 3)
        h2 = self.d2(h1)
        h3 = self.d3(h2)
        h4 = self.d4(h3)
        h4 = torch.cat([h4, cc2], dim=1)
        h5 = self.d5(h4)
        h5 = torch.cat([h5, cc1], dim=1)
        return self.d6(h5)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view())
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x,cond_x,factor=1.):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar,factor)
        res = self.decode(z,cond_x)
        return res, mu, logvar


class ShapeEncoder(nn.Module):
    """
    CNN network to encode ligand shape into a vectorized representation.
    """
    def __init__(self, in_layers=34):
        super(ShapeEncoder, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, kernel_size=3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x


class DecoderRNN(nn.Module):
    """
    RNN network to decode vectorized representation of a compound shape into SMILES string
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(1)
        self.device = device

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            if i == 0:
                predicted = outputs.max(1)[1]  # Do not start w/ different than start!
            else:
                probs = self.softmax(outputs)

                # Probabilistic sample tokens
                if self.device == 'cuda':
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for j in range(probs_np.shape[1]):
                    c_element = probs_np[:, j]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = j

                predicted = torch.tensor(tokens.astype(np.int), dtype=torch.int64, device=self.device)

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
