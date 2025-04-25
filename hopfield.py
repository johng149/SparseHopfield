import torch
import einops
from torch.nn import Module, Parameter, ParameterList
from torch import Tensor
from collections import namedtuple
from typing import List
import matplotlib.pyplot as plt

LayerInfo = namedtuple("LayerInfo", ["nodes", "memories"]) # num nodes, num mems per node

class SparseHopfield(Module):
  def __init__(self, receptive_fields, field_dim, layers: List[LayerInfo], t: int = 0, alpha: float=1.0, rho=1e-8):
    super().__init__()
    self.fields = receptive_fields
    self.field_dim = field_dim
    self.layers_info = layers
    self.t = t
    self.a = alpha
    self.rho = rho # used to prevent division by zero
    self.kwargs = {
        "receptive_fields": self.fields,
        "field_dim": self.field_dim,
        "layers": self.layers_info,
        "t": self.t,
        "alpha": self.a,
        "rho": self.rho
    }

    self.growth_threshold = self.a / (self.t + self.a)
    self.layers = []
    self.layer_counts = []

    self.create_layers()

  def reset_iteration(self, t: int = 0):
    self.t = t
    self.growth_threshold = self.a / (self.t + self.a)

  def update_iteration(self):
    self.t += 1
    self.growth_threshold = self.a / (self.t + self.a)

  def create_layers(self):
    for i in range(len(self.layers_info)):
      if i == 0: # assume we are in outer layer
        nodes, mems = self.layers_info[i]
        nodes = self.fields # we ignore what layers info says, we need to match the fields here
        self.layers_info[i] = LayerInfo(nodes=nodes, memories=mems)
        dim = self.field_dim
        counts = torch.ones(nodes, mems)
        mem_matrix = Parameter(torch.zeros((nodes, mems, dim)) + 0.5, requires_grad=False)
        self.layers.append(mem_matrix)
        self.layer_counts.append(counts)
      else:
        nodes, mems = self.layers_info[i]
        prev_nodes, prev_mems = self.layers_info[i - 1]
        assert prev_nodes % nodes == 0, f"At layer {i}, prev is {prev_nodes} but nodes is {nodes}. Cannot divide evenly"
        children_per_node = prev_nodes // nodes
        counts = torch.ones(nodes, mems)
        mem_matrix = Parameter(torch.zeros((nodes, children_per_node, mems, prev_mems)), requires_grad=False)
        self.layers.append(mem_matrix)
        self.layer_counts.append(counts)
      self.layers = ParameterList(self.layers)

  @staticmethod
  def maxi(x):
    max_indices = torch.argmax(x, dim=-1, keepdim=True)
    blank = torch.zeros_like(x)
    return torch.scatter(blank, -1, max_indices, torch.gather(x, -1, max_indices))

  @staticmethod
  def argmaxi(x, eps=1e-8):
    maxied = torch.abs(SparseHopfield.maxi(x))
    factors, indices = torch.max(maxied, dim=-1)
    factors = factors - eps
    return maxied / factors.unsqueeze(-1)

  @staticmethod
  def mark_reserved_indices(acts, usages, trigger_growth, mark=-2):
    used_values, used_indices = torch.sort(acts, dim=-1)
    reservations = used_indices[:,:,-1:]
    reservations[trigger_growth] = -1
    reservations = einops.rearrange(reservations, 'batch mems flag -> mems (batch flag)')
    sorts, sort_indices = torch.sort(usages, dim=-1)
    sorts, sort_indices, reservations
    reservation_mask = reservations != -1
    expanded_usages = sort_indices.unsqueeze(1).expand(-1, reservations.size(1), -1)
    expanded_res = reservations.unsqueeze(-1).expand(-1, -1, usages.size(1))
    matches = (expanded_res == expanded_usages) & reservation_mask.unsqueeze(-1)
    matches = matches.any(dim=1)
    return torch.where(matches, mark, sort_indices), sort_indices

  @staticmethod
  def move_value_to_back(x, value=-2):
    # moves all instances of `value` in `x` to the right most along the last dim
    # for example, if x is
    # [ [-2, 1, 2, 2, 3], [0, -2, 1, -2, 3]]
    # and value is -2, the result is
    # [ [1, 2, 2, 3, -2], [0, 1, 3, -2, -2]]
    # intended to be used with 2d tensors
    indices = (x == value).argsort(dim=-1, stable=True)
    return torch.gather(x, -1, indices)

  @staticmethod
  def expand_for_batches(x, batch_size):
    nodes, mems = x.shape
    full_expands = (batch_size // mems) + 1
    residual = batch_size % mems
    return x.unsqueeze(0).expand(full_expands, -1, -1).transpose(0, 1).reshape(nodes, -1)[:, :batch_size]

  @staticmethod
  def growth_argmaxi(x, counts, eps=1e-8, threshold=0.9):
    # for now we assume batch_size is 1
    batch_size, nodes, mems = x.shape
    normal_path = SparseHopfield.argmaxi(x, eps)
    trigger_growth = torch.sum(x > threshold, dim=-1, keepdim=True) <= 0
    mark = -2
    avail, all = SparseHopfield.mark_reserved_indices(normal_path, counts, trigger_growth, mark)
    avail = SparseHopfield.move_value_to_back(avail, mark)
    avail = SparseHopfield.expand_for_batches(avail, batch_size)
    all = SparseHopfield.expand_for_batches(all, batch_size)
    final = torch.where(avail == mark, all, avail)
    indices_sg = final.transpose(-1, -2).reshape(batch_size, -1, 1)
    growth_path = torch.zeros_like(normal_path)
    values_of_interest = torch.gather(x, -1, indices_sg)
    values_of_interest[values_of_interest == 0] = 1
    # taking a closer look, it doesn't seem like the exact values from
    # the original should matter, we were just using it to divide
    # by itself (minus an eps) in the argmaxi to turn them into 1s,
    # to just setting it to 1 should be fine?
    # follow up: No, need the exact values from original. If we don't we
    # won't break the network, but we will break the gradient flows
    # and although this network doesn't use backprop, the layers that
    # occur earlier might want those gradients
    growth_path = torch.scatter(growth_path, -1, indices_sg, values_of_interest)
    growth_path = SparseHopfield.argmaxi(torch.abs(growth_path), eps)
    grown = torch.where(trigger_growth, growth_path, normal_path)

    updated_counts = einops.reduce(grown, 'batch nodes mems -> nodes mems', 'sum')
    updated_counts = torch.where(updated_counts <= 0, counts, updated_counts)

    return grown, updated_counts.detach()

  @staticmethod
  def outer_forward_parallel(mems, xs, rho=1e-8):
    batch_size, fields, dim = xs.shape

    # if mems only has 3 dimensions, then we know it is missing the batch dim,
    # this is because the shape of mems should be
    # batch_size x fields x memories x dim
    m = mems - 0.5 if mems.dim() == 4 else (mems - 0.5).expand(batch_size, -1, -1, -1)
    x = xs - 0.5
    numerator = einops.einsum(m,x, 'batch fields memories dim, batch fields dim -> batch fields memories') * 0.5
    m_norm = torch.sqrt(einops.reduce(m ** 2, 'batch fields memories dim -> batch fields memories', 'sum'))
    x_norm = torch.sqrt(einops.reduce(x ** 2, 'batch field dim -> batch field', 'sum')).unsqueeze(-1)
    denom = m_norm * x_norm + rho
    return (numerator / denom) + 0.5

  @staticmethod
  def hidden_forward_parallel(hidden_mm, children_x, rho=1e-8):
    batch_size, total_children, children_mem_cols = children_x.shape
    if hidden_mm.dim() == 4:
      num_hidden_nodes, children_per_hidden, hidden_mems, children_mem_cols = hidden_mm.shape
      mm = hidden_mm.expand(batch_size, -1, -1, -1, -1)
    else:
      batch_size, num_hidden_nodes, children_per_hidden, hidden_mems, children_mem_cols = hidden_mm.shape
      mm = hidden_mm
    x = SparseHopfield.maxi(children_x).reshape(batch_size, num_hidden_nodes, children_per_hidden, children_mem_cols)
    propagation = einops.einsum(mm, x, 'batch hidden children h_mems c_mems, batch hidden children c_mems -> batch hidden h_mems')
    x_norm = torch.sqrt(einops.reduce(x**2,'batch hidden children c_mems -> batch hidden', 'sum')).unsqueeze(-1)
    norm_coeff = 1 / ((children_per_hidden * x_norm) + rho)
    return propagation * norm_coeff

  @staticmethod
  def down_prop_parallel(parent_h, parent_mm, child_h, coeff=0.5):
    # we assume `parent_h` has already been passed through argmaxi
    # in other words down_prop_parallel(argmaxi(parent_h), ...)
    batch_size, children, children_dim = child_h.shape
    if parent_mm.dim() == 4:
      mm = parent_mm.expand(batch_size, -1, -1, -1, -1)
    else:
      mm = parent_mm
    batch_size, parent_nodes, children_per_parent, parent_dim, child_dim = mm.shape
    argmaxi_parent_h = parent_h.unsqueeze(-2).expand(-1,-1, children_per_parent, -1)
    orig = child_h * coeff
    new = (1 - coeff) * einops.einsum(argmaxi_parent_h, mm, 'batch parents children pdim, batch parents children pdim cdim -> batch parents children cdim')
    new = new.reshape(batch_size, children, -1)
    return orig + new

  @staticmethod
  def pred(parent_down_prop, parent_mem_matrix):
    mm = parent_mem_matrix if parent_mem_matrix.dim() != 3 else parent_mem_matrix.unsqueeze(-3)
    nodes, children_per_node, memories, dim = mm.shape
    batch_size, nodes, memories = parent_down_prop.shape
    prediction = einops.einsum(mm, parent_down_prop, 'nodes children_per_node memories dim, batch nodes memories -> batch nodes children_per_node dim')
    prediction = prediction.reshape(batch_size, nodes * children_per_node, dim)
    return prediction

  @staticmethod
  def mem_delta(parent_down_prop, parent_mem_matrix, child_down_prop):
    """
    Calculates the delta that should be added to the given `parent_mem_matrix`
    to optimize the model. For example, to optimize the given matrix, use
    `new_mm = old_mm + lr * mem_delta(p_prop, old_mm, c_prop)`
    Note that the `lr` scaling should be done outside of this function

    Note also that, according to the paper / reference implementation, when
    calculating the matrix update for the outer layer, the child down prop is
    just the raw inputs, not one that is one-hot encoded
    """
    # shape is nodes memories dim, but we want
    # nodes children_per_node memories dim
    # in this case, we'll assume that children_per_node is 1
    mm = parent_mem_matrix if parent_mem_matrix.dim() != 3 else parent_mem_matrix.unsqueeze(-3)
    nodes, children_per_node, memories, dim = mm.shape
    batch_size, nodes, memories = parent_down_prop.shape

    prediction = SparseHopfield.pred(parent_down_prop, mm)
    error = child_down_prop - prediction
    error = error.reshape(batch_size, nodes, children_per_node, dim)
    delta = einops.einsum(error, parent_down_prop, 'batch nodes children_per_node dim, batch nodes memories -> nodes children_per_node memories dim')
    return delta.reshape(parent_mem_matrix.shape)

  def outer_up(self, sensor_input):
    outer_mm = self.layers[0]
    h_sub_l = SparseHopfield.outer_forward_parallel(outer_mm, sensor_input, self.rho)
    return h_sub_l

  def up(self, sensor_input):
    upwards = [self.outer_up(sensor_input)]
    for i in range(1, len(self.layers)):
      mm = self.layers[i]
      h_sub_l = SparseHopfield.hidden_forward_parallel(mm, upwards[i-1], self.rho)
      upwards.append(h_sub_l)
    return upwards

  def root_down(self, upwards, eps=1e-6):
    root_h_sub_l = upwards[-1]
    root_counts = self.layer_counts[-1]
    root_h_sub_l_star, root_counts = SparseHopfield.growth_argmaxi(root_h_sub_l, root_counts, eps, self.growth_threshold)
    self.layer_counts[-1] = root_counts
    return root_h_sub_l_star

  def down(self, upwards, eps=1e-6, coeff=0.5):
    downwards = [self.root_down(upwards, eps)]
    for i in range(len(upwards)-2, -1, -1):
      h_sub_l = upwards[i]
      counts = self.layer_counts[i]
      downed = SparseHopfield.down_prop_parallel(downwards[-1],self.layers[i+1],h_sub_l,coeff)
      h_sub_l_star, counts = SparseHopfield.growth_argmaxi(downed, counts, eps, self.growth_threshold)
      self.layer_counts[i] = counts
      downwards.append(h_sub_l_star)
    return downwards

  def pred_root_down(self, upwards, eps=1e-6):
    root_h_sub_l = upwards[-1]
    root_counts = self.layer_counts[-1]
    root_h_sub_l_star = SparseHopfield.argmaxi(root_h_sub_l, eps)
    return root_h_sub_l_star

  def pred_down(self, upwards, eps=1e-6, coeff=0.5):
    downwards = [self.pred_root_down(upwards, eps)]
    for i in range(len(upwards)-2, -1, -1):
      h_sub_l = upwards[i]
      counts = self.layer_counts[i]
      downed = SparseHopfield.down_prop_parallel(downwards[-1],self.layers[i+1],h_sub_l,coeff)
      h_sub_l_star = SparseHopfield.argmaxi(downed, eps)
      downwards.append(h_sub_l_star)
    return downwards

  def delta_outer(self, downwards, sensory_input):
    outer_mm = self.layers[0]
    outer_h_sub_l_star = downwards[-1]
    outer_delta = SparseHopfield.mem_delta(outer_h_sub_l_star, outer_mm, sensory_input)
    return outer_delta

  def delta(self, downwards, sensory_input):
    deltas = [self.delta_outer(downwards, sensory_input)]
    for i in range(1, len(downwards)):
      child_h_sub_l_star = downwards[len(downwards) - i]
      h_sub_l_star = downwards[len(downwards) - i - 1]
      mm = self.layers[i]
      delta = SparseHopfield.mem_delta(h_sub_l_star, mm, child_h_sub_l_star)
      deltas.append(delta)
    return deltas

  def optim_outer(self, deltas):
    outer_delta = deltas[0]
    outer_mm = self.layers[0]
    outer_counts = self.layer_counts[0]
    delta = outer_delta / (outer_counts).unsqueeze(-1)
    self.layers[0] = outer_mm + delta

  def optim(self, deltas):
    self.optim_outer(deltas)
    for i in range(1, len(deltas)):
      delta = deltas[i]
      mm = self.layers[i]
      count = self.layer_counts[i]
      delta = delta / (count).unsqueeze(-1).unsqueeze(-3)
      self.layers[i] = mm + delta
    self.update_iteration()

  def optimize(self, sensory_input, eps=1e-6, coeff=0.5):
    upwards = self.up(sensory_input)
    downwards = self.down(upwards, eps, coeff)
    deltas = self.delta(downwards, sensory_input)
    self.optim(deltas)

  def predict(self, sensory_input, eps=1e-6, coeff=0.5):
    with torch.no_grad():
      upwards = self.up(sensory_input)
      downwards = self.pred_down(upwards, eps, coeff)
      prediction = SparseHopfield.pred(downwards[-1],self.layers[0])
      return prediction

  def prediction_error(self, sensory_input, eps=1e-6, coeff=0.5):
    with torch.no_grad():
      pred = self.predict(sensory_input, eps, coeff)
      return torch.mean(torch.square(pred - sensory_input))
      
def train(net, data):
    net.optimize(data)
    # batch_size, fields, dim = data.shape
    # for i in range(batch_size):
    #     net.optimize(data[i].unsqueeze(0))

def loss(net, data):
    batch_size, fields, dim = data.shape
    total_loss = 0
    errors = []
    for i in range(batch_size):
        err = net.prediction_error(data[i].unsqueeze(0))
        errors.append(err)
        total_loss += err
    return total_loss / batch_size, errors