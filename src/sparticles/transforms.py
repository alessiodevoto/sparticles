from torch_geometric.transforms import BaseTransform
import torch
import warnings

class ParticlesTransform(BaseTransform):

    def __init__(self):
        super(ParticlesTransform, self).__init__()

    def forward(self, data):
        pass 

class MakeHomogeneous(BaseTransform):
    """
    Transform that makes the input tensor homogeneous by:
        - replacing NaN values with `nan_to_num` value
        - adding a feature to the tensor that indicates if the node was NaN or not. The feature will be 1 if the node was NaN, 0 otherwise.

    Args:
        nan_to_num (int): Value to replace NaN values with. Default is 0.
    
        
    Examples:
        
        
        g = torch_geometric.data.Data(x=torch.tensor([[torch.nan, -10]]))
        print(g.x)
        >>> tensor([[ nan, -10.]])

        g = MakeHomogeneous(nan_to_num=3)(g)
        print(g.x)

        >>> tensor([[  3.,   1., -10.,   0.]])
    """

    def __init__(self, nan_to_num: int=0):
        self.nan_to_num = nan_to_num
        super(MakeHomogeneous, self).__init__()

    def __call__(self, data):
        """
        Forward pass of the transform.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch_geometric.data.Data: The transformed data.
        """
        # mask of nan nodes
        nan_nodes = torch.isnan(data.x).type(data.x.dtype)

        # concat and replace nan nodes with zeros
        new_nodes = torch.cat((data.x.unsqueeze(2), nan_nodes.unsqueeze(2)), dim=2).reshape(data.x.shape[0], -1).nan_to_num_(self.nan_to_num)
        data.x = new_nodes.float()
        return data


class RevertMakeHomogeneous(BaseTransform):
    """
    Transform that reverts the MakeHomogeneous transform by removing the feature that indicates if the node was NaN or not.

    """

    def __init__(self):
        super(RevertMakeHomogeneous, self).__init__()
        warnings.warn("Warning: RevertMakeHomogeneous does not revert the NaNs. It only removes the feature that indicates if the node was NaN or not.")

        

    def __call__(self, data):
        """
        Forward pass of the transform.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch_geometric.data.Data: The transformed data.
        """
        # even columns mask
        mask = torch.tensor([i for i in range(0, data.x.shape[1], 2)])
        mask = torch.tensor([0,2,4,6,8,10])

        # Use boolean indexing to select only even columns
        data.x = data.x[:, mask]

        return data