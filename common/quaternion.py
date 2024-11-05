import torch

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4, "Quaternion input q should have shape (*, 4)"
    assert v.shape[-1] == 3, "Vector input v should have shape (*, 3)"
    assert q.shape[:-1] == v.shape[:-1], "Shapes of q and v should match except for the last dimension"

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinverse(q, inplace=False):
    """
    Returns the inverse of quaternion q.
    Assumes the quaternion to be normalized.
    If inplace=True, the operation is done in-place, modifying the input tensor.
    """
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)
