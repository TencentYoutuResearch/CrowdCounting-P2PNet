from .models.p2pnet import build


# build the P2PNet model
# set training to 'True' during training
def build_model(args, training: bool = False):
    """Build P2PNet model."""
    return build(args, training)
