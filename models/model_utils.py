
from models.armnet_1h import ARMNetModel as ARMNet1H

def create_model(args):
        model = ARMNet1H(args.nfeat, args.nfield, args.nemb, args.alpha, args.h, args.nemb, args.mlp_nlayer,
                     args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)

        return model