from .edcoder import PreModel, TransferModel


def build_model(args, spot_num=0):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_gene_rate = args.mask_gene_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    remask_rate = args.remask_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    beta_l = args.beta_l
    num_features = args.num_features
    warm_up = args.warm_up
    K = args.num_classes


    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_gene_rate=mask_gene_rate,
        remask_rate = remask_rate,
        spot_num = spot_num,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        warm_up = warm_up,
        alpha_l=alpha_l,
        beta_l = beta_l,
        K=K,
    )
    return model

def build_Transfer_model(args, num_classes, spot_num=0):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_gene_rate = args.mask_gene_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    remask_rate = args.remask_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    beta_l = args.beta_l
    num_features = args.num_features
    warm_up = args.warm_up
    K = num_classes
    balance_class = args.balance_class


    model = TransferModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_gene_rate=mask_gene_rate,
        remask_rate = remask_rate,
        spot_num = spot_num,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        warm_up = warm_up,
        alpha_l=alpha_l,
        beta_l = beta_l,
        K=K,
        balance_class = balance_class
    )
    return model