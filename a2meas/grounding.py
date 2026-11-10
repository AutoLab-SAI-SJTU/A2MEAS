"""Shared SAM multiplex detector from the pinned public SAM package."""

from pathlib import Path


def load_grounder(checkpoint, device, *, threshold):
    import torch
    from sam3 import model_builder as b
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = Path(b.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
    backbone = b.SAM3VLBackboneTri(
        scalp=0,
        visual=b._create_multiplex_tri_backbone(),
        text=b._create_text_encoder(str(bpe)),
    )
    model = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=b._create_sam3_transformer(),
        segmentation_head=b._create_segmentation_head(),
        semantic_segmentation_head=None,
        input_geometry_encoder=b._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=b._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "model" in saved:
        saved = saved["model"]
    state = {
        k.removeprefix("detector."): v
        for k, v in saved.items()
        if k.startswith("detector.")
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    processor = Sam3Processor(
        model, resolution=1008, device=device, confidence_threshold=threshold
    )
    return model, processor
