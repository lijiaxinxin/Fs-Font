import argparse
import os.path

from sconf import Config
import json
from evaluator import *
from transform import setup_transforms
from datasets import load_json, get_fixedref_loader
from models import generator_dispatch


def get_test_dict_from_train_json(train_json_path, seen=True):
    with open(train_json_path, 'r') as f:
        original_meta = json.load(f)['valid']

    if seen:
        gen_fonts = original_meta['seen_fonts']
    else:
        gen_fonts = original_meta['unseen_fonts']

    test_dict = {
        "gen_fonts": gen_fonts,
        "gen_unis": original_meta['unseen_unis'],
        "ref_unis": original_meta['seen_unis']
    }
    return test_dict


def eval_ckpt(args, cfg, test_dict):
    logger = Logger.get()
    content_name = args.content_name
    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    test_meta = test_dict

    g_kwargs = cfg.get('g_args', {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg['C'], 1, cfg, **g_kwargs)
    if cfg.use_half:
        gen.half()
    gen.to('gpu')

    weight = paddle.load(args.weight)

    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.set_state_dict(weight)
    logger.info(f"load checkpoint from {args.weight}")
    writer = None

    evaluator = Evaluator(
        env,
        env_get,
        cfg,
        logger,
        writer,
        cfg["batch_size"],
        val_transform,
        content_name,
        use_half=cfg.use_half
    )

    img_dir = Path(args.saving_root)
    ref_unis = test_meta["ref_unis"]
    gen_unis = test_meta["gen_unis"]
    gen_fonts = test_meta["gen_fonts"]
    target_dict = {f: gen_unis for f in gen_fonts}
    loader = get_fixedref_loader(env=env,
                                 env_get=env_get,
                                 target_dict=target_dict,
                                 ref_unis=ref_unis,
                                 cfg=cfg,
                                 transform=val_transform,
                                 num_workers=cfg.n_workers,
                                 shuffle=False
                                 )[1]

    logger.info("Save CV results to {} ...".format(img_dir))
    # saving_folder = evaluator.save_each_imgs(gen, loader, args.img_path, save_dir=args.saving_root, reduction='mean')
    evaluator.save_imgs(gen, loader, save_dir=args.saving_root, reduction='mean')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate")
    parser.add_argument("--content_name", default='id_000_simsun.ttc', help="name to content font")
    # parser.add_argument("--img_path", help="path of the your test img directory.")
    parser.add_argument("--saving_root", help="saving directory.")

    args = parser.parse_args()
    cfg = Config(*args.config_paths, default="./cfgs/defaults.yaml")

    if args.saving_root is None:
        args.saving_root = os.path.join(os.path.dirname(args.weight), 'samples', os.path.basename(args.weight))

    with open(cfg.content_reference_json, 'r') as f:
        cr_mapping = json.load(f)

    test_dict = get_test_dict_from_train_json(cfg.data_meta)
    eval_ckpt(args, cfg, test_dict)
