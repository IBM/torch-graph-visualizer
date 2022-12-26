import argparse
import torch
import torch.jit

from torchvision.models.vgg import vgg11

from torch_graph_visualizer.profile import parse_ncu_file
from torch_graph_visualizer import (
    attr,
    default_draw_model,
    run_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["profile", "draw"],
        required=True,
        help="mode of execution"
    )
    parser.add_argument(
        "--jit",
        choices=["none", "script", "trace", "aot"],
        default="none",
        help="activate JIT for the model",
    )
    parser.add_argument(
        "-i",
        "--attr-gen",
        choices=["none", "bottleneck", "duration"],
        default="none",
        help="information to be displayed"
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="profile of the module"
    )
    parser.add_argument(
        "-f",
        "--fuser",
        choices=["none", "fuser1", "fuser2", "fuser3", "fuser4", "fuser5"],
        default="none",
        help="fuser to be used"
    )
    args = parser.parse_args()

    dev = "cuda"
    model = vgg11()
    model = model.to(dev)
    model.train()

    input_data = [
        (torch.randn(16, 3, 64, 64, device=dev),)
    ]

    def factory(max_duration: int) -> attr.AttributeGenerator:
        if args.attr_gen == "none":
            return attr.Nop()
        elif args.attr_gen == "bottleneck":
            return attr.MemoryAndCompute()
        elif args.attr_gen == "duration":
            return attr.Duration(max_duration)
        raise ValueError(f"invalid attribute generator: {args.attr_gen}")

    if args.mode == "profile":
        run_model(model, input_data, fuser=args.fuser, jit=args.jit, warmup_steps=3, number=10)
    elif args.mode == "draw":
        kernels = parse_ncu_file(args.profile)[0].kernels
        default_draw_model(
            model=model,
            input_data=input_data,
            profiled_kernels=kernels,
            attribute_generator_fn=factory,
            fuser=args.fuser,
            jit=args.jit,
            profiled_run_number=7
        )


if __name__ == "__main__":
    main()
