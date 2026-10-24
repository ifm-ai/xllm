"""
Fault injection for the in-band fault-tolerance (ft) tests.

Publishes one injection request (xllm/ft/inject.py) to the ft control store; the
per-node agent that the request targets carries it out.

Usage:
    python tests/ft_inject.py --store HOST:PORT --kill-rank 3 --at-step 20
"""

import argparse
import datetime
import logging
from typing import List, Optional

from xllm.ft.inject import KINDS, Injection, publish


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="control store HOST:PORT")
    parser.add_argument("--at-step", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kill-rank", type=int, metavar="R")
    group.add_argument("--kill-process", type=int, metavar="N")
    group.add_argument("--kill-node", type=int, metavar="N")
    group.add_argument("--hang-rank", type=int, metavar="R")
    group.add_argument("--torn-write", type=int, metavar="R")
    group.add_argument("--bitflip", type=int, metavar="R")
    group.add_argument("--xid", type=int, nargs=2, metavar=("N", "XID"))
    group.add_argument("--gpu-lost", type=int, metavar="N")
    group.add_argument("--fail-spare-health", type=int, metavar="N")
    group.add_argument("--dump-kill", type=int, metavar="N")
    group.add_argument("--lease-cut", type=int, metavar="N")
    return parser.parse_args(argv)


def injection_from_args(args) -> Injection:
    for kind, target in KINDS.items():
        value = getattr(args, kind)
        if value is None:
            continue
        if kind == "xid":
            return Injection(kind, node=value[0], xid=value[1], step=args.at_step, gpu=args.gpu)
        return Injection(kind, step=args.at_step, gpu=args.gpu, **{target: value})
    raise ValueError("no injection given")


def main(argv: Optional[List[str]] = None):
    from torch.distributed import TCPStore

    args = parse_args(argv)
    host, port = args.store.rsplit(":", 1)
    store = TCPStore(host, int(port), is_master=False, timeout=datetime.timedelta(seconds=60))
    publish(store, injection_from_args(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
