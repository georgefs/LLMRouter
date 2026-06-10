#!/usr/bin/env python3
"""啟動 LLMRouter Endpoint Server

用法：
  python3 start_endpoint.py --router test_data/test_router.pkl --port 8888
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LLMRouter.endpoint import LLMRouterEndpointServer


def main():
    parser = argparse.ArgumentParser(description="Start LLMRouter Endpoint Server")
    parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Path to saved router (.pkl file)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Server port (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    router_path = Path(args.router)
    if not router_path.exists():
        print(f"❌ Router file not found: {router_path}")
        sys.exit(1)

    print("=" * 60)
    print("Starting LLMRouter Endpoint Server")
    print("=" * 60)
    print(f"Router: {router_path}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"URL: http://localhost:{args.port}")
    print("=" * 60)
    print("\nEndpoints:")
    print(f"  POST http://localhost:{args.port}/route")
    print(f"  GET  http://localhost:{args.port}/health")
    print(f"  GET  http://localhost:{args.port}/models")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        server = LLMRouterEndpointServer(
            router_path=router_path,
            port=args.port,
            host=args.host,
        )
        server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
