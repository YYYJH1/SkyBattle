#!/usr/bin/env python3
"""Server entry point for SkyBattle."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="SkyBattle Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting SkyBattle Server at http://{args.host}:{args.port}")
    print(f"📖 API docs at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "backend.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
