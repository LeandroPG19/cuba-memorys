import asyncio
import sys

from cuba_memorys.server import main as server_main

def main() -> None:
    try:
        asyncio.run(server_main())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
