if __name__ == "__main__":
    try:
        import sys
        from .cli import CLI
        import fire
        fire.Fire(CLI)
    except ImportError as e:
        print(e.name, " - Missing dependency, run uv sync", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Program interrupted by the user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
